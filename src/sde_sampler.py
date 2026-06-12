import torch
import logging

logger = logging.getLogger(__name__)


class SDESampler:
    """ODE-to-SDE conversion + Euler-Maruyama sampling for flow model.

    dx_t = [v_θ + (σ_t²/2)·∇log p_t] dt + σ_t dw
    Tweedie score: ∇log p_t ≈ -(x_t - t·v_θ) / (1-t)²
    Noise schedule: σ_t = η · √(1-t) / (√t + δ)
    """

    def __init__(self, velocity_net, n_steps=20, eta=0.5, delta=0.01,
                 sigma_min=0.0, score_clamp=None):
        """
        Args:
            sigma_min: floor on σ_t. The closed-form log-ratio and KL both divide by
                2·σ_t²·Δt, which vanishes as t→1 and explodes any policy drift. A small
                positive floor (e.g. 0.05) bounds that denominator. Default 0.0 keeps the
                original (unfloored) behavior.
            score_clamp: optional per-coordinate magnitude clamp on the Tweedie score
                term -(x - t·v)/(1-t)², which is magnified ~1/(1-t)² near t→1 and feeds
                both the drift and the KL. Default None disables clamping.
        """
        self.velocity_net = velocity_net
        self.n_steps = n_steps
        self.eta = eta
        self.delta = delta
        self.dt = 1.0 / n_steps
        self.sigma_min = float(sigma_min)
        self.score_clamp = None if score_clamp is None else float(score_clamp)
        self.score_eps = 1e-6

    def noise_schedule(self, t):
        """σ_t = max(η · √(1-t) / (√t + δ), σ_min)"""
        sigma = self.eta * torch.sqrt(1 - t) / (torch.sqrt(t) + self.delta)
        if self.sigma_min > 0.0:
            sigma = sigma.clamp_min(self.sigma_min)
        return sigma

    def _tweedie_score(self, x, v, t_val):
        """∇log p_t ≈ -(x - t·v)/(1-t)², floored and optionally magnitude-clamped.

        Keeps gradients flowing through v. t_val is a python float.
        """
        if t_val <= 1e-6:
            return torch.zeros_like(x)
        score = -(x - t_val * v) / max((1 - t_val) ** 2, self.score_eps)
        if self.score_clamp is not None:
            score = score.clamp(-self.score_clamp, self.score_clamp)
        return score

    @torch.no_grad()
    def sample_trajectories(self, user_emb, n_trajectories=8):
        """Sample SDE trajectories via Euler-Maruyama.

        Args:
            user_emb: (B, d) user embeddings
            n_trajectories: G, number of trajectories per user
        Returns:
            final_emb: (B, G, d) final generated embeddings
            trajectories: list of (B, G, d) at each step (length n_steps+1)
            noises: list of (B, G, d) noise vectors at each step
        """
        B, d = user_emb.shape
        G = n_trajectories

        u_exp = user_emb.unsqueeze(1).expand(B, G, d).reshape(B * G, d)

        x = torch.randn(B * G, d, device=user_emb.device)
        trajectories = [x.reshape(B, G, d).clone()]
        noises = []

        for step in range(self.n_steps):
            t_val = step * self.dt
            t = torch.full((B * G,), t_val, device=user_emb.device)
            sigma_t = self.noise_schedule(t[0:1]).item()

            v = self.velocity_net(x, u_exp, t)

            # Tweedie score approximation
            score = self._tweedie_score(x, v, t_val)

            drift = v + 0.5 * sigma_t ** 2 * score
            eps = torch.randn_like(x)
            x = x + drift * self.dt + sigma_t * (self.dt ** 0.5) * eps

            trajectories.append(x.reshape(B, G, d).clone())
            noises.append(eps.reshape(B, G, d))

        final_emb = x.reshape(B, G, d)
        return final_emb, trajectories, noises

    def sample_trajectories_with_grad(self, user_emb, n_trajectories=8):
        """Same as sample_trajectories but keeps grad for GRPO."""
        B, d = user_emb.shape
        G = n_trajectories

        u_exp = user_emb.unsqueeze(1).expand(B, G, d).reshape(B * G, d)

        x = torch.randn(B * G, d, device=user_emb.device)
        trajectories = [x.reshape(B, G, d)]
        noises = []

        for step in range(self.n_steps):
            t_val = step * self.dt
            t = torch.full((B * G,), t_val, device=user_emb.device)
            sigma_t = self.noise_schedule(t[0:1]).item()

            v = self.velocity_net(x, u_exp, t)

            score = self._tweedie_score(x, v, t_val)

            drift = v + 0.5 * sigma_t ** 2 * score
            eps = torch.randn_like(x)
            x = x + drift * self.dt + sigma_t * (self.dt ** 0.5) * eps

            trajectories.append(x.reshape(B, G, d))
            noises.append(eps.reshape(B, G, d))

        final_emb = x.reshape(B, G, d)
        return final_emb, trajectories, noises

    def compute_log_ratio(self, trajectories, noises, velocity_net_old, user_emb):
        """Compute closed-form log importance ratio per step.

        log r_t = (||x_{t+1} - x_t - ṽ_old·Δt||² - ||x_{t+1} - x_t - ṽ_new·Δt||²)
                  / (2·σ_t²·Δt)

        Args:
            trajectories: list of (B, G, d), length n_steps+1
            noises: list of (B, G, d), length n_steps
            velocity_net_old: old velocity net (before GRPO update)
            user_emb: (B, d)
        Returns:
            log_ratios: (B, G, T) per-step log importance ratios
        """
        B, G, d = trajectories[0].shape
        T = self.n_steps
        u_exp = user_emb.unsqueeze(1).expand(B, G, d).reshape(B * G, d)

        log_ratios = []
        for step in range(T):
            t_val = step * self.dt
            t = torch.full((B * G,), t_val, device=user_emb.device)
            sigma_t = self.noise_schedule(t[0:1]).item()

            x_t = trajectories[step].reshape(B * G, d)
            x_next = trajectories[step + 1].reshape(B * G, d)
            dx = x_next - x_t

            with torch.no_grad():
                v_old = velocity_net_old(x_t, u_exp, t)
                score_old = self._tweedie_score(x_t, v_old, t_val)
                drift_old = v_old + 0.5 * sigma_t ** 2 * score_old

            v_new = self.velocity_net(x_t, u_exp, t)
            score_new = self._tweedie_score(x_t, v_new, t_val)
            drift_new = v_new + 0.5 * sigma_t ** 2 * score_new

            diff_old = (dx - drift_old * self.dt).pow(2).sum(dim=-1)
            diff_new = (dx - drift_new * self.dt).pow(2).sum(dim=-1)

            denom = 2 * max(sigma_t ** 2, 1e-8) * self.dt
            log_r = (diff_old - diff_new) / denom  # (B*G,)
            log_ratios.append(log_r.reshape(B, G))

        return torch.stack(log_ratios, dim=-1)  # (B, G, T)

    def compute_per_step_kl(self, trajectories, velocity_net_ref, user_emb):
        """Per-step KL: D_t = ||ṽ_θ - ṽ_ref||²·Δt / (2·σ_t²).

        Args:
            trajectories: list of (B, G, d)
            velocity_net_ref: reference velocity net
            user_emb: (B, d)
        Returns:
            kl_per_step: (B, G, T)
        """
        B, G, d = trajectories[0].shape
        T = self.n_steps
        u_exp = user_emb.unsqueeze(1).expand(B, G, d).reshape(B * G, d)

        kl_steps = []
        for step in range(T):
            t_val = step * self.dt
            t = torch.full((B * G,), t_val, device=user_emb.device)
            sigma_t = self.noise_schedule(t[0:1]).item()

            x_t = trajectories[step].reshape(B * G, d)

            v_theta = self.velocity_net(x_t, u_exp, t)
            with torch.no_grad():
                v_ref = velocity_net_ref(x_t, u_exp, t)

            # Full drift including score term
            score_theta = self._tweedie_score(x_t, v_theta, t_val)
            score_ref = self._tweedie_score(x_t.detach(), v_ref, t_val)

            drift_theta = v_theta + 0.5 * sigma_t ** 2 * score_theta
            drift_ref = v_ref + 0.5 * sigma_t ** 2 * score_ref

            kl_t = (drift_theta - drift_ref).pow(2).sum(dim=-1) * self.dt / (
                2 * max(sigma_t ** 2, 1e-8)
            )
            kl_steps.append(kl_t.reshape(B, G))

        return torch.stack(kl_steps, dim=-1)  # (B, G, T)
