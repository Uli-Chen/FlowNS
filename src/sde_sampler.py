import math

import torch
import logging

logger = logging.getLogger(__name__)


class SDESampler:
    """ODE-to-SDE conversion + Euler-Maruyama sampling for flow generation.

    dx_t = [v_θ + (σ_t²/2)·∇log p_t] dt + σ_t dw
    Tweedie score: ∇log p_t = -(x_t - t·E[x_1|x_t]) / (1-t)². Under the linear
    CFM path, E[x_1|x_t] = x_t + (1-t)·v_θ, which simplifies to
    ∇log p_t ≈ -(x_t - t·v_θ) / (1-t).
    Noise schedule: σ_t = η · √(1-t) / (√t + δ)

    Discretization uses the exact per-step average of σ_s² over [t, t+Δt]
    (closed form below) instead of the left-endpoint value: σ_0 = η/δ is huge
    (50 with the defaults) and σ_0²·Δt would inject ~125 units of variance in
    the first Euler-Maruyama step, even though ∫σ_s²ds over that step is O(1).
    """

    def __init__(self, velocity_net, n_steps=20, eta=0.5, delta=0.01,
                 sigma_min=0.0, score_clamp=None, guidance_scale=1.0):
        """
        Args:
            sigma_min: floor on σ_t, which vanishes as t→1. A small positive floor
                (e.g. 0.05) bounds the late-step transition variance. Default 0.0
                keeps the original (unfloored) behavior.
            score_clamp: optional per-coordinate magnitude clamp on the Tweedie score
                term -(x - t·v)/(1-t), which is magnified ~1/(1-t) near t→1.
                Default None disables clamping.
            guidance_scale: classifier-free guidance weight w. 1.0 (default) is the
                plain conditional field (single forward pass, no CFG). w != 1.0
                extrapolates v = v_uncond + w·(v_cond - v_uncond); w > 1 sharpens
                the user conditioning, w = 0 is unconditional. Requires the flow to
                have been trained with cfg_dropout_prob > 0.
        """
        self.velocity_net = velocity_net
        self.n_steps = n_steps
        self.eta = eta
        self.delta = delta
        self.dt = 1.0 / n_steps
        self.sigma_min = float(sigma_min)
        self.score_clamp = None if score_clamp is None else float(score_clamp)
        self.score_eps = 1e-6
        self.guidance_scale = float(guidance_scale)

    def noise_schedule(self, t):
        """σ_t = max(η · √(1-t) / (√t + δ), σ_min)"""
        sigma = self.eta * torch.sqrt(1 - t) / (torch.sqrt(t) + self.delta)
        if self.sigma_min > 0.0:
            sigma = sigma.clamp_min(self.sigma_min)
        return sigma

    def _sigma_sq_integral(self, a, b):
        """Exact ∫_a^b σ_s² ds = η²·[F(√b) - F(√a)] for the η,δ schedule.

        With u = √s, w = u + δ:
        F(u) = -w² + 6δw + 2(1-3δ²)·ln(w) + 2(δ-δ³)/w.
        Pinned against numeric quadrature in tests/test_p0_fixes.py.
        """
        d = self.delta

        def F(u):
            w = u + d
            return (
                -w * w + 6.0 * d * w
                + 2.0 * (1.0 - 3.0 * d * d) * math.log(w)
                + 2.0 * (d - d ** 3) / w
            )

        return (self.eta ** 2) * (F(math.sqrt(b)) - F(math.sqrt(a)))

    def step_sigma(self, step):
        """√ of the average σ_s² over the Euler-Maruyama step [t, t+Δt]."""
        t0 = step * self.dt
        t1 = min(t0 + self.dt, 1.0)
        var = self._sigma_sq_integral(t0, t1) / self.dt
        sigma = math.sqrt(max(var, 0.0))
        if self.sigma_min > 0.0:
            sigma = max(sigma, self.sigma_min)
        return sigma

    def _guided_velocity(self, x, user_emb, t):
        """Velocity field, optionally sharpened by classifier-free guidance.

        With guidance_scale == 1.0 this is the plain conditional field
        v_θ(x, u, t): one forward pass, identical to the non-CFG path (so a
        velocity_net whose forward does not accept cond_drop_mask still works).
        Otherwise we also evaluate the unconditional field v_θ(x, ∅, t) — the
        condition replaced by the learned null embedding — and extrapolate:
            v = v_uncond + scale·(v_cond - v_uncond).
        scale > 1 pushes samples further onto the user-conditional manifold;
        scale = 0 recovers the unconditional field.
        """
        v_cond = self.velocity_net(x, user_emb, t)
        if self.guidance_scale == 1.0:
            return v_cond
        drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        v_uncond = self.velocity_net(x, user_emb, t, cond_drop_mask=drop)
        return v_uncond + self.guidance_scale * (v_cond - v_uncond)

    def _tweedie_score(self, x, v, t_val):
        """∇log p_t ≈ -(x - t·v)/(1-t), floored and optionally magnitude-clamped.

        Tweedie gives ∇log p_t = -(x - t·E[x_1|x])/(1-t)². With the linear CFM
        path, v = E[x_1 - x_0|x] implies E[x_1|x] = x + (1-t)·v, so the
        numerator carries a (1-t) factor and the denominator is first order:
        x - t·E[x_1|x] = (1-t)·(x - t·v). At t=0 this is just -x, the standard
        normal score, so no special case is needed. Verified against the
        closed-form Gaussian score in tests/test_p0_fixes.py.

        t_val is a python float.
        """
        score = -(x - t_val * v) / max(1 - t_val, self.score_eps)
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
            sigma_t = self.step_sigma(step)

            v = self._guided_velocity(x, u_exp, t)

            # Tweedie score approximation (uses the guided velocity so the score
            # term stays consistent with the drift when guidance is active).
            score = self._tweedie_score(x, v, t_val)

            drift = v + 0.5 * sigma_t ** 2 * score
            eps = torch.randn_like(x)
            x = x + drift * self.dt + sigma_t * (self.dt ** 0.5) * eps

            trajectories.append(x.reshape(B, G, d).clone())
            noises.append(eps.reshape(B, G, d))

        final_emb = x.reshape(B, G, d)
        return final_emb, trajectories, noises
