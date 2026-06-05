import copy
import torch
import torch.optim as optim
import logging

from .sde_sampler import SDESampler
from .reward import BoundaryAwareReward
from .neg_sampling import EmbeddingToItemMapper

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Group Relative Policy Optimization for flow-based negative sampling.

    GRPO objective (per user, per step):
      J = (1/G)(1/T) Σ_i Σ_t [ min(r_t·Â, clip(r_t)·Â) - β·D_t ]

    where:
      r_t: importance ratio (closed-form Gaussian)
      Â: trajectory-level advantage (R - mean) / std
      D_t: per-step KL divergence to reference policy
    """

    def __init__(self, flow_model, sde_sampler, reward_fn, mapper,
                 group_size=8, clip_eps=0.2, beta=0.1, lr=1e-4,
                 max_pos_samples=10):
        self.flow_model = flow_model
        self.sde_sampler = sde_sampler
        self.reward_fn = reward_fn
        self.mapper = mapper
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.beta = beta
        self.lr = lr
        self.max_pos_samples = max_pos_samples
        self.optimizer = optim.Adam(
            flow_model.velocity_net.parameters(), lr=lr
        )
        self._ref_net = None
        self._old_state = None

    def _ensure_ref_net(self):
        if self._ref_net is None:
            self._ref_net = self.flow_model.create_ref_net()

    def _save_old_policy(self):
        self._old_state = copy.deepcopy(
            self.flow_model.velocity_net.state_dict()
        )

    def _get_old_net(self):
        from .flow_model import ConditionalVelocityNet
        net = ConditionalVelocityNet(
            self.flow_model.emb_dim,
            hidden_dim=self.flow_model._hidden_dim,
            n_layers=self.flow_model._n_layers,
        ).to(self.flow_model.device)
        net.load_state_dict(self._old_state)
        net.eval()
        return net

    def _get_pos_embs(self, user_ids, item_emb_all, user_pos_items):
        """Get positive item embeddings for a batch of users.

        Args:
            user_ids: list of user IDs
            item_emb_all: (n_items, d)
            user_pos_items: dict {uid: set(item_ids)}
        Returns:
            pos_embs: (B, K, d) padded positive embeddings
            K: number of positive samples per user (capped)
        """
        K = self.max_pos_samples
        B = len(user_ids)
        d = item_emb_all.shape[1]
        device = item_emb_all.device

        pos_embs = torch.zeros(B, K, d, device=device)
        for i, uid in enumerate(user_ids):
            pos_ids = list(user_pos_items.get(uid, []))
            if not pos_ids:
                continue
            if len(pos_ids) > K:
                indices = torch.randperm(len(pos_ids))[:K]
                pos_ids = [pos_ids[j] for j in indices]
            pos_embs[i, :len(pos_ids)] = item_emb_all[pos_ids]

        return pos_embs

    def grpo_step(self, user_emb, user_ids, item_emb_all, user_pos_items):
        """Single GRPO update step.

        Args:
            user_emb: (B, d) user embeddings
            user_ids: list of user IDs
            item_emb_all: (n_items, d)
            user_pos_items: dict {uid: set}
        Returns:
            loss: scalar loss value
            stats: dict of training statistics
        """
        B, d = user_emb.shape
        G = self.group_size
        self._ensure_ref_net()

        # 1. Sample G trajectories per user
        with torch.no_grad():
            final_emb, trajectories, noises = self.sde_sampler.sample_trajectories(
                user_emb, n_trajectories=G
            )

        # 2. Compute rewards for each trajectory
        pos_embs = self._get_pos_embs(user_ids, item_emb_all, user_pos_items)
        # final_emb: (B, G, d), pos_embs: (B, K, d)
        rewards = []
        win_rates = []
        for g in range(G):
            gen_g = final_emb[:, g, :]  # (B, d)
            W = self.reward_fn.win_rate(user_emb, gen_g, pos_embs)
            R = self.reward_fn.compute_reward(W)
            rewards.append(R)
            win_rates.append(W)

        rewards = torch.stack(rewards, dim=1)  # (B, G)
        win_rates = torch.stack(win_rates, dim=1)  # (B, G)

        # 3. Compute group advantages: Â = (R - mean) / (std + ε)
        r_mean = rewards.mean(dim=1, keepdim=True)
        r_std = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - r_mean) / (r_std + 1e-8)  # (B, G)

        # 4. Compute importance ratios (closed-form)
        old_net = self._get_old_net()
        log_ratios = self.sde_sampler.compute_log_ratio(
            trajectories, noises, old_net, user_emb
        )  # (B, G, T)

        # Ratio normalization per step
        log_r_mean = log_ratios.mean(dim=1, keepdim=True)
        log_r_std = log_ratios.std(dim=1, keepdim=True)
        log_ratios_norm = (log_ratios - log_r_mean) / (log_r_std + 1e-8)
        ratios = torch.exp(log_ratios_norm)  # (B, G, T)

        # 5. Per-step KL
        kl_per_step = self.sde_sampler.compute_per_step_kl(
            trajectories, self._ref_net, user_emb
        )  # (B, G, T)

        # 6. Clipped surrogate objective
        adv_expanded = advantages.unsqueeze(-1)  # (B, G, 1)
        surr1 = ratios * adv_expanded  # (B, G, T)
        surr2 = torch.clamp(
            ratios, 1 - self.clip_eps, 1 + self.clip_eps
        ) * adv_expanded
        surrogate = torch.min(surr1, surr2)

        # J = mean over (B, G, T) of [surrogate - β·KL]
        objective = (surrogate - self.beta * kl_per_step).mean()
        loss = -objective

        # 7. Backward + step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.flow_model.velocity_net.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        stats = {
            'loss': loss.item(),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'win_rate_mean': win_rates.mean().item(),
            'kl_mean': kl_per_step.mean().item(),
            'ratio_mean': ratios.mean().item(),
        }
        return loss.item(), stats

    def train(self, user_emb_all, item_emb_all, user_pos_items,
              epochs=10, batch_size=64):
        """Phase 2: Full GRPO training loop.

        Args:
            user_emb_all: (n_users, d) — detached, used for conditioning
            item_emb_all: (n_items, d) — detached, used for reward/mapping
            user_pos_items: dict {uid: set(item_ids)}
            epochs: GRPO epochs
            batch_size: users per batch
        """
        self.flow_model.velocity_net.train()
        user_ids = [uid for uid in user_pos_items if uid < user_emb_all.shape[0]]

        for epoch in range(epochs):
            self._save_old_policy()
            total_loss = 0.0
            total_stats = {}
            n_batches = 0

            indices = torch.randperm(len(user_ids))
            for start in range(0, len(user_ids), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_uids = [user_ids[i] for i in batch_idx]

                u_emb = user_emb_all[batch_uids]
                loss, stats = self.grpo_step(
                    u_emb, batch_uids, item_emb_all, user_pos_items
                )

                total_loss += loss
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_stats = {k: v / n_batches for k, v in total_stats.items()}
            logger.info(
                f'GRPO epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}, '
                f'R={avg_stats.get("reward_mean", 0):.4f}, '
                f'W={avg_stats.get("win_rate_mean", 0):.4f}, '
                f'KL={avg_stats.get("kl_mean", 0):.4f}'
            )

    def train_steps(self, user_emb_all, item_emb_all, user_pos_items,
                    n_steps=2, batch_size=64):
        """Phase 3: A few GRPO steps (used during joint training)."""
        self.flow_model.velocity_net.train()
        user_ids = [uid for uid in user_pos_items if uid < user_emb_all.shape[0]]

        self._save_old_policy()
        for step in range(n_steps):
            batch_idx = torch.randperm(len(user_ids))[:batch_size]
            batch_uids = [user_ids[i] for i in batch_idx]
            u_emb = user_emb_all[batch_uids]

            loss, stats = self.grpo_step(
                u_emb, batch_uids, item_emb_all, user_pos_items
            )
            logger.debug(
                f'GRPO step {step+1}/{n_steps}, loss={loss:.4f}, '
                f'W={stats["win_rate_mean"]:.4f}'
            )
