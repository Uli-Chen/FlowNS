import copy
import time
import torch
import torch.nn.functional as F
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
                 max_pos_samples=10, log_interval=1,
                 old_policy_scope='batch', normalize_log_ratio=False,
                 reward_mode='mapped_item', mapping_strategy='nearest',
                 mapping_topk=50, boundary_safe_w=0.5):
        self.flow_model = flow_model
        self.sde_sampler = sde_sampler
        self.reward_fn = reward_fn
        self.mapper = mapper
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.beta = beta
        self.lr = lr
        self.max_pos_samples = max_pos_samples
        self.log_interval = log_interval
        if old_policy_scope not in {'batch', 'epoch'}:
            raise ValueError('old_policy_scope must be "batch" or "epoch"')
        if reward_mode not in {'continuous', 'mapped_item', 'score_topk'}:
            raise ValueError(
                'reward_mode must be "continuous", "mapped_item", or "score_topk"'
            )
        self.old_policy_scope = old_policy_scope
        self.normalize_log_ratio = normalize_log_ratio
        self.reward_mode = reward_mode
        self.mapping_strategy = mapping_strategy
        self.mapping_topk = mapping_topk
        self.boundary_safe_w = boundary_safe_w
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

    @torch.no_grad()
    def _reward_embs(self, final_emb, user_emb, user_ids, item_emb_all,
                     user_pos_items, pos_embs):
        """Return embeddings used by the reward.

        The recommender is trained with discrete item IDs after nearest-neighbor
        mapping. Using mapped item embeddings here keeps GRPO's reward aligned
        with the negatives that Phase 4 actually feeds to RecBole.
        """
        if self.reward_mode == 'continuous':
            return final_emb
        if self.reward_mode == 'score_topk':
            B, G, _ = final_emb.shape
            scores = user_emb @ item_emb_all.T
            scores[:, 0] = -torch.inf
            for row, uid in enumerate(user_ids):
                ids = user_pos_items.get(uid, set())
                if not ids:
                    continue
                ids = torch.as_tensor(
                    list(ids), device=scores.device, dtype=torch.long,
                )
                ids = ids[(ids >= 0) & (ids < scores.shape[1])]
                if ids.numel() > 0:
                    scores[row, ids] = -torch.inf

            k = min(max(int(self.mapping_topk), 1), scores.shape[1])
            cand_ids = scores.topk(k=k, dim=-1).indices
            cand_emb = item_emb_all[cand_ids]
            sim = torch.bmm(
                F.normalize(final_emb, dim=-1),
                F.normalize(cand_emb, dim=-1).transpose(1, 2),
            )
            chosen = sim.argmax(dim=-1)
            mapped_ids = cand_ids.gather(1, chosen)
            return item_emb_all[mapped_ids]

        B, G, d = final_emb.shape
        flat = final_emb.reshape(B * G, d)
        flat_user = user_emb.unsqueeze(1).expand(B, G, d).reshape(B * G, d)
        flat_pos = pos_embs.unsqueeze(1).expand(
            B, G, pos_embs.shape[1], d,
        ).reshape(B * G, pos_embs.shape[1], d)
        forbidden = []
        for uid in user_ids:
            ids = user_pos_items.get(uid, set())
            forbidden.extend([ids] * G)

        mapped_ids = self.mapper.map_to_items(
            flat,
            forbidden_item_ids=forbidden,
            exclude_item_ids=(0,),
            user_emb=flat_user,
            pos_item_embs=flat_pos,
            reward_fn=self.reward_fn,
            strategy=self.mapping_strategy,
            candidate_topk=self.mapping_topk,
            boundary_safe_w=self.boundary_safe_w,
        ).reshape(B, G)
        return item_emb_all[mapped_ids]

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
        if self.old_policy_scope == 'batch' or self._old_state is None:
            self._save_old_policy()

        # 1. Sample G trajectories per user
        with torch.no_grad():
            final_emb, trajectories, noises = self.sde_sampler.sample_trajectories(
                user_emb, n_trajectories=G
            )

        # 2. Compute rewards for each trajectory
        pos_embs = self._get_pos_embs(user_ids, item_emb_all, user_pos_items)
        reward_emb = self._reward_embs(
            final_emb, user_emb, user_ids, item_emb_all, user_pos_items,
            pos_embs,
        )
        # final_emb: (B, G, d), pos_embs: (B, K, d)
        rewards = []
        win_rates = []
        for g in range(G):
            gen_g = reward_emb[:, g, :]  # (B, d)
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

        if self.normalize_log_ratio:
            log_r_mean = log_ratios.mean(dim=1, keepdim=True)
            log_r_std = log_ratios.std(dim=1, keepdim=True)
            log_ratios = (log_ratios - log_r_mean) / (log_r_std + 1e-8)
        ratios = torch.exp(torch.clamp(log_ratios, -20.0, 20.0))  # (B, G, T)

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
            'log_ratio_abs_mean': log_ratios.abs().mean().item(),
        }
        return loss.item(), stats

    def train(self, user_emb_all, item_emb_all, user_pos_items,
              epochs=10, batch_size=64, stopping_step=0):
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
        n_batches_total = (len(user_ids) + batch_size - 1) // batch_size

        logger.info(
            'GRPO training started: users=%d, epochs=%d, batch_size=%d, '
            'batches/epoch=%d, group_size=%d, clip_eps=%.4f, beta=%.4f, '
            'lr=%s, old_policy_scope=%s, normalize_log_ratio=%s, '
            'reward_mode=%s, mapping_strategy=%s, mapping_topk=%s',
            len(user_ids), epochs, batch_size, n_batches_total,
            self.group_size, self.clip_eps, self.beta, self.lr,
            self.old_policy_scope, self.normalize_log_ratio, self.reward_mode,
            self.mapping_strategy, self.mapping_topk,
        )

        best_reward = -float('inf')
        best_state = None
        best_epoch = -1
        patience = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            if self.old_policy_scope == 'epoch':
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

            epoch_reward = avg_stats.get('reward_mean', 0)
            if epoch_reward > best_reward:
                best_reward = epoch_reward
                best_epoch = epoch + 1
                best_state = copy.deepcopy(
                    self.flow_model.velocity_net.state_dict()
                )
                patience = 0
            else:
                patience += 1

            should_log = (
                epoch == 0
                or epoch + 1 == epochs
                or (self.log_interval and (epoch + 1) % self.log_interval == 0)
            )
            if should_log:
                logger.info(
                    'GRPO epoch %d/%d: loss=%.4f, reward=%.4f±%.4f, '
                    'win_rate=%.4f, kl=%.4f, ratio=%.4f, best_reward=%.4f@%d, '
                    'patience=%d/%d, batches=%d, time=%.2fs',
                    epoch + 1, epochs, avg_loss,
                    avg_stats.get('reward_mean', 0),
                    avg_stats.get('reward_std', 0),
                    avg_stats.get('win_rate_mean', 0),
                    avg_stats.get('kl_mean', 0),
                    avg_stats.get('ratio_mean', 0),
                    best_reward, best_epoch,
                    patience, stopping_step or 0,
                    n_batches, time.time() - epoch_start,
                )

            if stopping_step and patience >= stopping_step:
                logger.info(
                    'GRPO early stopping at epoch %d '
                    '(no improvement for %d epochs).',
                    epoch + 1, stopping_step,
                )
                break

        if best_state is not None:
            self.flow_model.velocity_net.load_state_dict(best_state)
            logger.info(
                'GRPO training: restored best checkpoint from epoch %d '
                '(reward=%.4f).',
                best_epoch, best_reward,
            )

    def train_steps(self, user_emb_all, item_emb_all, user_pos_items,
                    n_steps=2, batch_size=64):
        """Phase 3: A few GRPO steps (used during joint training)."""
        self.flow_model.velocity_net.train()
        user_ids = [uid for uid in user_pos_items if uid < user_emb_all.shape[0]]

        logger.info(
            'GRPO joint update started: steps=%d, batch_size=%d, users=%d',
            n_steps, batch_size, len(user_ids),
        )
        if self.old_policy_scope == 'epoch':
            self._save_old_policy()
        for step in range(n_steps):
            batch_idx = torch.randperm(len(user_ids))[:batch_size]
            batch_uids = [user_ids[i] for i in batch_idx]
            u_emb = user_emb_all[batch_uids]

            loss, stats = self.grpo_step(
                u_emb, batch_uids, item_emb_all, user_pos_items
            )
            should_log = (
                step == 0
                or step + 1 == n_steps
                or (self.log_interval and (step + 1) % self.log_interval == 0)
            )
            if should_log:
                logger.info(
                    'GRPO joint step %d/%d: loss=%.4f, reward=%.4f, '
                    'win_rate=%.4f, kl=%.4f, ratio=%.4f',
                    step + 1, n_steps, loss, stats['reward_mean'],
                    stats['win_rate_mean'], stats['kl_mean'],
                    stats['ratio_mean'],
                )
