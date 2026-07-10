import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed (non-learned) sinusoidal embedding for continuous t ∈ [0, 1].

    Maps a scalar t to a multi-frequency basis {cos(ω_k·t̃), sin(ω_k·t̃)} so the
    main MLP can represent arbitrary-frequency dependence on t by a linear
    combination. Unlike a learned MLP on the raw scalar t, this provides the
    right inductive bias from the start and adds no trainable parameters.

    t is rescaled by `time_scale` (default 1000) so the continuous range [0, 1]
    spans the same effective range as DDPM's discrete timesteps, giving good
    frequency coverage; `max_period` controls the lowest frequency.
    """

    def __init__(self, dim, max_period=10000.0, time_scale=1000.0):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1)
        )
        self.register_buffer('freqs', freqs, persistent=False)
        self.time_scale = time_scale

    def forward(self, t):
        # t: (B,) or (B, 1) in [0, 1]
        t = t.reshape(-1, 1) * self.time_scale
        args = t * self.freqs.to(t.dtype).unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:  # pad to exact dim
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ConditionalVelocityNet(nn.Module):
    """v_θ(x_t, u, t): predicts velocity field for conditional flow matching.
    Input: concat(x_t, user_emb, t_embed) → MLP → velocity (emb_dim,)

    For classifier-free guidance (CFG) the net also learns the *unconditional*
    field v_θ(x_t, ∅, t): a learned `null_cond` embedding stands in for the user
    condition. At train time rows are dropped to ∅ with some probability; at
    sample time the conditional and unconditional fields are extrapolated
    (see SDESampler.guidance_scale).
    """

    def __init__(self, emb_dim, hidden_dim=256, n_layers=3, time_embed_dim=16):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_embed_dim = time_embed_dim

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        input_dim = emb_dim + emb_dim + time_embed_dim
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = emb_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

        # Learned null condition for CFG. Stays at its init (zeros) and is never
        # read unless conditions are actually dropped, so a flow trained/loaded
        # without CFG behaves exactly as before.
        self.null_cond = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x_t, user_emb, t, cond_drop_mask=None):
        """
        Args:
            x_t: (B, d) noisy embedding at time t
            user_emb: (B, d) user conditioning
            t: (B,) or (B, 1) time in [0, 1]
            cond_drop_mask: optional (B,) bool tensor. Rows that are True have
                their conditioning replaced by the learned null embedding — this
                is how CFG trains the unconditional branch (random drop) and
                samples it (all-True). None keeps every row conditional (the
                original behavior, no extra work).
        Returns:
            velocity: (B, d)
        """
        if cond_drop_mask is not None:
            null = self.null_cond.to(dtype=user_emb.dtype).expand_as(user_emb)
            user_emb = torch.where(cond_drop_mask.unsqueeze(-1), null, user_emb)
        t_embed = self.time_embed(t)
        inp = torch.cat([x_t, user_emb, t_embed], dim=-1)
        return self.net(inp)


class ConditionalFlowModel:
    """Conditional Flow Matching for learning negative item distribution."""

    def __init__(self, emb_dim, hidden_dim=256, n_layers=3, device='cpu',
                 cfg_dropout_prob=0.0):
        self.emb_dim = emb_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self.device = device
        # Classifier-free guidance: train-time probability of dropping the user
        # condition to the learned null embedding. 0.0 disables CFG entirely.
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.velocity_net = ConditionalVelocityNet(
            emb_dim, hidden_dim, n_layers
        ).to(device)
        self.ref_state_dict = None

    def save_as_reference(self):
        """Snapshot the current velocity net as the pretrained reference.

        Used as the "flow has been pretrained" sentinel and persisted as the
        flow-ref checkpoint so paired runs share one frozen reference policy.
        """
        self.ref_state_dict = {
            k: v.clone() for k, v in self.velocity_net.state_dict().items()
        }

    def cfm_loss(self, user_emb, neg_item_emb):
        """Conditional Flow Matching loss.
        L = E[||v_θ((1-t)x_0 + t·e_n, c, t) - (e_n - x_0)||²]

        With CFG enabled (cfg_dropout_prob > 0), the condition c is the user
        embedding for most rows but the learned null embedding for a random
        cfg_dropout_prob fraction, so the same net learns both the conditional
        and unconditional velocity fields.

        Args:
            user_emb: (B, d)
            neg_item_emb: (B, d) ground-truth negative item embeddings
        """
        B, d = neg_item_emb.shape
        t = torch.rand(B, device=neg_item_emb.device)
        x_0 = torch.randn_like(neg_item_emb)

        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * neg_item_emb
        target_velocity = neg_item_emb - x_0

        cond_drop_mask = None
        if self.cfg_dropout_prob > 0.0:
            cond_drop_mask = (
                torch.rand(B, device=neg_item_emb.device) < self.cfg_dropout_prob
            )

        pred_velocity = self.velocity_net(
            x_t, user_emb, t, cond_drop_mask=cond_drop_mask,
        )
        loss = (pred_velocity - target_velocity).pow(2).mean()
        return loss

    def pretrain(self, user_emb_all, item_emb_all, user_pos_items,
                 user_neg_items=None,
                 epochs=50, batch_size=256, lr=1e-4, log_interval=10,
                 stopping_step=0):
        """Pretrain flow model with CFM loss.

        Args:
            user_emb_all: (n_users, d) all user embeddings
            item_emb_all: (n_items, d) all item embeddings
            user_pos_items: dict {uid: set(item_ids)}
            user_neg_items: optional dict {uid: list(item_ids)} for explicit negatives
            epochs: training epochs
            batch_size: batch size
            lr: learning rate
        """
        optimizer = optim.Adam(self.velocity_net.parameters(), lr=lr)
        self.velocity_net.train()

        n_items = item_emb_all.shape[0]
        user_ids = [uid for uid in user_pos_items if uid < user_emb_all.shape[0]]
        n_batches_total = (len(user_ids) + batch_size - 1) // batch_size

        logger.info(
            'Flow pretrain started: users=%d, items=%d, emb_dim=%d, '
            'epochs=%d, batch_size=%d, batches/epoch=%d, lr=%s, '
            'explicit_neg_users=%d',
            len(user_ids), n_items, self.emb_dim, epochs, batch_size,
            n_batches_total, lr,
            sum(1 for uid in user_ids if user_neg_items and user_neg_items.get(uid)),
        )

        best_loss = float('inf')
        best_state = None
        best_epoch = -1
        patience = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0.0
            n_batches = 0

            indices = torch.randperm(len(user_ids))
            for start in range(0, len(user_ids), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_uids = [user_ids[i] for i in batch_idx]

                u_emb = user_emb_all[batch_uids]

                neg_ids = []
                for uid in batch_uids:
                    pos = user_pos_items.get(uid, set())
                    neg_pool = user_neg_items.get(uid) if user_neg_items else None
                    if neg_pool:
                        nid = random.choice(neg_pool)
                    else:
                        while True:
                            nid = random.randint(1, n_items - 1)
                            if nid not in pos:
                                break
                    neg_ids.append(nid)
                neg_emb = item_emb_all[neg_ids]

                optimizer.zero_grad()
                loss = self.cfm_loss(u_emb, neg_emb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                best_state = {
                    k: v.clone() for k, v in self.velocity_net.state_dict().items()
                }
                patience = 0
            else:
                patience += 1

            should_log = (
                epoch == 0
                or epoch + 1 == epochs
                or (log_interval and (epoch + 1) % log_interval == 0)
            )
            if should_log:
                logger.info(
                    'Flow pretrain epoch %d/%d: loss=%.6f, best=%.6f@%d, '
                    'patience=%d/%d, batches=%d, time=%.2fs',
                    epoch + 1, epochs, avg_loss, best_loss, best_epoch,
                    patience, stopping_step or 0,
                    n_batches, time.time() - epoch_start,
                )

            if stopping_step and patience >= stopping_step:
                logger.info(
                    'Flow pretrain early stopping at epoch %d '
                    '(no improvement for %d epochs).',
                    epoch + 1, stopping_step,
                )
                break

        if best_state is not None:
            self.velocity_net.load_state_dict(best_state)
            logger.info(
                'Flow pretrain: restored best checkpoint from epoch %d (loss=%.6f).',
                best_epoch, best_loss,
            )
        self.save_as_reference()
        logger.info('Flow pretrain complete: reference policy saved.')
