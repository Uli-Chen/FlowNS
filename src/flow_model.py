import random

import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


class ConditionalVelocityNet(nn.Module):
    """v_θ(x_t, u, t): predicts velocity field for conditional flow matching.
    Input: concat(x_t, user_emb, t_embed) → MLP → velocity (emb_dim,)
    """

    def __init__(self, emb_dim, hidden_dim=256, n_layers=3, time_embed_dim=16):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        input_dim = emb_dim + emb_dim + time_embed_dim
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = emb_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, user_emb, t):
        """
        Args:
            x_t: (B, d) noisy embedding at time t
            user_emb: (B, d) user conditioning
            t: (B,) or (B, 1) time in [0, 1]
        Returns:
            velocity: (B, d)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_embed = self.time_mlp(t)
        inp = torch.cat([x_t, user_emb, t_embed], dim=-1)
        return self.net(inp)


class ConditionalFlowModel:
    """Conditional Flow Matching for learning negative item distribution."""

    def __init__(self, emb_dim, hidden_dim=256, n_layers=3, device='cpu'):
        self.emb_dim = emb_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self.device = device
        self.velocity_net = ConditionalVelocityNet(
            emb_dim, hidden_dim, n_layers
        ).to(device)
        self.ref_state_dict = None

    def save_as_reference(self):
        """Save current velocity net as π_ref for GRPO KL computation."""
        self.ref_state_dict = {
            k: v.clone() for k, v in self.velocity_net.state_dict().items()
        }

    def create_ref_net(self):
        """Create reference net with identical architecture from saved state."""
        if self.ref_state_dict is None:
            raise RuntimeError('No reference state saved. Call save_as_reference() first.')
        net = ConditionalVelocityNet(
            self.velocity_net.emb_dim,
            hidden_dim=self._hidden_dim,
            n_layers=self._n_layers,
            time_embed_dim=self.velocity_net.time_embed_dim,
        ).to(self.device)
        net.load_state_dict(self.ref_state_dict)
        net.eval()
        return net

    def cfm_loss(self, user_emb, neg_item_emb):
        """Conditional Flow Matching loss.
        L = E[||v_θ((1-t)x_0 + t·e_n, u, t) - (e_n - x_0)||²]

        Args:
            user_emb: (B, d)
            neg_item_emb: (B, d) ground-truth negative item embeddings
        """
        B, d = neg_item_emb.shape
        t = torch.rand(B, device=neg_item_emb.device)
        x_0 = torch.randn_like(neg_item_emb)

        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * neg_item_emb
        target_velocity = neg_item_emb - x_0

        pred_velocity = self.velocity_net(x_t, user_emb, t)
        loss = (pred_velocity - target_velocity).pow(2).mean()
        return loss

    def pretrain(self, user_emb_all, item_emb_all, user_pos_items,
                 epochs=50, batch_size=256, lr=1e-4):
        """Phase 1: Pretrain flow model with CFM loss.

        Args:
            user_emb_all: (n_users, d) all user embeddings
            item_emb_all: (n_items, d) all item embeddings
            user_pos_items: dict {uid: set(item_ids)}
            epochs: training epochs
            batch_size: batch size
            lr: learning rate
        """
        optimizer = optim.Adam(self.velocity_net.parameters(), lr=lr)
        self.velocity_net.train()

        n_items = item_emb_all.shape[0]
        user_ids = [uid for uid in user_pos_items if uid < user_emb_all.shape[0]]

        for epoch in range(epochs):
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
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f'Flow pretrain epoch {epoch+1}/{epochs}, loss={avg_loss:.6f}')

        self.save_as_reference()
        logger.info('Flow pretrain complete. Reference policy saved.')
