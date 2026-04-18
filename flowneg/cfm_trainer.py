"""
CFM (Conditional Flow Matching) training logic for FlowNeg.

Trains the velocity network on the CondOT objective using exposure negatives:
  For (u, e_n) from exposure-negative set E_u^neg:
    x_0 ~ N(0, I),  t ~ U(0, 1)
    x_t = (1-t) * x_0 + t * e_n          # CondOT straight path
    target = e_n - x_0                     # velocity target
    loss = ||v_t^theta(x_t, u) - target||^2
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flowneg.velocity_net import VelocityNet

logger = logging.getLogger(__name__)


class ExposureNegDataset(torch.utils.data.Dataset):
    """Yields (user_id, neg_item_id) pairs from the exposure-negative cache."""

    def __init__(self, exposure_negs: dict, min_negs: int = 1):
        """
        Args:
            exposure_negs: {user_id (int): list/array of neg_item_ids (int)}
            min_negs: skip users with fewer than this many exposure negatives
        """
        self.pairs = []
        for uid, neg_ids in exposure_negs.items():
            neg_ids = list(neg_ids)
            if len(neg_ids) >= min_negs:
                for nid in neg_ids:
                    self.pairs.append((uid, nid))
        self.pairs = np.array(self.pairs, dtype=np.int64)
        logger.info(
            f"ExposureNegDataset: {len(self.pairs)} pairs "
            f"from {len(exposure_negs)} users"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # (user_id, neg_item_id)


class CFMTrainer:
    """Trains the velocity network on CondOT objective."""

    def __init__(self, config, device):
        fn_config = config.get("flowneg", {}) if isinstance(config, dict) else (
            config["flowneg"] if "flowneg" in config.final_config_dict else {}
        )

        self.d = config.get("embedding_size", 64) if isinstance(config, dict) else config["embedding_size"]
        self.device = device
        self.lr = fn_config.get("cfm_lr", 1e-3)
        self.train_steps = fn_config.get("cfm_train_steps", 200)
        self.batch_size = fn_config.get("cfm_batch_size", 1024)
        self.cache_path = fn_config.get(
            "exposure_cache_path", "exposure_cache/mindsmall_exposure_negs.pkl"
        )

        hidden_mult = fn_config.get("velocity_hidden_mult", 4)
        n_layers = fn_config.get("velocity_layers", 2)

        self.velocity_net = VelocityNet(
            d=self.d, hidden_mult=hidden_mult, n_layers=n_layers
        ).to(self.device)

        self.optimizer = optim.Adam(self.velocity_net.parameters(), lr=self.lr)
        self.exposure_dataset = None  # Set via load_exposure_data()

    def load_exposure_data(self, uid_map=None, iid_map=None):
        """Load exposure negatives, optionally remapping string IDs to ints.

        Args:
            uid_map: dict mapping string user IDs to int IDs (from RecBole dataset)
            iid_map: dict mapping string item IDs to int IDs (from RecBole dataset)
        """
        if not Path(self.cache_path).exists():
            logger.warning(f"Exposure cache not found at {self.cache_path}")
            return

        with open(self.cache_path, "rb") as f:
            raw_negs = pickle.load(f)

        # Remap string IDs to RecBole integer IDs if mappings provided
        if uid_map is not None and iid_map is not None:
            remapped = {}
            for uid_str, neg_set in raw_negs.items():
                uid_int = uid_map.get(uid_str)
                if uid_int is None:
                    continue
                remapped_negs = set()
                for iid_str in neg_set:
                    iid_int = iid_map.get(iid_str)
                    if iid_int is not None:
                        remapped_negs.add(iid_int)
                if remapped_negs:
                    remapped[uid_int] = remapped_negs
            exposure_negs = remapped
        else:
            # Assume IDs are already integers
            exposure_negs = raw_negs

        self.exposure_dataset = ExposureNegDataset(exposure_negs, min_negs=1)
        logger.info(
            f"Loaded exposure negatives from {self.cache_path}: "
            f"{len(exposure_negs)} users"
        )

    def train_step(self, user_embs, item_embs, n_steps=None):
        """Run one round of CFM training using current embeddings.

        Args:
            user_embs: [n_users, d] detached tensor (all user embeddings)
            item_embs: [n_items, d] detached tensor (all item embeddings)
            n_steps: override number of training steps

        Returns:
            float: average loss over the training steps
        """
        if self.exposure_dataset is None or len(self.exposure_dataset) == 0:
            logger.warning("No exposure data available for CFM training")
            return 0.0

        n_steps = n_steps or self.train_steps
        dataloader = torch.utils.data.DataLoader(
            self.exposure_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        self.velocity_net.train()
        total_loss = 0.0
        step_count = 0

        data_iter = iter(dataloader)
        for _ in range(n_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            user_ids = batch[:, 0].long()
            neg_item_ids = batch[:, 1].long()

            # Clamp IDs to valid range
            user_ids = user_ids.clamp(0, user_embs.size(0) - 1)
            neg_item_ids = neg_item_ids.clamp(0, item_embs.size(0) - 1)

            u_emb = user_embs[user_ids].to(self.device)  # [B, d]
            e_n = item_embs[neg_item_ids].to(self.device)  # [B, d]

            # CondOT: x_0 ~ N(0,I), t ~ U(0,1)
            x_0 = torch.randn_like(e_n)
            t = torch.rand(u_emb.size(0), device=self.device)

            # Straight path interpolation
            t_expand = t.unsqueeze(1)  # [B, 1]
            x_t = (1 - t_expand) * x_0 + t_expand * e_n  # [B, d]

            # Target velocity = e_n - x_0 (constant along CondOT path)
            target = e_n - x_0  # [B, d]

            # Predict
            pred = self.velocity_net(x_t, u_emb, t)  # [B, d]

            # MSE loss
            loss = ((pred - target) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(step_count, 1)
        logger.info(f"CFM training: {step_count} steps, avg loss = {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def generate(self, user_embs, n_samples=1, n_ode_steps=4):
        """Generate negative query points via flow.

        Args:
            user_embs: [batch, d] user embeddings
            n_samples: number of samples per user (currently 1)
            n_ode_steps: number of ODE solver steps

        Returns:
            [batch * n_samples, d] generated points in item embedding space
        """
        from flowneg.utils import midpoint_solver

        self.velocity_net.eval()
        batch_size, d = user_embs.shape

        if n_samples > 1:
            user_embs = user_embs.repeat_interleave(n_samples, dim=0)

        x_0 = torch.randn(user_embs.size(0), d, device=user_embs.device)
        x_1 = midpoint_solver(
            self.velocity_net, x_0, user_embs, n_steps=n_ode_steps
        )
        return x_1
