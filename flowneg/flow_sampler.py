"""
FlowNeg Sampler — extends RecBole's Sampler for flow-based negative generation.

During warm-up: delegates to uniform sampling (standard RecBole behavior).
After warm-up: uses generate-then-reweight pipeline:
  1. Generate query via CFM flow
  2. Retrieve M nearest items via FAISS
  3. Reweight by recommender score with temperature tau_h
  4. Sample from reweighted distribution
"""

import copy
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from recbole.sampler.sampler import Sampler

logger = logging.getLogger(__name__)


class FlowNegSampler(Sampler):
    """Flow-based negative sampler with exposure grounding.

    Inherits from RecBole's Sampler to maintain full compatibility with
    the data pipeline (get_used_ids, set_phase, sample_by_user_ids).
    """

    def __init__(self, phases, datasets, distribution="uniform", alpha=1.0,
                 config=None):
        # Store config before super().__init__ which calls get_used_ids
        self.config = config or {}
        fn_config = self._get_fn_config()

        self.warm_up_epochs = fn_config.get("warm_up_epochs", 5)
        self.tau_h = fn_config.get("tau_h", 0.5)
        self.faiss_M = fn_config.get("faiss_M", 50)
        self.ode_steps = fn_config.get("ode_steps", 4)

        # These will be set later by FlowNegTrainer
        self.cfm_trainer = None
        self.faiss_index = None
        self.current_epoch = 0
        self._model = None
        self._item_embs = None
        self._user_embs = None
        self._flow_active = False

        # Exposure negatives loaded later via load_exposure_data after ID remapping
        self.exposure_negs = {}

        super().__init__(phases, datasets, distribution, alpha)

    def _get_fn_config(self):
        if isinstance(self.config, dict):
            return self.config.get("flowneg", {})
        try:
            return self.config["flowneg"] if "flowneg" in self.config.final_config_dict else {}
        except (AttributeError, KeyError):
            return {}

    def set_flow_components(self, cfm_trainer, faiss_index):
        """Called by FlowNegTrainer after initialization."""
        self.cfm_trainer = cfm_trainer
        self.faiss_index = faiss_index

    def set_model(self, model):
        """Store reference to the backbone model for score computation."""
        self._model = model

    def update_embeddings(self, user_embs, item_embs):
        """Update cached embeddings (called after FAISS rebuild)."""
        self._user_embs = user_embs.detach().cpu()
        self._item_embs = item_embs.detach().cpu()

    def set_epoch(self, epoch):
        """Update current epoch for warm-up logic."""
        self.current_epoch = epoch
        self._flow_active = (
            epoch >= self.warm_up_epochs
            and self.cfm_trainer is not None
            and self.faiss_index is not None
            and self._item_embs is not None
        )

    def set_phase(self, phase):
        """Override to preserve FlowNeg attributes in the copy."""
        if phase not in self.phases:
            raise ValueError(f"Phase [{phase}] not exist.")
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Main sampling interface called by the dataloader.

        During warm-up or if flow is not active: uniform sampling.
        After warm-up: generate-then-reweight pipeline.
        """
        if not self._flow_active:
            return super().sample_by_user_ids(user_ids, item_ids, num)

        try:
            return self._flow_sample(user_ids, num)
        except Exception as e:
            logger.warning(f"Flow sampling failed: {e}. Falling back to uniform.")
            return super().sample_by_user_ids(user_ids, item_ids, num)

    @torch.no_grad()
    def _flow_sample(self, user_ids, num):
        """Generate-then-reweight negative sampling pipeline.

        Args:
            user_ids: numpy array of user IDs [batch]
            num: number of negatives per user

        Returns:
            torch.LongTensor of sampled negative item IDs
        """
        user_ids_np = np.array(user_ids)
        batch_size = len(user_ids_np)
        total_num = batch_size * num
        device = next(self.cfm_trainer.velocity_net.parameters()).device

        # Repeat user_ids for num samples each
        if num > 1:
            user_ids_expanded = np.tile(user_ids_np, num)
        else:
            user_ids_expanded = user_ids_np

        # Clamp user IDs
        user_ids_clamped = np.clip(user_ids_expanded, 0, self._user_embs.size(0) - 1)

        # 1. Get user embeddings
        u_emb = self._user_embs[user_ids_clamped].to(device)  # [total, d]

        # 2. Generate query points via flow
        queries = self.cfm_trainer.generate(
            u_emb, n_samples=1, n_ode_steps=self.ode_steps
        )  # [total, d]

        # 3. FAISS retrieval: M nearest items per query
        queries_np = queries.cpu().numpy().astype(np.float32)
        _, candidate_indices = self.faiss_index.search(
            queries_np, self.faiss_M
        )  # [total, M]

        # 4. Score candidates with recommender
        item_embs_device = self._item_embs.to(device)
        result_ids = np.zeros(total_num, dtype=np.int64)

        # Process in chunks to avoid OOM
        chunk_size = min(1024, total_num)
        for start in range(0, total_num, chunk_size):
            end = min(start + chunk_size, total_num)
            chunk_u = u_emb[start:end]  # [chunk, d]
            chunk_cands = candidate_indices[start:end]  # [chunk, M]

            # Clamp candidate indices
            chunk_cands = np.clip(chunk_cands, 0, self._item_embs.size(0) - 1)
            cand_tensor = torch.from_numpy(chunk_cands).long().to(device)
            cand_embs = item_embs_device[cand_tensor]  # [chunk, M, d]

            # Compute scores: dot product
            scores = torch.bmm(
                cand_embs, chunk_u.unsqueeze(2)
            ).squeeze(2)  # [chunk, M]

            # 5. Softmax reweighting with tau_h
            if self.tau_h > 0 and self.tau_h < float("inf"):
                probs = torch.softmax(scores / self.tau_h, dim=1)  # [chunk, M]
            else:
                # Uniform over candidates
                probs = torch.ones_like(scores) / scores.size(1)

            # 6. Sample from categorical distribution
            sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(1)  # [chunk]

            # Map back to item IDs
            for i in range(end - start):
                result_ids[start + i] = chunk_cands[i, sampled_idx[i].item()]

        # Filter out positive items (rejection sampling for used_ids)
        for i in range(total_num):
            uid = user_ids_clamped[i]
            max_tries = 10
            tries = 0
            while result_ids[i] in self.used_ids[uid] and tries < max_tries:
                # Resample from the same candidate set
                cands = candidate_indices[i]
                cands = cands[cands >= 0]
                valid_cands = [c for c in cands if c not in self.used_ids[uid]]
                if valid_cands:
                    result_ids[i] = np.random.choice(valid_cands)
                else:
                    # Fallback to uniform
                    result_ids[i] = np.random.randint(1, self.item_num)
                tries += 1

        return torch.tensor(result_ids, dtype=torch.long)
