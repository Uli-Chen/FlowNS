import torch
import torch.nn.functional as F


class EmbeddingToItemMapper:
    """Map generated continuous embeddings to discrete item IDs."""

    def __init__(self, item_embeddings, temperature=0.1, chunk_size=16384):
        """
        Args:
            item_embeddings: (n_items, d) all item embeddings
            temperature: softmax temperature for soft mapping
            chunk_size: max generated embeddings to map in one similarity block
        """
        self.item_embeddings = item_embeddings
        self.temperature = temperature
        self.chunk_size = chunk_size
        self._norm_emb = F.normalize(item_embeddings, dim=-1)

    def update_embeddings(self, item_embeddings):
        self.item_embeddings = item_embeddings
        self._norm_emb = F.normalize(item_embeddings, dim=-1)

    @torch.no_grad()
    def map_to_items(self, gen_embeddings, forbidden_item_ids=None,
                     exclude_item_ids=None, user_emb=None, pos_item_embs=None,
                     reward_fn=None, strategy='nearest', candidate_topk=50,
                     boundary_safe_w=0.5):
        """Nearest-neighbor mapping to item IDs.

        Args:
            gen_embeddings: (B, d) or (B, G, d)
            forbidden_item_ids: optional iterable of per-sample item-id sets to mask
            exclude_item_ids: optional global item ids to mask for every sample
            user_emb: optional user embeddings for score-aware selection
            pos_item_embs: optional positives for reward-aware selection
            reward_fn: optional reward function with win_rate/compute_reward
            strategy: nearest, hard_topk, reward_topk, score_topk, or boundary_topk
            candidate_topk: number of nearest generated-embedding neighbors to rerank
            boundary_safe_w: for boundary_topk, the max tolerated win-rate against the
                user's positives. Candidates with W > boundary_safe_w are treated as
                likely false negatives and dropped; among the rest the boundary-aware
                reward R = W^a(1-W)^gamma picks the hardest still-safe negative.
        Returns:
            item_ids: same shape as gen_embeddings without last dim
        """
        if strategy not in {
            'nearest', 'hard_topk', 'reward_topk', 'score_topk', 'boundary_topk',
        }:
            raise ValueError(
                'strategy must be "nearest", "hard_topk", "reward_topk", '
                '"score_topk", or "boundary_topk"'
            )
        orig_shape = gen_embeddings.shape[:-1]
        flat = gen_embeddings.reshape(-1, gen_embeddings.shape[-1])
        if self.chunk_size and flat.shape[0] > self.chunk_size:
            flat_user = None
            if user_emb is not None:
                flat_user = user_emb.reshape(-1, user_emb.shape[-1])
                if flat_user.shape[0] != flat.shape[0]:
                    raise ValueError('user_emb must align with gen_embeddings')

            flat_pos = None
            if pos_item_embs is not None:
                flat_pos = pos_item_embs.reshape(
                    flat.shape[0],
                    pos_item_embs.shape[-2],
                    pos_item_embs.shape[-1],
                )

            chunks = []
            for start in range(0, flat.shape[0], self.chunk_size):
                end = min(start + self.chunk_size, flat.shape[0])
                chunk_forbidden = (
                    None if forbidden_item_ids is None
                    else forbidden_item_ids[start:end]
                )
                chunks.append(self.map_to_items(
                    flat[start:end],
                    forbidden_item_ids=chunk_forbidden,
                    exclude_item_ids=exclude_item_ids,
                    user_emb=None if flat_user is None else flat_user[start:end],
                    pos_item_embs=None if flat_pos is None else flat_pos[start:end],
                    reward_fn=reward_fn,
                    strategy=strategy,
                    candidate_topk=candidate_topk,
                    boundary_safe_w=boundary_safe_w,
                ))
            return torch.cat(chunks, dim=0).reshape(orig_shape)

        flat_norm = F.normalize(flat, dim=-1)

        sim = flat_norm @ self._norm_emb.T  # (N, n_items)
        n_items = sim.shape[1]
        if exclude_item_ids is not None:
            ids = torch.as_tensor(
                list(exclude_item_ids), device=sim.device, dtype=torch.long,
            )
            ids = ids[(ids >= 0) & (ids < n_items)]
            if ids.numel() > 0:
                sim[:, ids] = -torch.inf

        if forbidden_item_ids is not None:
            if len(forbidden_item_ids) != flat.shape[0]:
                raise ValueError(
                    'forbidden_item_ids must have one entry per generated embedding'
                )
            for row, ids in enumerate(forbidden_item_ids):
                if not ids:
                    continue
                ids = torch.as_tensor(
                    list(ids), device=sim.device, dtype=torch.long,
                )
                ids = ids[(ids >= 0) & (ids < n_items)]
                if ids.numel() > 0:
                    sim[row, ids] = -torch.inf

        if strategy == 'nearest':
            item_ids = sim.argmax(dim=-1)
            return item_ids.reshape(orig_shape)

        if user_emb is None:
            raise ValueError(f'user_emb is required for {strategy}')

        flat_user = user_emb.reshape(-1, user_emb.shape[-1])
        if flat_user.shape[0] != flat.shape[0]:
            raise ValueError('user_emb must align with gen_embeddings')

        k = min(max(int(candidate_topk), 1), n_items)
        if strategy == 'score_topk':
            score_logits = flat_user @ self.item_embeddings.T
            if exclude_item_ids is not None:
                ids = torch.as_tensor(
                    list(exclude_item_ids), device=score_logits.device,
                    dtype=torch.long,
                )
                ids = ids[(ids >= 0) & (ids < n_items)]
                if ids.numel() > 0:
                    score_logits[:, ids] = -torch.inf

            if forbidden_item_ids is not None:
                for row, ids in enumerate(forbidden_item_ids):
                    if not ids:
                        continue
                    ids = torch.as_tensor(
                        list(ids), device=score_logits.device, dtype=torch.long,
                    )
                    ids = ids[(ids >= 0) & (ids < n_items)]
                    if ids.numel() > 0:
                        score_logits[row, ids] = -torch.inf

            cand_ids = score_logits.topk(k=k, dim=-1).indices
            cand_emb = self.item_embeddings[cand_ids]
            cand_sim = (
                F.normalize(cand_emb, dim=-1) * flat_norm.unsqueeze(1)
            ).sum(dim=-1)
            chosen = cand_sim.argmax(dim=-1)
            item_ids = cand_ids.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
            return item_ids.reshape(orig_shape)

        cand_ids = sim.topk(k=k, dim=-1).indices  # (N, k)
        cand_emb = self.item_embeddings[cand_ids]  # (N, k, d)
        cand_scores = (cand_emb * flat_user.unsqueeze(1)).sum(dim=-1)

        needs_reward = strategy in {'reward_topk', 'boundary_topk'}
        if strategy == 'hard_topk' or (
            needs_reward and (pos_item_embs is None or reward_fn is None)
        ):
            # hard_topk, or reward/boundary asked for without positives: fall back
            # to the highest-scoring nearby candidate.
            chosen = cand_scores.argmax(dim=-1)
        else:
            flat_pos = pos_item_embs.reshape(
                flat.shape[0], pos_item_embs.shape[-2], pos_item_embs.shape[-1],
            )
            score_pos = (
                flat_user.unsqueeze(1) * flat_pos
            ).sum(dim=-1).unsqueeze(1)  # (N, 1, K)
            W = torch.sigmoid(cand_scores.unsqueeze(-1) - score_pos).mean(dim=-1)
            rewards = reward_fn.compute_reward(W)
            if strategy == 'boundary_topk':
                # Drop candidates that beat the positives too often (likely false
                # negatives), then keep the hardest still-safe one. cand_ids is
                # similarity-sorted, so column 0 is the nearest fallback when every
                # candidate is unsafe.
                unsafe = W > boundary_safe_w
                rewards = rewards.masked_fill(unsafe, float('-inf'))
                chosen = rewards.argmax(dim=-1)
                chosen = torch.where(
                    unsafe.all(dim=-1), torch.zeros_like(chosen), chosen,
                )
            else:
                chosen = rewards.argmax(dim=-1)

        item_ids = cand_ids.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
        return item_ids.reshape(orig_shape)

    def soft_map(self, gen_embeddings):
        """Soft probability mapping (differentiable).

        Args:
            gen_embeddings: (B, d)
        Returns:
            probs: (B, n_items)
        """
        gen_norm = F.normalize(gen_embeddings, dim=-1)
        sim = gen_norm @ self._norm_emb.T / self.temperature
        return F.softmax(sim, dim=-1)
