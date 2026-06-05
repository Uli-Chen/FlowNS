import torch
import torch.nn.functional as F


class EmbeddingToItemMapper:
    """Map generated continuous embeddings to discrete item IDs."""

    def __init__(self, item_embeddings, temperature=0.1):
        """
        Args:
            item_embeddings: (n_items, d) all item embeddings
            temperature: softmax temperature for soft mapping
        """
        self.item_embeddings = item_embeddings
        self.temperature = temperature
        self._norm_emb = F.normalize(item_embeddings, dim=-1)

    def update_embeddings(self, item_embeddings):
        self.item_embeddings = item_embeddings
        self._norm_emb = F.normalize(item_embeddings, dim=-1)

    @torch.no_grad()
    def map_to_items(self, gen_embeddings, forbidden_item_ids=None,
                     exclude_item_ids=None):
        """Nearest-neighbor mapping to item IDs.

        Args:
            gen_embeddings: (B, d) or (B, G, d)
            forbidden_item_ids: optional iterable of per-sample item-id sets to mask
            exclude_item_ids: optional global item ids to mask for every sample
        Returns:
            item_ids: same shape as gen_embeddings without last dim
        """
        orig_shape = gen_embeddings.shape[:-1]
        flat = gen_embeddings.reshape(-1, gen_embeddings.shape[-1])
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
        item_ids = sim.argmax(dim=-1)
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
