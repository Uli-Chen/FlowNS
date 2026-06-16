import torch
import numpy as np


def compute_fn_rate(generated_item_ids, test_positive_dict, user_ids):
    """Compute false-negative rate of generated negative samples.

    Args:
        generated_item_ids: list/tensor of generated negative item IDs per user
        test_positive_dict: dict {uid: set(test_positive_item_ids)}
        user_ids: list of user IDs corresponding to generated_item_ids
    Returns:
        fn_rate: float, proportion of generated negatives that are test positives
    """
    fn_count = 0
    total = 0
    for uid, neg_id in zip(user_ids, generated_item_ids):
        uid_val = uid.item() if isinstance(uid, torch.Tensor) else uid
        neg_val = neg_id.item() if isinstance(neg_id, torch.Tensor) else neg_id
        test_pos = test_positive_dict.get(uid_val, set())
        if neg_val in test_pos:
            fn_count += 1
        total += 1
    return fn_count / max(total, 1)


def compute_w_statistics(user_emb, gen_emb, pos_item_embs, pos_mask=None):
    """Compute win rate distribution statistics.

    Args:
        user_emb: (B, d)
        gen_emb: (B, d)
        pos_item_embs: (B, K, d)
        pos_mask: optional (B, K) bool marking real (non-padded) positives
    Returns:
        dict with mean, std, histogram counts for W
    """
    score_gen = (user_emb * gen_emb).sum(dim=-1, keepdim=True)
    score_pos = (user_emb.unsqueeze(1) * pos_item_embs).sum(dim=-1)
    wins = torch.sigmoid(score_gen - score_pos)
    if pos_mask is None:
        W = wins.mean(dim=-1)
    else:
        mask = pos_mask.to(wins.dtype)
        W = (wins * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    W_np = W.detach().cpu().numpy()
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(W_np, bins=bins)

    return {
        'mean': float(W_np.mean()),
        'std': float(W_np.std()),
        'median': float(np.median(W_np)),
        'pct_above_0.8': float((W_np > 0.8).mean()),
        'histogram': hist.tolist(),
        'bin_edges': bins.tolist(),
    }


def theoretical_fn_bound(fn_rate_ref, r_max, beta):
    """Compute theoretical FN rate upper bound.

    FN(π_θ) ≤ exp(R_max / β) · FN(π_ref)

    Args:
        fn_rate_ref: FN rate of reference policy
        r_max: maximum reward value
        beta: KL penalty coefficient
    Returns:
        upper_bound: float
    """
    if beta <= 0:
        return float('inf')
    return np.exp(r_max / beta) * fn_rate_ref
