"""Metrics for evaluating negative sampling quality in FlowNeg."""

import torch


def fn_rate(neg_item_ids, positive_item_sets):
    """False negative rate: fraction of sampled negatives that are actually positive.

    A false negative occurs when an item sampled as a "negative" is in fact
    a positive (interacted) item for that user.

    Args:
        neg_item_ids: [batch] tensor of sampled negative item IDs.
        positive_item_sets: List of sets, length batch. Each set contains
                           the positive item IDs for the corresponding user.

    Returns:
        Float in [0, 1]. Lower is better.
    """
    if len(neg_item_ids) == 0:
        return 0.0

    neg_ids = neg_item_ids.detach().cpu().tolist()
    false_neg_count = sum(
        1 for idx, nid in enumerate(neg_ids) if nid in positive_item_sets[idx]
    )
    return false_neg_count / len(neg_ids)


def candidate_purity(candidate_ids, positive_item_sets):
    """Fraction of candidate items that are truly negative (not in the positive set).

    Args:
        candidate_ids: [batch, M] tensor of candidate item IDs.
        positive_item_sets: List of sets, length batch.

    Returns:
        Float in [0, 1]. Higher is better.
    """
    ids = candidate_ids.detach().cpu().tolist()
    total = 0
    true_neg = 0
    for i, row in enumerate(ids):
        pos_set = positive_item_sets[i]
        for cid in row:
            total += 1
            if cid not in pos_set:
                true_neg += 1

    return true_neg / total if total > 0 else 1.0


def hardness_score(neg_scores, pos_scores):
    """Average ratio of negative scores to positive scores.

    Measures how "hard" the sampled negatives are. Higher values indicate
    negatives that the model scores close to (or above) positives.

    Args:
        neg_scores: [batch] tensor of model scores for negative items.
        pos_scores: [batch] tensor of model scores for positive items.

    Returns:
        Float. Higher means harder negatives.
    """
    # Avoid division by zero by clamping positive scores
    pos_clamped = pos_scores.detach().clamp(min=1e-8)
    ratios = neg_scores.detach() / pos_clamped
    return ratios.mean().item()


def diversity_coverage(neg_item_ids, total_items):
    """Coverage: fraction of the total item catalog appearing in sampled negatives.

    Args:
        neg_item_ids: [N] tensor (or any shape) of sampled negative item IDs,
                      possibly from multiple batches concatenated.
        total_items: Int, total number of items in the catalog.

    Returns:
        Float in [0, 1]. Higher means more diverse sampling.
    """
    if total_items <= 0:
        return 0.0

    unique_count = neg_item_ids.detach().cpu().unique().numel()
    return unique_count / total_items


def exposure_recall(sampled_neg_ids, exposure_neg_sets):
    """Recall of exposure-based negatives among the sampled negatives.

    Measures the overlap between the set of sampled negatives and a
    reference set of exposure negatives (items shown but not clicked).

    Args:
        sampled_neg_ids: [batch] tensor of sampled negative item IDs.
        exposure_neg_sets: List of sets, length batch. Each set contains
                          the exposure-based negative item IDs for that user.

    Returns:
        Float in [0, 1]. Higher means more exposure negatives are captured.
    """
    if len(sampled_neg_ids) == 0:
        return 0.0

    sampled = sampled_neg_ids.detach().cpu().tolist()
    recall_sum = 0.0
    valid_count = 0

    for i, sid in enumerate(sampled):
        exp_set = exposure_neg_sets[i]
        if len(exp_set) == 0:
            continue
        valid_count += 1
        if sid in exp_set:
            recall_sum += 1.0

    return recall_sum / valid_count if valid_count > 0 else 0.0
