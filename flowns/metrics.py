from __future__ import annotations

import math
from collections.abc import Iterable


def topk_metrics(
    ranked_items_by_user: dict[int, list[int]],
    targets_by_user: dict[int, int],
    topks: Iterable[int],
) -> dict[str, float]:
    topks = sorted(set(int(k) for k in topks))
    if not topks:
        raise ValueError("topks must not be empty")
    if not targets_by_user:
        return {f"{name}@{k}": 0.0 for k in topks for name in ("recall", "ndcg")}

    totals = {f"recall@{k}": 0.0 for k in topks}
    totals.update({f"ndcg@{k}": 0.0 for k in topks})

    evaluated = 0
    for user, target in targets_by_user.items():
        ranked_items = ranked_items_by_user.get(user)
        if not ranked_items:
            continue
        evaluated += 1
        for k in topks:
            window = ranked_items[:k]
            if target in window:
                rank = window.index(target)
                totals[f"recall@{k}"] += 1.0
                totals[f"ndcg@{k}"] += 1.0 / math.log2(rank + 2)

    if evaluated == 0:
        return {key: 0.0 for key in totals}
    return {key: value / evaluated for key, value in totals.items()}
