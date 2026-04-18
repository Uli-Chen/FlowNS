import math

from flowns.metrics import topk_metrics


def test_topk_metrics():
    metrics = topk_metrics(
        ranked_items_by_user={0: [3, 1, 2], 1: [4, 5, 6]},
        targets_by_user={0: 1, 1: 6},
        topks=[1, 3],
    )

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    expected_ndcg = (1 / math.log2(3) + 1 / math.log2(4)) / 2
    assert round(metrics["ndcg@3"], 6) == round(expected_ndcg, 6)
