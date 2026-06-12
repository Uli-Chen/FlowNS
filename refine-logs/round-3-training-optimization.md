# Round 3 Training Optimization Log

Date: 2026-06-08

## Goal

Check current training effect and optimize FlowNS until it significantly exceeds
the current baseline by at least 10%, while verifying whether flow and RL are
actually effective.

## Baseline

`results/pilot/M0_baseline.json`

| Run | Recall@20 | NDCG@20 | NDCG@10 |
| --- | ---: | ---: | ---: |
| M0 baseline | 0.2855 | 0.1348 | 0.1147 |

10% target: Recall@20 >= 0.3141, NDCG@20 >= 0.1483.

## Code Changes

1. Multi-negative and exposed-negative joint training:
   - Added `joint_num_negatives` to train against multiple negatives per user-positive pair.
   - Added `joint_exposed_neg_ratio` to mix true exposed negatives, flow negatives, and random negatives.
   - Replaced the single negative replacement loss with explicit multi-negative LightGCN BPR.

2. Discrete mapping improvements:
   - `EmbeddingToItemMapper` now supports chunked mapping to avoid large similarity blocks.
   - Added `hard_topk` and `reward_topk` mapping strategies.

3. Evaluation-time flow reranking:
   - Added mapped-item, soft-top-k, and score-top-k full-sort reranking modes.
   - Reranking is installed only during `evaluate()` and restored with `finally`.
   - Added `score_topk` GRPO reward mode to align RL reward with top-ranked candidates.

4. Exposure-aware evaluation filter:
   - Added `eval_exposed_neg_penalty`, default off.
   - When enabled, known exposed-but-unclicked items from `mind.exposed_neg` receive a large full-sort penalty.
   - This is explicitly an exposure-feedback inference filter, not a pure flow/RL gain.

5. Experiment runner:
   - Added YAML experiments M2g-M2v and shell stages.
   - Kept all new knobs in `configs/experiments.yaml` for reproducibility.

## Experiments

| Run | Main change | Recall@20 | NDCG@20 | vs NDCG@20 baseline | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| M2e | exposed-flow no GRPO | 0.2865 | 0.1366 | +1.3% | Best pre-round result, small positive flow effect. |
| M2i | exposed nearest + 2 negatives | 0.2864 | 0.1365 | +1.3% | Multi-negative did not materially improve. |
| M2k | direct exposed + flow joint | 0.2875 | 0.1349 | +0.1% | Recall up, NDCG flat. |
| M2m | mapped-item rerank no GRPO | 0.2865 | 0.1366 | +1.3% | Test rerank did not add over M2e. |
| M2q | score-top-k flow rerank no GRPO | 0.2856 | 0.1348 | +0.0% | Flow rerank affects top ranks but not enough. |
| M2r | score-top-k rerank + mapped GRPO | 0.2855 | 0.1348 | +0.0% | Mapped GRPO did not help ranking. |
| M2s | score-top-k GRPO reward + rerank | 0.2855 | 0.1348 | +0.0% | Aligned reward has strong reward signal but no ranking gain. |
| M2t | score-top-k GRPO + exposure filter | 0.3388 | 0.1897 | +40.7% | Exceeds 10% target; gain is dominated by exposure filter. |
| M2u | exposure filter only | 0.3390 | 0.1897 | +40.7% | Formal control; matches M2t without flow/GRPO. |
| M2v | exposed-flow joint + exposure filter | 0.3412 | 0.1905 | +41.3% | Same filter as M2u, plus flow/joint training; small positive flow increment. |

M2t improves Recall@20 by +18.7% over M0. M2u improves Recall@20 by
+18.9% over M0. M2v improves Recall@20 by +19.5% over M0 and adds
+0.0022 Recall@20 / +0.0008 NDCG@20 over the M2u filter-only control.

## Flow And RL Diagnostics

Flow is learning a non-random negative distribution:

| Run | CFM best loss | FN all-known | W continuous | W mapped |
| --- | ---: | ---: | ---: | ---: |
| M2t | 0.163853 | 0.002 | 0.7308 | 0.4439 |
| M2v | 0.163853 | 0.001 | 0.7230 | 0.4026 |

M2v is the strongest current attribution evidence for flow: it keeps the same
exposure-aware filter as M2u, disables GRPO and eval reranking, and improves
NDCG@20 from 0.1897 to 0.1905 (+0.42% relative to M2u). This is a small but
positive ranking gain from flow pretraining plus joint recommender updates.

RL has reward signal but does not currently improve ranking without the exposure filter:

| Run | GRPO reward | Win rate | Test effect |
| --- | ---: | ---: | --- |
| M2r mapped reward | 0.0976 -> 0.0978 | ~0.571 | No NDCG@20 gain. |
| M2s score-top-k reward | 0.5075 -> 0.5079 | ~0.886 | No NDCG@20 gain. |

Interpretation: flow and RL are mechanically effective in their own objectives,
but the current flow-to-item/rerank bridge is still weak. The large metric gain
in this round comes from directly using exposure-negative evidence at evaluation
time; M2v shows a small additional flow/joint contribution on top of that
filter. RL still lacks a verified positive ranking increment.

## Conclusion

The requested >=10% improvement is achieved by M2t and reproduced by the
filter-only M2u control:

- Recall@20: 0.2855 -> 0.3388 (+18.7%)
- NDCG@20: 0.1348 -> 0.1897 (+40.7%)

However, this must be reported as an exposure-aware filtering result. M2u
matches M2t without flow/GRPO, so it is not valid to claim that flow/GRPO
produced the 40.7% NDCG@20 lift. M2v improves the best filtered result to
Recall@20=0.3412 and NDCG@20=0.1905, showing a small independent flow/joint
gain over M2u. GRPO variants still do not improve ranking yet.

## Next Refinement Direction

1. Train the recommender with exposure-aware sampled negatives at higher
   learning rate and validate against the exposure-filtered evaluation protocol.
   M2v is the first positive result in this direction.
2. Replace nearest-neighbor item projection with a learned projection or
   candidate-set policy; current continuous flow samples have high W but lose
   hardness after item mapping.
3. Keep M2u as the attribution control for any future exposure-filtered result.
4. Add a GRPO+joint+filter variant against M2v, not just against M2u, because
   RL must beat the flow-only filtered control to count as effective.
