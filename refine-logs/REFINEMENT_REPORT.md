# Refinement Report

**Problem**: exposure-grounded hard negative sampling for implicit-feedback recommendation.
**Initial Approach**: conditional flow matching plus ANN retrieval and recommender-score softmax.
**Date**: 2026-04-07
**Rounds**: 1 / 5
**Final Score**: 8.4 / 10
**Final Verdict**: REVISE

## Problem Anchor

- **Bottom-line problem**: Negative sampling in recommender systems with implicit feedback cannot simultaneously guarantee that negative samples are (1) truly negative (not false negatives) and (2) hard enough to provide strong training gradients. Exposure data (items shown but not interacted with) provides a direct signal for true negatives but is sparse and underutilized.
- **Must-solve bottleneck**: Existing methods conflate realness and hardness. RNS introduces false negatives. DNS amplifies false negatives. Generative methods (IRGAN, MixGCF) do not model the true negative distribution and ignore exposure data.
- **Non-goals**: New recommendation architecture. Content-based filtering. Positive augmentation.
- **Constraints**: Neurips. Data-centric framing. Million-item scale. Standard GPU.
- **Success condition**: Exposure-grounded negatives with smooth hardness control and improved recommendation quality.

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 initial | 9 | 8 | 7 | 8 | 8 | 8 | 7 | 7.8 | REVISE |
| 1 refined | 9 | 8.5 | 8.5 | 8.5 | 8 | 8.5 | 8 | 8.4 | REVISE |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|---|---|---|---|
| 1 | ANN+score softmax makes hardness look like standard hard mining. | Replaced it with KL-constrained exposure-distribution energy tilt and beta-conditioned weighted CFM. | Partial; method is cleaner, experiments still needed. |

## Final Proposal Snapshot

- Canonical clean version lives in `refine-logs/FINAL_PROPOSAL.md`.
- Final thesis: hard negatives should be sampled from the KL-constrained hardness tilt of the exposure-negative distribution.
- Realness is preserved through `q << p_E` and the KL penalty.
- Hardness is controlled by beta in the closed-form tilted target.
- ANN is retained only for nearest-neighbor projection, not score sorting.

## Method Evolution Highlights

1. Replaced post-hoc ANN+score reweighting with a target-distribution-level hardness mechanism.
2. Added a closed-form variational objective with beta as the smooth hardness knob.
3. Rejected RL as overbuilt and retained one new trainable component.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|---|---|---|---|
| 1 | Consider RL/bandit policy. | Rejected as main route because it shifts the contribution to training control and weakens authenticity. | Rejected. |
| 1 | Consider constrained noise optimization. | Kept as possible ablation/diagnostic, not main route, due inference cost and density-penalty design. | Rejected as main method. |

## Remaining Weaknesses

The method still needs empirical proof that weighted beta-conditioned CFM does more than score-reweight exposed negatives. The critical ablation is distance-only decoding across centroid, KNN, unweighted flow, and beta-tilted flow.

## Raw Reviewer Responses

<details>
<summary>Round 1 Review</summary>

External Claude review failed:

```text
jobId: 75e5a59ea1544b7797c3d3a1de2dabe7
status: failed
error: API Error: 400 default channel restriction: interface only available for Claude Code client
```

Local review is saved in `round-1-review.md`.

</details>

## Next Steps

- Proceed to `/experiment-plan` if the KL-constrained tilting route is accepted.
- First pilot ablation: beta-tilted flow vs exposure centroid/KNN vs unweighted flow under the same distance-only decoder.
- If beta-tilted flow does not beat simpler baselines, pivot to a simpler exposure-weighted sampler instead of adding RL.

