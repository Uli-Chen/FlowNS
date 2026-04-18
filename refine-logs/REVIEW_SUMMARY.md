# Review Summary

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

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|---|---|---|---|---|
| 1 | ANN+score softmax is too trivial as the hardness mechanism. | Replaced post-hoc sorting with KL-constrained energy tilting and beta-conditioned weighted CFM. | Partial | Need empirical proof that weighted CFM beats centroid/KNN and old generate-then-sort. |

## Overall Evolution

- The method moved from "generate exposure-like negatives, then sort candidates" to "learn the hardness-tilted exposure-negative target distribution."
- ANN is demoted to catalog quantization rather than presented as novelty.
- RL was rejected as the main method because it adds instability and contribution sprawl.
- Per-sample noise optimization was rejected as the main method because it adds inference-time optimization and support-density engineering.

## Final Status

- Anchor status: preserved.
- Focus status: tighter than the initial proposal, but still needs validation.
- Modernity status: appropriately frontier-aware through flow matching and variational distribution tilting.
- Strongest part: the closed-form KL-constrained target `q_beta*(x | u)`.
- Remaining weakness: score-weighted CFM must avoid weight collapse and must empirically justify the flow over simpler exposure reweighting.

