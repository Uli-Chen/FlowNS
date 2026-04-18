# Research Proposal: FlowNeg -- Exposure-Grounded Negative Generation with Controllable Hardness via Conditional Flow Matching

Source: `proposal/FINAL_PROPOSAL.md`

## Problem Anchor

- **Bottom-line problem**: Negative sampling in recommender systems with implicit feedback cannot simultaneously guarantee that negative samples are (1) truly negative (not false negatives) and (2) hard enough to provide strong training gradients. Exposure data (items shown but not interacted with) provides a direct signal for true negatives but is sparse and underutilized.
- **Must-solve bottleneck**: Existing methods conflate realness and hardness. RNS introduces false negatives. DNS amplifies false negatives. Generative methods (IRGAN, MixGCF) do not model the true negative distribution and ignore exposure data.
- **Non-goals**: New recommendation architecture. Content-based filtering. Positive augmentation.
- **Constraints**: Neurips. Data-centric framing. Million-item scale. Standard GPU.
- **Success condition**: Exposure-grounded negatives with smooth hardness control and improved recommendation quality.

## Initial Method Summary

The initial proposal trains a conditional flow matching model on exposed-but-not-clicked negatives and uses a generate-then-reweight decoder:

```text
x_0 ~ N(0, I)
x_1 = ODESolve(v_theta, x_0, u)
C_M = FAISS_topM(x_1)
p(k) proportional to exp(s_rec(u, e_k) / tau_h)
j ~ p(k)
```

## User Concern

The ANN candidate retrieval plus recommender-score softmax is likely too trivial as the core hardness mechanism. It makes hardness a post-hoc hard-mining step rather than a property of the learned generative distribution.

