# Round 1 Review

## External Reviewer Status

The Claude review MCP job was attempted but failed in this environment:

```text
jobId: 75e5a59ea1544b7797c3d3a1de2dabe7
status: failed
error: API Error: 400 default channel restriction: interface only available for Claude Code client
```

The review below applies the same `research-refine` criteria locally.

## Scores

| Dimension | Score | Notes |
|---|---:|---|
| Problem Fidelity | 9 | The proposal preserves the realness-vs-hardness bottleneck and uses exposure data directly. |
| Method Specificity | 8 | CFM training and decoding are concrete, but hardness is not integrated into the generative objective. |
| Contribution Quality | 7 | Exposure-conditioned CFM is plausible, but ANN+score reweighting reads like standard hard mining. |
| Frontier Leverage | 8 | Flow matching is appropriate; the hardness mechanism is less elegant than the generative framing. |
| Feasibility | 8 | One MLP plus FAISS is feasible, but alternating retraining and score sorting need careful cost control. |
| Validation Focus | 8 | The flow query value test is strong; add a direct test that hardness is learned without post-hoc sorting. |
| Venue Readiness | 7 | Current paper risks being perceived as "CFM proposal + existing hard negative selection." |
| Overall | 7.8 | Promising but should revise the hardness mechanism. |

Verdict: REVISE.

## Is ANN + Sorting/Reweighting Too Trivial?

Yes, if it remains the named hardness contribution. ANN is acceptable as a continuous-to-discrete projection step, but recommender-score sorting/softmax should not be the core method. The core method should make hardness a property of the target negative distribution learned by the flow.

## Alternative Comparison

### A. Constrained noise optimization / guided flow

Define a generated embedding `x = G_theta(z, u)` and optimize:

```text
z* = argmax_z h_phi(u, G_theta(z, u))
     - lambda D_E(G_theta(z, u), u)
     - eta ||z||_2^2
```

where `h_phi` is a stopped-gradient recommender hardness energy and `D_E` is an exposure-support penalty.

Pros: Directly mathematical; keeps authenticity as a constraint.

Cons: Adds per-sample inference-time optimization, risks mode collapse, and requires a reliable differentiable exposure-density penalty. It is better as an ablation or diagnostic than the main route.

### B. RL / bandit hardness policy

Learn a policy that chooses a hardness temperature or candidate item using reward:

```text
R = Delta validation proxy or BPR loss improvement
    - lambda false_negative_penalty
    - gamma diversity_penalty
```

Pros: Adapts hardness to training dynamics.

Cons: High variance, delayed reward, million-item action space, weaker authenticity guarantee, and contribution sprawl. It also resembles a training controller rather than a clean negative distribution model.

### C. Energy-tilted exposure distribution / constrained variational projection

Let `p_E(x | u)` be the exposure-grounded true-negative distribution and `h_phi(u, x)` be a stopped-gradient hardness energy from the current recommender. Define the target negative distribution as:

```text
q_beta*(x | u) = argmax_{q << p_E}
    E_{x ~ q}[h_phi(u, x)] - (1 / beta) KL(q || p_E)
```

with closed-form solution:

```text
q_beta*(x | u) = p_E(x | u) exp(beta h_phi(u, x)) / Z_u(beta)
```

Pros: The method cleanly separates realness and hardness. Realness is preserved by absolute continuity and KL proximity to `p_E`; hardness is controlled by `beta`; increasing `beta` monotonically increases expected hardness under mild conditions. It can be trained with weighted CFM and does not require post-hoc sorting.

Cons: Needs score normalization and clipping to avoid collapse toward a few high-score negatives.

## Recommendation

Use route C as the main method. Delete recommender-score softmax over FAISS candidates as the hardness mechanism. Retain FAISS only for nearest-neighbor quantization from continuous generated embedding to catalog item.

Top action items:

1. Replace generate-then-reweight with beta-conditioned energy-tilted CFM.
2. Add the variational objective and closed-form tilted target distribution as the main novelty.
3. Add validation showing hardness increases with beta without post-hoc score sorting, while exposure-support purity remains stable.

Drift warning: NONE. The recommended route still solves exposure-grounded hard negative sampling.

