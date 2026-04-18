# Research Proposal: FlowNeg-Tilt -- KL-Constrained Exposure-Grounded Hard Negative Generation via Conditional Flow Matching

## Problem Anchor

- **Bottom-line problem**: Negative sampling in recommender systems with implicit feedback cannot simultaneously guarantee that negative samples are (1) truly negative (not false negatives) and (2) hard enough to provide strong training gradients. Exposure data (items shown but not interacted with) provides a direct signal for true negatives but is sparse and underutilized.
- **Must-solve bottleneck**: Existing methods conflate realness and hardness. RNS introduces false negatives. DNS amplifies false negatives. Generative methods (IRGAN, MixGCF) do not model the true negative distribution and ignore exposure data.
- **Non-goals**: New recommendation architecture. Content-based filtering. Positive augmentation.
- **Constraints**: Neurips. Data-centric framing. Million-item scale. Standard GPU.
- **Success condition**: Exposure-grounded negatives with smooth hardness control and improved recommendation quality.

## Technical Gap

The original FlowNeg proposal uses conditional flow matching to generate exposure-grounded negatives, then increases difficulty by ANN retrieval plus recommender-score softmax. This is feasible but makes hardness a post-hoc hard-mining heuristic. FlowNeg-Tilt instead defines hardness inside the target distribution learned by the flow.

## Method Thesis

FlowNeg-Tilt learns a beta-conditioned conditional flow whose target distribution is the KL-constrained energy tilt of the exposure-negative distribution. Increasing beta makes negatives harder, while the KL constraint and exposure base distribution preserve authenticity.

## Contribution Focus

- Dominant contribution: KL-constrained exposure-distribution tilting for controllable hard negative generation, implemented with beta-conditioned weighted conditional flow matching.
- Supporting contribution: distance-only catalog projection, where ANN is only a continuous-to-discrete quantizer.
- Explicit non-contributions: new recommender backbone, RL item policy, adversarial discriminator, content features, or new ANN method.

## Core Mechanism

Let `p_E(x | u)` be the exposure-grounded negative distribution in item-embedding space and let `h_phi(u, x)` be a stopped-gradient hardness energy from the current recommender:

```text
h_phi(u, x) = clip((s_phi(u, x) - mean_E[s_phi(u, e)]) / std_E[s_phi(u, e)], -c, c)
```

For hardness level `beta >= 0`, define:

```text
q_beta*(x | u) = argmax_{q << p_E}
    E_{x ~ q}[h_phi(u, x)] - (1 / beta) KL(q || p_E)
```

The solution is:

```text
q_beta*(x | u) = p_E(x | u) exp(beta h_phi(u, x)) / Z_u(beta)
```

At `beta = 0`, the target is the exposure-negative distribution. Larger beta shifts mass toward harder negatives while staying grounded in exposure negatives.

## Training Plan

Train one beta-conditioned velocity MLP `v_theta(x_t, u, beta)` with weighted CondOT:

```text
For each (u, e_n in E_u^neg):
  beta ~ Uniform(0, beta_max)
  x_0 ~ N(0, I), t ~ Uniform(0, 1)
  x_t = (1 - t) x_0 + t e_n
  w_beta(u, e_n) = normalize(exp(beta stopgrad(h_phi(u, e_n))))
  w_beta = clip(w_beta, w_min, w_max)

Minimize:
  E[w_beta(u, e_n) ||v_theta(x_t, u, beta) - (e_n - x_0)||_2^2]
```

The recommender score is used only to define stopped-gradient target weights. It is not used for inference-time candidate sorting.

## Inference Path

```text
Input: user u, hardness beta
x_0 ~ N(0, I)
x_1 = ODESolve(v_theta(., u, beta), x_0)
C_M = FAISS_topM_by_distance(x_1)
Filter positives and optionally reject items outside exposure-support density threshold
Return nearest valid item, or sample using distance-only weights
```

ANN remains only a projection from continuous generated embedding to discrete item ID.

## Integration

1. Warm up the recommender with RNS for 5-10 epochs.
2. Compute stopped-gradient exposure-negative hardness scores.
3. Train the beta-conditioned flow with weighted CFM.
4. Train the recommender with BPR using generated negatives at a scheduled beta.
5. Refresh scores and fine-tune the flow every `K` epochs.

## Failure Modes and Diagnostics

- Weight collapse: monitor effective sample size and clip weights.
- Over-hardening: monitor support purity and FN proxy as beta increases.
- Sparse-user exposure: mix with beta=0 exposure flow or a population exposure prior.
- Stale hardness scores: refresh periodically and monitor score drift.

## Claim-Driven Validation Sketch

### Claim 1: Authenticity is preserved

- Minimal experiment: compare FN proxy and exposure-support density across beta values.
- Baselines / ablations: RNS, DNS, exposure-weighted sampler, original generate-then-reweight FlowNeg.
- Metric: FN proxy, support purity, unique negatives.
- Expected evidence: support purity remains close to beta=0 exposure flow and better than standard hard mining.

### Claim 2: Hardness is controlled without post-hoc sorting

- Minimal experiment: sweep beta with no recommender-score sorting in decoding.
- Baselines / ablations: beta=0, unweighted CFM, original ANN+softmax reweighting.
- Metric: average negative score, BPR gradient magnitude, Recall@20/NDCG@20.
- Expected evidence: hardness increases smoothly with beta while FN proxy grows more slowly than DNS/hard mining.

### Claim 3: The learned tilted flow matters

- Minimal experiment: compare query mechanisms with the same distance-only decoder.
- Baselines / ablations: user embedding, exposure centroid, exposure KNN, unweighted exposure flow, beta-tilted flow.
- Metric: Recall@20/NDCG@20, support purity, diversity.
- Expected evidence: beta-tilted flow beats centroid/KNN and unweighted flow.

## Compute & Timeline Estimate

- Estimated GPU-hours: 80-150 A100 GPU-hours for full validation; lower for ML-100K/MIND-small pilot.
- Data / annotation cost: zero with public exposure datasets.
- Timeline: 1 week implementation, 1 week pilot, 2-3 weeks main experiments, 1-2 weeks ablations and writing.
