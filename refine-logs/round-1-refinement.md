# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**: Negative sampling in recommender systems with implicit feedback cannot simultaneously guarantee that negative samples are (1) truly negative (not false negatives) and (2) hard enough to provide strong training gradients. Exposure data (items shown but not interacted with) provides a direct signal for true negatives but is sparse and underutilized.
- **Must-solve bottleneck**: Existing methods conflate realness and hardness. RNS introduces false negatives. DNS amplifies false negatives. Generative methods (IRGAN, MixGCF) do not model the true negative distribution and ignore exposure data.
- **Non-goals**: New recommendation architecture. Content-based filtering. Positive augmentation.
- **Constraints**: Neurips. Data-centric framing. Million-item scale. Standard GPU.
- **Success condition**: Exposure-grounded negatives with smooth hardness control and improved recommendation quality.

## Anchor Check

- Original bottleneck: true-negative grounding and useful hardness are currently entangled.
- Why the revised method still addresses it: the flow learns a KL-constrained hardness-tilted distribution whose base measure is the exposure-negative distribution.
- Reviewer suggestions rejected as drift: full RL item-selection policy, because it shifts the paper toward training-controller design and weakens the authenticity guarantee.

## Simplicity Check

- Dominant contribution after revision: energy-tilted exposure-conditioned flow matching for controllable hard negative generation.
- Components removed or merged: post-hoc recommender-score softmax over ANN candidates is removed as a core mechanism.
- Reviewer suggestions rejected as unnecessary complexity: per-sample latent optimization is not the main route because it adds inference cost and a new support-density penalty.
- Why the remaining mechanism is the smallest adequate route: it changes only the flow target distribution and adds `beta` conditioning; the recommender, BPR loss, item embeddings, and FAISS projection remain reused.

## Changes Made

### 1. Replaced post-hoc sorting with KL-constrained hardness tilting

- Reviewer said: ANN+score reweighting is too trivial as the core hardness mechanism.
- Action: define `q_beta*(x | u) proportional to p_E(x | u) exp(beta h_phi(u, x))`.
- Reasoning: this is the entropy-regularized solution to "increase hardness while staying close to exposure negatives."
- Impact on core method: hardness becomes part of the learned generative distribution, not a downstream candidate-ranking heuristic.

### 2. Kept ANN only as quantization

- Reviewer said: delete sorting as contribution.
- Action: keep nearest-neighbor retrieval only to map continuous generated embeddings to discrete catalog items.
- Reasoning: every continuous embedding generator needs a catalog projection; this should not be sold as novelty.
- Impact on core method: the proposal is cleaner and less engineering-driven.

### 3. Rejected RL as the main route

- Reviewer said: RL/bandit is possible but risky.
- Action: use RL only as a possible future scheduler for `beta`, not as the core method.
- Reasoning: RL adds delayed rewards, instability, a large action space, and weaker realness guarantees.
- Impact on core method: one-paper focus is preserved.

## Revised Proposal

# Research Proposal: FlowNeg-Tilt -- KL-Constrained Exposure-Grounded Hard Negative Generation via Conditional Flow Matching

## Problem Anchor

- **Bottom-line problem**: Negative sampling in recommender systems with implicit feedback cannot simultaneously guarantee that negative samples are (1) truly negative (not false negatives) and (2) hard enough to provide strong training gradients. Exposure data (items shown but not interacted with) provides a direct signal for true negatives but is sparse and underutilized.
- **Must-solve bottleneck**: Existing methods conflate realness and hardness. RNS introduces false negatives. DNS amplifies false negatives. Generative methods (IRGAN, MixGCF) do not model the true negative distribution and ignore exposure data.
- **Non-goals**: New recommendation architecture. Content-based filtering. Positive augmentation.
- **Constraints**: Neurips. Data-centric framing. Million-item scale. Standard GPU.
- **Success condition**: Exposure-grounded negatives with smooth hardness control and improved recommendation quality.

## Technical Gap

The previous generate-then-reweight route learns an exposure-negative flow, then makes samples harder by sorting ANN candidates with the recommender score. That preserves feasibility but makes the key hardness mechanism look like standard hard negative mining. A sharper mechanism is to define the target distribution itself as the closest hardness-tilted distribution to the exposure-negative base distribution.

## Method Thesis

FlowNeg-Tilt learns a beta-conditioned conditional flow whose target distribution is the KL-constrained energy tilt of the exposure-negative distribution. The flow samples harder negatives as beta increases while staying absolutely continuous with, and close to, the exposure-grounded true-negative distribution.

## Contribution Focus

- Dominant contribution: KL-constrained exposure-distribution tilting for controllable hard negative generation, implemented with beta-conditioned weighted conditional flow matching.
- Supporting contribution: a simple discrete decoding rule that uses ANN only as quantization, not as a hardness selector.
- Explicit non-contributions: new recommender backbone, adversarial discriminator, learned RL policy, content features, or new ANN indexing.

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: MF or LightGCN/SASRec recommender, item embeddings, BPR loss, FAISS nearest-neighbor search.
- New trainable components: one beta-conditioned velocity MLP `v_theta(x_t, u, beta)`.
- Tempting additions intentionally not used: RL item policy, discriminator, per-sample latent optimization, LLM/VLM features, score-sorted candidate selection.

### Core Variational Mechanism

Let `p_E(x | u)` denote the exposure-grounded negative distribution over item-embedding space, estimated from exposed-but-not-clicked items for user `u`.

Let `h_phi(u, x)` be a stopped-gradient hardness energy from the current recommender. A practical choice is a normalized score:

```text
h_phi(u, x) = clip((s_phi(u, x) - mean_E[s_phi(u, e)]) / std_E[s_phi(u, e)], -c, c)
```

For a hardness parameter `beta >= 0`, define the target negative distribution:

```text
q_beta*(x | u) = argmax_{q << p_E}
    E_{x ~ q}[h_phi(u, x)] - (1 / beta) KL(q || p_E)
```

The closed-form solution is:

```text
q_beta*(x | u) = p_E(x | u) exp(beta h_phi(u, x)) / Z_u(beta)
```

At `beta = 0`, the target is the exposure-negative distribution. As `beta` increases, probability mass shifts toward harder negatives, but the KL term and absolute-continuity constraint keep the distribution exposure-grounded.

### Weighted Conditional Flow Matching

Train the beta-conditioned velocity network with a weighted CondOT objective:

```text
For each (u, e_n in E_u^neg):
  beta ~ Uniform(0, beta_max)
  x_0 ~ N(0, I), t ~ Uniform(0, 1)
  x_t = (1 - t) x_0 + t e_n
  w_beta(u, e_n) = normalize_user_or_batch(exp(beta stopgrad(h_phi(u, e_n))))
  w_beta = clip(w_beta, w_min, w_max)

Minimize:
  E[w_beta(u, e_n) ||v_theta(x_t, u, beta) - (e_n - x_0)||_2^2]
```

The recommender score is stopped-gradient and periodically refreshed. It is used to define the tilted target distribution, not to sort generated candidates at inference.

### Inference

```text
Input: user u, hardness beta
x_0 ~ N(0, I)
x_1 = ODESolve(v_theta(., u, beta), x_0)
C_M = FAISS_topM_by_distance(x_1)
Filter positives and optionally reject items outside an exposure-support density threshold
Return nearest valid item, or sample by distance-only weights
```

No recommender-score sorting is used in the decoder. The only role of ANN is to project a continuous generated embedding back to the discrete item catalog.

### Integration into Recommender Training

1. Warm up the base recommender with RNS for 5-10 epochs.
2. Compute stopped-gradient exposure-negative hardness scores.
3. Train `v_theta(x_t, u, beta)` with weighted CFM over beta values.
4. Train the recommender with BPR using generated negatives at a scheduled beta.
5. Refresh scores and fine-tune the flow every `K` epochs.

### Why This Is Smaller Than RL or Noise Optimization

RL would require a policy, reward design, credit assignment, and exploration over a million-item action space. Latent noise optimization would require per-sample inference optimization and a differentiable exposure-support penalty. FlowNeg-Tilt instead changes the target measure of the existing flow and adds only beta conditioning plus sample weights.

### Failure Modes and Diagnostics

- Weight collapse: monitor effective sample size of `w_beta`; mitigate with clipping and beta schedule.
- Over-hardening: monitor exposure-support purity and false-negative proxies as beta increases.
- Sparse exposure users: mix with `beta = 0` exposure flow or population-level exposure prior.
- Stale scores: refresh stopped-gradient hardness scores every `K` epochs and monitor score drift.

### Novelty and Elegance Argument

The novelty is not "generate then sort." The paper's mechanism claim is that exposure-grounded hard negatives can be obtained as an entropy-regularized variational projection: stay close to the exposure-negative distribution while tilting toward high recommender-score negatives. Conditional flow matching is the amortized sampler for this target family.

## Claim-Driven Validation Sketch

### Claim 1: Authenticity is preserved by exposure grounding

- Minimal experiment: compare false-negative proxy rate and exposure-support density across beta values.
- Baselines / ablations: RNS, DNS, exposure-weighted sampler, original generate-then-reweight FlowNeg.
- Metric: FN proxy, support purity, unique negatives.
- Expected evidence: support purity remains close to beta=0 exposure flow and better than DNS/hard mining.

### Claim 2: Hardness is controlled by beta without post-hoc sorting

- Minimal experiment: sweep beta with no recommender-score sorting in the decoder.
- Baselines / ablations: beta=0, original ANN+softmax reweighting, unweighted CFM.
- Metric: average negative score, BPR gradient magnitude, Recall@20/NDCG@20.
- Expected evidence: hardness increases smoothly with beta while FN proxy rises slower than standard hard mining.

### Claim 3: The flow target matters

- Minimal experiment: compare query mechanisms under the same distance-only decoder.
- Baselines / ablations: user embedding, exposure centroid, exposure KNN, unweighted exposure flow, beta-tilted flow.
- Metric: Recall@20/NDCG@20, diversity, support purity.
- Expected evidence: beta-tilted flow beats centroid/KNN and unweighted flow, showing value beyond ANN projection.

## Experiment Handoff Inputs

- Must-prove claims: beta controls hardness; exposure grounding preserves realness; learned tilted flow beats centroid/KNN and generate-then-sort.
- Must-run ablations: remove beta, remove weights, restore post-hoc sorting, vary weight clipping, vary beta schedule.
- Critical datasets / metrics: KuaiRand and Coat for exposure/FN analysis; ML-100K or MIND-small for fast pipeline debugging; Recall@20/NDCG@20 plus FN proxy/support purity.
- Highest-risk assumptions: exposure-not-clicked reliably approximates true negatives; recommender scores are stable enough for periodic stopped-gradient target updates.

## Compute & Timeline Estimate

- Estimated GPU-hours: 80-150 A100 GPU-hours for full validation; much lower for ML-100K/MIND-small pilots.
- Data / annotation cost: zero if using public exposure datasets.
- Timeline: 1 week implementation, 1 week pilot, 2-3 weeks main experiments, 1-2 weeks ablations and writing.

