# Research Proposal: FlowNeg -- Exposure-Grounded Negative Generation with Controllable Hardness via Conditional Flow Matching

## Problem Anchor

- **Bottom-line problem**: Negative sampling in recommender systems with implicit feedback cannot simultaneously guarantee that negative samples are (1) truly negative (not false negatives) and (2) hard enough to provide strong training gradients. Exposure data (items shown but not interacted with) provides a direct signal for true negatives but is sparse and underutilized.
- **Must-solve bottleneck**: Existing methods conflate realness and hardness. RNS introduces false negatives. DNS amplifies false negatives. Generative methods (IRGAN, MixGCF) do not model the true negative distribution and ignore exposure data.
- **Non-goals**: New recommendation architecture. Content-based filtering. Positive augmentation.
- **Constraints**: SIGMOD 2026 Round 2. Data-centric framing. Million-item scale. Standard GPU.
- **Success condition**: Exposure-grounded negatives with smooth hardness control and improved recommendation quality.

## Technical Gap

Current methods each fail in a specific way: RNS (false negatives, no hardness), DNS (amplifies FN), PNS (popularity bias), IRGAN (adversarial instability, no exposure grounding), MixGCF (interpolation, no distribution modeling).

**Core claim**: Conditional Flow Matching (CondOT) learns an exposure-grounded negative proposal from exposed-but-not-clicked items. A generate-then-reweight decoding applies candidate-set-restricted hardening: the flow generates diverse points in the negative support, ANN retrieval restricts to proximal items, recommender-score reweighting selects harder items. Realness and hardness are controlled by structurally independent mechanisms.

## Method Thesis

FlowNeg learns an exposure-grounded negative proposal via conditional Flow Matching and applies generate-then-reweight decoding for candidate-set-restricted hardness control. Smallest adequate intervention: one small MLP + ANN + score reweighting.

## Contribution Focus

- **Dominant**: Exposure-conditioned CFM for negative generation + generate-then-reweight for structurally independent hardness control.
- **Non-contributions**: Backbone design, hardness scheduling (impl detail), sparse-user handling (appendix), ANN indexing (off-the-shelf).

## Proposed Method

### Complexity Budget
- **Frozen/reused**: Item embeddings, backbone recommender, BPR loss, FAISS.
- **New**: One velocity network v_t^theta(x_t, u) -- small MLP (~35K params for d=64). The ONLY new trainable component.
- **Excluded**: Adversarial discriminator, learned exposure model, score-stratified training (appendix), consistency distillation.

### System Overview

CFM Training (Standard CondOT):
  For each (u, e_n in E_u^neg):
    x_0 ~ N(0, I), t ~ U(0, 1)
    x_t = (1-t)*x_0 + t*e_n
    Minimize: ||v_t^theta(x_t, u) - (e_n - x_0)||^2

Generate-Then-Reweight:
  x_0 ~ N(0, I)
  x_1 = Midpoint4(v_t^theta, x_0, u)     4-step midpoint, t: 0->1
  C_M = FAISS_topM(x_1)                   M nearest items
  p(k) proportional to exp(s_rec(u,e_k)/tau_h)   hardness reweighting
  j ~ p(k)

Recommender Training:
  L_BPR = -ln sigma(r_ui - r_uj)
  Every K epochs: rebuild FAISS, fine-tune CFM

### Core Mechanism

Step 1 -- Generate (Realness + Diversity): v_t^theta transports Gaussian noise to exposure-negative distribution conditioned on user. 4-step midpoint solver on CondOT straight paths. Different x_0 -> different x_1 -> diverse negatives.

Step 2 -- Retrieve (Candidate Restriction): FAISS top-M restricts to neighborhood of generated point.

Step 3 -- Reweight (Hardness): p(k) proportional to exp(s_rec(u,e_k)/tau_h). Low tau_h = harder. Structurally independent from generation.

### Velocity Network
- Input: [x_t; u; PE(t)]. 2 layers SiLU width 4d. Output R^d. ~35K params.
- Loss: E[||v_t^theta(x_t, u) - (e_n - x_0)||^2], CondOT path.

### 4-Step Midpoint Solver
- Time grid: {0, 0.25, 0.5, 0.75, 1.0}. 4 MLP evaluations. Negligible cost.

### Integration
- Replaces negative sampling. Backbone unchanged.
- Warm-up RNS (5-10 epochs) -> train CFM (50-100 epochs) -> alternate: generate + train recommender + update CFM every K=5 epochs.
- tau_h schedule (impl detail): decrease over epochs.

### Failure Modes
- Mode collapse: monitor entropy, unique items.
- Candidate set quality: monitor FN contamination.
- Stale embeddings: monitor cosine drift.

### Novelty
(1) First CFM for negative sampling. (2) Exposure grounding. (3) Generate-then-reweight for structural hardness independence.
One MLP + 4-step solver + FAISS + softmax. No adversarial, no gradients through recommender.

## Validation

### Claim 1: Exposure-grounded realness
KuaiRand/Coat. FN rate and candidate-set purity vs. RNS, DNS, MixGCF, exposure-weighted.

### Claim 2: Candidate-set-restricted hardness
Vary tau_h. Plot hardness/FN/Recall@20. Ablation: no reweighting.

### Claim 3: Flow query value (CRITICAL TEST)
5 query baselines (same retrieve-reweight pipeline):
(a) Random Gaussian, (b) user embedding, (c) exposure-negative centroid, (d) exposure-negative KNN, (e) FlowNeg.
Must clearly beat centroid/KNN to justify generative model.

### Claim 4: End-to-end improvement
KuaiRand/Coat. MF/LightGCN. Recall@20, NDCG@20 vs. RNS, DNS, MixGCF, IRGAN, exposure-weighted.
Supplementary: Gowalla, Yelp2018.

### Diversity Diagnostics
Noise sensitivity, candidate set diversity, mode coverage.

### Dense vs. Sparse
Reported separately. Dense-user is the headline; sparse-user is boundary analysis.

## Compute and Timeline
- ~80-150 A100 GPU-hours. Zero data cost. 6-7 weeks total.
