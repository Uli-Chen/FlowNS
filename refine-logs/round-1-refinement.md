# Round 1 Refinement

## Problem Anchor
[Verbatim from round 0]
- **Bottom-line problem**: In implicit-feedback recommender systems trained with pairwise ranking losses (e.g., BPR), negative sampling quality directly governs ranking performance. Current methods either sample uniformly (uninformative), heuristically select hard negatives (false-negative contamination), or ignore exposure data.
- **Must-solve bottleneck**: There is no principled mechanism that simultaneously (1) generates negatives faithful to the true negative distribution, (2) controls hardness in a mathematically grounded way, and (3) provides formal guarantees against false-negative contamination.
- **Non-goals**: We are NOT building a general-purpose generative recommendation model, NOT replacing the recommendation backbone, and NOT pursuing state-of-the-art on every benchmark through engineering tricks. We are NOT designing a flashy multi-module system.
- **Constraints**: Single-GPU training feasible (A100 or equivalent); standard RecSys datasets (Yelp, Amazon, Gowalla); flow model operates in the item embedding space (dimensionality d ≤ 128); the recommendation backbone is a standard collaborative filtering model (e.g., LightGCN, MF).
- **Success condition**: A single, mathematically principled mechanism that provably generates negatives that are both real (from the true negative distribution) and hard (near the decision boundary), with formal analysis of the reward's properties and empirical validation on standard benchmarks.

## Anchor Check
- **Original bottleneck**: The hardness-fidelity tradeoff in negative sampling — no existing method formally controls both hardness and false-negative contamination.
- **Why the revised method still addresses it**: The core reward R = W^a(1-W)^γ and its dual KL+shaping guarantee remain unchanged. The revisions sharpen the validation, not the mechanism.
- **Reviewer suggestions rejected as drift**: None — all reviewer suggestions are orthogonal to the problem anchor.

## Simplicity Check
- **Dominant contribution after revision**: Unchanged — the boundary-aware shaped reward R = W^a(1-W)^γ with formal false-negative control.
- **Components removed or merged**:
  1. ODE-to-SDE conversion demoted from dedicated section to sampling procedure subsection
  2. GRPO-Guard ratio normalization moved to implementation detail
  3. Training phases simplified from 4 to 3 (warmup merged into flow pre-training)
- **Reviewer suggestions rejected as unnecessary complexity**: None
- **Why the remaining mechanism is still the smallest adequate route**: The shaped reward is the only new equation; everything else is reused. Now the paper presentation matches this reality by de-emphasizing the reused components.

## Changes Made

### 1. Added low-dimensional generator justification + ablation
- **Reviewer said**: No argument for why flow matching is necessary in d=64-128 embedding space. Simpler models (GMM, rejection sampling) might suffice.
- **Action**: Added a principled argument (multimodal negative distributions across diverse users cannot be captured by GMMs) AND added an ablation experiment (Claim 4) comparing flow model vs. conditional GMM vs. rejection sampling, all using the same shaped reward.
- **Reasoning**: The reviewer is right that this needs justification. However, we argue flow matching is necessary because: (a) the negative distribution is user-conditional and multimodal (different users have different negative preference clusters), (b) GMMs require pre-specifying the number of modes, (c) rejection sampling from the recommender's score distribution reintroduces the circular dependency. The ablation will empirically validate this.
- **Impact on core method**: None — the reward design (core contribution) is unchanged. This adds one ablation experiment.

### 2. Added guarantee-tightness experiment
- **Reviewer said**: Risk that formal guarantees feel disconnected from practical performance.
- **Action**: Added to Claim 1 a sub-experiment measuring actual FN rate vs. theoretical bound across β values. This bridges theory and practice by showing the bound is empirically useful, not just a mathematical artifact.
- **Reasoning**: This is a low-cost addition (same experimental setup, just measure and plot one additional quantity) that significantly strengthens the theoretical contribution's practical relevance.
- **Impact on core method**: None — adds a measurement, not a mechanism change.

### 3. Simplified presentation structure
- **Reviewer said**: Three simplification opportunities (ODE-to-SDE emphasis, GRPO-Guard subsection, 4-phase training).
- **Action**: Accepted all three.
  - ODE-to-SDE conversion is now a subsection under "Sampling Procedure" rather than a highlighted component
  - Ratio normalization is noted as an implementation detail, not a named mechanism
  - Training pipeline simplified to 3 phases (flow pre-training with warmup, GRPO fine-tuning, joint training)
- **Reasoning**: These changes reduce perceived complexity without changing the method. The paper now reads as "one new reward function applied to a standard pipeline."
- **Impact on core method**: Presentation only. No mechanism change.

### 4. Added one generative baseline
- **Reviewer said**: Missing modern generative baseline (MINOR).
- **Action**: Added conditional VAE baseline to experiment plan. This is one additional row in the results table.
- **Reasoning**: A CVAE is the simplest generative baseline and contextualizes the flow matching choice without expanding the experiment matrix significantly.
- **Impact on core method**: None — adds one baseline.

## Revised Proposal

# Research Proposal: FlowNS — Flow Matching with Principled RL-Guided Hardness for Negative Sampling

## Problem Anchor

- **Bottom-line problem**: In implicit-feedback recommender systems trained with pairwise ranking losses (e.g., BPR), negative sampling quality directly governs ranking performance. Current methods either sample uniformly (uninformative), heuristically select hard negatives (false-negative contamination), or ignore exposure data.
- **Must-solve bottleneck**: There is no principled mechanism that simultaneously (1) generates negatives faithful to the true negative distribution, (2) controls hardness in a mathematically grounded way, and (3) provides formal guarantees against false-negative contamination.
- **Non-goals**: We are NOT building a general-purpose generative recommendation model, NOT replacing the recommendation backbone, and NOT pursuing state-of-the-art on every benchmark through engineering tricks. We are NOT designing a flashy multi-module system.
- **Constraints**: Single-GPU training feasible (A100 or equivalent); standard RecSys datasets (Yelp, Amazon, Gowalla); flow model operates in the item embedding space (dimensionality $d \leq 128$); the recommendation backbone is a standard collaborative filtering model (e.g., LightGCN, MF).
- **Success condition**: A single, mathematically principled mechanism that provably generates negatives that are both real (from the true negative distribution) and hard (near the decision boundary), with formal analysis of the reward's properties and empirical validation on standard benchmarks.

## Technical Gap

### Where Current Methods Fail

Current negative sampling methods in recommender systems fall into three categories, each with a clear failure mode:

1. **Uniform / popularity-based sampling** (BPR): Most sampled negatives have near-zero gradient contribution. The BPR gradient weight is $\sigma(r_{uj} - r_{ui^+})$, which is exponentially small when $r_{uj} \ll r_{ui^+}$.

2. **Hard negative mining** (DNS, ANCE, MixGCF): Selects negatives with the highest recommender scores. Shi et al. (WWW 2023) prove this implicitly optimizes One-way Partial AUC (OPAUC), which better aligns with Top-K ranking. However, the hardest negatives are precisely those most likely to be false negatives, creating a hardness-fidelity tradeoff with no formal resolution.

3. **Generative negative sampling** (IRGAN, AdvInfoNCE): Uses adversarial training to generate hard negatives. Suffers from GAN training instability and lacks explicit control over the hardness-fidelity tradeoff.

### Why Naive Extensions Fail

- **Larger models / more data**: Does not address the fundamental hardness-fidelity tradeoff.
- **Exposure-aware heuristics**: Exposure data is sparse and cannot cover the full negative space.
- **Simply adding a flow model**: A vanilla flow model can learn the negative distribution but provides no mechanism to control hardness.
- **Naively applying RL to increase hardness**: Using the recommender's raw score as reward creates circular dependency, risks reward hacking (unbounded reward → mode collapse), and conflates hardness with informativeness.
- **Simpler generative models (GMM, rejection sampling)**: The negative preference distribution is user-conditional and multimodal — different users dislike items for qualitatively different reasons. GMMs require pre-specifying mode count, cannot capture complex geometry in embedding space, and scale poorly with user diversity. Rejection sampling from the recommender's score distribution reintroduces the circular dependency that we aim to avoid.

### The Smallest Adequate Intervention

A conditional flow matching model to capture the true negative distribution, combined with a single RL fine-tuning stage (GRPO) whose reward is derived from first principles.

## Method Thesis

- **One-sentence thesis**: FlowNS learns the true negative distribution via conditional flow matching, then applies GRPO with a mathematically derived *boundary-aware shaped reward* that maximizes training informativeness while provably controlling false-negative contamination through a dual mechanism of reward shaping and distributional anchoring.
- **Why this is the smallest adequate intervention**: The flow model handles realness; a single reward function handles hardness; the KL constraint handles fidelity. No additional modules, discriminators, or auxiliary losses are needed.
- **Why this route is timely**: Flow matching provides the cleanest generative framework for continuous embeddings; GRPO avoids the need for a learned value function; the ODE-to-SDE conversion enables exploration while preserving marginals.

## Contribution Focus

- **Dominant contribution**: A principled reward function for RL-guided negative sample generation, derived from OPAUC theory with formal analysis of its boundedness, false-negative correction, and optimality properties.
- **Supporting contribution**: The complete FlowNS pipeline demonstrating that this minimal architecture suffices.
- **Explicit non-contributions**: We do not claim novelty in flow matching itself, in GRPO itself, or in the ODE-to-SDE conversion. Our contribution is the principled integration via the reward design.

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: Recommendation model (LightGCN / MF) — frozen during GRPO, updated during joint training; item embeddings — shared between recommender and flow model.
- **New trainable components**: (1) Conditional flow matching velocity network $v_\theta(\mathbf{x}_t, u, t)$; (2) Same network fine-tuned via GRPO (no additional parameters).
- **Tempting additions intentionally not used**: Learned reward model (unnecessary — reward is analytical); discriminator (adds instability); separate hardness predictor (redundant with shaped reward); multi-stage curriculum (subsumed by GRPO's natural progression).

### System Overview

```
Phase 1 (Flow Pre-training): Learn True Negative Distribution
  Train recommender with uniform negatives → obtain initial embeddings
  Simultaneously train flow model on exposed negatives
  Noise x_0 ~ N(0,I) --[Cond. Flow Matching]--> Negative embedding x_1
  Loss: L_CFM(θ) = E[||v_θ(x_t, u, t) - (e_n - x_0)||^2]
  Output: π_ref (reference flow policy)

Phase 2 (GRPO Fine-tuning): Add Hardness via Shaped Reward
  For each user u:
    Sample G trajectories via SDE: {x_0^i -> ... -> x_1^i}_{i=1}^G
    Compute reward: R(x_1^i, u) = W^a · (1-W)^γ  [boundary-aware shaped reward]
    Compute group advantage: Â^i = (R^i - mean) / std
    Update v_θ via clipped surrogate + KL penalty
  Output: π_θ (hardness-enhanced flow policy)

Phase 3 (Joint Training): Co-evolve Recommender and Generator
  Alternate:
    (a) Generate hard negatives with π_θ, train recommender
    (b) Re-run GRPO with updated recommender scores (few steps)
```

### Core Mechanism: Boundary-Aware Shaped Reward

This is the central technical contribution. We derive the reward function from first principles.

#### Step 1: Define the Win Rate

For a generated negative embedding $\mathbf{x}_1$ and user $u$ with positive item set $\mathcal{I}_u^+$, define the **win rate** against positives:

$$W(\mathbf{x}_1, u) = \frac{1}{|\mathcal{I}_u^+|} \sum_{i \in \mathcal{I}_u^+} \sigma\bigl(s(u, \mathbf{x}_1) - s(u, i)\bigr)$$

where $s(u, \cdot)$ is the recommender's scoring function and $\sigma(\cdot)$ is the sigmoid function.

**Properties of $W$:**
- $W \in (0, 1)$ (bounded, since sigmoid outputs are in $(0,1)$)
- $W \approx 0$: the negative scores well below all positives (easy, uninformative)
- $W \approx 0.5$: the negative is at the decision boundary (maximally informative for BPR)
- $W \approx 1$: the negative beats most positives (likely a false negative)

**Connection to OPAUC**: Shi et al. (2023) prove that BPR + hard negative sampling implicitly optimizes OPAUC over FPR $\in [0, \beta]$. The win rate $W$ is the empirical estimate of the false positive rate for a single generated negative, directly connecting to the OPAUC framework.

#### Step 2: Shape the Reward to Penalize False Negatives

The naive reward "maximize $W$" would push toward false negatives. Instead, we apply a **boundary-aware shaping function**:

$$R(\mathbf{x}_1, u) = W(\mathbf{x}_1, u)^a \cdot \bigl(1 - W(\mathbf{x}_1, u)\bigr)^\gamma$$

where $a > 0$ controls the hardness incentive and $\gamma > 0$ controls the false-negative penalty strength.

**Mathematical properties:**

1. **Boundedness**: $R \in [0, R_{\max}]$ where $R_{\max} = \frac{a^a \gamma^\gamma}{(a+\gamma)^{a+\gamma}}$. This prevents reward hacking.

2. **Optimal hardness point**: Setting $\frac{\partial R}{\partial W} = 0$ gives:
$$W^* = \frac{a}{a + \gamma}$$
The reward naturally peaks at $W^*$ and decays on both sides.

3. **Asymmetric penalty**: When $\gamma > a$, false negatives (high $W$) are penalized more severely than easy negatives (low $W$).

4. **Connection to Beta distribution**: $R(W) \propto \text{Beta}(a+1, \gamma+1)$ density. The GRPO objective encourages the generator to produce negatives whose win-rate distribution concentrates around $W^*$.

5. **Special cases**:
   - $a = 1, \gamma = 1$: $R = W(1-W)$, symmetric, peaks at $W^* = 0.5$. The simplest choice — the reward equals the variance of a Bernoulli with parameter $W$.
   - $a = 1, \gamma = 2$: $R = W(1-W)^2$, peaks at $W^* = 1/3$ (stronger FN penalty)
   - $a = 1, \gamma = 3$: peaks at $W^* = 1/4$ (very conservative)

**Default**: $a = 1, \gamma = 1$ (symmetric), tuning $\gamma \in \{1, 2, 3\}$ via validation.

#### Step 3: Formal Analysis of False-Negative Control

**Dual mechanism against false negatives:**

**Mechanism 1 (Reward shaping)**: $\frac{\partial R}{\partial W}\big|_{W=1} = 0$ for any $\gamma > 0$. The GRPO policy gradient provides zero learning signal at the false-negative boundary.

**Mechanism 2 (KL anchoring)**: The KL penalty constrains the RL policy near the pre-trained flow model. By the Donsker-Varadhan representation, for any measurable set $A$:

$$\pi_\theta(A | u) \leq \pi_{\text{ref}}(A | u) \cdot \exp\left(\frac{R_{\max}}{\beta}\right)$$

Since $R$ is bounded, the density ratio is bounded by $\exp(R_{\max}/\beta)$.

**Combined false-negative guarantee**: Let $\mathcal{F}_u$ be the false negative set. The expected false-negative rate under the RL policy satisfies:

$$\mathbb{E}_{\mathbf{x} \sim \pi_\theta(\cdot|u)}[\mathbb{1}[\mathbf{x} \in \mathcal{F}_u]] \leq \exp\left(\frac{R_{\max}}{\beta}\right) \cdot \mathbb{E}_{\mathbf{x} \sim \pi_{\text{ref}}(\cdot|u)}[\mathbb{1}[\mathbf{x} \in \mathcal{F}_u]]$$

#### Step 4: Why Not Alternative Rewards?

| Alternative | Problem |
|------------|---------|
| Raw score $s(u, \mathbf{x})$ | Unbounded → reward hacking; no FN correction; scale-dependent |
| Sigmoid of score $\sigma(\alpha \cdot s)$ | No reference point; can't distinguish boundary-hard vs false-negative |
| Gradient norm $\|\nabla_\Theta \ell\|$ | Expensive; doesn't separate FN |
| Rank-based reward | Requires sorting; discontinuous |
| **Win-rate + Beta shaping (ours)** | Bounded; OPAUC-connected; closed-form FN penalty; single hyperparameter $\gamma$ |

### Flow Matching for True Negative Distribution

The velocity field $v_\theta(\mathbf{x}_t, u, t)$ is conditioned on user $u$ and trained with:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1],\, \mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I}),\, \mathbf{e}_n \sim \mathcal{E}_u^{\text{neg}}} \left[\left\| v_\theta\bigl((1-t)\mathbf{x}_0 + t\,\mathbf{e}_n,\, u,\, t\bigr) - (\mathbf{e}_n - \mathbf{x}_0) \right\|^2\right]$$

**Convention**: $t = 0$ is noise, $t = 1$ is data. Generation proceeds forward.

**Why flow matching over simpler models**: The user-conditional negative distribution is multimodal and varies across the user population. Flow matching captures arbitrary geometry in embedding space without mode-count assumptions (unlike GMMs) and without posterior collapse (unlike VAEs). We empirically validate this choice via ablation (Claim 4).

### Sampling Procedure

For GRPO training, we convert the deterministic flow ODE to an equivalent SDE that preserves marginal distributions while enabling stochastic exploration:

$$d\mathbf{x}_t = \left[\mathbf{v}_\theta + \frac{\sigma_t^2}{2} \nabla\log p_t(\mathbf{x}_t)\right] dt + \sigma_t\,d\mathbf{w}$$

where the score is approximated via Tweedie's formula as $\nabla\log p_t(\mathbf{x}_t) \approx -\frac{\mathbf{x}_t - t\,\mathbb{E}[\mathbf{x}_1 | \mathbf{x}_t]}{(1-t)^2}$ with $\mathbb{E}[\mathbf{x}_1 | \mathbf{x}_t] \approx \mathbf{x}_t + (1-t)\mathbf{v}_\theta$.

Noise schedule: $\sigma_t = \eta \cdot \frac{\sqrt{1-t}}{\sqrt{t} + \delta}$ with $\delta > 0$ preventing divergence at $t = 0$.

Euler-Maruyama discretization with step size $\Delta t$:
$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \tilde{\mathbf{v}}_\theta(\mathbf{x}_t, u, t)\,\Delta t + \sigma_t \sqrt{\Delta t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

### GRPO Objective

Sample $G$ trajectories per user. Compute trajectory-level advantage:

$$\hat{A}^i = \frac{R^i - \bar{R}}{\sigma_R + \epsilon_{\text{std}}}$$

The GRPO objective with clipped surrogate and per-step KL:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left(\min\bigl(r_t^i \hat{A}^i,\, \text{clip}(r_t^i, 1-\epsilon, 1+\epsilon)\hat{A}^i\bigr) - \beta\, D_t^i\right)\right]$$

Importance ratios and KL are computed in closed form from the Gaussian transitions:
$$\log r_t^i(\theta) = \frac{\|\mathbf{x}_{t+\Delta t}^i - \mathbf{x}_t^i - \tilde{\mathbf{v}}_{\theta_{\text{old}}} \Delta t\|^2 - \|\mathbf{x}_{t+\Delta t}^i - \mathbf{x}_t^i - \tilde{\mathbf{v}}_\theta \Delta t\|^2}{2\sigma_t^2 \Delta t}$$

$$D_t^i = \frac{\|\tilde{\mathbf{v}}_\theta(\mathbf{x}_t^i) - \tilde{\mathbf{v}}_{\text{ref}}(\mathbf{x}_t^i)\|^2 \Delta t}{2\sigma_t^2}$$

Implementation detail: per-step ratio normalization $\tilde{r}_t^i = (r_t^i - \mu_r^{(t)}) / \sigma_r^{(t)} + 1$ for stability.

### Embedding-to-Item Mapping

$$p(j \mid \mathbf{x}_1) = \frac{\exp(\mathbf{x}_1^\top \mathbf{e}_j / \tau)}{\sum_{k \in \mathcal{I}} \exp(\mathbf{x}_1^\top \mathbf{e}_k / \tau)}$$

In practice, approximate nearest-neighbor search (FAISS) for efficiency.

### Training Plan

1. **Phase 1 (Flow pre-training)**: Train recommender with uniform negatives + flow model with CFM loss simultaneously. Adam, lr=1e-4, 50 epochs, batch 256.
2. **Phase 2 (GRPO fine-tuning)**: G=8, clip ε=0.2, β=0.1, SDE steps T=20, η=0.5, δ=0.01. 10 GRPO epochs.
3. **Phase 3 (Joint training)**: Alternate every 5 recommender epochs with 2 GRPO epochs. Total 100 recommender epochs.
4. **Reward parameters**: Default a=1, γ=1. Tune γ ∈ {1, 2, 3} via validation.

### Failure Modes and Diagnostics

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| Reward hacking (mode collapse) | Entropy of generated distribution | Increase β; decrease η |
| False negative explosion | W distribution; alert if W>0.8 for >10% | Increase γ |
| Flow underfitting | CFM loss plateau | Increase capacity/epochs |
| Importance ratio instability | Var(log r_t) across t | Verify ratio normalization; reduce Δt |
| Joint training oscillation | Recommender degradation | Reduce GRPO frequency; increase warm-start |

### Novelty and Elegance Argument

**Closest work**: DNS/ANCE (heuristic, no generative model), IRGAN (GAN-based, unstable), DiffuRec/DreamRec (positive generation), Flow-GRPO for images (different domain)

**Exact difference**: FlowNS derives the reward from OPAUC theory with formal false-negative guarantees. The entire novelty is in $R = W^a(1-W)^\gamma$ and its dual KL+shaping guarantee. The pipeline is assembled from known components — this is a mechanism-level contribution, not a system-level contribution.

## Claim-Driven Validation Sketch

### Claim 1: Shaped reward generates harder negatives without increasing false negatives

- **Experiment**: Compare FlowNS vs. baselines on Yelp2018/Amazon-Book/Gowalla.
- **Baselines**: Uniform, Popularity, DNS, MixGCF, IRGAN, AdvInfoNCE, CVAE-NS (conditional VAE with shaped reward)
- **Metrics**: Recall@20, NDCG@20; win rate W distribution; FN rate
- **Sub-experiment (guarantee tightness)**: Measure actual FN rate vs. theoretical bound $\exp(R_{\max}/\beta) \cdot \text{FN}_{\text{ref}}$ across $\beta \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$. Plot both on the same axis to show the bound is empirically useful.
- **Expected evidence**: FlowNS achieves win rate concentrated near W* with lower FN rate than DNS/MixGCF. The theoretical bound tracks the actual FN rate within a constant factor.

### Claim 2: Shaped reward is necessary — ablation of reward components

- **Ablations**:
  - (A) Full FlowNS: $R = W(1-W)^\gamma$
  - (B) No shaping: $R = W$ (raw win rate)
  - (C) No RL: Pre-trained flow model directly (no GRPO)
  - (D) Symmetric: $R = W(1-W)$ (γ=1)
  - (E) Vary γ ∈ {0.5, 1, 2, 3}
- **Metrics**: Recall@20, NDCG@20, FN rate, reward distribution
- **Expected evidence**: (B) high W but high FN → worse Recall. (C) realistic but not hard enough. (A) with tuned γ best.

### Claim 3 (Simplification check): KL constraint is load-bearing

- **Experiment**: Set β=0, observe distribution drift, FN rate, and recommender performance.
- **Expected evidence**: Without KL, mode collapse and FN spike → performance degradation.

### Claim 4: Flow matching is the right generator for this setting

- **Experiment**: Replace flow model with (a) conditional Gaussian mixture model (b) conditional VAE, both using the same shaped reward and GRPO-style optimization where applicable. For GMM: sample from the mixture, apply reward-weighted re-sampling. For CVAE: REINFORCE on the latent space.
- **Metrics**: Win rate distribution quality, FN rate, Recall@20
- **Expected evidence**: Flow matching produces better-calibrated win rate distributions due to its ability to capture multimodal, user-conditional negative geometry. If the gap is small, the paper still stands on the reward contribution — the generator is then presented as interchangeable.

## Experiment Handoff Inputs

- **Must-prove claims**: (1) Shaped reward > raw score; (2) RL > no RL; (3) KL necessary; (4) Flow matching justified
- **Must-run ablations**: Reward shape; KL coefficient; generator type; γ sensitivity
- **Critical datasets**: Yelp2018, Amazon-Book, Gowalla
- **Critical metrics**: Recall@20, NDCG@20, win rate distribution, FN rate, guarantee tightness
- **Highest-risk assumptions**: (a) W is a reliable OPAUC proxy; (b) GRPO converges stably; (c) theoretical FN bound is empirically meaningful

## Compute & Timeline Estimate

- **Flow pre-training**: ~2 GPU-hours per dataset
- **GRPO fine-tuning**: ~4 GPU-hours per dataset
- **Joint training + all baselines + ablations**: ~30 GPU-hours total
- **Generator ablation**: ~6 GPU-hours (GMM + CVAE variants)
- **Total**: ~40-50 A100 GPU-hours
- **Timeline**: 4-6 weeks
