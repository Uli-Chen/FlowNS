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

2. **Hard negative mining** (DNS, ANCE, MixGCF): Selects negatives with the highest recommender scores. Shi et al. (WWW 2023) prove this implicitly optimizes One-way Partial AUC (OPAUC), which better aligns with Top-K ranking. However, the hardest negatives are precisely those most likely to be false negatives (items the user would like but hasn't seen), creating a hardness-fidelity tradeoff with no formal resolution.

3. **Generative negative sampling** (IRGAN, AdvInfoNCE): Uses adversarial training to generate hard negatives. Suffers from GAN training instability and lacks explicit control over the hardness-fidelity tradeoff.

### Why Naive Extensions Fail

- **Larger models / more data**: Does not address the fundamental hardness-fidelity tradeoff.
- **Exposure-aware heuristics**: Exposure data is sparse and cannot cover the full negative space.
- **Simply adding a flow model**: A vanilla flow model can learn the negative distribution but provides no mechanism to control hardness.
- **Naively applying RL to increase hardness**: Using the recommender's raw score as reward creates circular dependency (the generator optimizes toward the recommender's blind spots), risks reward hacking (unbounded reward → mode collapse), and conflates hardness with informativeness.

### The Smallest Adequate Intervention

A conditional flow matching model to capture the true negative distribution, combined with a single RL fine-tuning stage (GRPO) whose reward is derived from first principles to be:
- **OPAUC-motivated**: Directly connected to the ranking objective
- **Bounded and shaped**: Prevents reward hacking via sigmoid saturation
- **Self-correcting for false negatives**: Penalizes negatives that exceed the positive score boundary
- **KL-anchored**: The KL regularization against the pre-trained flow model mathematically guarantees proximity to the true negative distribution

## Method Thesis

- **One-sentence thesis**: FlowNS learns the true negative distribution via conditional flow matching, then applies GRPO with a mathematically derived *boundary-aware shaped reward* that maximizes training informativeness while provably controlling false-negative contamination through a dual mechanism of reward shaping and distributional anchoring.
- **Why this is the smallest adequate intervention**: The flow model handles realness; a single reward function handles hardness; the KL constraint handles fidelity. No additional modules, discriminators, or auxiliary losses are needed.
- **Why this route is timely**: Flow matching provides the cleanest generative framework for continuous embeddings; GRPO avoids the need for a learned value function; the ODE-to-SDE conversion enables exploration while preserving marginals — all leveraging recent advances that make this pipeline natural rather than forced.

## Contribution Focus

- **Dominant contribution**: A principled reward function for RL-guided negative sample generation, derived from OPAUC theory with formal analysis of its boundedness, false-negative correction, and optimality properties.
- **Supporting contribution**: The complete FlowNS pipeline (flow matching → SDE conversion → GRPO) with closed-form KL computation, demonstrating that this minimal architecture suffices.
- **Explicit non-contributions**: We do not claim novelty in flow matching itself, in GRPO itself, or in the ODE-to-SDE conversion. Our contribution is the principled integration via the reward design.

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: Recommendation model (LightGCN / MF) — frozen during GRPO, updated during joint training; item embeddings — shared between recommender and flow model.
- **New trainable components**: (1) Conditional flow matching velocity network $v_\theta(\mathbf{x}_t, u, t)$; (2) Same network fine-tuned via GRPO (no additional parameters).
- **Tempting additions intentionally not used**: Learned reward model (unnecessary — reward is analytical); discriminator (adds instability); separate hardness predictor (redundant with shaped reward); multi-stage curriculum (subsumed by GRPO's natural progression).

### System Overview

```
Training Phase 1: Learn True Negative Distribution
  Noise x_0 ~ N(0,I) --[Cond. Flow Matching]--> Negative embedding x_1
  Supervision: exposed-but-not-clicked item embeddings e_n
  Loss: L_CFM(theta) = E[||v_theta(x_t, u, t) - (e_n - x_0)||^2]

Training Phase 2: RL Fine-Tuning for Hardness (GRPO)
  For each user u:
    Sample G trajectories via SDE: {x_0^i -> ... -> x_1^i}_{i=1}^G
    Compute reward: R(x_1^i, u) = h(W(x_1^i, u))  [boundary-aware shaped reward]
    Compute group advantage: A^i = (R^i - mean) / std
    Update v_theta via clipped surrogate + KL penalty

Joint Training:
  Alternate between:
    (a) Train recommender with generated hard negatives
    (b) Update flow model via GRPO with frozen recommender
```

### Core Mechanism: Boundary-Aware Shaped Reward

This is the central technical contribution. We derive the reward function from first principles.

#### Step 1: Define the Win Rate

For a generated negative embedding $\mathbf{x}_1$ and user $u$ with positive item set $\mathcal{I}_u^+$, define the **win rate** against positives:

$$W(\mathbf{x}_1, u) = \frac{1}{|\mathcal{I}_u^+|} \sum_{i \in \mathcal{I}_u^+} \sigma\bigl(s(u, \mathbf{x}_1) - s(u, i)\bigr)$$

where $s(u, \cdot)$ is the recommender's scoring function (e.g., $s(u, \mathbf{x}) = \mathbf{u}^\top \mathbf{x}$) and $\sigma(\cdot)$ is the sigmoid function.

**Properties of $W$:**
- $W \in (0, 1)$ (bounded, since sigmoid outputs are in $(0,1)$)
- $W \approx 0$: the negative scores well below all positives (easy, uninformative)
- $W \approx 0.5$: the negative is at the decision boundary (maximally informative for BPR)
- $W \approx 1$: the negative beats most positives (likely a false negative)

**Connection to OPAUC**: Shi et al. (2023) prove that BPR + hard negative sampling implicitly optimizes OPAUC over FPR $\in [0, \beta]$. The win rate $W$ is exactly the empirical estimate of the false positive rate for a single generated negative — it measures what fraction of positives the negative "beats." Thus, $W$ directly connects to the OPAUC framework.

#### Step 2: Shape the Reward to Penalize False Negatives

The naive reward "maximize $W$" would push toward false negatives. Instead, we apply a **boundary-aware shaping function**:

$$R(\mathbf{x}_1, u) = W(\mathbf{x}_1, u)^a \cdot \bigl(1 - W(\mathbf{x}_1, u)\bigr)^\gamma$$

where:
- $a > 0$ controls the hardness incentive
- $\gamma > 0$ controls the false-negative penalty strength

**Mathematical properties:**

1. **Boundedness**: $R \in [0, R_{\max}]$ where $R_{\max} = \frac{a^a \gamma^\gamma}{(a+\gamma)^{a+\gamma}}$. This prevents reward hacking.

2. **Optimal hardness point**: Setting $\frac{\partial R}{\partial W} = 0$:

$$W^* = \frac{a}{a + \gamma}$$

This is the target hardness level. The reward naturally peaks at $W^*$ and decays on both sides.

3. **Asymmetric penalty**: When $\gamma > a$, the reward penalizes false negatives (high $W$) more severely than easy negatives (low $W$). This matches the practical requirement that false negatives are more harmful than uninformative samples.

4. **Connection to Beta distribution**: $R(W) \propto \text{Beta}(a+1, \gamma+1)$ density evaluated at $W$. This means the shaped reward defines a proper probability density over win rates, with the mode at $W^*$. The GRPO objective then encourages the generator to produce negatives whose win-rate distribution concentrates around $W^*$.

5. **Special cases**:
   - $a = 1, \gamma = 1$: $R = W(1-W)$, symmetric, peaks at $W^* = 0.5$ (decision boundary). This is the simplest and most elegant choice — the reward is exactly the variance of a Bernoulli with parameter $W$.
   - $a = 1, \gamma = 2$: $R = W(1-W)^2$, peaks at $W^* = 1/3$ (conservative, stronger FN penalty)
   - $a = 1, \gamma = 3$: peaks at $W^* = 1/4$ (very conservative)

**Default recommendation**: $a = 1, \gamma = 1$ (symmetric boundary reward $R = W(1-W)$), unless the dataset has high false-negative rates, in which case increase $\gamma$.

#### Step 3: Formal Analysis of False-Negative Control

The FlowNS framework provides a **dual mechanism** against false negatives:

**Mechanism 1 (Reward shaping)**: The shaped reward $R = W^a(1-W)^\gamma$ has zero gradient at $W = 1$ (pure false negative). More precisely, $\frac{\partial R}{\partial W}\big|_{W=1} = 0$ for any $\gamma > 0$. This means the GRPO policy gradient provides zero learning signal for samples that beat all positives — the generator receives no incentive to produce false negatives.

**Mechanism 2 (KL anchoring)**: The KL penalty $\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ constrains the RL-fine-tuned policy to remain close to the pre-trained flow model, which was trained on the true negative distribution. Formally, for any measurable set $A$ in the embedding space:

$$\pi_\theta(A | u) \leq \pi_{\text{ref}}(A | u) \cdot \exp\left(\frac{1}{\beta} \sup_{\mathbf{x} \in A} R(\mathbf{x}, u)\right)$$

(by Donsker-Varadhan variational representation of KL). Since $R$ is bounded by $R_{\max}$, the density ratio between the RL policy and the reference policy is bounded by $\exp(R_{\max}/\beta)$. This guarantees the RL policy cannot place arbitrarily high probability mass on any region, including false-negative regions.

**Combined guarantee**: Let $\mathcal{F}_u = \{j : j \text{ is a false negative for } u\}$ be the false negative set. The expected false-negative rate under the RL policy satisfies:

$$\mathbb{E}_{\mathbf{x} \sim \pi_\theta(\cdot|u)}[\mathbb{1}[\mathbf{x} \in \mathcal{F}_u]] \leq \exp\left(\frac{R_{\max}}{\beta}\right) \cdot \mathbb{E}_{\mathbf{x} \sim \pi_{\text{ref}}(\cdot|u)}[\mathbb{1}[\mathbf{x} \in \mathcal{F}_u]]$$

Since the reference policy (pre-trained on true negatives) has low false-negative rate by construction, and $R_{\max}/\beta$ can be controlled, the RL policy's false-negative rate is bounded.

#### Step 4: Why Not Alternative Rewards?

| Alternative | Problem |
|------------|---------|
| Raw score $s(u, \mathbf{x})$ | Unbounded → reward hacking; no FN correction; scale-dependent |
| Sigmoid of score $\sigma(\alpha \cdot s)$ | No reference point; doesn't distinguish hard-from-boundary vs hard-from-false-negative |
| Gradient norm $\|\nabla_\Theta \ell\|$ | Expensive (requires backprop through recommender per sample); doesn't separate FN |
| Rank-based reward | Requires sorting over item set; discontinuous gradients |
| **Win-rate + Beta shaping (ours)** | Bounded; OPAUC-connected; closed-form FN penalty; single scalar hyperparameter $\gamma$ |

### Flow Matching for True Negative Distribution

We adopt the Conditional Optimal Transport (CondOT) formulation of flow matching. The velocity field $v_\theta(\mathbf{x}_t, u, t)$ is conditioned on user $u$ and trained with:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1],\, \mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I}),\, \mathbf{e}_n \sim \mathcal{E}_u^{\text{neg}}} \left[\left\| v_\theta\bigl((1-t)\mathbf{x}_0 + t\,\mathbf{e}_n,\, u,\, t\bigr) - (\mathbf{e}_n - \mathbf{x}_0) \right\|^2\right]$$

where $\mathcal{E}_u^{\text{neg}}$ is the set of exposed-but-not-clicked item embeddings for user $u$.

**Convention**: $t = 0$ corresponds to noise, $t = 1$ corresponds to data. Generation proceeds forward from $t = 0$ to $t = 1$.

### ODE-to-SDE Conversion for RL Exploration

The deterministic ODE $d\mathbf{x}_t = \mathbf{v}_\theta(\mathbf{x}_t, u, t)\,dt$ is converted to an equivalent SDE that preserves marginal distributions while enabling stochastic exploration:

$$d\mathbf{x}_t = \underbrace{\left[\mathbf{v}_\theta(\mathbf{x}_t, u, t) + \frac{\sigma_t^2}{2} \nabla\log p_t(\mathbf{x}_t)\right]}_{\text{drift (preserves marginals)}} dt + \underbrace{\sigma_t\,d\mathbf{w}}_{\text{diffusion (enables exploration)}}$$

For rectified flow with the learned velocity, the score is approximated as:

$$\nabla\log p_t(\mathbf{x}_t) \approx -\frac{\mathbf{x}_t - t\,\mathbf{v}_\theta(\mathbf{x}_t, u, t)}{(1-t)^2}$$

This follows from Tweedie's formula: given $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ with $\mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I})$, the conditional score of $p_t(\mathbf{x}_t)$ is:

$$\nabla\log p_t(\mathbf{x}_t) = -\frac{\mathbf{x}_t - t\,\mathbb{E}[\mathbf{x}_1 | \mathbf{x}_t]}{(1-t)^2}$$

and the learned velocity provides $\mathbb{E}[\mathbf{x}_1 | \mathbf{x}_t] \approx \mathbf{x}_t + (1-t)\mathbf{v}_\theta$.

**Noise schedule**: $\sigma_t = \eta \cdot \frac{\sqrt{1-t}}{\sqrt{t} + \delta}$ where $\delta > 0$ is a small constant preventing divergence at $t = 0$. The parameter $\eta$ controls the exploration-exploitation tradeoff.

**Euler-Maruyama discretization** with step size $\Delta t$:

$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \tilde{\mathbf{v}}_\theta(\mathbf{x}_t, u, t)\,\Delta t + \sigma_t \sqrt{\Delta t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

where $\tilde{\mathbf{v}}_\theta$ is the SDE drift incorporating the score correction.

### GRPO Objective

Given user $u$, sample $G$ trajectories $\{(\mathbf{x}_0^i, \ldots, \mathbf{x}_1^i)\}_{i=1}^G$ via the SDE under the current policy. Compute the shaped reward $R^i = R(\mathbf{x}_1^i, u)$ for each trajectory's final sample, and the group-normalized advantage:

$$\hat{A}^i = \frac{R^i - \bar{R}}{\sigma_R + \epsilon_{\text{std}}}$$

where $\bar{R} = \frac{1}{G}\sum_{i=1}^G R^i$, $\sigma_R = \text{std}(\{R^i\})$, and $\epsilon_{\text{std}}$ prevents division by zero.

**Note**: The advantage $\hat{A}^i$ is trajectory-level (independent of timestep $t$), since the reward is defined on the final generated embedding $\mathbf{x}_1^i$ only.

The GRPO objective:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{u, \{\mathbf{x}^i\} \sim \pi_{\theta_{\text{old}}}} \left[\frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left(\min\bigl(r_t^i \hat{A}^i,\, \text{clip}(r_t^i, 1-\epsilon, 1+\epsilon)\hat{A}^i\bigr) - \beta\, D_t^i\right)\right]$$

where the importance ratio at step $t$ is:

$$r_t^i(\theta) = \frac{p_\theta(\mathbf{x}_{t+\Delta t}^i \mid \mathbf{x}_t^i, u)}{p_{\theta_{\text{old}}}(\mathbf{x}_{t+\Delta t}^i \mid \mathbf{x}_t^i, u)}$$

Since both transition densities are Gaussian (from the Euler-Maruyama discretization):

$$\log r_t^i(\theta) = \frac{\|\mathbf{x}_{t+\Delta t}^i - \mathbf{x}_t^i - \tilde{\mathbf{v}}_{\theta_{\text{old}}} \Delta t\|^2 - \|\mathbf{x}_{t+\Delta t}^i - \mathbf{x}_t^i - \tilde{\mathbf{v}}_\theta \Delta t\|^2}{2\sigma_t^2 \Delta t}$$

And the per-step KL divergence:

$$D_t^i = D_{\text{KL}}\bigl(p_\theta(\cdot | \mathbf{x}_t^i, u) \| p_{\text{ref}}(\cdot | \mathbf{x}_t^i, u)\bigr) = \frac{\|\tilde{\mathbf{v}}_\theta(\mathbf{x}_t^i) - \tilde{\mathbf{v}}_{\text{ref}}(\mathbf{x}_t^i)\|^2 \Delta t}{2\sigma_t^2}$$

### Importance Ratio Stabilization

Following GRPO-Guard (2025), the importance ratio distribution in flow models is systematically biased below 1 with variance that fluctuates across timesteps. We apply **per-step ratio normalization**:

$$\tilde{r}_t^i = \frac{r_t^i - \mu_r^{(t)}}{\sigma_r^{(t)}} + 1$$

where $\mu_r^{(t)}$ and $\sigma_r^{(t)}$ are the mean and standard deviation of $\{r_t^i\}_{i=1}^G$ at timestep $t$. This ensures the ratio distribution at each step is centered at 1, preventing implicit over-optimization.

### Embedding-to-Item Mapping

The generated embedding $\mathbf{x}_1$ is mapped to discrete items via:

$$p(j \mid \mathbf{x}_1) = \frac{\exp(\mathbf{x}_1^\top \mathbf{e}_j / \tau)}{\sum_{k \in \mathcal{I}} \exp(\mathbf{x}_1^\top \mathbf{e}_k / \tau)}$$

where $\tau$ is the temperature parameter. In practice, we use approximate nearest-neighbor search (FAISS) for efficiency.

### Modern Primitive Usage

- **Flow matching**: Acts as the **generative backbone** for learning the true negative distribution. This is more natural than GANs (no adversarial instability) or VAEs (no posterior collapse) for embedding-space generation.
- **GRPO**: Acts as the **hardness controller**. More natural than PPO (no learned value function needed) or direct reward optimization (which lacks exploration).
- **ODE-to-SDE conversion**: Provides **exploration without retraining**. This is a mathematical property of the flow, not an added component.

### Integration into Recommendation Pipeline

```
Phase 1 (Warmup): Train recommender with uniform negatives → obtain initial embeddings
Phase 2 (Flow pre-training): Train flow model on exposed negatives → π_ref
Phase 3 (GRPO fine-tuning): Fine-tune flow model with shaped reward → π_theta
Phase 4 (Joint training): Alternate:
  (a) Generate hard negatives with π_theta, train recommender
  (b) Re-run GRPO with updated recommender scores (few steps)
```

### Training Plan

1. **Flow pre-training**: Standard CFM loss, Adam optimizer, lr=1e-4, 50 epochs, batch size 256.
2. **GRPO fine-tuning**: Group size $G = 8$, clip $\epsilon = 0.2$, KL coefficient $\beta = 0.1$, SDE steps $T = 20$, noise parameter $\eta = 0.5$, $\delta = 0.01$. Train for 10 GRPO epochs.
3. **Joint training**: Alternate every 5 recommender epochs with 2 GRPO epochs. Total 100 recommender epochs.
4. **Reward parameters**: Default $a = 1, \gamma = 1$ (symmetric shaped reward). Tune $\gamma \in \{1, 2, 3\}$ via validation.

### Failure Modes and Diagnostics

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| Reward hacking (mode collapse) | Monitor entropy of generated distribution; check if all negatives map to same items | Increase $\beta$ (KL penalty); decrease $\eta$ (less exploration) |
| False negative explosion | Track win rate $W$ distribution; alert if $W > 0.8$ for >10% of samples | Increase $\gamma$ in shaped reward |
| Flow model underfitting | CFM loss plateau with high reconstruction error | Increase model capacity or training epochs |
| Importance ratio instability | Monitor $\text{Var}(\log r_t)$ across timesteps | Verify RatioNorm is active; reduce $\Delta t$ |
| Recommender-generator oscillation | Recommender performance degrades after joint training steps | Reduce GRPO update frequency; increase warm-start period |

### Novelty and Elegance Argument

**Closest work**: 
- DNS/ANCE: Heuristic hard negative selection, no generative model, no formal hardness-fidelity tradeoff
- IRGAN: GAN-based negative generation, but adversarial training is unstable and lacks the clean OPAUC connection
- DiffuRec/DreamRec: Diffusion models for recommendation, but focused on positive item generation, not negative sampling
- Flow-GRPO (for images): Applies GRPO to flow models for aesthetic/text-alignment, not for recommendation

**Exact difference**: FlowNS is the first to derive a reward function for RL-guided negative generation from OPAUC theory, providing formal analysis of boundedness, false-negative control, and the dual KL+shaping guarantee. The $R = W^a(1-W)^\gamma$ reward is not an ad-hoc design choice — it emerges naturally from requiring (1) OPAUC-motivated hardness, (2) bounded reward, and (3) zero gradient at $W = 1$ (false-negative boundary).

**Why this is focused**: The entire novelty is in one equation ($R = W^a(1-W)^\gamma$) and its formal properties. The pipeline (flow matching + SDE + GRPO) is assembled from known components. This is a mechanism-level contribution, not a system-level contribution.

## Claim-Driven Validation Sketch

### Claim 1: The shaped reward generates harder negatives than baselines without increasing false negatives

- **Minimal experiment**: Compare FlowNS vs. uniform, popularity, DNS, MixGCF on Yelp/Amazon/Gowalla. Measure: (a) Recall@K, NDCG@K for the trained recommender; (b) average win rate $W$ of generated negatives; (c) false-negative rate (fraction of generated "negatives" that are actually in the test positive set).
- **Baselines**: Uniform, Popularity, DNS, MixGCF, IRGAN, AdvInfoNCE
- **Metric**: Recall@20, NDCG@20; win rate distribution; FN rate
- **Expected evidence**: FlowNS achieves win rate concentrated near $W^*$ with lower FN rate than DNS/MixGCF, translating to higher Recall/NDCG.

### Claim 2: The shaped reward is necessary — ablation of reward components

- **Minimal experiment**: Ablate the reward function:
  - (A) Full FlowNS: $R = W(1-W)^\gamma$
  - (B) No shaping: $R = W$ (raw win rate, equivalent to "higher score = better")
  - (C) No RL: Use pre-trained flow model directly (no GRPO)
  - (D) Symmetric shaping: $R = W(1-W)$ (fixed $\gamma = 1$)
  - (E) Vary $\gamma \in \{0.5, 1, 2, 3\}$ to study the FN penalty strength
- **Metric**: Recall@20, NDCG@20, FN rate, reward distribution
- **Expected evidence**: (B) achieves high win rate but also high FN rate, hurting recommendation quality. (C) shows flow model alone generates realistic but insufficiently hard negatives. (A) with appropriate $\gamma$ achieves the best quality.

### Claim 3 (Simplification check): The KL constraint is load-bearing

- **Minimal experiment**: Set $\beta = 0$ (no KL penalty) and observe:
  - Does the generator drift far from the true negative distribution?
  - Does the FN rate increase?
  - Does the recommender performance degrade?
- **Expected evidence**: Without KL, the generator collapses to a narrow mode of adversarial negatives, FN rate spikes, and recommender performance degrades.

## Experiment Handoff Inputs

- **Must-prove claims**: (1) Shaped reward > raw score reward; (2) RL fine-tuning > no fine-tuning; (3) KL is necessary
- **Must-run ablations**: Reward ablation ($W$ vs $W(1-W)$ vs $W(1-W)^\gamma$); KL ablation ($\beta = 0$); group size ($G$)
- **Critical datasets**: Yelp2018, Amazon-Book, Gowalla
- **Critical metrics**: Recall@20, NDCG@20, win rate distribution, FN rate
- **Highest-risk assumptions**: (a) The win rate $W$ is a reliable proxy for OPAUC-relevant hardness; (b) GRPO converges stably with the shaped reward; (c) the flow model has sufficient capacity to capture the negative distribution

## Compute & Timeline Estimate

- **Flow pre-training**: ~2 GPU-hours per dataset (small embedding-space model)
- **GRPO fine-tuning**: ~4 GPU-hours per dataset (G=8 trajectories, 10 epochs)
- **Joint training + baselines**: ~20 GPU-hours total across 3 datasets
- **Total**: ~30-40 A100 GPU-hours
- **Timeline**: 4-6 weeks (implementation + experiments + analysis)
