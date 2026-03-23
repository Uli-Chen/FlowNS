GENERATING BOTH HIGH-CONFIDENCE AND HARD NEGATIVES FOR RECOMMENDER SYSTEMS

**SIGMOD 26 Round 2 (Revised Draft)**  
**March 2026**

---

## 1. Introduction

Pairwise recommendation objectives (e.g., BPR) rely critically on negative samples. In implicit-feedback settings, most user-item pairs are unobserved, and exposure logs are biased by ranking position, serving policies, and context. Therefore, "not clicked after exposure" cannot be treated as guaranteed true negatives.

This proposal revises the objective from **true negative generation** to **debiased high-confidence negative generation**, then introduces hardness in a controlled way.

We target three practical requirements:

1. **Realness (confidence)**: generated negatives should likely reflect non-preference.
2. **Hardness**: generated negatives should provide strong training gradients.
3. **Scalability**: sampling must work under large item vocabularies.

---

## 2. Problem Setup and Assumptions

Let:

- \(U\): users
- \(\mathcal{I}\): items
- \(x\): exposure context (e.g., position, device, time, traffic source)
- \(E_{ui}\in\{0,1\}\): exposure indicator
- \(C_{ui}\in\{0,1\}\): click/interaction indicator
- \(R_{ui}\in\{0,1\}\): latent relevance (unobserved)

Goal: learn \(G(\cdot \mid u,x)\) that samples items approximating
\[
q^{-}(i\mid u,x)\approx P(R_{ui}=0 \mid E_{ui}=1,\ C_{ui}=0,\ u,\ x),
\]
i.e., **debiased high-confidence negatives**, not guaranteed true negatives.

### Assumptions

1. Exposure logs contain at least partial context (or a proxy is available).
2. Propensity \(p_{ui}=P(E_{ui}=1\mid u,i,x)\) is estimable (directly or approximately).
3. Recommendation model and generator are trained in alternating stages for stability.

---

## 3. Method

### 3.1 Debiased Confidence from Exposure Data

For exposed non-clicked pairs \((u,i,x)\), define a confidence weight:
\[
w_{ui}^{-}=\mathrm{clip}\left(\frac{1}{p_{ui}},\ w_{\min},\ w_{\max}\right)\cdot \mathbb{1}[E_{ui}=1,\ C_{ui}=0].
\]

This weight corrects exposure bias (IPS-style). We do not claim these are always negatives; we model them as **high-confidence debiased negatives**.

### 3.2 Conditional Flow Matching for Negative Preference

Let \(x_0\sim\mathcal{N}(0,I)\), \(e_i\) item embedding, and
\[
x_t=t\,e_i+(1-t)\,x_0.
\]
Train velocity field \(v_\theta\) by weighted CFM:
\[
\mathcal{L}_{\mathrm{CFM}}(\theta)=
\mathbb{E}_{t,x_0,(u,i,x)}
\left[
w_{ui}^{-}\cdot\left\|v_\theta(t,x_t,u,x)-(e_i-x_0)\right\|_2^2
\right].
\]

Interpretation: this learns a transport map toward a debiased high-confidence negative region under stated assumptions.

### 3.3 Controlled Hardness Guidance with False-Negative Control

Naively pushing generated negatives toward the user embedding can increase false negatives. We use a constrained objective:
\[
\hat{x}_1(x_0)=x_0+v_\theta(0,x_0,u,x),
\]
\[
J(x_0)=s(\hat{x}_1,u)-\beta\cdot\max_{i^+\in\mathcal{P}_u}s(\hat{x}_1,e_{i^+}),
\]
where \(\mathcal{P}_u\) is user's positive history.

Guidance:
\[
g=\nabla_{x_0}J(x_0),\quad
g\leftarrow g\cdot\min\left(1,\frac{\gamma}{\|g\|_2}\right),
\]
\[
x_0'=x_0+\alpha_t g,\quad \|x_0'-x_0\|_2\le \rho_t.
\]

Here \(\gamma\) is gradient clip norm and \(\rho_t\) is trust-region radius.

### 3.4 Two-Stage Discrete Sampling (Scalable)

Full softmax over \(|\mathcal{I}|\) is expensive and norm/popularity-sensitive.

1. **ANN retrieval**: retrieve candidate set \(\mathcal{C}_u=\mathrm{ANN}(\hat{x}_1,K)\).
2. **Candidate reweighting**:
\[
\tilde{s}_j = \frac{\hat{x}_1^\top e_j}{\|e_j\|^\eta} - \lambda_{\mathrm{pop}}\log(\mathrm{pop}_j+1),\quad j\in\mathcal{C}_u.
\]
\[
p(j\mid \hat{x}_1)=\frac{\exp(\tilde{s}_j/\tau_t)}{\sum_{k\in\mathcal{C}_u}\exp(\tilde{s}_k/\tau_t)}.
\]

This reduces compute from \(O(|\mathcal{I}|d)\) to ANN retrieval + \(O(Kd)\).

### 3.5 Unified Training Objective

\[
\mathcal{L}_{\text{total}}=
\mathcal{L}_{\text{recommend}}
\ +\ \lambda_1\mathcal{L}_{\mathrm{CFM}}
\ +\ \lambda_2\mathcal{L}_{\mathrm{debias}}
\ +\ \lambda_3\mathcal{L}_{\mathrm{FN\_control}}.
\]

Recommended instantiation:

- \(\mathcal{L}_{\text{recommend}}\): weighted BPR / pairwise loss.
- \(\mathcal{L}_{\mathrm{debias}}\): propensity regularization/calibration term.
- \(\mathcal{L}_{\mathrm{FN\_control}}\): margin or penalty against historical positives.

### 3.6 Curriculum Schedule

Use staged hardness:

- \(\alpha_t\): piecewise-linear increase (warmup then plateau).
- \(\tau_t\): piecewise decrease (high temperature early, low late).
- report stability: gradient norm stats, divergence rate, and loss oscillation.

---

## 4. Experimental Protocol (Must-Have)

### 4.1 Datasets

- MIND
- ML-100K
- Yelp2018 (implicit)
- Amazon-Books (implicit)

At least two datasets beyond MIND/ML-100K are required in final paper.

### 4.2 Split and Leakage Control

1. Prefer **time-based split** per dataset.
2. Keep exposure/context features aligned with training timeline.
3. Evaluate with full ranking protocol (not sampled-only metrics).

### 4.3 Baselines

- RNS (random negatives)
- DNS/HNS (dynamic hard negatives)
- PNS (popularity-based negatives)
- Exposure-only negatives (no debias)
- IPS-weighted pairwise baseline
- HDCCF-style hard negative baseline
- DiffRec / conditional diffusion recommendation baseline (if reproducible)

### 4.4 Metrics

Accuracy:

- Recall@K
- NDCG@K

Beyond accuracy:

- Coverage@K
- AvgPop@K (popularity bias)
- Novelty@K
- Long-tail hit rate

Efficiency:

- Training time/epoch
- Inference sampling latency
- GPU memory usage

### 4.5 Ablation Matrix

1. w/o exposure debias
2. w/o hardness guidance
3. w/o false-negative control
4. full softmax vs ANN two-stage sampling
5. schedule variants for \(\alpha_t\), \(\tau_t\)

### 4.6 Negative Quality Analysis

Report a realness-hardness tradeoff:

- hardness proxy: model score on sampled negatives
- false-negative proxy: overlap/similarity with held-out positives
- subgroup analysis: cold users vs active users, head vs tail items

### 4.7 Statistical Testing

Use multi-seed runs (>=3, ideally 5), and report:

- mean ± std
- paired significance test (paired t-test or Wilcoxon signed-rank)

---

## 5. Failure Modes and Mitigation

1. **Propensity misspecification**  
   Mitigation: clipping, calibration check, sensitivity analysis.

2. **Sparse exposure context**  
   Mitigation: context proxy features and robustness ablation.

3. **Over-hard negatives causing collapse**  
   Mitigation: trust region, gradient clipping, delayed hardness warmup.

4. **ANN recall errors**  
   Mitigation: monitor recall@K of ANN candidate set and tune \(K\).

---

## 6. Expected Contributions

1. A debiased formulation of exposure-based generative negative sampling.
2. Controlled hardness guidance with explicit false-negative control.
3. Scalable two-stage discrete sampler for large item spaces.
4. Reproducible evaluation protocol that covers accuracy, bias, and efficiency.

---

## 7. Acceptance Checklist

1. No unverifiable claim such as "guaranteed true negatives."
2. Method section includes assumptions, objective, complexity, and failure modes.
3. Experiment section includes complete baselines, ablations, statistics, and cost.
4. Final contribution is clearly positioned as:
   **exposure debias + generative negative sampling + controlled hardness curriculum**.

---

## 8. Related Work References

- Flow Matching (ICLR 2023): https://arxiv.org/abs/2210.02747
- Multisample Flow Matching (2023): https://arxiv.org/abs/2304.14772
- Riemannian Flow Matching (ICLR 2024): https://iclr.cc/virtual/2024/oral/19740
- DiffRec (SIGIR 2023): https://arxiv.org/abs/2304.04971
- Diffusion Rec Survey (2024): https://arxiv.org/abs/2409.05033
- HDCCF (IJCAI 2022): https://www.ijcai.org/proceedings/2022/0327.pdf
- Recommendations as Treatments (ICML 2016): https://proceedings.mlr.press/v48/schnabel16.html
- Attribute-based Propensity (KDD 2020): https://research.google/pubs/attribute-based-propensity-for-unbiased-learning-in-recommender-systems-algorithm-and-case-studies/
- Practically Unbiased Pairwise Loss (2025): https://pubmed.ncbi.nlm.nih.gov/40030662/
- Time to Split (RecSys 2025): https://arxiv.org/abs/2507.16289
- ICPNS (2026 preprint): https://arxiv.org/abs/2602.18759
