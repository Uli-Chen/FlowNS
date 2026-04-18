# Experiment Plan: FlowNeg on RecBole — MINDsmall Pipeline Test

**Problem**: Negative sampling in implicit-feedback recommenders cannot simultaneously guarantee realness (no false negatives) and hardness (strong training gradients). Exposure data (shown-but-not-clicked) is sparse and underutilized.

**Method Thesis**: FlowNeg learns an exposure-grounded negative proposal via Conditional Flow Matching (CondOT) and applies generate-then-reweight decoding for structurally independent hardness control. One small MLP + FAISS + score reweighting.

**Date**: 2026-03-30

**Target**: SIGMOD 2026 Round 2, data-centric framing.

**Scope of this plan**: MINDsmall for pipeline validation. Full MIND and KuaiRand deferred to future work.

---

## Why MINDsmall

MINDsmall (Microsoft News Dataset, small version) is ideal for FlowNeg pipeline testing:

1. **Explicit exposure data**: Each impression in `behaviors.tsv` encodes `NewsID-1` (clicked) and `NewsID-0` (displayed but not clicked). The `-0` items are natural exposure negatives — the user saw them but chose not to engage.
2. **Rich exposure-negative signal**: Average ~1.7 clicks per impression out of ~20-40 displayed articles → roughly 1:10-1:20 positive-to-negative ratio per impression.
3. **Manageable size**: 50K users, ~51K news articles, ~236K train impressions — fits on a single GPU, fast iteration.
4. **RecBole compatible**: MIND already has RecBole conversion tools via RecSysDatasets.
5. **Natural upgrade path**: MINDsmall → MINDlarge (1M users) for final paper results.

**Key difference from KuaiRand**: MIND provides impression-level grouping (which items were co-displayed), while KuaiRand provides per-item random exposure flags. Both give true exposure negatives, but MIND's impression structure is richer and the dataset is more widely used.

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| **C1**: Exposure-grounded CFM generates negatives with lower FN rate than RNS/DNS | Core novelty — impression-level exposure data as grounding signal | FN rate < DNS on MINDsmall; candidate-set purity measured | B1, B2 |
| **C2**: Generate-then-reweight provides structurally independent hardness control | Decoupling claim — hardness tunable without corrupting realness | tau_h sweep shows smooth hardness-FN tradeoff; ablation confirms reweighting is necessary | B3 |
| **Anti-claim to rule out**: The flow query adds no value beyond simpler query strategies | Justifies generative model complexity | FlowNeg beats centroid/KNN/user-emb baselines under identical retrieve-reweight pipeline | B4 |

## Paper Storyline

**Main paper must prove (to be validated on MINDsmall first):**
1. Exposure-grounded negatives are more real (lower FN rate) — Block B1
2. Flow-generated queries retrieve better candidate sets than simpler alternatives — Block B4
3. Hardness is independently controllable via tau_h — Block B3
4. End-to-end Recall@20/NDCG@20 improvement — Block B2

**Appendix can support:**
- Dense vs. sparse user breakdown
- Noise sensitivity analysis
- Candidate-set diversity metrics

**Deferred to future work:**
- Full MIND dataset (1M users) — scale validation
- KuaiRand dataset — random exposure setting (different exposure mechanism)
- Coat dataset — explicit rating feedback
- Gowalla/Yelp2018 — no exposure data (proxy negatives only)

---

## Implementation Architecture on RecBole

### File Structure

```
flowns/
├── recbole/                          # Existing RecBole framework (DO NOT MODIFY core files)
│   ├── model/general_recommender/
│   │   ├── bpr.py                    # Backbone 1 (existing)
│   │   └── lightgcn.py              # Backbone 2 (existing)
│   ├── sampler/
│   │   └── sampler.py               # AbstractSampler base (existing)
│   ├── data/
│   │   ├── dataloader/
│   │   │   ├── abstract_dataloader.py   # NegSampleDataLoader (existing)
│   │   │   └── general_dataloader.py    # TrainDataLoader (existing)
│   │   └── utils.py                 # data_preparation, create_samplers (existing)
│   └── trainer/
│       └── trainer.py               # Trainer._train_epoch (existing)
│
├── flowneg/                         # NEW: All FlowNeg-specific code
│   ├── __init__.py
│   ├── velocity_net.py              # Velocity MLP v_t^theta
│   ├── cfm_trainer.py               # CFM training logic (CondOT loss)
│   ├── flow_sampler.py              # FlowNegSampler (extends AbstractSampler)
│   ├── flowneg_trainer.py           # FlowNegTrainer (extends Trainer)
│   ├── flowneg_dataloader.py        # FlowNegTrainDataLoader (extends TrainDataLoader)
│   ├── faiss_index.py               # FAISS ANN wrapper
│   ├── metrics.py                   # FN rate, candidate purity, diversity diagnostics
│   └── utils.py                     # tau_h schedule, midpoint solver, logging helpers
│
├── dataset/                         # Datasets in RecBole atomic format
│   ├── mindsmall/                   # PRIMARY: Pipeline testing
│   │   ├── mindsmall.inter          # user_id:token  item_id:token  label:float  impression_id:token
│   │   ├── mindsmall.item           # item_id:token  category:token  subcategory:token  title:token_seq
│   │   └── mindsmall.yaml           # Dataset-specific config
│   └── ml-100k/                     # Sanity check (existing)
│
├── data_raw/                        # Raw downloaded data (not in git)
│   └── MINDsmall_train/
│       ├── behaviors.tsv
│       ├── news.tsv
│       ├── entity_embedding.vec
│       └── relation_embedding.vec
│   └── MINDsmall_dev/
│       └── ...
│
├── exposure_cache/                  # Pre-computed exposure-negative sets
│   └── mindsmall_exposure_negs.pkl  # {user_id: [exposed_neg_item_ids]}
│
├── configs/                         # Experiment YAML configs
│   ├── base.yaml                    # Shared defaults
│   ├── flowneg_mf_mindsmall.yaml
│   ├── flowneg_lgcn_mindsmall.yaml
│   ├── baselines/
│   │   ├── rns_mf_mindsmall.yaml
│   │   ├── dns_mf_mindsmall.yaml
│   │   ├── pns_mf_mindsmall.yaml
│   │   └── expweight_mf_mindsmall.yaml
│   └── ablations/
│       ├── no_reweight.yaml
│       ├── query_baselines.yaml
│       └── tau_sweep.yaml
│
├── scripts/                         # Run scripts
│   ├── download_mind.py             # Download MINDsmall from Azure Blob Storage
│   ├── prepare_mindsmall.py         # Raw MIND -> RecBole atomic format + exposure cache
│   ├── run_flowneg.py               # Main entry: config -> dataset -> FlowNeg pipeline
│   ├── run_baseline.py              # Baseline runner (RNS, DNS, PNS)
│   ├── run_query_ablation.py        # 5-query-baseline experiment (Claim 3)
│   ├── run_tau_sweep.py             # tau_h sweep (Claim 2)
│   └── run_all.sh                   # Full experiment orchestration
│
└── refine-logs/                     # This file and tracker
    ├── EXPERIMENT_PLAN.md
    └── EXPERIMENT_TRACKER.md
```

---

## Detailed Implementation Plan

### Module 1: Velocity Network (`flowneg/velocity_net.py`)

```python
class VelocityNet(nn.Module):
    """
    v_t^theta(x_t, u) : R^d x R^d x R -> R^d
    Input:  [x_t; user_emb; PE(t)]  in R^{2d + d_pe}
    Layers: Linear(3d, 4d) -> SiLU -> Linear(4d, 4d) -> SiLU -> Linear(4d, d)
    ~35K params for d=64
    """
```

**Key design decisions:**
- Positional encoding of t: sinusoidal PE with `d_pe = d` (so input is 3d)
- User embedding: detached from backbone (no gradient flow to recommender during CFM training)
- Weight init: Xavier normal
- Config params: `velocity_hidden_mult: 4`, `velocity_layers: 2`, `velocity_d_pe: 64`

### Module 2: CFM Training Logic (`flowneg/cfm_trainer.py`)

```python
class CFMTrainer:
    """
    Trains the velocity network on CondOT objective:
      For (u, e_n) from exposure-negative set E_u^neg:
        x_0 ~ N(0, I)
        t ~ U(0, 1)
        x_t = (1-t) * x_0 + t * e_n          # CondOT straight path
        target = e_n - x_0                     # velocity target
        loss = ||v_t^theta(x_t, u) - target||^2
    """
```

**Integration with RecBole:**
- Reads item embeddings from `model.item_embedding.weight` (detached)
- Reads user embeddings from `model.user_embedding.weight` (detached)
- For LightGCN: uses propagated embeddings from `model.forward()`, not raw embeddings
- Exposure-negative set: loaded from `exposure_cache/mindsmall_exposure_negs.pkl`
- Optimizer: separate Adam with lr=1e-3, independent of backbone optimizer
- Training: batch of (user_id, neg_item_id) pairs -> look up embeddings -> compute CondOT loss

### Module 3: FlowNeg Sampler (`flowneg/flow_sampler.py`)

```python
class FlowNegSampler(AbstractSampler):
    """
    Replaces RecBole's default Sampler for training phase.
    
    generate_then_reweight(user_ids) -> neg_item_ids:
      1. x_0 ~ N(0, I)                         # [batch, d]
      2. x_1 = midpoint4(v_theta, x_0, u_emb)  # 4-step midpoint solver
      3. C_M = faiss_index.search(x_1, M)       # M nearest items per query
      4. scores = dot(u_emb, item_emb[C_M])     # recommender scores
      5. p(k) = softmax(scores / tau_h)          # hardness reweighting
      6. j ~ Categorical(p)                      # sample from candidates
      return item_ids[C_M[j]]
    """
```

**Key integration point:** Must conform to `AbstractSampler` interface:
- `get_used_ids()` -> returns per-user positive item sets (same as standard Sampler)
- `sample_by_user_ids(user_ids, item_ids, num)` -> returns `torch.tensor` of neg item ids
- `_uni_sampling()` -> fallback to uniform for warm-up phase

**Warm-up handling:** During epochs 0..warm_up_epochs, delegates to standard uniform sampling. After warm-up, switches to flow-based generation.

**4-step Midpoint Solver:**
```python
def midpoint4(v_theta, x_0, u_emb):
    """Time grid: {0, 0.25, 0.5, 0.75, 1.0}"""
    dt = 0.25
    x = x_0
    for t_start in [0.0, 0.25, 0.5, 0.75]:
        t_mid = t_start + dt / 2
        k1 = v_theta(x, u_emb, t_start)
        x_mid = x + (dt / 2) * k1
        k2 = v_theta(x_mid, u_emb, t_mid)
        x = x + dt * k2
    return x  # x_1
```

### Module 4: FAISS Index Wrapper (`flowneg/faiss_index.py`)

```python
class FAISSIndex:
    """
    Wraps faiss.IndexFlatIP (inner product) or IndexFlatL2.
    Rebuilt every K epochs from current item embeddings.
    
    Methods:
      build(item_embeddings: np.ndarray)  # [n_items, d]
      search(queries: np.ndarray, M: int) -> (distances, indices)  # [batch, M]
    """
```

**Config params:** `faiss_M: 50` (candidate set size), `faiss_rebuild_every: 5` (epochs), `faiss_metric: 'IP'`

### Module 5: FlowNeg Trainer (`flowneg/flowneg_trainer.py`)

```python
class FlowNegTrainer(Trainer):
    """
    Extends RecBole's Trainer to support the alternating training schedule:
    
    Phase 1 (epochs 0..W):    Standard RNS warm-up, train backbone only
    Phase 2 (epochs W..end):  Alternating:
      - Every epoch:  train backbone with FlowNeg-sampled negatives
      - Every K epochs: 
        a) rebuild FAISS index from current item embeddings
        b) fine-tune CFM velocity network on current embeddings
    """
```

**Overrides:**
- `__init__`: creates CFMTrainer, FlowNegSampler, FAISSIndex alongside backbone
- `_train_epoch`: checks epoch to decide warm-up vs. flow sampling; manages FAISS rebuild and CFM update
- `fit`: adds CFM training loop interleaved with backbone training

**Critical:** The backbone's `calculate_loss(interaction)` remains UNCHANGED. Only the negative item IDs fed into the batch change.

### Module 6: FlowNeg DataLoader (`flowneg/flowneg_dataloader.py`)

```python
class FlowNegTrainDataLoader(TrainDataLoader):
    """
    Overrides _neg_sampling to use FlowNegSampler instead of standard sampler.
    Minimal override: collate_fn -> _neg_sampling chain stays the same.
    """
```

### Module 7: MINDsmall Dataset Preparation (`scripts/prepare_mindsmall.py`)

#### 7a. Download

```python
# MINDsmall download URLs (Azure Blob Storage):
TRAIN_URL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
DEV_URL   = "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
# Download to data_raw/MINDsmall_train/ and data_raw/MINDsmall_dev/
```

#### 7b. MIND Impression Format

Raw `behaviors.tsv` has 5 columns:
```
ImpressionID \t UserID \t Time \t History \t Impressions
123          \t U131   \t 11/13/2019 8:36:57 AM \t N11 N21 N103 \t N4-1 N34-1 N156-0 N207-0 N198-0
```

Where in Impressions column:
- `N4-1` → user clicked news N4 (POSITIVE)
- `N156-0` → news N156 was displayed but NOT clicked (EXPOSURE NEGATIVE)

#### 7c. Conversion to RecBole Atomic Format

```python
# Step 1: Parse behaviors.tsv
# For each impression:
#   - clicked items (label=1) → positive interactions
#   - non-clicked items (label=0) → exposure negatives

# Step 2: Create mindsmall.inter (only positive interactions for RecBole training)
# user_id:token    item_id:token    timestamp:float
# U131             N4               1573634217
# U131             N34              1573634217

# Step 3: Create exposure_cache/mindsmall_exposure_negs.pkl
# Dictionary: {user_id_remapped: set(exposed_neg_item_ids_remapped)}
# Built from ALL non-clicked items across ALL impressions of each user
# This is the grounding set for CFM training

# Step 4: Create mindsmall.item (optional, for content-aware extensions)
# item_id:token    category:token    subcategory:token    title:token_seq
# N4               lifestyle         travel               "The 10 best..."
```

#### 7d. Key Conversion Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Which interactions are "positive"? | Clicked items (`-1` label) | Standard implicit feedback convention |
| Where do exposure negatives come from? | Non-clicked items in impressions (`-0` label) | Ground-truth exposure data |
| How to split? | MINDsmall train → train/valid (temporal: first 4 days / last day). MINDsmall dev → test. | Follows MIND convention; dev labels are available |
| User history field? | Stored separately, not in .inter | RecBole handles interaction history internally |
| Item content features? | Optional .item file; BPR/LightGCN don't use them | Keep it simple for pipeline test; content-aware models are future work |

#### 7e. Expected MINDsmall Statistics After Conversion

| Metric | Expected Value |
|--------|---------------|
| Users | ~50,000 |
| Items (news articles) | ~51,000 |
| Positive interactions (clicks) | ~347,000 (train) |
| Exposure negatives per user | ~50-200 (from impressions) |
| Avg. impressions per user | ~4.7 |
| Avg. clicks per impression | ~1.7 |
| Avg. displayed items per impression | ~20-40 |
| Positive-to-exposure-negative ratio | ~1:10 to 1:20 |

**This is excellent for FlowNeg**: rich exposure-negative signal, clear ground-truth, manageable scale.

### Module 8: Custom Metrics (`flowneg/metrics.py`)

```python
def fn_rate(neg_item_ids, ground_truth_positives):
    """Fraction of sampled negatives that are actually positive (false negative rate)"""
    
def candidate_purity(candidate_set, ground_truth_positives):
    """Fraction of candidate set that is truly negative"""
    
def hardness_score(neg_scores, pos_scores):
    """Average neg_score / pos_score ratio (higher = harder negatives)"""
    
def diversity_coverage(neg_item_ids, total_items):
    """Unique items sampled / total items (mode coverage)"""
    
def exposure_recall(sampled_negs, exposure_neg_set):
    """How well sampled negatives overlap with exposure-negative set"""
```

---

## Experiment Blocks

### Block B1: Exposure-Grounded Realness (Claim C1)

- **Claim tested:** Exposure-grounded CFM generates negatives with lower FN rate
- **Why this block exists:** Core differentiator — impression-level exposure data improves negative quality
- **Dataset / split / task:** MINDsmall. Train/valid/test from temporal split.
- **Compared systems:**
  | System | Description |
  |--------|-------------|
  | RNS | Uniform random (RecBole default) |
  | DNS | Dynamic negative sampling (RecBole built-in, `candidate_num: 100`) |
  | PNS | Popularity-biased sampling (RecBole built-in) |
  | Exposure-Weighted | Sampling weighted by exposure frequency |
  | **FlowNeg** | Our method |
- **Metrics:** FN rate (primary), candidate-set purity, exposure recall
- **Setup details:** BPR backbone (MF, d=64), 100 epochs (fewer for pipeline test), eval every 5 epochs, 1 seed (pipeline), 3 seeds (final)
- **Success criterion:** FlowNeg FN rate lower than DNS and PNS
- **Failure interpretation:** Exposure grounding does not help → re-examine impression data quality or CFM capacity
- **Table / figure target:** Table 2 (FN analysis)
- **Priority:** MUST-RUN

### Block B2: End-to-End Recommendation Quality (Claim C1 + C4)

- **Claim tested:** FlowNeg improves downstream recommendation quality
- **Why this block exists:** Must show improved negatives translate to better recommendations
- **Dataset / split / task:** MINDsmall.
- **Compared systems:**
  | System | Description |
  |--------|-------------|
  | RNS-MF | BPR + MF + uniform negatives |
  | DNS-MF | BPR + MF + dynamic negatives |
  | PNS-MF | BPR + MF + popularity negatives |
  | Exposure-Weighted-MF | BPR + MF + exposure-frequency weighting |
  | **FlowNeg-MF** | Our method with MF backbone |
  | RNS-LightGCN | BPR + LightGCN + uniform |
  | DNS-LightGCN | BPR + LightGCN + dynamic |
  | **FlowNeg-LightGCN** | Our method with LightGCN backbone |
- **Metrics:** Recall@20, NDCG@20 (primary); Recall@50, MRR@20 (secondary)
- **Setup details:**
  - MF: embedding_size=64, lr=1e-3, batch_size=2048, 100 epochs
  - LightGCN: embedding_size=64, n_layers=3, lr=1e-3, reg_weight=1e-4, 100 epochs
  - FlowNeg: warm_up=5, cfm_lr=1e-3, tau_h=0.5, M=50, K=5
  - Pipeline test: 1 seed. Final: 3 seeds, mean ± std.
- **Success criterion:** FlowNeg beats RNS/DNS on Recall@20 for both MF and LightGCN
- **Failure interpretation:** Flow does not help → check for mode collapse or insufficient exposure negatives
- **Table / figure target:** Table 1 (main results)
- **Priority:** MUST-RUN

### Block B3: Hardness Control Ablation (Claim C2)

- **Claim tested:** tau_h provides smooth, independent hardness control
- **Why this block exists:** Structural independence claim — hardness does not corrupt realness
- **Dataset / split / task:** MINDsmall with MF backbone
- **Compared systems:**
  | Variant | tau_h | Reweighting |
  |---------|-------|-------------|
  | FlowNeg-uniform | inf (uniform over candidates) | No |
  | FlowNeg-soft | 1.0 | Yes |
  | FlowNeg-medium | 0.5 | Yes |
  | FlowNeg-hard | 0.1 | Yes |
  | FlowNeg-no-retrieve | N/A | Reweight over ALL items |
- **Metrics:** Recall@20, FN rate, hardness score, all plotted jointly vs tau_h
- **Setup details:** Same as B2 MF config, vary tau_h in {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, inf}
- **Success criterion:** Monotonic hardness increase as tau_h decreases; FN rate stays low across all tau_h
- **Failure interpretation:** FN rate increases with hardness → decoupling is incomplete
- **Table / figure target:** Figure 4 (3-axis plot: hardness, FN rate, Recall@20 vs tau_h)
- **Priority:** MUST-RUN

### Block B4: Flow Query Value — CRITICAL TEST (Anti-claim)

- **Claim tested:** Flow-generated queries are better than simpler query strategies
- **Why this block exists:** Justifies the generative model — if a centroid works just as well, FlowNeg is over-engineered
- **Dataset / split / task:** MINDsmall with MF backbone
- **Compared systems (ALL use the same retrieve-reweight pipeline):**
  | Query Strategy | Description |
  |----------------|-------------|
  | (a) Random Gaussian | x_0 ~ N(0, I), no flow |
  | (b) User embedding | query = user_emb directly |
  | (c) Exposure-neg centroid | query = mean(exposure_neg_embs per user) |
  | (d) Exposure-neg KNN | query = nearest exposure neg to user |
  | (e) **FlowNeg** | query = midpoint4(v_theta, x_0, u) |
- **Metrics:** Recall@20, NDCG@20, FN rate, candidate diversity
- **Setup details:** Identical retrieve-reweight pipeline (same M=50, same tau_h=0.5). ONLY query generation differs.
- **Success criterion:** FlowNeg clearly beats centroid (c) and KNN (d) on Recall@20
- **Failure interpretation:** Flow adds no value → simplify to centroid-based method (major pivot)
- **Table / figure target:** Table 3 (query ablation)
- **Priority:** MUST-RUN (highest risk)

### Block B5: Diagnostics (Supplementary)

- **Claim tested:** Understanding FlowNeg's behavior on MINDsmall
- **Analyses:**
  - Dense vs. sparse user breakdown (impression count quartiles)
  - Mode coverage: unique items sampled per epoch
  - Embedding drift: cosine similarity between FAISS rebuild epochs
  - ODE steps sensitivity: {1, 2, 4, 8}
- **Priority:** NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost Est. | Risk |
|-----------|------|------|---------------|-----------|------|
| **M0: Sanity** | Data pipeline works, BPR+RNS on ml-100k reproduces, MINDsmall loads correctly | R001-R005 | BPR Recall@20 on ml-100k reasonable; MINDsmall loads; exposure-neg stats look correct | 1 GPU-hr | Low |
| **M1: Baselines** | RNS, DNS, PNS baselines on MINDsmall | R006-R011 | DNS/PNS Recall@20 in reasonable range for MINDsmall | 5 GPU-hrs | Low |
| **M2: FlowNeg Core** | CFM training converges, flow generates valid embeddings, end-to-end training works | R012-R016 | CFM loss converges; generated points near item space; FN rate < RNS | 8 GPU-hrs | High: first full integration |
| **M3: Critical Test** | Query ablation (Block B4) — does flow add value? | R017-R021 | FlowNeg query beats centroid/KNN | 5 GPU-hrs | **Highest risk** |
| **M4: Main Results** | Full comparison on MINDsmall with MF + LightGCN | R022-R029 | FlowNeg wins on Recall@20 for both backbones | 10 GPU-hrs | Medium |
| **M5: Ablations** | tau_h sweep, hardness analysis | R030-R038 | Smooth hardness-FN tradeoff | 8 GPU-hrs | Low |
| **M6: Diagnostics** | Dense/sparse, diversity, drift | R039-R043 | No show-stoppers | 3 GPU-hrs | Low |

**Total estimated: ~40 GPU-hours (single GPU, RTX 3090/4090 or A100)**

### Stop/Go Gates

- **After M0:** If MINDsmall exposure negatives per user < 10 on average, consider relaxing filters or using impression-level augmentation.
- **After M2:** If CFM loss does not converge below 0.5 after 30 epochs, check embedding normalization and learning rate.
- **After M3 (CRITICAL):** If FlowNeg query does not beat centroid by a meaningful margin, STOP and reconsider. Possible pivots:
  (a) use flow for diversity only, not query quality
  (b) switch to diffusion-based generation
  (c) reframe contribution around exposure grounding alone

---

## RecBole-Specific Implementation Details

### 1. How FlowNeg Plugs Into RecBole's Pipeline

```python
# scripts/run_flowneg.py
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.sampler import Sampler
from recbole.model.general_recommender.bpr import BPR
from flowneg.flow_sampler import FlowNegSampler
from flowneg.flowneg_dataloader import FlowNegTrainDataLoader
from flowneg.flowneg_trainer import FlowNegTrainer

# 1. Standard config + dataset creation
config = Config(model='BPR', dataset='mindsmall',
                config_file_list=['configs/flowneg_mf_mindsmall.yaml'])
dataset = create_dataset(config)
built_datasets = dataset.build()
train_dataset, valid_dataset, test_dataset = built_datasets

# 2. Standard samplers for valid/test (unchanged)
standard_sampler = Sampler(['train', 'valid', 'test'], built_datasets, 'uniform')
valid_sampler = standard_sampler.set_phase('valid')
test_sampler = standard_sampler.set_phase('test')

# 3. FlowNeg sampler for training
flowneg_sampler = FlowNegSampler(
    phases=['train', 'valid', 'test'],
    datasets=built_datasets,
    config=config,
    distribution='uniform',  # fallback for warm-up
)
train_sampler = flowneg_sampler.set_phase('train')

# 4. Custom train dataloader, standard eval dataloaders
train_data = FlowNegTrainDataLoader(config, train_dataset, train_sampler, shuffle=True)
valid_data = FullSortEvalDataLoader(config, valid_dataset, valid_sampler, shuffle=False)
test_data = FullSortEvalDataLoader(config, test_dataset, test_sampler, shuffle=False)

# 5. Model (standard BPR or LightGCN — UNCHANGED)
model = BPR(config, dataset).to(config['device'])

# 6. Custom trainer
trainer = FlowNegTrainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
test_result = trainer.evaluate(test_data)
```

### 2. MINDsmall Exposure-Negative Set Construction

```python
# In scripts/prepare_mindsmall.py:

import pickle
from collections import defaultdict

exposure_negs = defaultdict(set)  # {user_id: set(neg_item_ids)}
positive_items = defaultdict(set)  # {user_id: set(pos_item_ids)}

# Parse behaviors.tsv
for line in open('data_raw/MINDsmall_train/behaviors.tsv'):
    imp_id, user_id, time, history, impressions = line.strip().split('\t')
    for entry in impressions.split():
        news_id, label = entry.rsplit('-', 1)
        if label == '1':
            positive_items[user_id].add(news_id)
        else:  # label == '0'
            exposure_negs[user_id].add(news_id)

# Remove any items that are positive for the user (safety check)
for user_id in exposure_negs:
    exposure_negs[user_id] -= positive_items[user_id]

# Save after ID remapping (to match RecBole's remapped IDs)
# ... remap using dataset.token2id ...
pickle.dump(dict(exposure_negs), open('exposure_cache/mindsmall_exposure_negs.pkl', 'wb'))
```

### 3. MINDsmall .inter File Format

```
# mindsmall.inter
user_id:token	item_id:token	timestamp:float
U131	N4	1573634217.0
U131	N34	1573634217.0
U245	N129416	1573548900.0
```

Only clicked items go into .inter. Exposure negatives stored separately.

### 4. MINDsmall Split Strategy

| Source | RecBole Split | Purpose |
|--------|---------------|---------|
| MINDsmall train (days 1-4) | train | Training |
| MINDsmall train (day 5) | valid | Validation (hyperparameter selection) |
| MINDsmall dev | test | Testing (final evaluation) |

Implementation: Use `eval_args: split: {'RS': [8,1,1]}` or create separate .inter files with `benchmark_filename`.

**Recommended approach**: Use `benchmark_filename` with pre-split files:
```
dataset/mindsmall/
  mindsmall.train.inter    # days 1-4 of MINDsmall_train
  mindsmall.valid.inter    # day 5 of MINDsmall_train
  mindsmall.test.inter     # MINDsmall_dev
```

Config:
```yaml
benchmark_filename: ['train', 'valid', 'test']
```

### 5. Embedding Access Pattern

```python
# For BPR backbone:
item_embs = model.item_embedding.weight.detach()  # [n_items, d]
user_embs = model.user_embedding.weight.detach()  # [n_users, d]

# For LightGCN backbone:
user_all_embs, item_all_embs = model.forward()
item_embs = item_all_embs.detach()
user_embs = user_all_embs.detach()
```

### 6. Config Example (`configs/flowneg_mf_mindsmall.yaml`)

```yaml
# Dataset
dataset: mindsmall
data_path: dataset/
benchmark_filename: ['train', 'valid', 'test']

# Model (standard BPR)
model: BPR
embedding_size: 64

# Training
epochs: 100
train_batch_size: 2048
learner: adam
learning_rate: 0.001
weight_decay: 0.0
eval_step: 5
stopping_step: 15
train_neg_sample_args:
  distribution: uniform
  sample_num: 1

# Evaluation
eval_args:
  group_by: user
  order: RO
  mode:
    valid: full
    test: full
metrics: ['Recall', 'NDCG', 'MRR']
topk: [10, 20, 50]
valid_metric: Recall@20

# FlowNeg-specific
flowneg:
  warm_up_epochs: 5
  velocity_hidden_mult: 4
  velocity_layers: 2
  cfm_lr: 0.001
  cfm_epochs_per_update: 5
  cfm_train_steps: 200
  faiss_M: 50
  faiss_rebuild_every: 5
  tau_h: 0.5
  tau_h_schedule: 'cosine'
  ode_steps: 4
  n_flow_samples: 1
  exposure_cache_path: 'exposure_cache/mindsmall_exposure_negs.pkl'
```

### 7. Baseline Implementation Notes

| Baseline | RecBole Support | Action Needed |
|----------|----------------|---------------|
| RNS | Built-in (`distribution: uniform`) | Config only |
| DNS | Built-in (`dynamic: True, candidate_num: 100`) | Config only |
| PNS | Built-in (`distribution: popularity`) | Config only |
| Exposure-Weighted | NOT built-in | Simple custom sampler: weight by user's exposure-neg frequency |

**Deferred baselines (for future full-paper experiments):**
- MixGCF: requires custom implementation (~100 LOC)
- IRGAN: requires custom trainer (~300 LOC)

For pipeline testing, RNS + DNS + PNS + Exposure-Weighted provide sufficient baselines to validate the FlowNeg pipeline.

---

## Compute and Data Budget

- **Total estimated GPU-hours:** ~40 (single GPU, suitable for RTX 3090/4090 or A100)
- **Data preparation:** MINDsmall download (~50MB compressed), conversion ~10 min
- **Human evaluation needs:** None
- **Biggest bottleneck:** M3 (query ablation) — determines if the generative component is justified

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Flow query adds no value over centroid (M3 fails) | Medium | Critical | Run M3 early. Pivot to exposure-grounding-only contribution |
| MINDsmall exposure-neg set is too uneven (some users have many, some few) | Medium | Medium | Filter users with <5 exposure negatives; report stats |
| CFM mode collapse | Medium | Medium | Monitor entropy; add noise regularization |
| MINDsmall too small for LightGCN to converge well | Low | Medium | Graph may be sparse; tune n_layers, use d=32 if needed |
| FAISS stale embeddings | Low | Medium | Increase rebuild frequency |

## Future Work (Deferred)

| Dataset | Purpose | When |
|---------|---------|------|
| Full MIND (MINDlarge) | Scale validation: 1M users, 161K articles | After pipeline validated on MINDsmall |
| KuaiRand | Random-exposure setting (different exposure mechanism from impressions) | After MIND results are solid |
| Coat | Explicit ratings as proxy for exposure | After KuaiRand |
| Gowalla / Yelp2018 | No exposure data — proxy-negative regime | Supplementary appendix |

## Final Checklist

- [ ] MINDsmall download and conversion pipeline works
- [ ] Exposure-negative cache built correctly from impressions
- [ ] Main paper tables covered (Table 1: main results, Table 2: FN analysis, Table 3: query ablation)
- [ ] Novelty isolated (Block B4: flow query vs 4 simpler alternatives)
- [ ] Simplicity defended (Block B3: tau_h sweep)
- [ ] Frontier contribution justified (CFM proven by query ablation)
- [ ] Nice-to-have runs separated from must-run runs
- [ ] Baselines use RecBole built-ins (RNS, DNS, PNS)
- [ ] Custom code isolated in `flowneg/` — no RecBole core modifications
- [ ] All configs are YAML-based and reproducible
- [ ] Clear upgrade path to full MIND and KuaiRand
