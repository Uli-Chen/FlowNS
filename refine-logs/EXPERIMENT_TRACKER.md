# Experiment Tracker — MINDsmall Pipeline Test

## Milestone M0: Sanity

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R001 | M0 | Sanity: BPR+RNS on ml-100k | BPR + uniform | ml-100k | Recall@20, NDCG@20 | MUST | TODO | Verify RecBole pipeline works |
| R002 | M0 | Download MINDsmall | N/A | mindsmall | Download success | MUST | TODO | Azure Blob Storage |
| R003 | M0 | Convert MINDsmall to RecBole format | N/A | mindsmall | .inter file stats | MUST | TODO | Run prepare_mindsmall.py |
| R004 | M0 | Build exposure-negative cache | N/A | mindsmall | Exposure-neg stats | MUST | TODO | Verify per-user exposure-neg counts |
| R005 | M0 | BPR+RNS on MINDsmall loads and trains | BPR + uniform | mindsmall | Recall@20 | MUST | TODO | Verify pipeline end-to-end |

## Milestone M1: Baselines

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R006 | M1 | Baseline: RNS-MF | BPR + uniform | mindsmall | Recall@20, NDCG@20, MRR@20 | MUST | TODO | |
| R007 | M1 | Baseline: DNS-MF | BPR + DNS(100) | mindsmall | Recall@20, NDCG@20, MRR@20 | MUST | TODO | |
| R008 | M1 | Baseline: PNS-MF | BPR + popularity | mindsmall | Recall@20, NDCG@20, MRR@20 | MUST | TODO | |
| R009 | M1 | Baseline: Exposure-Weighted-MF | BPR + exp-weight | mindsmall | Recall@20, NDCG@20, MRR@20 | MUST | TODO | Custom sampler |
| R010 | M1 | Baseline: RNS-LightGCN | LightGCN + uniform | mindsmall | Recall@20, NDCG@20, MRR@20 | MUST | TODO | |
| R011 | M1 | Baseline: DNS-LightGCN | LightGCN + DNS(100) | mindsmall | Recall@20, NDCG@20, MRR@20 | MUST | TODO | |

## Milestone M2: FlowNeg Core

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R012 | M2 | CFM training converges | FlowNeg-MF (CFM only) | mindsmall | CFM loss curve | MUST | TODO | CondOT loss < 0.5 after 30 epochs |
| R013 | M2 | Flow generates valid embeddings | FlowNeg-MF | mindsmall | Cosine sim to item space | MUST | TODO | Generated points near real items |
| R014 | M2 | FAISS retrieval works | FlowNeg-MF | mindsmall | Candidates contain relevant items | MUST | TODO | |
| R015 | M2 | End-to-end FlowNeg trains | FlowNeg-MF | mindsmall | Recall@20, FN rate | MUST | TODO | Compare against R006 |
| R016 | M2 | End-to-end FlowNeg-LightGCN | FlowNeg-LightGCN | mindsmall | Recall@20, FN rate | MUST | TODO | Compare against R010 |

## Milestone M3: Critical Test (Query Ablation)

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R017 | M3 | Query: Random Gaussian | Gaussian + retrieve-reweight | mindsmall | Recall@20, FN rate | MUST | TODO | Baseline (a) |
| R018 | M3 | Query: User embedding | UserEmb + retrieve-reweight | mindsmall | Recall@20, FN rate | MUST | TODO | Baseline (b) |
| R019 | M3 | Query: Exposure-neg centroid | Centroid + retrieve-reweight | mindsmall | Recall@20, FN rate | MUST | TODO | Baseline (c) — MUST beat |
| R020 | M3 | Query: Exposure-neg KNN | KNN + retrieve-reweight | mindsmall | Recall@20, FN rate | MUST | TODO | Baseline (d) — MUST beat |
| R021 | M3 | Query: FlowNeg | FlowNeg + retrieve-reweight | mindsmall | Recall@20, FN rate | MUST | TODO | Must clearly beat R019/R020 |

## Milestone M4: Main Results

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R022 | M4 | Main: FlowNeg-MF (3 seeds) | FlowNeg + BPR | mindsmall | Recall@20, NDCG@20 | MUST | TODO | |
| R023 | M4 | Main: FlowNeg-LightGCN (3 seeds) | FlowNeg + LightGCN | mindsmall | Recall@20, NDCG@20 | MUST | TODO | |
| R024 | M4 | Main: RNS-MF (3 seeds) | BPR + uniform | mindsmall | Recall@20, NDCG@20 | MUST | TODO | |
| R025 | M4 | Main: DNS-MF (3 seeds) | BPR + DNS | mindsmall | Recall@20, NDCG@20 | MUST | TODO | |
| R026 | M4 | Main: PNS-MF (3 seeds) | BPR + popularity | mindsmall | Recall@20, NDCG@20 | MUST | TODO | |
| R027 | M4 | Main: ExpWeight-MF (3 seeds) | BPR + exp-weight | mindsmall | Recall@20, NDCG@20 | MUST | TODO | |
| R028 | M4 | FN analysis: all systems | All systems | mindsmall | FN rate, purity | MUST | TODO | Table 2 |
| R029 | M4 | Main: all LightGCN (3 seeds) | RNS/DNS/FlowNeg + LightGCN | mindsmall | Recall@20, NDCG@20 | MUST | TODO | |

## Milestone M5: Ablations

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R030 | M5 | tau_h sweep: 0.05 | FlowNeg(tau=0.05) | mindsmall | Recall@20, FN, hardness | MUST | TODO | |
| R031 | M5 | tau_h sweep: 0.1 | FlowNeg(tau=0.1) | mindsmall | Recall@20, FN, hardness | MUST | TODO | |
| R032 | M5 | tau_h sweep: 0.2 | FlowNeg(tau=0.2) | mindsmall | Recall@20, FN, hardness | MUST | TODO | |
| R033 | M5 | tau_h sweep: 0.5 | FlowNeg(tau=0.5) | mindsmall | Recall@20, FN, hardness | MUST | TODO | |
| R034 | M5 | tau_h sweep: 1.0 | FlowNeg(tau=1.0) | mindsmall | Recall@20, FN, hardness | MUST | TODO | |
| R035 | M5 | tau_h sweep: 2.0 | FlowNeg(tau=2.0) | mindsmall | Recall@20, FN, hardness | MUST | TODO | |
| R036 | M5 | tau_h sweep: inf | FlowNeg(no reweight) | mindsmall | Recall@20, FN, hardness | MUST | TODO | |
| R037 | M5 | Ablation: no retrieve | FlowNeg(all items) | mindsmall | Recall@20, FN | MUST | TODO | |
| R038 | M5 | Candidate set size: M sweep | FlowNeg(M=10,25,50,100) | mindsmall | Recall@20, FN | NICE | TODO | |

## Milestone M6: Diagnostics

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R039 | M6 | Dense vs sparse users | FlowNeg-MF by quartile | mindsmall | Recall@20 per group | NICE | TODO | |
| R040 | M6 | Mode coverage | FlowNeg-MF | mindsmall | Unique items/epoch | NICE | TODO | |
| R041 | M6 | Embedding drift | FlowNeg-MF | mindsmall | Cosine drift per rebuild | NICE | TODO | |
| R042 | M6 | ODE steps: {1, 2, 4, 8} | FlowNeg(steps=N) | mindsmall | Recall@20, FN | NICE | TODO | |
| R043 | M6 | t-SNE visualization | FlowNeg queries vs baselines | mindsmall | Figure | NICE | TODO | |
