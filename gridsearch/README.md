# Grid search: flow negatives (no GRPO) across backbones

Sweeps the **"flow, no GRPO"** FlowNS setting over RecBole backbones on MIND and
writes JSON reports to `gridsearch/reports/`. Built to find the best
lr / flow / neg-sampling hyperparameters per backbone before committing to a
formal run.

Each grid point is a self-contained pipeline (`rec_pretrain -> flow_pretrain ->
joint`, GRPO disabled, trained from scratch — `use_paired_m0: false`) executed
through the same code path as the pilot runner
(`src.pilot_runner._run_flowns_config`), so results are comparable to the named
experiments.

## Backbones

| Archetype | Model | Status |
| --- | --- | --- |
| MF | `BPR` | supported |
| GCN | `LightGCN` | supported |
| VAE | `MultiVAE` | supported |
| Transformer (sequential) | `SASRec` | **disabled** — needs a sequential data pipeline and a per-user representation for the flow; the harness will record a clean error if enabled. |

(`MultiDAE` / `RecVAE` use the same autoencoder path as `MultiVAE` and can be
added the same way.)

## Search space

Edit `gridsearch/search_space.yaml`. Structure:

- `base`: shared "flow, no GRPO" settings.
- `backbones.<Name>`: `enabled`, `model`, `fixed` (per-backbone constants), and
  `grid` (the searched lists). Anything not set falls back to
  `configs/experiments.yaml` defaults.

The default `grid` searches **lr** (`learning_rate`, `joint_lr`), a **flow/neg
param** (`flow_neg_ratio`, or `flow_neg_loss_weight` for the VAE), and a
**neg-sampling param** (`mapping_strategy` ∈ {`nearest`, `boundary_topk`}). Move
any extra key into a `grid` block to search it (examples are listed at the bottom
of the YAML).

## Usage

```bash
# quick smoke: 1 epoch per phase, first grid point per enabled backbone
.venv/bin/python -m gridsearch.grid_search --smoke

# print the plan without running
.venv/bin/python -m gridsearch.grid_search --list

# formal sweep (default space)
.venv/bin/python -m gridsearch.grid_search

# restrict backbones / cap runs / point at a custom space
.venv/bin/python -m gridsearch.grid_search --backbones BPR,MultiVAE --max-runs 8
.venv/bin/python -m gridsearch.grid_search --space gridsearch/search_space.yaml
```

## Report format

`gridsearch/reports/{grid,smoke}_<dataset>_<timestamp>.json`, written
incrementally (survives crashes / Ctrl-C):

```json
{
  "dataset": "mind", "seed": 2020, "smoke": false,
  "n_planned": 48, "n_ok": 47, "n_error": 1,
  "runs": [
    {"backbone": "BPR", "model": "BPR", "params": {"learning_rate": 0.001, ...},
     "status": "ok", "best_valid_score": 0.09,
     "test_result": {"recall@20": ..., "ndcg@20": ...},
     "generation_diagnostics": {"fn_rate_all_known": ..., "w_mapped_mean": ...},
     "runtime_sec": 123.4, "error": null}
  ],
  "best_per_backbone": {"BPR": {"params": {...}, "best_valid_score": ..., "test_result": {...}}}
}
```

A summary table is also printed to stdout at the end.
