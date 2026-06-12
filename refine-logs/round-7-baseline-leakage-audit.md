# Round 7 Baseline Leakage Audit

Date: 2026-06-08

## Question

The current MIND numbers were abnormally high, especially the MultiVAE and
flow-negative runs from Round 6. This round checks whether the baseline itself
is contaminated, using ml-100k and split-level diagnostics.

## Root Cause

The bug was in `src/recbole_utils.py`.

Before the fix, the code did:

```python
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = model_class(config, dataset).to(config['device'])
```

The model constructor received the full unsplit dataset.

This is a hard leakage bug:

- `MultiVAE.__init__()` calls `build_histroy_items(dataset)`, so the user input
  history contained train, valid, and test interactions.
- `LightGCN.__init__()` calls `dataset.inter_matrix(...)`, so the graph also
  contained train, valid, and test interactions.

## Fixes

- `src/recbole_utils.py`
  - build RecBole models from the split-local train dataset attached to
    `train_data`;
  - keep returning the full dataset for token lookup and global diagnostics;
  - isolate RecBole `Config` from the runner's `sys.argv`.
- `src/pilot_runner.py`
  - added `ml-100k` to supported pilot datasets;
  - made dataset-level `data_path` overrides explicit.
- `configs/lightgcn_ml100k.yaml`
  - added ml-100k sanity config.
- `scripts/check_recbole_split_leakage.py`
  - added a no-training diagnostic that compares held-out pairs against the
    model constructor state.

## Leakage Diagnostics

Command:

```bash
.venv/bin/python scripts/check_recbole_split_leakage.py \
  --dataset ml-100k \
  --experiment M0_multivae_baseline
```

ml-100k MultiVAE:

| Quantity | Value |
| --- | ---: |
| Full unique pairs | 100000 |
| Train unique pairs | 80808 |
| Held-out unique pairs | 19192 |
| Old full-dataset constructor overlap | 19192 pairs / 943 users |
| Current model constructor overlap | 0 pairs / 0 users |

The same check on ml-100k LightGCN produced the same overlap result:
old constructor overlap `19192`, current constructor overlap `0`.

MIND MultiVAE:

| Quantity | Value |
| --- | ---: |
| Full unique pairs | 345188 |
| Train unique pairs | 240275 |
| Held-out unique pairs | 104913 |
| Old full-dataset constructor overlap | 104913 pairs / 59581 users |
| Current model constructor overlap | 0 pairs / 0 users |

This invalidates the Round 6 MIND MultiVAE metrics.

Note: RecBole special-cases `dataset == ml-100k` and resolves it to its bundled
`dataset_example/ml-100k`. The loaded data still contains the full 100000
MovieLens-100K interactions and is sufficient for this baseline sanity check.

## ml-100k Sanity Runs

These are short 20-epoch checks, not final tuned benchmarks.

Commands:

```bash
bash scripts/run_pilot.sh --dataset ml-100k --stage M0vae \
  --set epochs=20 \
  --set show_progress=False \
  --set stopping_step=5

bash scripts/run_pilot.sh --dataset ml-100k --stage M0 \
  --set epochs=20 \
  --set show_progress=False \
  --set stopping_step=5
```

Results:

| Run | Model | Best valid score | Test Recall@20 | Test NDCG@20 |
| --- | --- | ---: | ---: | ---: |
| `M0_multivae_baseline_ml_100k` | MultiVAE | 0.2118 | 0.3129 | 0.2555 |
| `M0_baseline_ml_100k` | LightGCN | 0.1571 | 0.2382 | 0.1997 |

These values are no longer near 1.0 and are consistent with a non-leaking
baseline sanity run.

## Cleanup

Removed the invalid Round 6 MIND artifacts:

- old MultiVAE MIND checkpoints under `saved/`;
- `results/pilot/M0_multivae_baseline.json`;
- `results/pilot/M2_multivae_flow_no_grpo.json`;
- `results/pilot/M2_multivae_random_neg_control.json`;
- old MIND flow cache checkpoints;
- stale MIND M0 pointer files.

Kept the ml-100k sanity artifacts:

- `results/pilot/M0_multivae_baseline_ml_100k.json`;
- `results/pilot/M0_baseline_ml_100k.json`;
- `results/pilot/logs/M0_multivae_baseline_ml_100k.log`;
- `results/pilot/logs/M0_baseline_ml_100k.log`;
- current ml-100k checkpoints in `saved/`.

## Conclusion

The baseline was broken. The abnormal Round 6 MIND numbers were caused by
train/valid/test leakage during model construction, not by a genuine FlowNS
gain.

The fix cuts the leakage at the RecBole setup boundary. The ml-100k checks now
show normal-scale results, and the model constructor state contains zero
held-out pairs.

Next valid experiment: rerun MIND `M0_multivae_baseline`,
`M2_multivae_flow_no_grpo`, and `M2_multivae_random_neg_control` from scratch
with the fixed setup before making any claim about flow-generated negatives.
