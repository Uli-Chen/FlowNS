# Round 6 MultiVAE Backbone Pilot

Date: 2026-06-08

Status update: invalidated on 2026-06-08 by the Round 7 baseline leakage
audit. `setup_recbole()` was constructing models from the full RecBole dataset
after train/valid/test splitting. For MultiVAE this put held-out interactions
into `history_item_matrix`; for LightGCN it put held-out interactions into the
graph. Do not use the MIND metrics below as valid experimental evidence.

## Motivation

LightGCN's embedding space may be a poor fit for FlowNS because nearest item
projection collapses continuous flow hardness into weak discrete negatives. This
round tests whether an autoencoder recommender space is more compatible.

SASRec was considered, but it is a sequential recommender and would require a
sequential dataset pipeline. The current MIND setup is a general user-item
RecBole dataset. MultiVAE is a better first backbone swap because it is already
available in RecBole's general recommender stack and supports full-sort
evaluation on the current data.

No GRPO was used in this round. The goal was only to test whether flow-generated
negative items help training.

## Implementation

Added support for RecBole autoencoder backbones:

- `src/recbole_utils.py`
  - added model loading for `MultiVAE`, `MultiDAE`, `RecVAE`, and `SASRec`;
  - switched trainer creation to RecBole `get_trainer`;
  - added autoencoder-compatible embedding extraction.
- `src/flowns_trainer.py`
  - flow dimension now comes from the backbone scoring space instead of assuming
    `embedding_size`;
  - added an autoencoder joint loss:
    native MultiVAE CE/KL loss plus a sampled-positive BPR penalty on generated
    negatives;
  - supports random-negative control through the same BPR penalty path.
- `configs/experiments.yaml`
  - added `M0_multivae_baseline`;
  - added `M2_multivae_flow_no_grpo`;
  - added `M2_multivae_random_neg_control`.
- `scripts/run_pilot.sh`
  - added stages `M0vae`, `M2vae`, and `M2vaeR`.

For MultiVAE, the flow space is the decoder-logit space:

- user representation: hidden vector before the decoder's final item-output
  linear layer;
- item representation: final decoder linear layer's item weight vector.

This keeps flow samples and mapped items aligned with the scoring function used
by `full_sort_predict`.

## Commands

Smoke test:

```bash
.venv/bin/python -m src.pilot_runner run \
  --experiment M0_multivae_baseline \
  --dataset mind \
  --set epochs=1 \
  --set evaluate_test=False \
  --set save_rec_checkpoint_pointer=False \
  --set train_batch_size=256 \
  --set eval_batch_size=204800
```

Pilot runs:

```bash
.venv/bin/python -m src.pilot_runner run \
  --experiment M0_multivae_baseline \
  --dataset mind \
  --set epochs=20 \
  --set show_progress=False

.venv/bin/python -m src.pilot_runner run \
  --experiment M2_multivae_flow_no_grpo \
  --dataset mind \
  --set epochs=20 \
  --set show_progress=False \
  --set flow_pretrain_epochs=40 \
  --set flow_log_interval=5 \
  --set joint_rec_epochs=10 \
  --set stopping_step=5

.venv/bin/python -m src.pilot_runner run \
  --experiment M2_multivae_random_neg_control \
  --dataset mind \
  --set epochs=20 \
  --set show_progress=False \
  --set joint_rec_epochs=10 \
  --set stopping_step=5
```

## Results

Dataset: `mind`

| Run | Backbone | Flow | GRPO | Valid NDCG@20 | Recall@20 | NDCG@20 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `M0_multivae_baseline` | MultiVAE | no | no | 0.6690 | 0.9566 | 0.8275 |
| `M2_multivae_random_neg_control` | MultiVAE | no | no | 0.6910 | 0.9682 | 0.8529 |
| `M2_multivae_flow_no_grpo` | MultiVAE | yes | no | 0.7905 | 0.9916 | 0.9719 |

Flow pretraining:

- CFM loss: `0.984762 -> 0.677289` over 40 epochs.
- Active checkpoint:
  `results/pilot/M2_multivae_flow_no_grpo_flow.pt`
- Reference checkpoint:
  `results/pilot/M2_multivae_flow_no_grpo_flow_ref.pt`

Generation diagnostics for `M2_multivae_flow_no_grpo`:

| Diagnostic | Value |
| --- | ---: |
| `fn_rate_all_known` | 0.0000 |
| `mapped_unique_ratio` | 0.8670 |
| `W_continuous` mean | 0.4881 |
| `W_mapped` mean | 0.2029 |
| `W_mapped` pct > 0.8 | 0.0490 |

## Interpretation

This is the first strong positive signal for flow-generated negatives under a
standard no-filter protocol:

- MultiVAE baseline is already much stronger than LightGCN on the current MIND
  split.
- Adding a random sampled-negative BPR penalty gives a small improvement:
  NDCG@20 `0.8275 -> 0.8529`.
- Replacing random negatives with exposed-flow negatives gives a much larger
  improvement:
  NDCG@20 `0.8275 -> 0.9719`.

The random control is important: the improvement is not just from extra joint
training or from adding a BPR penalty to MultiVAE. Flow-generated negatives are
substantially more useful than random negatives in this backbone space.

However, the mapped hardness diagnostic is still low (`W_mapped=0.2029`). This
means the gain may come from realistic, diverse, low-FN negatives rather than
from hard boundary negatives. The continuous-to-discrete projection issue is not
fully solved, but it is less damaging in MultiVAE's decoder-logit space than it
was in LightGCN's embedding space.

## Caveats

- The current MIND random split appears very easy for MultiVAE; test NDCG is
  much higher than the LightGCN runs. This should be sanity-checked before
  making broad claims.
- The pilot used only one seed.
- The run used a 20-epoch MultiVAE baseline and 10 joint epochs. A longer,
  repeated run is needed before treating this as a stable result.
- SASRec remains untested because it requires converting the data pipeline to a
  sequential setup.

## Next Steps

1. Repeat the three MultiVAE runs with seeds 2020/2021/2022.
2. Run a longer baseline and longer flow-joint schedule to check whether the
   flow gain persists after stronger MultiVAE convergence.
3. Add an exposure-only MultiVAE control if we want to separate "flow learned
   exposed negatives" from direct exposed-negative supervision.
4. If the result holds, build a proper sequential MIND pipeline and then test
   SASRec.
