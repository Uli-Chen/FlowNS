# Round 4 No-Filter Redesign Log

Date: 2026-06-08

## Reason For Reset

The previous large gain came from `eval_exposed_neg_penalty`, which subtracts a
large score from user-specific exposed-but-unclicked items during full-sort
evaluation. This changes the evaluation candidate scores directly and is not a
learned flow/RL improvement.

M2u proved the issue: filter-only evaluation matched M2t, so the +40% NDCG@20
lift cannot be attributed to FlowNS or GRPO.

## Revised Protocol

Primary experiments now use standard RecBole full-sort evaluation:

- `eval_exposed_neg_penalty = 0.0`
- no evaluation-time exposure masking
- no exposure-filtered result can be used as the main >=10% claim

M2t-M2w are retained only as audit/history runs. They were removed from the
`all` stage in `scripts/run_pilot.sh`.

## Checkpoint Audit

Paired experiments read the same M0 recommender checkpoint:

- pointer files: `results/pilot/M0_model_path.mind.txt` and
  `results/pilot/M0_model_path.txt`
- target: `saved/LightGCN-Jun-05-2026_21-11-07.pth`
- file mtime: `2026-06-05 21:11:28 +0800`
- sha256: `1b37ff465f88d7dd6e21178cf30c21a197a3fedfcb7500a4e53baa59905dd66f`
- checkpoint metadata: `epoch=23`, `best_valid_score=0.0926`, `cur_step=0`

Later paired runs copy this M0 checkpoint into a new timestamped
`saved/LightGCN-Jun-08-...pth` file before joint updates. They do not overwrite
the original M0 checkpoint.

## Flow Cache

Flow CFM pretraining has not changed materially across the recent experiments;
the repeated runs were retraining the same exposed-negative flow from the same
M0 embeddings. To remove that waste and reduce run-to-run ambiguity:

- Added `flow_checkpoint_path` and `flow_ref_checkpoint_path` to
  `FlowNSTrainer`.
- Phase 2 now loads these files when present.
- If the files are missing, Phase 2 trains flow once and saves both the velocity
  state and the GRPO reference state.
- Added `M2flow_exposed_cache` as the explicit cache-builder stage.
- M2x/M2y now point to
  `results/pilot/M2flow_exposed_s20_cache_flow.pt` and
  `results/pilot/M2flow_exposed_s20_cache_flow_ref.pt`.

Cache audit note: an initial cache was created with inherited
`stopping_step=10`, stopping at epoch 247 with best loss 0.176742. That is not
used by M2x/M2y. The active cache path was renamed with `s20`, and
`M2flow_exposed_cache` now explicitly sets `stopping_step=20` to match the
previous M2 flow-pretrain protocol that reached best loss 0.163853.

Active cache files:

- flow state:
  `results/pilot/M2flow_exposed_s20_cache_flow.pt`
  - sha256:
    `eacb52c6cd0e7335d5bbf193334190093efae7999bb797bef50d7b8fbe8117ef`
  - mtime: `2026-06-08 20:56:45 +0800`
- reference state:
  `results/pilot/M2flow_exposed_s20_cache_flow_ref.pt`
  - sha256:
    `d3bb49aef94819e5185c4f87ed3c1db3a4545237e9891453d46aa023e597f500`
  - mtime: `2026-06-08 20:56:45 +0800`

The cache was verified by rerunning `M2flow_exposed_cache`; Phase 2 loaded the
existing flow checkpoint and skipped repeated CFM pretraining.

## Redesign

The new no-trick path is to make flow/RL affect the recommender during training,
not evaluation:

1. Add `score_topk` item mapping for joint training.
   - For each generated flow embedding, first take the user's current
     high-scoring non-training-positive candidates.
   - Then choose the candidate nearest to the flow embedding.
   - This aligns joint-training negatives with the top-ranked items that can
     affect Recall/NDCG.
2. M2x: no-filter score-top-k hard-negative joint training, no GRPO.
   - `joint_num_negatives=1`
   - `flow_neg_ratio=0.5`
   - `mapping_strategy=score_topk`
   - `mapping_topk=100`
3. M2y: M2x plus score-top-k GRPO.
   - `grpo_reward_mode=score_topk`
   - `grpo_old_policy_scope=epoch`
   - same standard full-sort evaluation as M2x

## Success Criteria

The new main result must beat M0 under standard full-sort:

- M0 Recall@20 = 0.2855, target >= 0.3141
- M0 NDCG@20 = 0.1348, target >= 0.1483

Flow is counted as effective only if M2x improves over M0 and prior no-filter
flow variants. RL is counted as effective only if M2y improves over M2x under
the same no-filter protocol.

## Attempt Notes

- Initial M2x attempt used `joint_num_negatives=4`, `flow_neg_ratio=1.0`,
  `mapping_topk=200`. It was stopped after 10 joint epochs because validation
  stayed at 0.0923-0.0926 and trended below M0 best=0.0926. This showed the
  hard-negative pressure was too strong for stable full-sort ranking.
- M2x/M2y were reduced to one generated negative, 50% flow/random mixing, and
  top-100 candidate mapping to keep the no-filter protocol while avoiding early
  ranking collapse.
- The reduced M2x attempt still trended down under standard full-sort
  validation: epoch 1-2 valid=0.0926, then epoch 3-12 fell to 0.0922. No JSON
  result was produced because the run was stopped before completing. This
  suggests `score_topk` joint mapping makes the negative pressure too close to
  the current ranking boundary and hurts the recommender before producing a
  measurable top-k gain.

## Follow-Up No-Filter Experiments

Because `score_topk` joint mapping hurt validation, the next branch separates
stable flow training from more aggressive boundary mining:

- `M2z_nearest_cached_no_filter_no_grpo`: reproduces the best stable
  no-filter flow setup using the cached exposed-flow checkpoint. This is the
  new flow-only control.
- `M2aa_nearest_cached_no_filter_grpo`: same as M2z, with mapped-item GRPO.
  This tests whether RL adds anything when the joint mapping is already stable.
- `M2ab_hardtopk_cached_no_filter_no_grpo`: uses `hard_topk` mapping, which
  first stays near the flow-generated embedding and only then chooses the
  highest-scored item among nearby candidates. This is less aggressive than
  `score_topk`, which starts from the recommender's current top-ranked items.
- `M2ac_hardtopk_cached_no_filter_grpo`: same as M2ab, with mapped-item GRPO.

All four experiments:

- load the same clean M0 recommender checkpoint
- load the same exposed-flow cache
- keep `eval_exposed_neg_penalty=0.0`
- use standard RecBole full-sort evaluation

## M2z/M2aa Results

M2z reproduced the stable flow-only control with cached flow:

- best valid: 0.0933
- test Recall@20: 0.2864
- test NDCG@20: 0.1365
- improvement over M0 NDCG@20: +1.3%
- diagnostics: `fn_rate_all_known=0.002`, `W_cont=0.7260`,
  `W_mapped=0.3948`, `mapped_unique_ratio=0.3510`

This confirms that exposed-flow nearest negatives are mildly useful under the
standard no-filter protocol, but the effect is far below the >=10% target.

M2aa added mapped-item GRPO to the same setup:

- GRPO pretraining reward increased from 0.0994 to 0.1019
- KL and importance ratios were unstable:
  - epoch 1: `kl=526.7204`, `ratio=102917843.2374`
  - epoch 2: `kl=1356.5982`, `ratio=102086452.1775`
  - epoch 3: `kl=1818.7602`, `ratio=102043921.2991`
- best valid never exceeded the M0 checkpoint value 0.0926
- test Recall@20: 0.2855
- test NDCG@20: 0.1348

So mapped-item GRPO with `old_policy_scope=epoch` is not counted as effective.
It optimizes the internal reward but does not improve full-sort ranking, and it
removes the small M2z flow-only gain.

## M2ab Result

M2ab tested a harder but still no-filter mapping:

- `mapping_strategy=hard_topk`
- `mapping_topk=50`
- no GRPO
- same clean M0 and same flow cache

Result:

- best valid: 0.0928
- test Recall@20: 0.2867
- test NDCG@20: 0.1360
- diagnostics: `fn_rate_all_known=0.014`, `W_cont=0.7127`,
  `W_mapped=0.6574`, `mapped_unique_ratio=0.2190`

The mapped negatives were much harder (`W_mapped` rose from M2z's 0.3948 to
0.6574), but NDCG@20 fell from M2z's 0.1365 to 0.1360. This indicates that
stronger boundary negatives alone are not enough; they can hurt calibration and
do not produce the required top-k gain.

`M2ac_hardtopk_cached_no_filter_grpo` is not run yet because its no-GRPO
control is worse than M2z and the latest GRPO configuration is already shown to
be unstable.

## Training-Time Exposure Branch

The audit filter showed that exposed-but-unclicked items carry a strong signal,
but applying that signal at evaluation time is not acceptable. The next branch
moves the signal into training:

- `M2ad_exposed_only_no_filter_no_grpo`: train with user-specific exposed
  negatives only (`joint_exposed_neg_ratio=1.0`, four negatives per positive),
  without flow and without any evaluation-time score edit. This is a control to
  measure the value of exposure supervision itself.
- `M2ae_exposed_flow_mix_no_filter_no_grpo`: use the same exposed-negative
  training pressure but reserve 25% of negatives for cached flow samples. This
  tests whether flow adds over direct exposure supervision.
- `M2af_exposed_flow_mix_conservative_grpo`: M2ae plus a conservative GRPO
  update (`grpo_lr=1e-5`, `grpo_beta=1.0`, `old_policy_scope=batch`) to avoid
  the ratio/KL explosion observed in M2aa.

Implementation change: Phase 4 now detects whether joint training actually
needs flow-generated negatives. If `joint_exposed_neg_ratio=1.0` or
`flow_neg_ratio=0.0`, it can run exposed/random negative training without
loading or sampling the flow model. This keeps M2ad a clean non-flow exposure
control.

M2ad result:

- best valid: 0.0928
- test Recall@20: 0.2854
- test NDCG@20: 0.1346

Pure training-time exposure supervision is therefore not enough. It does not
replicate the audit filter gain and is slightly below M0 on NDCG@20.
