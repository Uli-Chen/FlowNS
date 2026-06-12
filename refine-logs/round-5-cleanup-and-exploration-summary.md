# Round 5 Cleanup And Exploration Summary

Date: 2026-06-08

## Scope

This note records the current FlowNS exploration before deleting old runtime
artifacts from `saved/`, `log/`, `log_tensorboard/`, and `results/`.

The cleanup target is generated experiment data only. Source code, configs,
dataset files, and all `refine-logs/` notes are kept.

## Cleanup Snapshot

Before cleanup:

| Directory | File count | Size |
| --- | ---: | ---: |
| `saved/` | 41 | 4.1G |
| `log/` | 42 | 8.0K |
| `log_tensorboard/` | 42 | 340K |
| `results/` | 45 | 2.8M |

Important reproducibility identifiers recorded before cleanup:

- M0 checkpoint pointer: `saved/LightGCN-Jun-05-2026_21-11-07.pth`
- M0 checkpoint sha256: `1b37ff465f88d7dd6e21178cf30c21a197a3fedfcb7500a4e53baa59905dd66f`
- Active flow cache: `results/pilot/M2flow_exposed_s20_cache_flow.pt`
- Active flow cache sha256: `eacb52c6cd0e7335d5bbf193334190093efae7999bb797bef50d7b8fbe8117ef`
- Active flow reference sha256: `d3bb49aef94819e5185c4f87ed3c1db3a4545237e9891453d46aa023e597f500`

These files are intentionally deleted by this cleanup. The next run should
rebuild M0 and flow cache from the current configs.

After cleanup:

| Directory | Remaining file count | Size |
| --- | ---: | ---: |
| `saved/` | 0 | 4.0K |
| `log/` | 0 | 4.0K |
| `log_tensorboard/` | 0 | 4.0K |
| `results/` | 0 | 4.0K |

## Baseline

Dataset: `mind`

| Run | Recall@20 | NDCG@20 | Best valid |
| --- | ---: | ---: | ---: |
| M0 LightGCN baseline | 0.2855 | 0.1348 | 0.0926 |

The original 10% target was:

- Recall@20 >= 0.3141
- NDCG@20 >= 0.1483

## No-Filter Main Results

The current valid reporting protocol is standard RecBole full-sort evaluation:

- `eval_exposed_neg_penalty = 0.0`
- no evaluation-time exposure masking
- no exposure-filtered result is used as the main FlowNS improvement claim

Completed no-filter runs:

| Run | Main change | Recall@20 | NDCG@20 | Interpretation |
| --- | --- | ---: | ---: | --- |
| M2_flowns_full | original full pipeline | 0.2857 | 0.1347 | no gain |
| M2b | flow negatives, no GRPO | 0.2873 | 0.1363 | small flow-only gain |
| M2e | exposed-flow nearest, no GRPO | 0.2865 | 0.1366 | best small NDCG gain |
| M2i | exposed-flow nearest, 2 negatives | 0.2864 | 0.1365 | multi-negative did not help |
| M2k | direct exposed + flow training | 0.2875 | 0.1349 | recall up, NDCG flat |
| M2m | exposed-flow + rerank, no GRPO | 0.2865 | 0.1366 | rerank did not add over M2e |
| M2q | score-top-k rerank, no GRPO | 0.2856 | 0.1348 | no useful gain |
| M2r | score-top-k rerank + mapped GRPO | 0.2855 | 0.1348 | GRPO did not help |
| M2s | score-top-k reward + rerank | 0.2855 | 0.1348 | reward alignment did not transfer |
| M2z | cached nearest no-filter, no GRPO | 0.2864 | 0.1365 | reproducible small flow-only gain |
| M2aa | M2z + mapped-item GRPO | 0.2855 | 0.1348 | GRPO removed the small gain |
| M2ab | cached hard-top-k no GRPO | 0.2867 | 0.1360 | harder negatives did not improve NDCG |
| M2ad | training-time exposed negatives only | 0.2854 | 0.1346 | pure exposure supervision did not help |

Status of unfinished no-filter branch:

- `M2ae_exposed_flow_mix_no_filter_no_grpo` produced a log but no JSON result.
- The run stopped around joint epoch 11/500 and no `pilot_runner` process was
  active during cleanup.
- It is not counted in conclusions.

Conclusion: under standard full-sort evaluation, the best observed NDCG@20 is
0.1366, about +1.3% over M0. This is far below the 10% target.

## Exposure-Filter Audit Results

The large gains came from evaluation-time exposure filtering:

| Run | Setting | Recall@20 | NDCG@20 | Interpretation |
| --- | --- | ---: | ---: | --- |
| M2t | score-top-k GRPO + exposure filter | 0.3388 | 0.1897 | large gain, but filter-dominated |
| M2u | exposure filter only | 0.3390 | 0.1897 | matches M2t without FlowNS/GRPO |
| M2v | exposure filter + flow joint | 0.3412 | 0.1905 | small increment over filter-only |
| M2w | exposure filter + flow joint + GRPO | 0.3413 | 0.1905 | no material GRPO increment |

These results show that exposed-but-unclicked items are a strong signal, but
using them as an evaluation-time score penalty is not acceptable as the main
FlowNS claim. They are retained only as audit evidence.

## What We Learned

1. Flow pretraining can learn a non-random exposed-negative distribution.
   Typical diagnostics showed low all-known false negative rate and high
   continuous hardness, for example M2z had `fn_rate_all_known=0.002`,
   `W_cont=0.7260`, and `W_mapped=0.3948`.

2. The continuous-to-discrete bridge is the main bottleneck. Continuous flow
   samples look hard, but nearest item mapping substantially reduces hardness.
   `hard_topk` raised mapped hardness but did not improve NDCG.

3. Current GRPO variants optimize internal reward but do not improve standard
   full-sort ranking. M2aa returned exactly to the M0 test result, and earlier
   score-top-k reward variants had no ranking gain.

4. Direct training-time exposure supervision is not enough. M2ad slightly
   underperformed M0 on NDCG@20, so the exposure-filter audit gain cannot be
   recovered simply by treating exposed negatives as extra BPR negatives.

5. The original 10% improvement claim must not be attributed to FlowNS/GRPO.
   The only defensible no-filter claim so far is a small flow-only improvement
   around +1.3% NDCG@20.

## Next Direction

The next cycle should rebuild a clean M0 and flow cache, then focus on the
mapping/candidate policy instead of adding more GRPO variants:

- learn a user-conditioned discrete item projection instead of nearest mapping;
- train a candidate-set policy that selects useful boundary negatives without
  directly starting from the recommender's top-ranked items;
- keep M0, flow-only, GRPO, and exposure-only controls paired from the same
  checkpoint;
- report only standard full-sort results as main results, with exposure-filter
  numbers isolated in an audit section.
