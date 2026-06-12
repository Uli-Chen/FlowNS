# Round 11 跨 backbone 超参网格搜索（基础设施，已 smoke 通过）

日期：2026-06-09

本轮搭建「flow, no GRPO」设定下、MIND 数据集上跨 backbone 的超参网格搜索工具，并为此把负采样
管线推广到 MF（BPR）。**只跑了 1-epoch 的 smoke test 验证各 backbone 跑通**，正式搜索由用户进行。

## 1. 新增网格搜索工具（`gridsearch/`）

- `gridsearch/grid_search.py`：runner。每个网格点 = 一条自包含 FlowNS 流水线
  （`rec_pretrain → flow_pretrain → joint`，GRPO 关闭，`use_paired_m0: false` 从零训练），
  复用 `src.pilot_runner._run_flowns_config`（与 pilot 命名实验同一条代码路径）。
- `gridsearch/search_space.yaml`：搜索空间。`base`（共享 flow-no-grpo 设置）+
  每个 backbone 的 `enabled / model / fixed / grid`。默认搜索 **lr**（`learning_rate`,
  `joint_lr`）、**flow/负采样参数**（`flow_neg_ratio` 或 VAE 的 `flow_neg_loss_weight`）、
  **负采样离散化**（`mapping_strategy ∈ {nearest, boundary_topk}`）。其余参考 `experiments.yaml`。
- 输出：`gridsearch/reports/{grid,smoke}_<dataset>_<timestamp>.json`，**增量写盘**（崩溃/Ctrl-C
  可保留已完成结果），含每个 run 的 params/status/best_valid/test_result/诊断/耗时，以及
  `best_per_backbone` 汇总；结束打印汇总表。
- 鲁棒性：每个 run 包在 try/except 里，单点失败记录 error 并继续整轮。

用法（详见 `gridsearch/README.md`）：

```bash
.venv/bin/python -m gridsearch.grid_search --smoke      # 各 backbone 跑 1 epoch 验证
.venv/bin/python -m gridsearch.grid_search --list       # 只打印计划
.venv/bin/python -m gridsearch.grid_search              # 正式搜索（默认 48 runs）
.venv/bin/python -m gridsearch.grid_search --backbones BPR,MultiVAE --max-runs 8
```

## 2. 为支持 MF（BPR）做的管线推广

FlowNS 之前只支持 LightGCN（图 embedding 路径）和 MultiVAE 系（autoencoder 路径）。BPR 是纯 MF，
`forward()` 需要 `(user, item)` 参数、且无 `reg_loss/reg_weight`，会触发两处崩溃。修复：

- `src/recbole_utils.py`：`get_embeddings` 对 `forward()` 的 `TypeError` 回退到
  `user_embedding.weight / item_embedding.weight`；新增 `forward_all_embeddings()`（带梯度，
  供 joint loss 用），对无静态 user 表的序列模型（SASRec）显式抛错。
- `src/flowns_trainer.py`：joint 的 embedding-path loss 用 `forward_all_embeddings`，并对
  无 `reg_loss/reg_weight` 的 backbone（BPR 靠 optimizer weight decay）只返回 BPR 项。

## 3. pilot_runner 重构（DRY，零行为变化）

把 `run_experiment` 的核心抽成 `_run_flowns_config(merged_config, dataset, seed) -> (result, trainer)`
（不做任何文件 IO），CLI 与网格搜索共用。`run_experiment` 仍负责 checkpoint 指针/flow 保存与
写 JSON。已验证 `pilot_runner list` 与 8 个命名实验解析不变。

## 4. Backbone 支持现状

| 原型 | 模型 | 状态 |
| --- | --- | --- |
| MF | `BPR` | 支持（本轮新增） |
| GCN | `LightGCN` | 支持 |
| VAE | `MultiVAE` | 支持（`MultiDAE`/`RecVAE` 同 AE 路径，可同样加入） |
| Transformer（序列） | `SASRec` | **默认关闭**：FlowNS 对序列模型没有静态 per-user 表征，且当前 MIND 是非序列通用数据集；需要序列化数据管线 + 定义 user 表征（见 round-6）。开启后会在 report 中记录为 error。 |

## 5. Smoke 结果（1 epoch/phase，仅验证跑通，非有效指标）

`gridsearch/reports/smoke_mind_20260609_103935.json`：3/3 通过，各 ~35s。

| backbone | status | test NDCG@20 | W_cont → W_mapped |
| --- | --- | ---: | --- |
| BPR | ok | 0.0007 | 0.50 → 0.50 |
| LightGCN | ok | 0.1125 | 0.50 → 0.50 |
| MultiVAE | ok | 0.0993 | 0.51 → **0.36** |

BPR 1 epoch 自然很低（MF 收敛慢）；这些只验证三条 backbone 路径（含新增 BPR）端到端跑通、
曝光负样本加载、flow 预训练、诊断与增量 JSON 报告均正常。MultiVAE 的 `W_mapped<W_cont`
再次显现桥塌陷，与既往一致。

## 6. 下一步（由用户跑正式搜索）

1. 跑正式网格（可先 `--list` 确认规模，必要时按 README 把 `flow_lr / negative_source /
   boundary_safe_w / mapping_topk` 等加入 `grid`）。
2. 用 report 的 `best_per_backbone` 选各 backbone 最优超参，作为后续 round-10 桥修复
   （`boundary_topk`）与 round-11 稳定 GRPO 的统一基线。
