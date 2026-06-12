# Round 9 后泄漏-修复结果记录 + 工作区刷新

日期：2026-06-09

本轮：(1) 把 Round 7 泄漏修复**之后**重跑的 MIND MultiVAE 结果落盘记录（此前只存在于
`results/pilot/*.json`，没进任何 refine-log）；(2) 刷新工作区，清掉过期 checkpoint / 日志 /
缓存；(3) 据此重新校准下一轮方向。不做新实验。

## 1. 关键结果：修复泄漏后，MultiVAE exposed-flow 增益归零

这些是 2026-06-09 在修复 `setup_recbole()` 泄漏 bug（见 round-7）之后跑出的、量纲正常的结果：

| Run | dataset | seed | best_valid (NDCG@20) | Recall@20 | NDCG@20 | 说明 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `M0_multivae_baseline` | mind | 2020 | 0.1196 | 0.2955 | 0.1404 | 干净 MultiVAE 基线 |
| `M2_multivae_flow_no_grpo` | mind | 2020 | 0.1196 | 0.2955 | 0.1404 | **与基线逐字节相同** |

`M2vae` 的 joint 训练（40 epoch、exposed-flow 负样本）从未超过 paired-M0 的 valid 分（0.1196），
早停后保留了 M0 checkpoint，所以测试结果和基线完全一致 → **exposed-flow 在 MultiVAE 上零增益**。

M2vae 生成诊断（修复后）：

| 指标 | 值 |
| --- | ---: |
| `fn_rate_all_known` | 0.001 |
| `W_continuous` mean | 0.4731 |
| `W_mapped` mean | 0.2039 |

连续难度尚可（0.47），映射后塌到 0.20 —— **连续→离散瓶颈在 MultiVAE 的 decoder-logit 空间里同样存在**，
并没有像 Round 6（泄漏版）误以为的那样被缓解。

ml-100k sanity（round-7 保留，作为非泄漏量纲对照）：

| Run | Recall@20 | NDCG@20 |
| --- | ---: | ---: |
| `M0_multivae_baseline_ml_100k` | 0.3129 | 0.2555 |
| `M0_baseline_ml_100k` (LightGCN) | 0.2382 | 0.1997 |

## 2. 结论更新（对 round-8 的修正）

Round 8 把「干净重跑 MultiVAE 主干、验证 exposed-flow > random」列为下一步希望所在。**该重跑已部分完成，
答案是否定的**：

- LightGCN 标准 full-sort：exposed/flow 仅 +1.3% NDCG（round-5）。
- MultiVAE 标准 full-sort：exposed-flow **+0%**（本轮）。
- 两个 backbone 一致指向同一根因：**realness 的负样本经离散映射后不够 hard，无法转化为排序增益**。

注意残缺项：post-fix 的 `M2_multivae_random_neg_control`（M2vaeR）**尚未跑**，所以严格的
「flow vs random」配对对照还缺一半。但即便不跑，M2vae=baseline 已说明 flow 这条 joint 路径目前不产生增益。

## 3. 工作区刷新（本轮清理）

清理原则：只删**可重建的过期产物**，保留源码 / 配置 / refine-logs / data / 当前有效结果记录与
被指针引用的 checkpoint。

**删除（过期 / 可重建）：**
- `log/`、`log_tensorboard/`：旧训练日志与 tensorboard 事件。
- `src/__pycache__/`、`scripts/__pycache__/`：字节码缓存。
- `results/pilot/logs/`：旧 per-stage tee 日志。
- 孤立的旧 checkpoint（不被任何 M0 指针引用，且多为 Jun-08 泄漏期产物）：
  `saved/MultiVAE-Jun-08-2026_23-00-28.pth`、`saved/MultiVAE-Jun-08-2026_23-07-04.pth`、
  `saved/MultiVAE-Jun-08-2026_23-11-25.pth`、`saved/MultiVAE-Jun-09-2026_09-07-12.pth`（M2vae joint
  中间产物，未被引用）。

**保留（当前有效状态 + 证据）：**
- 全部 `results/pilot/*.json`（含上面的 post-fix 结果，体积小、是唯一记录）与 `M0_model_path.*.txt`。
- `results/pilot/M2_multivae_flow_no_grpo_flow.pt` / `_flow_ref.pt`（M2vae 配置引用的 flow cache）。
- `saved/MultiVAE-Jun-09-2026_08-44-25.pth`（当前 MIND M0vae，被 mind 指针引用）。
- `saved/LightGCN-Jun-08-2026_23-00-41.pth`（当前 ml-100k M0，被 ml_100k 指针引用）。

> 备注：`M0_model_path.mind.txt` 目前指向 MultiVAE checkpoint（最后在 MIND 上跑的是 M0vae，
> 覆盖了指针）。若下一轮要跑 LightGCN MIND 主干（M2/M2b），需先重跑 M0 重建指针 —— 这是
> round-8 记录过的「指针按 dataset 而非 model 命名」脚枪。

## 4. 下一轮方向（据本轮结果收敛）

两个 backbone 都证明「只靠 realness」拿不到排序增益，所以下一轮**不再做 realness 的重复验证**，
集中火力在两件事（沿用 round-8 第 5 节，按本轮证据重排优先级）：

1. **连续→离散桥（最高优先级）**：可微离散化（Gumbel-Softmax / straight-through）、
   把 reward 对齐到真实离散 item、或候选集策略，让 hardness 穿过映射而不被抹平。
   这是两个 backbone 共同的瓶颈，必须先解决。
2. **补齐并跑通稳定 GRPO**：先按 round-8 第 5.B 节修 SDE 数值（t→1 处 `2σ_t²Δt→0` 爆炸）、
   改 batch-scope、reward 对齐排序；硬门槛仍是「GRPO 必须在标准 full-sort 下打赢 no-GRPO 对照」。

补充：把 post-fix 的 `M2vaeR` 补跑掉，凑齐「flow vs random」配对对照，作为新方法的对照基线。
