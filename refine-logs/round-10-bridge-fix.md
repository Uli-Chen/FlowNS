# Round 10 连续→离散桥修复（代码改进，未跑实验）

日期：2026-06-09

本轮按 round-8 §5 / round-9 §4 的反思动手改代码，**不跑实验**。核心是给「连续→离散映射」这座
被反复确认的瓶颈一个有原则的修法，并顺手把 round-8 诊断出的 GRPO 数值爆炸点修掉（为下一轮 GRPO 铺路）。
所有改动**向后兼容、默认关闭/默认不变**，只新增一个实验 `M2vae_boundary` 用来检验修复。

## 1. 触发本轮的新证据：flow 负样本**比随机还差**

修复泄漏后补齐的 MIND / MultiVAE 配对结果：

| Run | 负样本来源 / 映射 | NDCG@20 | vs baseline |
| --- | --- | ---: | ---: |
| `M0_multivae_baseline` | 无额外负样本 | 0.1404 | — |
| `M2_multivae_flow_no_grpo` | exposed-flow, `nearest` | 0.1404 | **+0%** |
| `M2_multivae_random_neg_control` | random | **0.1435** | **+2.2%** |

**flow 负样本（+0%）严格劣于随机负样本（+2.2%）。** 这把 round-8/9 的判断坐实了：
`nearest` 映射把 flow 的连续 hardness 抹平（`W_mapped≈0.20`），映射后的负样本太「易」，
正样本打分本就远高于它 → BPR 梯度 ≈ 0 → flow 提供的信号还不如随机负样本的多样性。

## 2. 修复一：`boundary_topk` —— 保留 hardness 且规避假负样本的映射

**问题回顾**（round-8 §3 B 类）：映射在两个极端都失败 ——
`nearest` 太易（无梯度，劣于随机）；`hard_topk` / `score_topk` 太难（贴着排序边界，
本质是拿假负样本训练，伤排序）。需要的是「**最难、但仍然安全（非假负样本）的真实负样本**」。

**实现**（`src/neg_sampling.py`，新增 strategy `boundary_topk`）：
在 flow 向量的 top-k 最近候选里，对每个候选算其相对用户正样本的 win-rate
`W = mean_k σ(s(u,cand) − s(u,pos_k))`，然后：

1. **安全过滤**：丢掉 `W > boundary_safe_w` 的候选（赢正样本太多 → 很可能是假负样本）。
2. **取最难的安全候选**：在剩下的候选里按 boundary-aware reward `R = W^a(1−W)^γ` 取 argmax。

关键设计：把 `reward_a > reward_gamma`（峰值 `W* = a/(a+γ) > safe_w`），于是在安全区
`[0, safe_w]` 上 R 单调递增 → argmax 落在 `W → safe_w`，即**最难的安全负样本**。
`M2vae_boundary` 用 `a=2, γ=1`（W\*=0.667）、`safe_w=0.5`。

与既有策略的区别：
- `hard_topk`：只取打分最高 → 会选中假负样本。
- `reward_topk`：取 R 峰值（W\*），但 W\* 本身可能在安全线以上 → 仍可能选假负样本。
- `boundary_topk`：**先卡安全线，再在安全区里取最难** —— 这正是 realness+hardness 想要的。

单元测试（`item1` 是高分假负样本 W=.668，`item2` 是较难的安全负样本 W=.45）：
`hard_topk → item1`，`reward_topk → item1`，**`boundary_topk → item2`**，并验证了
全部不安全时回退到最近邻、`(B,G,d)` 形状保持。

**MultiVAE 路径打通**：之前 autoencoder 的 joint loss 调用 `map_to_items` 时**根本没传**
`pos_item_embs`/`reward_fn`，所以 reward/boundary 策略在 MultiVAE 上等于退化成 `hard_topk`。
本轮补传了正样本与 reward（对 MultiVAE，item embedding=decoder 权重、user=decoder 前隐层，
点积即 decoder logit，与打分函数一致），使 `boundary_topk` 能真正在 MultiVAE 上生效。

## 3. 修复二：SDE 数值稳定（为下一轮 GRPO 铺路）

**问题回顾**（round-8 §3 C 类）：闭式 log-ratio 与 KL 都除以 `2·σ_t²·Δt`，而
`σ_t = η√(1−t)/(√t+δ) → 0`（t→1）；Tweedie score `−(x−t·v)/(1−t)²` 又在末段放大策略差异。
两者叠加 epoch-scope 漂移 → ratio≈1e8、kl≈10³，GRPO 退回 M0。

**实现**（`src/sde_sampler.py`）：
- 把分散在 6 处的 score 计算收敛成 `_tweedie_score()`，新增可选幅值 `score_clamp`。
- `noise_schedule` 增加 `sigma_min` 下限，给 `2σ²Δt` 分母兜底。
- 两个旋钮默认 `sigma_min=0.0 / score_clamp=None`（**完全保持原行为**），通过 config 开启。

单元测试**复现了爆炸并验证了修复**：同一条漂移轨迹下，
`|log_ratio|` 峰值 `3.1e7 → 6.5e6`、`|kl|` 峰值 `3.1e7 → 6.5e6`（开启 `sigma_min=0.05, clamp=10`），
且全程有限。注意 raw 的 3.1e7 与 M2aa 实测 ratio≈1e8 同量级，确认诊断正确。

## 4. 新增实验与改动文件

新增实验（Round 10 实验臂，**不在 `all` 里**）：

- `M2vae_boundary`（stage `M2vaeB`）：MultiVAE + exposed-flow + `boundary_topk`，无 GRPO，
  复用 `M2_multivae_flow_no_grpo` 的 flow cache（**只改映射**），所以是对
  「映射方式」的干净消融。

运行（下一步由你决定是否跑）：

```bash
bash scripts/run_pilot.sh --dataset mind --stage M2vaeB
```

判定标准（标准 full-sort，无任何评测 trick）：
- **桥修复有效** ⟺ `M2vae_boundary` NDCG@20 同时 > `M2vae`(0.1404) 且 > `M2vaeR`(0.1435)。
- 若仍 ≤ random（0.1435）：说明仅靠「更聪明的离散选择」不够，需转向可微离散化
  （Gumbel-Softmax / straight-through，round-8 §5.A.1）。

改动文件：
- `src/neg_sampling.py`：新增 `boundary_topk` 策略 + `boundary_safe_w` 参数。
- `src/sde_sampler.py`：`_tweedie_score()` 收敛 + `sigma_min` / `score_clamp`（默认不变）。
- `src/flowns_trainer.py`：新 config 默认值；SDE 旋钮接线；LightGCN 与 **MultiVAE** 两条
  custom-loss 路径都为 reward/boundary 传入正样本、reward、`boundary_safe_w`。
- `src/grpo.py`：`GRPOTrainer` 透传 `boundary_safe_w`，使 GRPO 内部 reward 映射与 joint 一致。
- `configs/experiments.yaml`：新增 `boundary_safe_w` / `sde_sigma_min` / `sde_score_clamp` 默认；
  新增 `M2vae_boundary`。
- `scripts/run_pilot.sh`：新增 `M2vaeB` stage（标注为 Round 10 实验臂）。

## 5. 下一步

1. 跑 `M2vaeB`，对照 `M2vae`(+0%) / `M2vaeR`(+2.2%) 判桥修复是否成立。
2. 若成立：把 `boundary_topk` 推到 LightGCN 线（`--set mapping_strategy=boundary_topk`），
   并补跑 post-fix 的对照，凑齐跨 backbone 证据。
3. 若不成立：转可微离散化。
4. 桥确认能产出有效负样本后，再开 Round 11：用本轮已落地的 `sigma_min`/`score_clamp`
   + batch-scope + 排序对齐 reward 重启 GRPO，硬门槛仍是「必须打赢 no-GRPO 对照」。
