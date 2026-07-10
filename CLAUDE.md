# CLAUDE.md — FlowNS

> 给未来的 Claude 会话和协作者的项目说明。技术名词/命令/路径保留英文，叙述用中文。
> 最后更新：2026-06-16（移除 GRPO、回到地基验证的重做版本）。

## 这是什么

**FlowNS** 研究隐式反馈推荐里的**负采样**：用 Conditional Flow Matching (CFM) 学习"真实"
负样本分布（曝光未点击物品），生成连续 embedding，再通过"连续→离散桥"映射回物品 ID 作为
BPR 的负样本。研究提案见 `refine-logs/FINAL_PROPOSAL.md`。

**当前阶段（重要）**：原设计有第三阶段 GRPO（用 reward $R=W^a(1-W)^\gamma$ 给负样本加"硬度"）。
**本轮已把 GRPO 整体移除**，因为它的有效性以"flow 本身有效"为前提，而这个前提一直没被干净地
证实过。现在只验证两件地基：

- **F1 — backbone 指标是否正常**（ml-100k 快速检测）
- **F2 — flow sampler 本身是否有效**（MIND，曝光数据集）

GRPO 之后若 F1/F2 成立再考虑重新引入（git 历史里 `src/grpo.py` 可恢复）。

## 架构与流程

`FlowNSTrainer`（`src/flowns_trainer.py`）按 phase 组合 RecBole 训练 + flow：

| Phase | 方法 | 作用 |
|---|---|---|
| `rec_pretrain` (Phase 1) | `phase1` | 训练/加载推荐 backbone（LightGCN 等）；可加载配对 M0 checkpoint |
| `flow_pretrain` (Phase 2) | `phase2` | 在 backbone embedding 空间用 CFM loss 预训练 flow；缓存到 `results/pilot/` |
| `joint` (Phase 4) | `phase4` | 用 flow / exposed / random 负样本与推荐模型联合训练 |

> Phase 3（GRPO）已删除，方法名保留 `phase4`（历史编号），功能即"joint 联合训练"。

**关键文件**：
- `src/flowns_trainer.py` — 编排器（3 个 phase）、自定义 loss（discrete / continuous / dns 三种负样本模式）、eval 期 reranker（默认关闭）
- `src/flow_model.py` — `ConditionalFlowModel` + velocity net + CFM 预训练
- `src/sde_sampler.py` — ODE→SDE 转换、Euler-Maruyama 采样（**仅生成**；GRPO 的 log-ratio/KL 已删）
- `src/neg_sampling.py` — `EmbeddingToItemMapper`，连续→离散桥（`nearest`/`hard_topk`/`score_topk`/`reward_topk`/`boundary_topk`，`metric` ∈ {cosine, dot}）
- `src/reward.py` — `BoundaryAwareReward`（W 胜率 + R shaping）；现仅供桥的 FN-safe 策略和诊断使用
- `src/custom_metrics.py` — `compute_fn_rate`、`compute_w_statistics`
- `src/recbole_utils.py` — RecBole setup、embedding 提取（含 MultiVAE 等 AE backbone）、RecBole GPU-DNS 设备 bug 的 monkey-patch
- `src/pilot_runner.py` — CLI 入口（`run` / `list`），结果与诊断落盘到 `results/pilot/*.json`

**配置**：
- `configs/experiments.yaml` — 实验注册表（recbole base + flow defaults + S0/S1/S3 实验）
- `configs/lightgcn_<dataset>.yaml` — 各数据集的 RecBole 配置（ml100k / mind / kuairand / yelp / amazon / gowalla）

## 怎么跑

```bash
# 列出实验
.venv/bin/python -m src.pilot_runner list

# 单个实验（dataset ∈ mind | ml-100k | kuairand …）
.venv/bin/python -m src.pilot_runner run --experiment S0_m0_lgcn --dataset ml-100k
# 覆盖配置：--set KEY=VALUE（可重复）；--seed N；--tag T（避免 sweep 互相覆盖）

# 用 stage 预设批量跑（带日志 + 汇总表）
bash scripts/run_experiments.sh --dataset ml-100k backbone   # F1: S0_m0_lgcn S0_dns_lgcn
bash scripts/run_experiments.sh s1                           # F2 realness: S1_flow_lgcn (mind)
bash scripts/run_experiments.sh s3                           # F2 ranking: S3_{rand,exposed,flow,cont}_lgcn
bash scripts/run_experiments.sh smoke                        # 2-epoch 全链路冒烟（不污染指针/缓存）

# 汇总结果表
.venv/bin/python scripts/summarize_results.py --dataset mind --baseline S0_m0_lgcn --sort ndcg

# 单元测试（无需 pytest）
.venv/bin/python tests/test_p0_fixes.py
```

依赖关系：S1/S3 用 `use_paired_m0: true`，必须先为该 dataset 跑 `S0_m0_lgcn`（写配对指针
`results/pilot/M0_model_path.<dataset>.LightGCN.txt`）。预训练 flow 缓存按 dataset 自动加后缀
（`S1_flow_lgcn_flow_<dataset>.pt`），删掉即可强制重训。

环境：`.venv/`（python 3.12，torch 2.12 cu130，recbole 1.2.0，单卡 RTX 4060 8GB）。
RecBole 在 `/home/chen/workspace/RecBole`，ml-100k 数据也在那。

## 实验焦点与结果

### F1 — backbone 健康（ml-100k）✅
判据：LightGCN 指标落在正常区间，pipeline（数据/模型/eval）无异常。

**结果（2026-06-16）**：
- `S0_dns_lgcn`（DNS 硬负样本）：NDCG@20 **0.346** / Recall@20 0.396，valid 0.177→0.286 稳步爬升 106 epoch — **文献区间内，pipeline 健康**。
- `S0_m0_lgcn`（uniform 负样本）：NDCG@20 0.156 / Recall@20 0.194，valid 在 **epoch 0 见顶 0.139** 后只降不升（train loss 仍在降）。
- **解读**：uniform 负样本在 ml-100k 这种稠密数据上几乎不提供有效梯度（不是 bug，是数据特性）。ml-100k 负敏感性极强（DNS 比 uniform **+121%**），远超 MIND(+1.5%)/KuaiRand(+33%)，是验证负采样方法的好 testbed。

### F2 — flow sampler 有效性（MIND）
RL 有效的前提是 flow 本身有效。两步验证：realness 诊断（S1）+ ranking 收益（S3）。

**Step 1 — realness 诊断（`S1_flow_lgcn`，MIND/LightGCN，1000 users，2026-06-16）✅ 连续 flow 健康，但离散桥是瓶颈。**
flow CFM 收敛（loss 0.50→0.32）。同一缓存 flow 下三种桥（W_continuous 一致 ≈ 0.32）：

| 桥 (metric+strategy) | fn_all | W_mapped | unique_ratio | 解读 |
|---|---|---|---|---|
| cosine + nearest | 0.002 | 0.20 | 0.75 | FN-safe + 多样，但硬度坍缩（太 easy）|
| dot + nearest      | 0.038 | 0.80 | 0.026 | 硬，但模式坍缩 + FN 过冲（73.6% W>0.8）|
| dot + boundary_topk | 0.020 | 0.62 | 0.19 | 折中：较 FN-safe、中等硬度/多样 |

- **连续 flow 本身有效**：`W_cont≈0.32`（对 BPR 有信息量的硬度区间）、`fn_rate=0.005`（FN-safe）。
- **离散桥是瓶颈**：MIND 上没有任何桥同时满足 硬 + 多样 + FN-safe（cosine 太易、dot-nearest 坍缩+过冲、boundary 折中）。这正是 proposal 的 hardness-fidelity 权衡。
- **启示**：`continuous` 模式（S3_cont，直接 u·x_gen 不过桥）是用好这个 W_cont 0.32 分布、绕开桥病的最干净路径。

**Step 2 — ranking 收益（`S3_*`，待定）**：同一配对 M0、诚实 full-sort，`S3_flow`/`S3_cont` 对比 `S3_rand`/`S3_exposed`。**注意**：MIND ranking 已饱和（DNS 仅 +1.5% headroom），历史上 flow 在 MIND 输给 uniform/DNS，**ranking 大概率在 MIND 上偏平**——真正的 ranking 测试需要高 headroom 且有曝光日志的数据集（KuaiRand，已预处理就绪；ml-100k 无曝光日志无法承载 flow）。

## 血泪教训（不可妥协的实验纪律）

来自之前几十轮实验，违反任何一条都曾导致假结论：

1. **只信配对对照**：所有结论与同一 M0 出发的 paired 控制比（M-R/M-E/M-F），**绝不**跨 run 比数字。历史上两次 +40% "增益"分别死于 eval 期曝光过滤和 split 泄漏。
2. **诚实评估**：标准 full-sort，`rerank_*` 与 `eval_exposed_neg_penalty` 一律 0。每次改 pipeline 后跑 `scripts/check_recbole_split_leakage.py`。
3. **paired-M0 fine-tune 对硬负样本有敌意**：从已收敛 M0 继续用硬离散负样本训练会**崩溃**（valid 骤降），early-stopping 回退到 M0 → 看似"+0%"实为灾难性退化被回退。`continuous`（直接用 u·x_gen）更稳，能优雅退化而非崩溃。从头训练（warmup + 长 joint）则正常。
4. **MIND 太饱和**：DNS-vs-uniform 动态范围仅 +1.5%，难以体现负采样贡献。realness（fn≈0.002）成立但 ranking 收益在 MIND 上被证伪（flow 输给 uniform/DNS）。高 headroom testbed：ml-100k(+121%) > KuaiRand(+33%) ≫ MIND。
5. **度量一致性**：打分函数与 W 都是点积；桥若用 cosine 会丢掉范数信息、机械性坍缩硬度。LightGCN 上必须 `mapping_metric=dot`。
6. **FN-control 是前提不是可选项**：realness-only flow 无 FN 控制，在稠密数据上会把质量堆到高分区（硬负与 val/test 假负纠缠），nearest 映射必然命中假负 → 崩溃。`boundary_topk`（W>0.5 丢弃）或 reward shaping 是 load-bearing 的。
7. **3 个数学 bug 已修**（`tests/test_p0_fixes.py` 守护）：① Tweedie score 分母应为 (1-t) 不是 (1-t)²；② win-rate W 的 padding 必须 mask（否则稀疏用户的 W 被拉向 W*）；③ 噪声调度用步内平均 σ̄² 而非左端点（否则首步注入巨量方差）。

## 进展跟踪

- [x] 2026-06-16 移除 GRPO：删 `src/grpo.py`；`sde_sampler` 只留生成路径；`flowns_trainer` 删 phase3/grpo 配置/joint-grpo；`pilot_runner` 删 fn-guarantee；测试重写为 6 项（全绿）。
- [x] 2026-06-16 清理旧 artifacts（saved/results/log/log_tensorboard，释放 ~3.7G）。
- [x] 2026-06-16 重写 `experiments.yaml` v3 + `run_experiments.sh` + `summarize_results.py`，聚焦 LightGCN S0/S1/S3。
- [x] **F1 backbone（ml-100k）通过**：pipeline 健康，DNS 0.346（文献区间），uniform 0.156（数据特性）。
- [x] **F2 step 1 realness（MIND）完成**：连续 flow 健康（W_cont 0.32, fn 0.005）；离散桥是瓶颈（cosine 太易 / dot-nearest 坍缩+过冲 / boundary 折中）。continuous 路线最干净。
- [ ] **F2 step 2 ranking（待用户定夺数据集）**：S3 rand/exposed/flow/cont。MIND 已饱和大概率偏平；高 headroom ranking 测试用 KuaiRand（有曝光日志 + +33% headroom）。
- [ ] （视 F2 step 2 结果）若 flow ranking 成立再考虑重新引入 GRPO；否则可能 reframe 为 FN-guarantee/realness 贡献。

> 注：`refine-logs/EXPERIMENT_PLAN.md` 是上一版（含 GRPO 的 S0-S4 诊断链）的计划，已过时；
> 当前以本文件"实验焦点"为准。`FINAL_PROPOSAL.md` 仍是研究北极星，但其 GRPO 部分暂缓。
