# Round 8 主干清理与失败复盘

日期：2026-06-09

本轮不做任何实验。目标是：(1) 把 `run_pilot.sh` / `experiments.yaml` 里堆积的大量变体清理回
M0/M1/M2 主干 + 核心对照；(2) 系统复盘这些变体为什么失败；(3) 给出我认为合理、且与失败教训对应的
改进方向。

---

## 1. 项目现状（一句话版本）

FlowNS 想用「两段式」生成负样本：

1. **Realness（真实性）**：用 Conditional Flow Matching（CFM）从「曝光未点击」item 的 embedding
   学一个用户条件下的负偏好分布 `G(·|u)`，保证生成的负样本是真实负样本而非假负样本。
2. **Hardness（难度）**：把确定性 Flow-ODE 转成保边际的 SDE，用 GRPO（组相对策略优化）+
   boundary-aware reward `R = W^a (1-W)^γ` 把生成分布往「更难但仍合理」推。
3. 生成的连续向量 `x_1` 再通过与 item embedding 的相似度映射回离散 item id，喂给推荐模型做 BPR。

诚实的实验结论（去掉所有评测 trick 和泄漏 bug 之后）：

| 结论 | 证据 |
| --- | --- |
| **Realness 成立**：flow 确实学到了非随机、低假负率的负分布 | `fn_rate_all_known ≈ 0.002`，`W_cont ≈ 0.73` |
| **Hardness via GRPO 未被证明有效** | GRPO 要么数值爆炸，要么只提升内部 reward 不提升排序 |
| **连续→离散这座桥是真正的瓶颈** | `W_cont 0.73` 经 nearest 映射后塌到 `W_mapped 0.39` |
| **标准 full-sort 下最好成绩仅 +1.3% NDCG@20** | M2b/M2z：0.1348 → 0.1365，远低于 10% 目标 |

---

## 2. 清理记录

### 2.1 保留的主干（7 个，两条 backbone 线）

| Stage | experiment | 作用 |
| --- | --- | --- |
| M0 | `M0_baseline` | LightGCN 基线，M1/M2/M2b 的 paired-M0 来源 |
| M1 | `M1_flow_pretrain` | LightGCN + flow CFM 预训练 + 生成质量（realness）检查 |
| M2 | `M2_flowns_full` | 完整 FlowNS（flow + GRPO + joint） |
| M2b | `M2b_flow_no_grpo` | flow 负样本但**关掉 GRPO** —— 隔离 GRPO 贡献的关键对照 |
| M0vae | `M0_multivae_baseline` | MultiVAE 基线，M2vae/M2vaeR 的 paired-M0 来源 |
| M2vae | `M2_multivae_flow_no_grpo` | MultiVAE + **曝光负样本** flow，无 GRPO |
| M2vaeR | `M2_multivae_random_neg_control` | MultiVAE + 随机负样本对照（无 flow、无 GRPO） |

为什么 no-GRPO（M2b）和 random-neg（M2vaeR）也算「主干」：负采样方法的核心论断是
「flow 负样本 > 随机负样本」「GRPO 难度 > 仅 flow」。没有这两个配对对照，主结果根本无法归因。
它们不是消融，是主干本身的一部分。

### 2.2 删除的变体

- **M2c–M2af（约 30 个探索性变体）**：mapping 策略、rerank、曝光过滤、multi-negative、
  exposed/flow 混采的各种排列组合。结论已固化（见第 3 节），定义留着只是噪声。
- **M3 / M4 / M5（reward 消融 / KL-β 消融 / FN 理论保证）**：按你选择（「主干 + 核心对照」）
  一并移出 pilot 阶段。M5 的 FN-guarantee 理论是干净的贡献，建议作为理论结论保留（第 5 节 E），
  其 Python 实现 `src/pilot_runner.py::run_fn_guarantee` **未删除**（该文件未被 git 跟踪，删了不可恢复），
  只是 shell 不再暴露入口。

### 2.3 注意：paired-M0 指针按 dataset 而非 model 命名

`results/pilot/M0_model_path.<dataset>.txt` 不区分 backbone。所以 LightGCN 线和 MultiVAE 线
**共用**同一个指针文件。`all` 已排成依赖安全的顺序（M0→M1→M2→M2b→M0vae→M2vae→M2vaeR）：
LightGCN 主干在 M0vae 覆盖指针之前就已消费完毕。但**单独重跑某个 LightGCN 阶段**时，若指针
已被 MultiVAE 覆盖，需要先重跑 M0。这是既有设计的脚枪，本轮只做了文档标注，没有改代码。

---

## 3. 变体失败的分类与根因分析

我把所有失败的变体归成 6 类。前 5 类是方法/工程问题，第 6 类是评测纪律问题。

### A 类：评测作弊型（M2t / M2u / M2v / M2w）——「赢了但不算」

这些变体靠 `eval_exposed_neg_penalty`：在 full-sort 打分阶段，对该用户「曝光未点击」的 item
直接减去一个大常数。NDCG@20 从 0.1348 飙到 0.1897（+40.7%）。

**为什么不算**：M2u（只有过滤、没有 flow/GRPO）拿到了和 M2t 几乎一样的分数。增益来自
**评测时直接改了候选分**，不是学到的东西。这等于把测试期才知道的曝光反馈塞进打分函数，
是一种信息泄漏式的 inference filter。已在 Round 4 永久退回 audit-only。

> 教训：任何「主结果」必须在标准 full-sort 下成立，曝光过滤只能进 audit 章节，且必须配
> filter-only 对照（M2u）。

### B 类：连续→离散映射瓶颈（M2e/M2i/M2k/M2m/M2q/M2ab/M2x …）——**最核心的失败**

flow 在连续空间生成的样本是「难」的（`W_cont ≈ 0.73`），但推荐模型要的是离散 item：

| 映射策略 | W_mapped（映射后难度） | 对 NDCG 的效果 |
| --- | --- | --- |
| `nearest`（最近邻） | 0.39 | 难度塌掉，仅 +1.3% |
| `hard_topk` | 0.66 | 难度回来了，但 **NDCG 反而下降**（0.1365→0.1360） |
| `score_topk`（从推荐 top 候选里选） | 高 | **训练直接发散**，valid 跌破 M0 |

三种结局对应同一个根因：**连续空间的「难」和离散排序里的「有用」不是一回事。**

- `nearest`：离一个「难」的连续点最近的真实 item，往往恰好是个普通的简单负样本。难度在投影中被抹平。
- `hard_topk` / `score_topk`：强行恢复难度，就把负样本推到了「用户当前 top 排序里、且接近正样本」的
  区域。这本质上是在拿**接近真实正样本 / 假负样本**当负样本训练，破坏 calibration，伤排序。
  Round 4 的日志原话：`score_topk joint mapping makes the negative pressure too close to the current
  ranking boundary and hurts the recommender`。

这是整个方法的命门：**在连续空间生成、靠最近邻离散化，难度不可控地丢失或变成假负样本。**

### C 类：GRPO 数值爆炸 + reward 错配（M2aa / M2y / M2af / M2r / M2s）

**现象**（M2aa）：GRPO 内部 reward 0.0994→0.1019（在涨），但

```
epoch 1: kl=526.7,  ratio=1.0e8
epoch 2: kl=1356.6, ratio=1.0e8
epoch 3: kl=1818.8, ratio=1.0e8
```

best valid 从未超过 M0，测试结果精确退回 M0（NDCG 0.1348）。

**根因 1 —— SDE 闭式 log-ratio / KL 在 t→1 处被放大**（见 `src/sde_sampler.py`）：

- 每步 log-ratio 和 KL 都除以 `2·σ_t²·Δt`。噪声调度 `σ_t = η·√(1-t)/(√t+δ)`，当 t→1 时
  `σ_t → 0`，分母趋零 → 任意 `π_θ` 与 `π_old`/`π_ref` 的微小漂移都被放大。
- Tweedie score 项 `-(x - t·v)/(1-t)²` 进一步在末段放大 `v_θ - v_ref` 的差异，并同时进入
  ratio 和 KL。
- `ratios = exp(clamp(log_ratio, -20, 20))`，`exp(20) ≈ 4.85e8`，与观测到的 `ratio≈1.0e8`
  完全吻合 —— 说明 log-ratio 长期顶在 clamp 上界。

**根因 2 —— `old_policy_scope='epoch'`**：M2aa/M2y 用 epoch 级行为策略，一个 epoch 内策略持续漂移，
batch 越往后 `π_θ` 离 `π_old` 越远，ratio 必然爆。`'batch'` scope（每个 batch 重存 old policy）
能把 ratio 压在 1 附近，稳定的几次跑都用的是 batch scope。

**根因 3 —— reward 与排序指标错配**：`R = W^a(1-W)^γ` 在 `W* = a/(a+γ)` 取最大（a=2,γ=1 时
W*≈0.667）。即 GRPO 把负样本往「有 2/3 概率打分高过正样本」推 —— 这恰恰是**接近正样本 / 假负样本**
的区域。而且 reward 是在 mapped/连续 embedding 上算的，和「这个负样本能否提升 test NDCG」之间
没有建立因果联系。所以 M2s 的 win-rate 能冲到 0.886，NDCG 纹丝不动。

> 一句话：GRPO 在优化一个**数值不稳、且与最终指标不对齐**的内部目标。

### D 类：负样本压力过强（M2x num_neg=4 / M2ab hard_topk / 高 flow_neg_ratio）

即使负样本是真实的（低 FN），一旦又「真」又「难」，它就贴在决策边界上。对这种样本施加过大的
BPR 压力 ≈ 在跟「差一点就是正样本」的 item 对抗，推荐模型的排序校准被破坏。M2x 初版（4 负样本、
flow_ratio=1.0、topk=200）10 个 epoch 内 valid 持续低于 M0 best，只能砍回单负样本、50% 混采。

### E 类：直接曝光监督不足（M2ad / M2k）——「signal 在，但搬不进训练」

A 类证明「曝光未点击」是强信号。于是尝试把它当成额外 BPR 负样本直接训练（M2ad: 纯曝光负样本；
M2k: 曝光+flow 混采）。结果 M2ad 的 NDCG 0.1346，**还略低于 M0**。

根因：评测期过滤之所以有效，是因为它用到了**针对该用户、在测试候选上**的曝光信息；而训练期把
曝光 item 当负样本，模型本来大多就已经把它们排在正样本之后了，再加压边际收益极小。**训练期监督
无法复刻评测期过滤的增益**，这本身也说明 A 类增益是评测信息泄漏而非可学习的能力。

### F 类：backbone 泄漏 bug（M2vae，Round 6 → Round 7）

Round 6 换 MultiVAE backbone，NDCG 从 0.83 飙到 0.97，看似第一个强正信号。Round 7 审计发现是
`setup_recbole()` 用**未切分的全量 dataset** 构造模型，把 valid/test 交互灌进了
`history_item_matrix`（MultiVAE）/ 图（LightGCN）——硬泄漏。修复后异常分数消失，Round 6 作废。

> 这不是方法失败，是工程 bug。但它和 A 类一起构成本项目最大的教训：**好得离谱的结果，先怀疑评测/数据，
> 而不是先庆祝。**

---

## 4. 我的反思

**4.1 方法的两个支柱，强度严重不对称。**
Realness（CFM）是扎实的、可验证的（FN≈0.002），这是论文真正立得住的贡献。Hardness（GRPO）目前是
一个**未兑现的承诺**：它在数值上不稳、在目标上不对齐、在离散化后被销毁。论文叙事把两者并列，但实验
只支撑前者。

**4.2 真正的科学问题被埋在工程问题下面。**
团队大量精力花在 mapping 策略、rerank、混采比例的排列组合上（M2c–M2af），但所有这些都绕不开同一个
根因 —— **连续生成 + 最近邻离散化这条链路会丢失难度**。这才是该集中火力的地方，而不是再加第 31 个变体。

**4.3 名义上的「完整 FlowNS」其实没在测真正的方法。**
`M2_flowns_full`（LightGCN）用的是默认 `negative_source=random`，所以它的 flow CFM 目标是
**随机非正样本**（`src/flow_model.py:161`），等于用 flow 复刻了均匀负采样 —— 一个退化版本。它跟
M0 几乎打平毫不意外。论文真正主张的「曝光负样本 realness」只活在 `M2vae`（exposed 源）那条线上。
**所以 LightGCN 主干其实是对方法的弱检验，MultiVAE 的 exposed 线才是。** 这也是我保留 M2vae 的原因。

**4.4 让这个项目没翻车的，是配对对照的纪律。**
M2u（filter-only）、M2vaeR（random-neg）、Round 7 的泄漏审计 —— 每一次「大增益」最后都被一个对照
打回原形。这套纪律比任何单个变体都值钱，必须延续。

---

## 5. 我认为合理的改进方向（不含实验，按优先级）

### A. 优先级最高：解决连续→离散这座桥（B 类根因）

当前是「自由空间生成 → 最近邻投影」，难度在投影处丢失。可选路径：

1. **可微离散化**：用 Gumbel-Softmax / straight-through 在 item logits 上采样，把「snap 到 item」
   这一步也纳入可学习、可反传的范围，让难度信号穿过离散化而不是被它抹平。
2. **奖励对齐到真实离散负样本**：reward 不在连续 `x_1` 或最近邻 embedding 上算，而在**实际喂给推荐器的
   那个离散 item** 上算，关掉 proxy gap（C 类根因 3 的一半）。
3. **候选集策略而非自由生成**：让 flow/策略从一个**受控候选池**里选 item（例如「难但已知是真实负样本」
   的池子），而不是在全空间生成再投影。这直接回避「最近邻把难度抹平」。
4. **换到投影更友好的空间**：Round 6/7 暗示 MultiVAE 的 decoder-logit 空间里最近邻没那么伤难度
   （`W_mapped` 相对没塌那么狠）。先把 M2vae 在**修复泄漏后**干净重跑，确认这条更值得投入。

### B. 让 GRPO 既稳定又对齐（C 类根因）

1. **稳住 SDE 数值**：要么在 t→1 前提前停止轨迹 / 给 `σ_t` 设下界，要么 clip/重参 score 项，
   避免 `2σ_t²Δt → 0` 把 log-ratio、KL 放大到爆。
2. **行为策略用 `batch` scope** 或 PPO 式「每个 mini-batch 重算 ratio」，配合更小的 `grpo_lr`、
   自适应 β（KL-target）。
3. **reward 直接对齐排序**：用推荐器自身的 score margin、rank-based 信号，甚至「移除该负样本后某个
   留出正样本的 rank 变化」这类反事实信号，让「最大化 reward」真的等价于「提升 NDCG」。
4. **设硬性门槛**：GRPO 必须在**完全相同的标准 full-sort 评测**下打赢 no-GRPO 对照（M2b/M2vae），
   否则不计入有效。这把「内部 reward 涨了」和「指标涨了」彻底分开。

### C. 让 LightGCN 主干真正测方法

把 `M2`/`M2b` 也改成 `negative_source=exposed`（让 LightGCN 线真的用曝光负样本），或者干脆
**承认 MultiVAE 是主 backbone**，LightGCN 仅作 backbone-robustness 的次要验证。现状下 LightGCN 主干
测的是退化版本，结论参考价值有限。

### D. 固化评测纪律

- 主结果只认标准 full-sort；曝光过滤永远进 audit 且必带 filter-only 对照。
- 所有配对实验从**同一个 M0** 出发；多 seed（2020/2021/2022）。
- 每次换 backbone / 数据管线，先跑 Round 7 那个 split-leakage 检查再信任数字。

### E. 把理论保证（原 M5）当成独立贡献保留

FN-guarantee（KL 约束下生成分布偏离参考策略的闭式上界）是干净的理论结果，即使从 pilot 阶段移除，
也应作为论文的理论章节保留 —— 它正好支撑「realness 在加 hardness 后仍可控」这个叙事。

---

## 6. 建议的下一步（最小、干净）

1. **修复泄漏后，多 seed 干净重跑 MultiVAE 主干**（M0vae / M2vae / M2vaeR），判定
   「exposed-flow 负样本是否真的 > 随机负样本」。这是当前唯一有希望的正信号，必须先证实或证伪。
2. 若 `M2vae > M2vaeR` 成立 → realness 论断站稳；**再**在其上小心地加一个（按第 5.B 节修过的）稳定
   GRPO，并要求它打赢 M2vae。任何一步打不赢对照，就停下来分析而不是再加变体。
3. 在此之前，**不再新增 M2x 式的探索变体**。要动就动第 5.A 节那座桥。

---

## 附：本轮改动文件

- `configs/experiments.yaml`：从 30+ 个实验裁到 7 个主干 + 对照。
- `scripts/run_pilot.sh`：stage 列表、`all` 展开、`run_stage` 分支裁到 7 个；删除 `run_fn_guarantee`
  bash 函数与 M5 相关的结果打印；补充 paired-M0 指针共享的依赖顺序说明。
- `src/pilot_runner.py`：**未改**（fn-guarantee 能力保留，因该文件未被 git 跟踪）。
