# 实验计划

**问题**: 推荐系统负采样的硬度-保真度权衡
**方法论文**: FlowNS 通过条件 flow matching 学习真实负样本分布，配合 GRPO + 边界感知 shaped reward $R = W^a(1-W)^\gamma$ 控制硬度，双重机制防止假负样本
**日期**: 2026-05-20
**框架**: 基于 RecBole（`../RecBole`），不修改框架源码，通过 import 复用

---

## 声称映射 (Claim Map)

| 声称 | 为何重要 | 最低说服性证据 | 关联实验块 |
|------|---------|--------------|-----------|
| C1: FlowNS 生成的负样本比基线更硬且假负样本率更低 | 主导声称——证明方法解决了核心瓶颈 | 在 3 个数据集上 Recall@20/NDCG@20 显著优于最强基线，同时 FN 率更低 | B1, B5 |
| C2: Shaped reward $R=W^a(1-W)^\gamma$ 是必要的 | 证明核心机制是负样本质量提升的原因 | 消融实验：无 shaping ($R=W$) 导致 FN 率上升和推荐质量下降 | B2 |
| C3: KL 约束是承重的 | 证明双重机制的第二部分不可或缺 | $\beta=0$ 时模式坍缩，FN 率飙升 | B3 |
| C4: Flow matching 是合适的生成器 | 防御"为何不用更简单模型"的审稿质疑 | Flow vs GMM vs CVAE 对比，相同 reward | B4 |
| Anti-claim: 收益不仅来自更大参数量 | 排除"只是多了个网络"的解释 | 参数匹配消融：同参数量的 MLP 生成器不如 flow | B4 |

## 论文故事线 (Paper Storyline)

**主论文必须证明:**
- B1: 主结果表——FlowNS vs 7 个基线 × 3 数据集
- B2: Reward 消融——shaped vs unshaped vs no-RL
- B3: KL 消融——$\beta \in \{0, 0.01, 0.05, 0.1, 0.5\}$
- B4: 生成器消融——Flow vs GMM vs CVAE
- B5: 保证紧致性——实际 FN 率 vs 理论上界

**附录可支持:**
- B6: $\gamma$ 敏感性分析
- B7: 训练过程 W 分布可视化
- B8: 不同推荐骨干 (BPR-MF vs LightGCN)

**有意不做:**
- 大规模工业数据集（不在约束范围内）
- 序列推荐设定（非目标）
- 基于 LLM 的用户建模（与 problem anchor 无关）

---

## 实验块 (Experiment Blocks)

### Block 1: 主锚结果 (Main Anchor Result)

- **声称测试**: C1 — FlowNS 生成更硬且更真实的负样本
- **为何存在**: 直接证明方法解决了核心瓶颈
- **数据集/分割/任务**:
  - Yelp2018 (`yelp-2018`), Amazon-Books (`amazon-books`), Gowalla (`gowalla-merged`)
  - RecBole 内置随机分割: train/valid/test = 8:1:1
  - 任务: Top-K 隐式推荐
- **对比系统**:
  | 系统 | 类型 | RecBole 内置？ | 实现方式 |
  |------|------|------------|---------|
  | BPR-MF (Uniform) | 均匀采样基线 | ✓ `BPR` | `run_recbole(model='BPR', ...)` |
  | LightGCN (Uniform) | GCN 均匀基线 | ✓ `LightGCN` | `run_recbole(model='LightGCN', ...)` |
  | LightGCN (Popularity) | 流行度采样 | ✓ config | `train_neg_sample_args.distribution: popularity` |
  | LightGCN (DNS) | 动态硬负采样 | ✓ config | `train_neg_sample_args.dynamic: true, candidate_num: 50` |
  | MixGCF | GCN 混合负样本 | ✗ | 自行实现：在 LightGCN 嵌入空间混合正样本 |
  | AdvInfoNCE | 对抗信息NCE | ✗ | 自行实现：对比学习 + 对抗负样本 |
  | CVAE-NS | 条件 VAE 负采样 | ✗ | 自行实现：CVAE 生成 + shaped reward |
  | **FlowNS (ours)** | Flow + GRPO | ✗ | 自行实现 |
- **指标**:
  - 主指标: Recall@20, NDCG@20 (RecBole `metrics: ['Recall', 'NDCG'], topk: [20]`)
  - 辅助: Recall@10, NDCG@10, Hit@20
  - 诊断: 平均 W (胜率), FN 率, 生成负样本的推荐分数分布
- **设置细节**:
  - 骨干: LightGCN, embedding_size=64, n_layers=3
  - FlowNS: 速度网络 3-layer MLP (64→256→256→64), GRPO G=8, β=0.1, T=20, a=1, γ=1
  - 训练: Phase 1 (50 epochs), Phase 2 (10 GRPO epochs), Phase 3 (100 rec epochs, alternate 5/2)
  - Seeds: 3 (2020, 2021, 2022), 报告 mean ± std
  - Early stopping: patience=10, valid_metric=NDCG@20
- **成功标准**: FlowNS 在 ≥2/3 数据集上 Recall@20 和 NDCG@20 显著优于最强基线 (p < 0.05)
- **失败解读**: 若 FlowNS 不优于 DNS，需检查 (1) flow model 是否学到了有意义的分布 (2) GRPO 是否收敛
- **表/图目标**: **Table 1** (主结果表，论文核心)
- **优先级**: MUST-RUN

---

### Block 2: Reward 消融 (Novelty Isolation)

- **声称测试**: C2 — Shaped reward 是性能提升的核心原因
- **为何存在**: 隔离核心贡献——证明不是 flow model 本身的功劳
- **数据集**: Yelp2018 (最常用，代表性)
- **对比系统**:
  | 变体 | Reward 定义 | 含义 |
  |------|-----------|------|
  | FlowNS-Full | $R = W(1-W)^\gamma$ | 完整方法 |
  | FlowNS-Unshaped | $R = W$ | 无 shaping，朴素"越硬越好" |
  | FlowNS-NoRL | 无 GRPO | 仅预训练 flow model，不做 RL 微调 |
  | FlowNS-Symmetric | $R = W(1-W)$ | 对称 shaping，$\gamma=1$ |
  | FlowNS-RawScore | $R = s(u, \mathbf{x})$ | 用推荐原始分数做 reward |
- **指标**: Recall@20, NDCG@20, FN 率, W 分布 (直方图)
- **设置**: 与 B1 相同骨干和超参，仅改变 reward 定义
- **成功标准**: FlowNS-Unshaped 和 FlowNS-RawScore 的 FN 率显著高于 FlowNS-Full；FlowNS-NoRL 的 W 分布集中在低值区间
- **失败解读**: 若 FlowNS-Unshaped 与 FlowNS-Full 性能接近，说明 shaped reward 的贡献有限，需重新审视核心声称
- **表/图目标**: **Table 2** (消融表) + **Figure 2** (W 分布直方图)
- **优先级**: MUST-RUN

---

### Block 3: KL 约束消融 (Simplicity Check)

- **声称测试**: C3 — KL 约束是双重假负样本控制的关键部分
- **为何存在**: 验证理论保证的实践意义
- **数据集**: Yelp2018
- **对比系统**:
  | 变体 | $\beta$ 值 | 含义 |
  |------|-----------|------|
  | β=0 | 0 | 无 KL 约束 |
  | β=0.01 | 0.01 | 弱约束 |
  | β=0.05 | 0.05 | 中等 |
  | β=0.1 | 0.1 | 默认 |
  | β=0.5 | 0.5 | 强约束 |
  | β=1.0 | 1.0 | 非常强约束 |
- **指标**: Recall@20, NDCG@20, FN 率, 生成分布熵
- **成功标准**: β=0 时 FN 率飙升且推荐质量下降；存在最优 β 区间
- **失败解读**: 若 β=0 仍表现良好，说明 reward shaping 单独足够，KL 非必需（仍可重新定位贡献）
- **表/图目标**: **Figure 3** (β vs Recall/NDCG/FN 率的三联图)
- **优先级**: MUST-RUN

---

### Block 4: 生成器消融 (Frontier Necessity Check)

- **声称测试**: C4 — Flow matching 是合适的生成器选择
- **为何存在**: 回答审稿人"低维空间为何需要 flow matching"
- **数据集**: Yelp2018
- **对比系统**:
  | 生成器 | 描述 | RL 优化方式 |
  |--------|------|-----------|
  | Flow (ours) | 条件 Flow Matching + GRPO | GRPO on SDE 轨迹 |
  | Cond-GMM | 条件高斯混合模型，K=10 components | Reward-weighted resampling |
  | Cond-CVAE | 条件 VAE，latent dim=32 | REINFORCE on latent z |
  | MLP-Gen | 参数匹配的 MLP 直接生成 | REINFORCE |
  所有生成器使用**相同的 shaped reward** $R = W(1-W)^\gamma$
- **指标**: Recall@20, NDCG@20, W 分布质量 (KL divergence to target Beta), FN 率
- **成功标准**: Flow 产生更好校准的 W 分布。若差距小，论文重新定位为"reward 是贡献，生成器可互换"
- **失败解读**: 若 MLP-Gen 与 Flow 持平，则 flow matching 的价值需降级
- **表/图目标**: **Table 3** (生成器对比表) + **Figure 4** (各生成器 W 分布对比)
- **优先级**: MUST-RUN

---

### Block 5: 保证紧致性 (Theory-Practice Bridge)

- **声称测试**: C1 理论部分——假负样本率理论上界是否有实际意义
- **为何存在**: 回答审稿人"理论保证是否只是 nice math"
- **数据集**: Yelp2018, Amazon-Books
- **实验设计**:
  1. 测量 $\pi_{\text{ref}}$ 的 FN 率（预训练 flow model 直接生成，检查生成负样本是否出现在测试正样本中）
  2. 对每个 $\beta \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$:
     - 测量 $\pi_\theta$ 的实际 FN 率
     - 计算理论上界 $\exp(R_{\max}/\beta) \cdot \text{FN}(\pi_{\text{ref}})$
  3. 绘制实际 FN 率 vs 理论上界
- **指标**: 实际 FN 率, 理论 FN 上界, 比值 (tightness ratio)
- **成功标准**: 理论上界在同一数量级内 (tightness ratio < 10x)
- **失败解读**: 若上界极为松弛 (>100x)，理论贡献的实际价值需降调
- **表/图目标**: **Figure 5** (FN 率 vs 理论上界，两个数据集)
- **优先级**: MUST-RUN

---

### Block 6: γ 敏感性 (Appendix)

- **声称测试**: 超参 γ 对性能的影响
- **数据集**: Yelp2018, Amazon-Books, Gowalla
- **对比**: γ ∈ {0.5, 1, 2, 3, 5}, a=1 固定
- **指标**: Recall@20, NDCG@20, FN 率, W 分布峰值位置
- **成功标准**: 存在稳健的 γ 区间 (如 γ ∈ [1, 3])
- **表/图目标**: **Figure A1** (γ vs 指标，三数据集叠加) — 附录
- **优先级**: NICE-TO-HAVE

---

### Block 7: 训练过程可视化 (Appendix)

- **目的**: 展示 GRPO 训练过程的动态
- **数据集**: Yelp2018
- **可视化内容**:
  1. W 分布随 GRPO epoch 的演变 (t-SNE 或直方图序列)
  2. Reward 均值和方差随 epoch 变化
  3. KL divergence 随 epoch 变化
  4. 推荐性能 (Recall@20) 随联合训练 epoch 变化
- **表/图目标**: **Figure A2** (四联图) — 附录
- **优先级**: NICE-TO-HAVE

---

### Block 8: 骨干鲁棒性 (Appendix)

- **目的**: 验证 FlowNS 不依赖特定推荐骨干
- **数据集**: Yelp2018
- **对比**: FlowNS + BPR-MF vs FlowNS + LightGCN vs FlowNS + NGCF
- **指标**: Recall@20, NDCG@20
- **表/图目标**: **Table A1** — 附录
- **优先级**: NICE-TO-HAVE

---

## RecBole 集成架构

### 目录结构

```
flowns/
├── configs/                      # YAML 配置文件
│   ├── base.yaml                 # 公共配置（数据集路径、评估设置）
│   ├── lightgcn_yelp.yaml        # LightGCN + Yelp2018
│   ├── lightgcn_amazon.yaml      # LightGCN + Amazon-Books
│   ├── lightgcn_gowalla.yaml     # LightGCN + Gowalla
│   └── ablation/                 # 消融配置
├── src/
│   ├── __init__.py
│   ├── flow_model.py             # 条件 Flow Matching 速度网络
│   ├── sde_sampler.py            # ODE→SDE 转换 + Euler-Maruyama 采样
│   ├── reward.py                 # Shaped reward R=W^a(1-W)^γ
│   ├── grpo.py                   # GRPO 训练逻辑
│   ├── flowns_trainer.py         # 3 阶段训练器 (继承 RecBole Trainer)
│   ├── custom_metrics.py         # 自定义指标 (W 分布, FN 率)
│   ├── baselines/
│   │   ├── mixgcf.py             # MixGCF 负采样
│   │   ├── advinfonce.py         # AdvInfoNCE
│   │   └── cvae_ns.py            # 条件 VAE 负采样
│   └── generators/
│       ├── gmm_generator.py      # 条件 GMM 生成器 (消融用)
│       └── mlp_generator.py      # MLP 生成器 (消融用)
├── scripts/
│   ├── run_baselines.py          # 运行 RecBole 内置基线
│   ├── run_flowns.py             # 运行 FlowNS 完整 pipeline
│   ├── run_ablation.py           # 运行消融实验
│   └── evaluate_fn_rate.py       # 计算 FN 率和理论上界
├── refine-logs/                  # 实验计划和结果
└── docs/
    └── main.md
```

### RecBole 复用点

| 组件 | RecBole 提供 | 我们的使用方式 |
|------|------------|-------------|
| `Config` | 配置系统 | 加载 YAML，传递超参 |
| `create_dataset()` | 数据加载 + 自动下载 | 加载 Yelp2018/Amazon-Books/Gowalla |
| `data_preparation()` | 数据分割 + DataLoader 创建 | 训练/验证/测试分割 |
| `LightGCN` | 推荐骨干模型 | 直接使用，提取用户/物品嵌入 |
| `BPR` | MF 骨干 | 骨干鲁棒性消融 |
| `BPRLoss`, `EmbLoss` | 损失函数 | 推荐模型训练 |
| `Trainer` | 训练循环 | Phase 1 直接使用；Phase 3 继承并扩展 |
| `Evaluator` | 指标计算 | Recall@K, NDCG@K, Hit@K |
| `FullSortEvalDataLoader` | 全排序评估 | 标准评估 pipeline |
| `Sampler` | 均匀/流行度采样 | 基线运行 |
| `init_seed`, `init_logger` | 工具 | 种子控制、日志 |

### 关键集成代码设计

#### 1. FlowNS Trainer（继承 RecBole Trainer）

```python
# src/flowns_trainer.py
from recbole.trainer import Trainer

class FlowNSTrainer(Trainer):
    """3-phase FlowNS trainer built on RecBole Trainer."""

    def __init__(self, config, model, flow_model, grpo_config):
        super().__init__(config, model)
        self.flow_model = flow_model      # ConditionalFlowModel
        self.grpo_config = grpo_config
        self.grpo_trainer = GRPOTrainer(flow_model, grpo_config)

    def fit(self, train_data, valid_data=None, ...):
        # Phase 1: 标准 RecBole 训练 (均匀负采样)
        # 直接调用 super().fit() 的前 N 个 epoch
        logger.info("Phase 1: Flow pre-training...")
        self._phase1_pretrain(train_data, valid_data)

        # Phase 2: GRPO 微调 flow model
        logger.info("Phase 2: GRPO fine-tuning...")
        self._phase2_grpo(train_data)

        # Phase 3: 联合训练
        logger.info("Phase 3: Joint training...")
        self._phase3_joint(train_data, valid_data)

    def _phase1_pretrain(self, train_data, valid_data):
        # 训练推荐模型 (标准 RecBole)
        super().fit(train_data, valid_data)
        # 提取嵌入，训练 flow model
        user_emb, item_emb = self.model.forward()
        self.flow_model.pretrain(train_data, user_emb, item_emb)

    def _phase2_grpo(self, train_data):
        user_emb, item_emb = self.model.forward()
        self.grpo_trainer.train(
            train_data, user_emb, item_emb,
            model_score_fn=lambda u, x: (u * x).sum(dim=-1)
        )

    def _phase3_joint(self, train_data, valid_data):
        for epoch in range(self.grpo_config['joint_epochs']):
            # (a) 用 flow 生成负样本，训练推荐模型
            neg_embs = self.flow_model.generate(...)
            self._train_rec_with_custom_negs(train_data, neg_embs)
            # (b) 更新推荐嵌入后，GRPO 微调几步
            if epoch % 5 == 0:
                user_emb, item_emb = self.model.forward()
                self.grpo_trainer.train_steps(
                    train_data, user_emb, item_emb, n_steps=2
                )
```

#### 2. 提取 LightGCN 嵌入

```python
# 在训练完成后提取嵌入
rec_model = LightGCN(config, dataset)
# ... 训练后 ...
with torch.no_grad():
    user_all_emb, item_all_emb = rec_model.forward()
    # user_all_emb: [n_users, emb_dim]
    # item_all_emb: [n_items, emb_dim]
```

#### 3. Shaped Reward

```python
# src/reward.py
class BoundaryAwareReward:
    def __init__(self, a=1.0, gamma=1.0):
        self.a = a
        self.gamma = gamma

    def win_rate(self, user_emb, gen_emb, pos_item_embs):
        """W(x, u) = mean_i sigma(s(u,x) - s(u,i))"""
        # user_emb: [B, d], gen_emb: [B, d], pos_item_embs: [B, K, d]
        score_gen = (user_emb * gen_emb).sum(dim=-1, keepdim=True)  # [B, 1]
        score_pos = (user_emb.unsqueeze(1) * pos_item_embs).sum(dim=-1)  # [B, K]
        W = torch.sigmoid(score_gen - score_pos).mean(dim=-1)  # [B]
        return W

    def reward(self, W):
        """R = W^a * (1-W)^gamma"""
        return W.pow(self.a) * (1 - W).pow(self.gamma)

    @property
    def r_max(self):
        a, g = self.a, self.gamma
        return (a**a * g**g) / (a+g)**(a+g)

    def optimal_W(self):
        return self.a / (self.a + self.gamma)
```

#### 4. 评估 Pipeline（复用 RecBole Evaluator）

```python
# 评估时使用 RecBole 标准 pipeline
from recbole.evaluator import Evaluator, Collector

evaluator = Evaluator(config)
# ... 在 FullSortEvalDataLoader 上迭代 ...
# 调用 rec_model.full_sort_predict(interaction)
# 结果通过 Collector 收集，Evaluator 计算指标
```

#### 5. 自定义 FN 率指标

```python
# src/custom_metrics.py
def compute_fn_rate(generated_item_ids, test_positive_dict, user_ids):
    """计算生成负样本中假负样本的比例"""
    fn_count = 0
    total = 0
    for uid, neg_ids in zip(user_ids, generated_item_ids):
        test_pos = test_positive_dict.get(uid.item(), set())
        fn_count += len(set(neg_ids.tolist()) & test_pos)
        total += len(neg_ids)
    return fn_count / max(total, 1)
```

---

## 运行顺序与里程碑 (Run Order)

| 里程碑 | 目标 | 运行 | 决策门 | 算力 | 风险 |
|-------|------|------|-------|------|------|
| M0: 数据 + 环境 | 数据下载、pipeline 验证、指标正确性 | R001-R003 | 数据加载成功，LightGCN 在 ml-100k 上收敛 | 0.5 GPU-hr | RecBole 版本兼容 |
| M1: 基线复现 | 复现 LightGCN 在 3 数据集上的结果 | R004-R015 | 与文献报告值误差 <2% | 6 GPU-hr | 超参调优可能需迭代 |
| M2: Flow 预训练 | 验证 flow model 能学到有意义的负样本分布 | R016-R018 | 生成样本的最近邻确实是合理的负样本 | 6 GPU-hr | 嵌入空间可能退化 |
| M3: GRPO + FlowNS | 完整 FlowNS pipeline 运行 | R019-R027 | FlowNS > LightGCN(Uniform) 在 ≥1 数据集 | 12 GPU-hr | GRPO 不收敛 |
| M4: 决策消融 | Reward/KL/生成器消融 | R028-R045 | shaped reward 确实优于 unshaped | 15 GPU-hr | 差异不显著 |
| M5: 非 RecBole 基线 | MixGCF, AdvInfoNCE 实现并运行 | R046-R054 | 基线实现正确且性能合理 | 6 GPU-hr | 实现 bug |
| M6: 完善 | 保证紧致性、γ 敏感性、可视化 | R055-R065 | 理论上界有实际意义 | 5 GPU-hr | 上界过松 |

**总计: ~50 A100 GPU-hours, 4-6 周**

### 详细运行顺序

#### M0: Sanity (Day 1-2)

| Run | 目的 | 命令 |
|-----|------|------|
| R001 | RecBole 环境验证 | `python -c "from recbole.quick_start import run_recbole; ..."` on ml-100k |
| R002 | 数据下载验证 | 下载 yelp-2018, amazon-books, gowalla-merged |
| R003 | Flow model 单元测试 | 在合成数据上验证 CFM loss 收敛 |

**决策门**: 全部通过 → 进入 M1; 否则修复环境

#### M1: 基线复现 (Day 3-7)

| Run | 系统 | 数据集 | 配置 |
|-----|------|--------|------|
| R004-R006 | LightGCN (Uniform) | Yelp/Amazon/Gowalla × 3 seeds |
| R007-R009 | BPR-MF (Uniform) | Yelp/Amazon/Gowalla × 3 seeds |
| R010-R012 | LightGCN (Popularity) | Yelp/Amazon/Gowalla × 3 seeds |
| R013-R015 | LightGCN (DNS) | Yelp/Amazon/Gowalla × 3 seeds |

所有运行使用 RecBole 标准 `run_recbole()` API:
```python
from recbole.quick_start import run_recbole
run_recbole(model='LightGCN', dataset='yelp-2018',
            config_file_list=['configs/lightgcn_yelp.yaml'])
```

**决策门**: LightGCN(Uniform) 结果与文献一致 → 进入 M2

#### M2: Flow 预训练 (Day 5-9, 与 M1 部分并行)

| Run | 目的 | 数据集 |
|-----|------|--------|
| R016 | Flow model 预训练 + 生成质量检查 | Yelp |
| R017 | Flow model 预训练 | Amazon |
| R018 | Flow model 预训练 | Gowalla |

需要先从 M1 获取训练好的 LightGCN 嵌入。
生成质量检查: 生成 1000 个负样本，最近邻查找映射到物品，检查是否为已知正样本。

**决策门**: 生成样本的 FN 率 < 5% 且分布覆盖合理 → 进入 M3

#### M3: GRPO + FlowNS (Day 8-14)

| Run | 目的 | 数据集 |
|-----|------|--------|
| R019-R021 | FlowNS 完整 pipeline | Yelp × 3 seeds |
| R022-R024 | FlowNS 完整 pipeline | Amazon × 3 seeds |
| R025-R027 | FlowNS 完整 pipeline | Gowalla × 3 seeds |

**决策门**: FlowNS > LightGCN(Uniform) 在 ≥1 数据集 → 进入 M4; 否则 debug GRPO

#### M4: 决策消融 (Day 12-20)

| Run | Block | 变体 | 数据集 |
|-----|-------|------|--------|
| R028-R030 | B2 | FlowNS-Unshaped (R=W) | Yelp × 3 seeds |
| R031-R033 | B2 | FlowNS-NoRL | Yelp × 3 seeds |
| R034-R036 | B2 | FlowNS-RawScore | Yelp × 3 seeds |
| R037-R042 | B3 | β ∈ {0, 0.01, 0.05, 0.5, 1.0} | Yelp × 1 seed each |
| R043-R045 | B4 | GMM / CVAE / MLP generators | Yelp × 3 seeds each |

**决策门**: Shaped reward 显著优于 Unshaped → 核心声称成立

#### M5: 非 RecBole 基线 (Day 15-22)

| Run | 系统 | 数据集 |
|-----|------|--------|
| R046-R048 | MixGCF | Yelp/Amazon/Gowalla |
| R049-R051 | AdvInfoNCE | Yelp/Amazon/Gowalla |
| R052-R054 | CVAE-NS (baseline) | Yelp/Amazon/Gowalla |

**决策门**: 基线合理 → Table 1 完整

#### M6: 完善 (Day 20-28)

| Run | Block | 目的 |
|-----|-------|------|
| R055-R056 | B5 | 保证紧致性 (Yelp + Amazon) |
| R057-R061 | B6 | γ 敏感性 (Yelp, 5 个 γ 值) |
| R062-R063 | B7 | 训练过程可视化 (Yelp) |
| R064-R065 | B8 | 骨干鲁棒性 (BPR-MF + NGCF on Yelp) |

---

## 计算与数据预算

- **总计 GPU-hours**: ~50 A100 GPU-hours
  - M0-M1 (基线): ~7 hr
  - M2 (Flow 预训练): ~6 hr
  - M3 (FlowNS): ~12 hr
  - M4 (消融): ~15 hr
  - M5 (非 RecBole 基线): ~6 hr
  - M6 (完善): ~5 hr
- **数据准备**: RecBole 自动下载，无额外标注
- **人工评估**: 无需
- **最大瓶颈**: M3 GRPO 收敛（若不收敛需调参迭代）

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|------|-------|------|------|
| GRPO 不收敛 | 中 | 高 | 降低学习率；减少 SDE 步数；增大 group size G |
| Flow model 在低维空间退化 | 低 | 高 | 增加网络容量；尝试不同条件编码方式 |
| 与最强基线差异不显著 | 中 | 高 | 增加 seed 数至 5；检查超参；尝试更大 γ |
| RecBole 版本不兼容 | 低 | 中 | 固定 RecBole 版本；使用虚拟环境 |
| 曝光数据不可用 | 中 | 中 | 用未交互物品子集模拟曝光负样本 |
| MixGCF/AdvInfoNCE 实现困难 | 中 | 低 | 优先完成 FlowNS 核心实验，基线可后补 |

## 最终清单

- [x] 主论文表格已覆盖 (Table 1: 主结果, Table 2: Reward 消融, Table 3: 生成器消融)
- [x] 新颖性已隔离 (B2: reward 是性能提升的原因)
- [x] 简洁性已防御 (B3: KL 是必要的，不能去掉)
- [x] 前沿贡献已论证 (B4: flow matching vs 更简单生成器)
- [x] MUST-RUN 与 NICE-TO-HAVE 已分离
- [x] 所有实验基于 RecBole，不修改框架源码
