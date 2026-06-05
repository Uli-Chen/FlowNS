# Research Proposal: FlowNS — Flow Matching with Principled RL-Guided Hardness for Negative Sampling

## Problem Anchor

- **核心问题**: 隐式反馈推荐系统中，基于 pairwise ranking loss (如 BPR) 训练时，负采样质量直接决定排序性能。现有方法要么均匀采样（无信息量）、启发式选择硬负样本（假负样本污染）、或忽略曝光数据。
- **必须解决的瓶颈**: 不存在同时满足以下三点的数学严密机制：(1) 生成忠于真实负样本分布的负样本；(2) 以数学上有依据的方式控制硬度；(3) 提供防止假负样本污染的形式化保证。
- **非目标**: 不构建通用生成式推荐模型；不替换推荐骨干网络；不通过工程技巧追求所有 benchmark SOTA；不设计花哨的多模块系统。
- **约束**: 单 GPU 可训练（A100）；标准 RecSys 数据集；flow model 在物品嵌入空间运行（$d \leq 128$）；推荐骨干为标准协同过滤模型。
- **成功标准**: 单一数学严密机制，可证明地生成真实且困难的负样本，含 reward 性质的形式化分析和标准 benchmark 上的实验验证。

## 技术缺口

### 现有方法的失败之处

1. **均匀/流行度采样** (BPR): 多数负样本梯度贡献接近零。BPR 梯度权重 $\sigma(r_{uj} - r_{ui^+})$ 在 $r_{uj} \ll r_{ui^+}$ 时指数衰减。

2. **硬负样本挖掘** (DNS, ANCE, MixGCF): Shi et al. (WWW 2023) 证明其隐式优化 OPAUC。但最硬负样本恰是最可能的假负样本，造成硬度-保真度权衡无形式化解决方案。

3. **生成式负采样** (IRGAN, AdvInfoNCE): GAN 训练不稳定，无法显式控制硬度-保真度权衡。

### 朴素扩展为何失败

- **更大模型/更多数据**: 不解决根本的硬度-保真度权衡
- **仅添加 flow model**: 可学习负样本分布但无硬度控制机制
- **朴素应用 RL**: 用推荐模型原始分数做 reward → 循环依赖、reward hacking、混淆硬度与信息量
- **更简单的生成模型 (GMM, 拒绝采样)**: 负偏好分布是用户条件下的多模态分布，GMM 需预设模态数，拒绝采样重引入循环依赖

### 最小充分干预

条件 Flow Matching 学习真实负样本分布 + GRPO 微调（使用从第一性原理推导的 shaped reward）。

## 方法论文

- **一句话论文**: FlowNS 通过条件 flow matching 学习真实负样本分布，然后用 GRPO 配合数学推导的*边界感知 shaped reward* 最大化训练信息量，同时通过 reward shaping 和分布锚定的双重机制可证明地控制假负样本污染。
- **为何是最小充分干预**: Flow model 处理真实性；单一 reward 函数处理硬度；KL 约束处理保真度。
- **为何当下合适**: Flow matching、GRPO、ODE-to-SDE 转换都是近期进展，自然匹配此问题。

## 贡献聚焦

- **主导贡献**: 从 OPAUC 理论推导的原则性 reward 函数 $R = W^a(1-W)^\gamma$，含有界性、假负样本修正和最优性的形式化分析。
- **支撑贡献**: 完整 FlowNS pipeline 证明此最小架构足够。
- **明确非贡献**: 不声称 flow matching、GRPO 或 ODE-to-SDE 转换本身的新颖性。

## 方法

### 复杂度预算

- **冻结/复用**: 推荐模型（GRPO 期间冻结）；物品嵌入（共享）
- **新可训练组件**: (1) 条件速度网络 $v_\theta(\mathbf{x}_t, u, t)$；(2) 同一网络经 GRPO 微调
- **有意不使用**: 学习的 reward model；判别器；独立硬度预测器；多阶段 curriculum

### 系统总览

```
阶段 1 (Flow 预训练):
  同时训练推荐模型（均匀负样本）+ flow model（CFM 损失）
  输出: π_ref（参考 flow 策略）

阶段 2 (GRPO 微调):
  每用户采样 G 条 SDE 轨迹
  计算 R = W^a·(1-W)^γ，组优势，更新 v_θ
  输出: π_θ（硬度增强策略）

阶段 3 (联合训练):
  交替: (a) 用 π_θ 负样本训练推荐模型, (b) 冻结推荐模型下 GRPO 更新
```

### 核心机制: 边界感知 Shaped Reward

#### 第一步: 定义胜率

对生成的负样本嵌入 $\mathbf{x}_1$ 和用户 $u$（正样本集 $\mathcal{I}_u^+$），定义**对正样本的胜率**:

$$W(\mathbf{x}_1, u) = \frac{1}{|\mathcal{I}_u^+|} \sum_{i \in \mathcal{I}_u^+} \sigma\bigl(s(u, \mathbf{x}_1) - s(u, i)\bigr)$$

**$W$ 的性质:**
- $W \in (0, 1)$（有界）
- $W \approx 0$: 简单负样本（无信息量）
- $W \approx 0.5$: 决策边界处（对 BPR 信息量最大）
- $W \approx 1$: 击败多数正样本（可能是假负样本）

**与 OPAUC 的联系**: $W$ 恰好是生成负样本的假正率经验估计，直接对接 Shi et al. (2023) 的 OPAUC 框架。

#### 第二步: 通过 Shaping 惩罚假负样本

朴素地最大化 $W$ 会推向假负样本。我们施加**边界感知 shaping 函数**:

$$R(\mathbf{x}_1, u) = W(\mathbf{x}_1, u)^a \cdot \bigl(1 - W(\mathbf{x}_1, u)\bigr)^\gamma$$

**数学性质:**

1. **有界性**: $R \in [0, R_{\max}]$，其中 $R_{\max} = \frac{a^a \gamma^\gamma}{(a+\gamma)^{a+\gamma}}$，防止 reward hacking。

2. **最优硬度点**: $W^* = \frac{a}{a + \gamma}$，reward 在此处达到峰值。

3. **非对称惩罚**: 当 $\gamma > a$ 时，假负样本（高 $W$）受到比简单负样本（低 $W$）更严重的惩罚。

4. **Beta 分布联系**: $R(W) \propto \text{Beta}(a+1, \gamma+1)$ 密度，GRPO 鼓励生成器产生胜率集中于 $W^*$ 附近的负样本。

5. **特殊情况**:
   - $a = 1, \gamma = 1$: $R = W(1-W)$，对称，$W^* = 0.5$（最简形式，等于 Bernoulli 方差）
   - $a = 1, \gamma = 2$: $W^* = 1/3$（更保守）
   - $a = 1, \gamma = 3$: $W^* = 1/4$（非常保守）

**默认**: $a = 1, \gamma = 1$，通过验证集调整 $\gamma \in \{1, 2, 3\}$。

#### 第三步: 双重假负样本控制的形式化分析

**机制 1（Reward shaping）**: $\frac{\partial R}{\partial W}\big|_{W=1} = 0$，在假负样本边界处 GRPO 策略梯度提供零学习信号。

**机制 2（KL 锚定）**: KL 惩罚约束 RL 策略接近预训练 flow model。由 Donsker-Varadhan 表示，对任意可测集 $A$:

$$\pi_\theta(A | u) \leq \pi_{\text{ref}}(A | u) \cdot \exp\left(\frac{R_{\max}}{\beta}\right)$$

**联合假负样本保证**: 设 $\mathcal{F}_u$ 为假负样本集，RL 策略下的期望假负样本率满足:

$$\mathbb{E}_{\mathbf{x} \sim \pi_\theta(\cdot|u)}[\mathbb{1}[\mathbf{x} \in \mathcal{F}_u]] \leq \exp\left(\frac{R_{\max}}{\beta}\right) \cdot \mathbb{E}_{\mathbf{x} \sim \pi_{\text{ref}}(\cdot|u)}[\mathbb{1}[\mathbf{x} \in \mathcal{F}_u]]$$

#### 第四步: 为何不选择替代 reward？

| 替代方案 | 问题 |
|---------|------|
| 原始分数 $s(u, \mathbf{x})$ | 无界 → reward hacking；无 FN 修正；尺度依赖 |
| 分数的 sigmoid $\sigma(\alpha \cdot s)$ | 无参考点；不区分边界硬和假负硬 |
| 梯度范数 $\|\nabla_\Theta \ell\|$ | 昂贵；不分离 FN |
| 基于排名的 reward | 需全物品排序；不连续 |
| **胜率 + Beta shaping（本文）** | 有界；OPAUC 联系；闭式 FN 惩罚；单超参 $\gamma$ |

### Flow Matching 学习真实负样本分布

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1],\, \mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I}),\, \mathbf{e}_n \sim \mathcal{E}_u^{\text{neg}}} \left[\left\| v_\theta\bigl((1-t)\mathbf{x}_0 + t\,\mathbf{e}_n,\, u,\, t\bigr) - (\mathbf{e}_n - \mathbf{x}_0) \right\|^2\right]$$

约定: $t = 0$ 为噪声，$t = 1$ 为数据。生成沿正向进行。

**为何选择 flow matching**: 用户条件下负样本分布是多模态的。Flow matching 无需预设模态数（vs GMM），无后验坍缩（vs VAE），可捕捉嵌入空间中任意几何结构。

### 采样过程

将确定性 flow ODE 转换为保持边缘分布的等效 SDE:

$$d\mathbf{x}_t = \left[\mathbf{v}_\theta + \frac{\sigma_t^2}{2} \nabla\log p_t(\mathbf{x}_t)\right] dt + \sigma_t\,d\mathbf{w}$$

Tweedie 公式近似 score: $\nabla\log p_t(\mathbf{x}_t) \approx -\frac{\mathbf{x}_t - t\,\mathbb{E}[\mathbf{x}_1 | \mathbf{x}_t]}{(1-t)^2}$

噪声调度: $\sigma_t = \eta \cdot \frac{\sqrt{1-t}}{\sqrt{t} + \delta}$

Euler-Maruyama 离散化:
$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \tilde{\mathbf{v}}_\theta(\mathbf{x}_t, u, t)\,\Delta t + \sigma_t \sqrt{\Delta t}\,\boldsymbol{\epsilon}$$

### GRPO 目标

轨迹级优势: $\hat{A}^i = (R^i - \bar{R}) / (\sigma_R + \epsilon_{\text{std}})$

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left(\min\bigl(r_t^i \hat{A}^i,\, \text{clip}(r_t^i, 1-\epsilon, 1+\epsilon)\hat{A}^i\bigr) - \beta\, D_t^i\right)\right]$$

Gaussian 转移密度下闭式计算 importance ratio 和 KL:
$$\log r_t^i(\theta) = \frac{\|\mathbf{x}_{t+\Delta t}^i - \mathbf{x}_t^i - \tilde{\mathbf{v}}_{\theta_{\text{old}}} \Delta t\|^2 - \|\mathbf{x}_{t+\Delta t}^i - \mathbf{x}_t^i - \tilde{\mathbf{v}}_\theta \Delta t\|^2}{2\sigma_t^2 \Delta t}$$

$$D_t^i = \frac{\|\tilde{\mathbf{v}}_\theta(\mathbf{x}_t^i) - \tilde{\mathbf{v}}_{\text{ref}}(\mathbf{x}_t^i)\|^2 \Delta t}{2\sigma_t^2}$$

实现细节: 逐步比率归一化 $\tilde{r}_t^i = (r_t^i - \mu_r^{(t)}) / \sigma_r^{(t)} + 1$ 保证稳定性。

### 嵌入到物品映射

$$p(j \mid \mathbf{x}_1) = \frac{\exp(\mathbf{x}_1^\top \mathbf{e}_j / \tau)}{\sum_{k \in \mathcal{I}} \exp(\mathbf{x}_1^\top \mathbf{e}_k / \tau)}$$

实践中使用 FAISS 近似最近邻检索。

### 训练计划

1. **阶段 1（Flow 预训练）**: Adam, lr=1e-4, 50 epochs, batch 256
2. **阶段 2（GRPO 微调）**: G=8, clip ε=0.2, β=0.1, T=20, η=0.5, δ=0.01, 10 epochs
3. **阶段 3（联合训练）**: 每 5 个推荐 epoch 交替 2 个 GRPO epoch，共 100 推荐 epoch
4. **Reward 参数**: 默认 a=1, γ=1，调参 γ ∈ {1, 2, 3}

### 失败模式与诊断

| 失败模式 | 检测 | 缓解 |
|---------|------|------|
| 模式坍缩 | 生成分布熵 | 增大 β，减小 η |
| 假负样本爆炸 | W 分布（W>0.8 占比>10% 报警） | 增大 γ |
| 比率不稳定 | Var(log r_t) | 验证比率归一化；减小 Δt |
| 联合训练振荡 | 推荐性能退化 | 减少 GRPO 频率 |

### 新颖性与优雅性论证

**最近相关工作**: DNS/ANCE（启发式）, IRGAN（GAN，不稳定）, DiffuRec/DreamRec（正样本生成）, Flow-GRPO for images（不同领域）

**精确区别**: FlowNS 首次从 OPAUC 理论推导 RL 引导负样本生成的 reward 函数，提供有界性、假负样本控制和双重 KL+shaping 保证的形式化分析。$R = W^a(1-W)^\gamma$ 不是即兴设计——它自然地从要求 (1) OPAUC 动机硬度、(2) 有界 reward、(3) $W=1$ 处零梯度三个条件中涌现。

**为何聚焦**: 全部新颖性集中在一个方程及其形式化性质。Pipeline 组装自已知组件。这是机制级贡献，非系统级贡献。

## 声称驱动的验证方案

### 声称 1: Shaped reward 生成更硬的负样本且不增加假负样本

- **基线**: Uniform, Popularity, DNS, MixGCF, IRGAN, AdvInfoNCE, CVAE-NS
- **指标**: Recall@20, NDCG@20; W 分布; FN 率
- **子实验（保证紧致性）**: 测量实际 FN 率 vs 理论上界 $\exp(R_{\max}/\beta) \cdot \text{FN}_{\text{ref}}$，across $\beta \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$

### 声称 2: Shaped reward 是必要的——reward 组件消融

- (A) 完整: $R = W(1-W)^\gamma$
- (B) 无 shaping: $R = W$
- (C) 无 RL: 直接用预训练 flow model
- (D) 对称: $R = W(1-W)$
- (E) 变化 γ ∈ {0.5, 1, 2, 3}

### 声称 3（简化检查）: KL 约束是承重的

- 设 β=0，观察分布漂移、FN 率上升、推荐性能退化

### 声称 4: Flow matching 是此设定下正确的生成器

- 将 flow model 替换为 (a) 条件 GMM (b) 条件 VAE，使用相同 shaped reward
- 若 flow 胜: 验证选择。若平局: reward 才是贡献，生成器可互换。

## 计算与时间估算

- 总计: ~40-50 A100 GPU-hours
- 时间线: 4-6 周
