# Generating Both Real and Hard Negatives for Recommender System

## Introduction

![Figure 1](fig1.png)

推荐系统本质上是在做 Embedding Learning，基于 Pairwise Ranking 损失（e.g. BPR Loss）训练的系统需要正例和负例来计算梯度，负采样决定了负例的选择。在这个任务当中，我们研究带有曝光数据（e.g. 被推荐了但是用户未交互的物品）情景下的负采样方法。

负采样面临着如下问题：

- **数据稀疏性：** 在 User-item Interaction Matrix 当中，有许多未观测值。
- **难以利用曝光数据：** 曝光数据往往也是稀疏的，不足以支撑训练，难以直接利用。
- **假负样本 False Negative Problem：** 选择的负样本实际上是未交互的用户感兴趣的物品。

## Methodology

### Problem Formulation

设用户集合为 $\mathcal{U}$，物品集合为 $\mathcal{I}$，曝光数据为 $\mathcal{E}$。

我们的目标是学习一个条件生成模型 $G(\cdot \mid u)$，用于刻画用户 $u$ 的**负偏好分布**。

对于给定用户 $u$，生成模型输出一组负样本 $I_u^-$，并用于训练推荐模型 $R$。
与传统基于启发式规则的负采样方法不同，我们显式建模“真实负样本”的分布，再在此基础上引入难度约束。

### Learning Negative Preference

在隐式反馈场景中，未交互数据并不等价于负样本。
因此，我们首先希望学习一个能够刻画“真实负样本”分布的生成模型。

具体而言，我们采用 CondOT 路径下的条件 Flow Matching 方法，学习从先验高斯噪声分布到用户负曝光集合 $\mathcal{E}_u^{\text{neg}}$ 的映射。

其优化目标定义为：

$$
\mathcal{L}_{\text{CFM}}(\theta)
=
\mathbb{E}_{t, \mathbf{x}_0, u, \mathbf{e}_n}
\left[
\left\|
v_t^\theta(\mathbf{x}_t, u)
-
(\mathbf{e}_n - \mathbf{x}_0)
\right\|^2
\right],
$$

其中 $v_t^\theta$ 表示时间相关的速度场，$\mathbf{e}_n$ 为真实负曝光样本。

通过最小化该损失，我们学习一个条件传输映射，使得噪声分布能够被映射到“真实负样本”分布上，从而保证生成样本的真实性（realness）。

### Adding Hardness to Negatives

为了进一步提升生成负样本的训练价值，我们希望在保证真实性（realness）的同时，引入负样本的难度（hardness）。
强化学习（RL）的目标是学习一个策略，使其最大化期望累积回报，这通常可以表述为带正则项的策略优化问题：

$$
\max_{\theta} \; \mathbb{E}_{(s_0, a_0, \ldots, s_T, a_T) \sim \pi_\theta}
\left[
\sum_{t=0}^{T}
\left(
R(s_t, a_t) - \beta D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot \mid s_t) \| \pi_{\mathrm{ref}}(\cdot \mid s_t)\bigr)
\right)
\right].
$$

不同于 PPO 等传统基于策略梯度的方法，GRPO 提供了一种更加轻量的替代方案。
它通过组相对（group-relative）的方式来估计 advantage，从而避免对单样本奖励进行直接建模。

在我们的设定中，生成负样本的过程同样可以视为一个马尔可夫决策过程（MDP）。
给定用户条件 $u$，流模型 $p_\theta$ 生成一组大小为 $G$ 的负样本表示 $\{\mathbf{x}_0^i\}_{i=1}^{G}$，以及对应的反向时间轨迹 $\{(\mathbf{x}_T^i, \mathbf{x}_{T-1}^i, \ldots, \mathbf{x}_0^i)\}_{i=1}^{G}$。随后，第 $i$ 个样本在时刻 $t$ 的 advantage 可以通过组内奖励归一化得到：

$$
\hat{A}_t^i =
\frac{R(\mathbf{x}_0^i, u) - \mathrm{mean}(\{R(\mathbf{x}_0^i, u)\}_{i=1}^{G})}
{\mathrm{std}(\{R(\mathbf{x}_0^i, u)\}_{i=1}^{G})}.
$$

基于此，我们通过最大化如下目标来优化生成策略，使模型逐步倾向于生成更困难但仍然合理的负样本：

$$
\mathcal{J}_{\mathrm{Flow\mbox{-}GRPO}}(\theta)
=
\mathbb{E}_{u \sim \mathcal{U},\{\mathbf{x}^i\}_{i=1}^{G} \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid u)}
f(r, \hat{A}, \theta, \epsilon, \beta),
$$

其中

$$
f(r, \hat{A}, \theta, \epsilon, \beta)
=
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{T}
\sum_{t=0}^{T-1}
\left(
\min\bigl(r_t^i(\theta)\hat{A}_t^i,
\mathrm{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^i\bigr)
- \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
\right),
$$

$$
r_t^i(\theta) =
\frac{p_\theta(\mathbf{x}_{t-1}^i \mid \mathbf{x}_t^i, u)}
{p_{\theta_{\mathrm{old}}}(\mathbf{x}_{t-1}^i \mid \mathbf{x}_t^i, u)}.
$$

这里，奖励函数 $R(\mathbf{x}_0^i, u)$ 用于衡量生成负样本的“困难程度”。
例如，我们可以利用当前推荐模型对生成样本的打分来定义奖励：分数越高，说明该样本越接近用户可能感兴趣但尚未交互的物品，因此也越“难”。
与此同时，KL 正则项约束更新后的策略不要偏离参考策略过远，从而在提升 hardness 的同时维持生成分布的稳定性与真实性。

#### From ODE to SDE

在式（4）和式（5）中，GRPO 依赖随机采样来生成多样化轨迹，以便进行 advantage 估计与探索。扩散模型天然支持这一点：其正向过程逐步加入高斯噪声，反向过程则可近似为一个方差递减的 score-based SDE 所对应的马尔可夫链。相比之下，flow matching 模型在正向生成过程中通常采用确定性的 ODE：

$$
\mathrm{d}\mathbf{x}_t = \mathbf{v}_t\,\mathrm{d}t,
$$

其中 $\mathbf{v}_t$ 由前述 flow matching 目标学习得到。

然而，这种确定性方法并不完全满足 GRPO 对策略更新的需求，主要体现在两个方面：
第一，式（5）中的比值项 $r_t^i(\theta)$ 需要计算条件概率 $p(\mathbf{x}_{t-1}^i \mid \mathbf{x}_t^i, u)$，而在确定性动力系统下，这通常需要借助 divergence estimation，计算代价较高；
第二，更重要的是，强化学习依赖探索机制，而完全确定性的采样过程除了初始随机种子外几乎不引入额外随机性，这会限制策略优化中的探索能力。

为了解决这一问题，我们将式（6）中的确定性 Flow-ODE 转化为一个在所有时间步都与原模型边缘分布一致的 SDE。按照这一思路，可以构造如下保持边缘分布不变的反向时间 SDE：

$$
\mathrm{d}\mathbf{x}_t =
\left(
\mathbf{v}_t(\mathbf{x}_t) - \frac{\sigma_t^2}{2} \nabla \log p_t(\mathbf{x}_t)
\right)\mathrm{d}t + \sigma_t\,\mathrm{d}\mathbf{w},
$$

其中 $\mathrm{d}\mathbf{w}$ 表示 Wiener 增量，$\sigma_t$ 控制生成过程中注入随机性的强弱。
对于 rectified flow，上式可进一步写为：

$$
\mathrm{d}\mathbf{x}_t =
\left[
\mathbf{v}_t(\mathbf{x}_t) + \frac{\sigma_t^2}{2t}
\bigl(\mathbf{x}_t + (1-t)\mathbf{v}_t(\mathbf{x}_t)\bigr)
\right]\mathrm{d}t + \sigma_t\,\mathrm{d}\mathbf{w}.
$$

进一步地，采用 Euler--Maruyama 离散化后，可得到最终的更新公式：

$$
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t +
\left[
\mathbf{v}_\theta(\mathbf{x}_t, t) + \frac{\sigma_t^2}{2t}
\bigl(\mathbf{x}_t + (1-t)\mathbf{v}_\theta(\mathbf{x}_t, t)\bigr)
\right]\Delta t + \sigma_t\sqrt{\Delta t}\,\boldsymbol{\epsilon},
$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ 用于向采样过程显式注入随机性。本文采用 $\sigma_t = a\sqrt{\frac{1-t}{t}}$，其中 $a$ 是控制噪声水平的标量超参数。

由式（9）可见，策略 $v_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, c)$ 实际对应于各向同性高斯分布，因此式（5）中的参考策略 KL 散度可以写成闭式形式：

$$
D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
=
\frac{\left\|\bar{\mathbf{x}}_{t+\Delta t,\theta} - \bar{\mathbf{x}}_{t+\Delta t,\mathrm{ref}}\right\|^2}{2\sigma_t^2 \Delta t}
=
\frac{\Delta t}{2}
\left(
\frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t}
\right)^2
\left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{v}_{\mathrm{ref}}(\mathbf{x}_t, t) \right\|^2.
$$

这一定式非常关键：它说明在引入随机探索之后，我们仍然能够以闭式方式计算策略与参考策略之间的偏离程度，从而将 GRPO 稳定地应用到 flow-based 负样本生成过程中。

### Sampling

生成模型输出的是连续表示 $\mathbf{x}_1 \in \mathbb{R}^d$，但推荐系统训练需要离散物品索引 $j \in \mathcal{I}$。

为此，我们将生成向量与物品嵌入空间进行匹配。具体而言，对于每个物品嵌入 $\mathbf{e}_j$，计算相似度：

$$
s_j = \mathbf{x}_1^\top \mathbf{e}_j.
$$

随后通过 softmax 归一化得到条件分布：

$$
p(j \mid \mathbf{x}_1)
=
\frac{\exp(s_j / \tau)}
{\sum_{k \in \mathcal{I}} \exp(s_k / \tau)},
$$

其中 $\tau$ 为温度参数。

最终，我们根据该分布进行采样或选择 top-$k$ 物品，作为生成的离散负样本。

### Training

#### Hardness Scheduling

直接生成极难负样本可能导致训练不稳定，尤其是在推荐模型尚未充分收敛的早期阶段。

因此，我们引入 hardness scheduling 机制，逐步增强负样本的难度。

我们采用逐步增加的引导强度 $\alpha_t$：

$$
\alpha_t = \alpha_{\max} \cdot \frac{t}{T},
$$

或等价地，通过逐步降低 softmax 温度 $\tau_t$，使生成分布在训练后期更加集中于高相似度区域。

该机制本质上是一种 curriculum learning，使模型从容易区分的负样本过渡到困难负样本。
