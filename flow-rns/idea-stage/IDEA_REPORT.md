# Flow-RNS 新一轮研究方案

生成时间：2026-08-02 13:01:42 +08:00。

## 结论先行

当前最值得检验的主线不是“让 Flow 直接替代推荐器”，而是先让一个充分容量的条件
Flow 只学习 `train` 中的用户条件曝光非点击分布，再把其样本作为负样本提议分布。第一阶段
严格按预注册顺序直接使用连续 embedding；若连续—离散 gap 或虚假负样本风险导致失败，
才进入有文献依据的软 catalog transport 和安全困难度倾斜。只有 Flow-only 的 validation
NDCG 达到 `0.721166`，才允许用 group-relative RL 优化 Flow 轨迹。

这不是一个已被证明新颖或有效的结论。FlowCF 已经把 flow matching 用于协同过滤，
FMRec、DiffuRec 和 DreamRec 已经研究连续推荐表示的生成；本项目目前可区分的组合是
“拟合曝光非点击分布用于负采样 + 不映射的连续 BPR + 安全困难度约束 + 门控后的
flow-RL”。官方 arXiv 页面已人工定位，但本机缺少 `verify_papers.py`，故引用仍统一标为
machine-unverified；外部交叉模型评审接口也不可用，不能把 novelty jury 标成通过。

## 冻结问题与成功标准

- 数据训练：参数只读取 `train`。Flow 的早停只读取 train 内固定曝光 holdout；推荐器的
  选择只读取官方 `validation`；方法冻结前禁止读取 `test`。
- 参考：RNS validation-NDCG-best 为 `0.671166`。
- Flow-only 门槛：validation NDCG 至少 `0.721166`，即绝对提高 5 个百分点。
- RL 门槛：先满足 Flow-only 门槛；RL 的冻结 validation NDCG 必须严格超过冻结
  Flow-only。之后才允许一次 test 确认。
- “Flow 充分训练”不以 FM loss 单独判定：必须同时报告 train-only holdout loss、端点
  MSE、sliced Wasserstein、均值/方差误差、solver 收敛和 catalog gap。

## 文献定位（均为 machine-unverified）

| 方向 | 代表工作 | 对本项目的约束 |
|---|---|---|
| 连续 Flow | Flow Matching；OT-CFM | 采用 simulation-free conditional vector-field regression，并检查路径/solver 误差 |
| Flow 推荐 | FlowCF；FMRec；FlowRec | “FM 用于推荐”以及用正负样本对齐 FM 本身不新；贡献必须限定在曝光负采样及其安全优化 |
| 连续表示到物品 | DiffuRec；DreamRec；TIGER；LETTER | 连续生成后映射/量化是强基线，但必须量化 rounding gap 和 code assignment bias |
| 困难负采样 | HNS theory；PDNS；Hard-BPR；UMA² | 难度与 top-K/OPAUC 相关，但过难负样本会放大 false negative；必须联合报告安全性/去偏 |
| Flow 的 RL | Flow-GRPO；RL for Flow-Matching Policies | 轨迹优化需要可计算的随机策略/参考 KL；不能把同一终点 reward 无差别复制给每一步 |

官方页面索引：`2210.02747`、`2302.00482`、`2502.07303`、`2505.16298`、
`2304.00686`、`2310.20453`、`2305.05065`、`2302.03472`、`2211.13912`、
`2403.19276`、`2505.05470`、`2507.15073`、`2508.17618`、`2405.07314`、
`2207.02468`。

## 候选方法排序

### 1. 条件连续曝光 Flow + 连续 BPR（当前主线）

对用户 \(u\)、曝光非点击物品的冻结 catalog embedding \(e_i\in\mathbb R^d\)，令
\(x_0\sim\mathcal N(0,I)\)、\(x_t=(1-t)x_0+t\tilde e_i\)，其中
\(\tilde e_i\) 是按冻结 catalog 坐标逐维标准化后的表示。训练

\[
  \mathcal L_{FM}=\mathbb E\|v_\phi(x_t,t,u)-(\tilde e_i-x_0)\|_2^2
  +\lambda_{end}\|x_t+(1-t)v_\phi(x_t,t,u)-\tilde e_i\|_2^2.
\]

Flow 端点反标准化得到连续负样本 \(z^-\)。在完全不映射物品 ID 的第一分支中，
冻结发布模型的 item table，只更新 GMF 的用户表示和打分向量：

\[
  s_\theta(u,z)=\langle p_u\odot h,z\rangle,\qquad
  \ell=-\log\sigma(s_\theta(u,e^+)-s_\theta(u,z^-)).
\]

冻结物品表不是便利性选择，而是坐标一致性条件；否则正样本物品坐标会在排序训练中
移动，Flow 仍生成旧坐标，形成隐蔽的目标漂移。需要补一个同样冻结 item table 的
uniform-RNS/BPR 公平对照，分离“冻结坐标”的影响。

淘汰条件：在大容量 Flow 通过训练充分性审计后，连续分支仍明显低于 RNS 或生成端点
最近 catalog 距离/最近训练点击率异常，则记录失败并进入候选 2，不能用继续调 test 的
方式挽救。

### 2. 安全困难度倾斜与软 catalog transport（Flow-only 备选）

P0 已表明原始曝光分布不是足够强的负样本分布，因此冻结 Flow 只作为 base proposal
\(p_F(z\mid u)\)。在 train-only 统计上定义困难度 \(H_\theta(u,z)\) 和安全支持度
\(S(u,z)\in[0,1]\)，用受限指数倾斜

\[
 q_\beta(z\mid u)=\frac{p_F(z\mid u)
 \exp\{\beta S(u,z)H_\theta(u,z)\}}{Z_\beta(u)},
 \quad D_{KL}(q_\beta\|p_F)\le\varepsilon,
\]

并同时约束有效样本量（ESS）与唯一物品率。这样样本难度进入分布而非任意相加的
reward，\(\varepsilon\) 给出偏离曝光 Flow 的可解释上界。

若必须落到真实物品，不直接 hard nearest-neighbor。先在训练数据上选择温度，使用

\[
 w_j(z)=\frac{\exp[-d_M(z,e_j)/\tau]}{\sum_{k\in\mathcal N_K(z)}
 \exp[-d_M(z,e_k)/\tau]}
\]

对 top-K catalog 邻域做软边际化 BPR；报告有效支持、最近邻距离、hard/soft 性能差和
长尾覆盖。只有在这些 gap 指标可接受后，才可评估 hard ID 映射。

### 3. 门控后的 group-relative Flow RL

RL 不能读取 test，也不能把 validation 标签直接变成每个样本的训练 reward。建立来自
train clicks 的固定 shadow holdout \(H\)。对一个候选负样本 \(x\)，令
\(g_H=\nabla_\theta L_H(\theta)\)、\(g_x=\nabla_\theta\ell_x(\theta)\)。一步 SGD 后的
shadow-loss 改善一阶近似为

\[
 L_H(\theta)-L_H(\theta-\eta g_x)
 \approx \eta\langle g_H,g_x\rangle.
\]

因此采用归一化/裁剪后的梯度对齐为核心 reward；其内积同时包含梯度方向和大小，样本
难度自然通过 \(\|g_x\|\) 进入，而方向相反的“很难但有害”样本会被惩罚。为防止纯大
梯度支配，另记录 cosine alignment、gradient norm band 和 false-negative proxy。

若轨迹状态的势函数为 \(\Phi(z_t)\)，使用逐步 reward

\[
 r_t=\Phi(z_{t+1})-\Phi(z_t),
\]

则总和望远镜化为 \(\Phi(z_T)-\Phi(z_0)\)，不会改变固定起点下的终点最优策略，同时
比把终点 reward 复制到每一步提供更合理的 credit assignment。实际更新采用同用户
group-relative advantage，并加入 reference path KL/clip、熵、ESS、唯一率和 exposure
支持约束。若确定性 ODE 无法给出稳定的轨迹概率，必须切换为带已知转移密度的 SDE
参数化或可算 CNF likelihood，禁止伪造 log-probability。

## 负面先验实验

P0 直接从真实 train exposure 抽负样本，10 epochs 后最佳 validation AUC/NDCG 为
`0.744288 / 0.665192`，比 RNS validation-NDCG-best 低 `0.005974`，距 Flow gate
`0.055974`。这否定了“只要准确复现原始曝光分布就自然超过 RNS”的工作假设，但不
否定 Flow 作为高覆盖 proposal；它使安全困难度倾斜成为必要而非装饰性组件。

## 当前实施状态与风险

- 10.5M 参数、8 层 width-512 的 user-conditioned Flow 正在全量训练，train 内固定
  holdout 131,072 个曝光对；官方 validation/test 未参与。
- 连续 ranker 已实现：连续 embedding 直接 GMF 打分、物品表冻结、可重复 Flow pool、
  rounding gap 仅诊断不参与训练、validation NDCG early stopping。
- 当前测试 23 项通过。
- 最大风险是预注册的 +0.05 NDCG 门槛远高于现有方法间差异。若所有充分训练的
  Flow-only 分支均失败，应诚实停止在 Flow 阶段，RL 保持门控，而不是降低门槛或读取
  test 追结果。

## 下一步固定顺序

1. 完成 Flow 训练，做 8/16/32 步 solver、分布拟合和 catalog-gap 审计。
2. 运行连续 embedding 直训及冻结-item uniform 公平对照。
3. 若失败，基于上述文献与 gap 指标运行 soft transport，再运行 KL/ESS 约束的安全
   困难度倾斜。
4. 只有 validation NDCG 达到 `0.721166` 才实现并运行 RL；否则报告门槛未达成。
5. 冻结全部选择后才读取一次 test，并做多 seed、bootstrap、消融和中间产物清理。
