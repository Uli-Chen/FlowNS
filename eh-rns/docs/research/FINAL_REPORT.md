# EH-RNS：曝光校准困难负采样的设计、实现与结果

> 最终状态：在冻结的 Zhihu / seed=1 / 完整早停 benchmark 上，EH-RNS 的 test
> AUC/NDCG 为 **0.707490/0.662385**，同时超过完整 ReinforceNS 的
> **0.707266/0.662166**。这是单 seed 点估计胜出；AUC 配对区间为正，NDCG 配对区间跨 0。

## 1. 问题与结论

初始 FlowRL 用连续 Flow Matching 学习曝光表示，再用策略梯度加强困难度。实际结果只有
0.697971/0.652676，低于当时五轮 RNS 的 0.705600/0.660613。进一步完整训练 RNS 后，
真正门槛提高为 0.707266/0.662166。后续研究因此聚焦两个问题：

1. 如何直接学习离散候选上的曝光分布，而不是先生成连续表示再做最近邻解码；
2. 如何增强困难样本，同时用曝光证据抑制追逐潜在假负例。

最终方法 EH-RNS（Exposure-calibrated Hard ReinforceNS）保留 RNS 已验证有效的在线策略
梯度、精确曝光奖励和多核 MMD 奖励，在其行为分布前增加：

- 充分训练、随后冻结的曝光密度比模型；
- 曝光先验与在线策略的计数退火反向 KL 重心；
- 曝光后验校准的正标准化 ranker 困难度 Gibbs 倾斜。

相对完整 RNS，test AUC 提高 0.0002240903，NDCG 提高 0.0002195163。

## 2. 数据与冻结实验协议

| 项目 | 值 |
|---|---:|
| 数据集 | Zhihu |
| 用户数 / item 数 | 16,015 / 45,782 |
| train click events | 2,433,969 |
| train unique exposures | 6,683,040 |
| validation / test 用户 | 15,928 / 16,015 |
| 初始化 | 发布的 BPR-GMF checkpoint |
| seed | 1 |
| list length | 160 |
| batch size | 1,024 |
| 每用户采样候选 | 30（29 未曝光 + 1 train-exposure anchor） |
| 最大联合 epoch | 400 |
| 选择规则 | validation AUC |
| early stopping | patience=10；实现为连续 11 次不改善后停止 |

完整 RNS 在第 9 轮达到最佳 validation 0.747364/0.670858，第 20 轮早停，冻结后 test
为 0.707266/0.662166。EH-RNS 的所有设计和实现均在读取其 test 之前完成；EH-RNS
只有在 validation AUC 越过 0.747364 后才执行冻结 test 点估计；随后对同一冻结模型做
确认性逐用户 bootstrap，二者都没有反馈到训练、方法或 checkpoint 选择。

## 3. 方法

### 3.1 候选集合

对用户 $u$，先构造有限候选集合

\[
\mathcal C_u=\{j_1^{U},\ldots,j_{29}^{U},j^{E}\},
\]

其中 $j^U$ 从既未点击也未在训练曝光中出现的 item 均匀抽取，$j^E$ 从该用户的
train exposure 抽取。训练点击、曝光和候选构造都只读取 train split。

下面所有策略分布都条件化在同一个有限集合 $\mathcal C_u$ 上。这样既继承 RNS 的
经验曝光覆盖，又能用学习模型在集合内部泛化曝光相似性。

### 3.2 曝光密度比学习

令 $p_E(j\mid u)$ 为训练曝光分布，$r(j\mid u)$ 为 catalog noise。用等先验二分类
训练 GMF logit $a_\psi(u,j)$：

\[
\mathcal L_{NCE}(\psi)=
\mathbb E_{(u,j)\sim p_E}\operatorname{softplus}(-a_\psi(u,j))+
\mathbb E_{(u,j)\sim r}\operatorname{softplus}(a_\psi(u,j)).
\]

noise 与 exposure 的支持允许重叠；这里的“负类”表示样本来自 noise process，而不是把
该 item 声明成语义负反馈。

**定理 1（等先验 NCE 密度比）。** 对固定 $(u,j)$，若函数类无限且达到人口风险最优，
则

\[
a^*(u,j)=\log\frac{p_E(j\mid u)}{r(j\mid u)},\qquad
\sigma(a^*)=P(Y=E\mid u,j)
\]

其中后一个概率使用训练时人工构造的等类别先验。

**证明。** 逐点风险为

\[
-p_E\log\sigma(a)-r\log(1-\sigma(a)).
\]

对 $a$ 求导并令零，得到
$(p_E+r)\sigma(a)=p_E$，故
$\sigma(a)=p_E/(p_E+r)$；取 logit 即得结论。二阶导为
$(p_E+r)\sigma(a)(1-\sigma(a))>0$，最优解唯一。∎

在有限候选集合上定义曝光能量分布

\[
p_{\psi,\mathcal C}(j\mid u)=
\frac{\exp a_\psi(u,j)}{\sum_{k\in\mathcal C_u}\exp a_\psi(u,k)}.
\]

由于候选集合使用分层 proposal，这个式子被解释为候选内曝光能量，而不冒充对完整
catalog 的无偏重要性估计。

### 3.3 曝光先验—在线策略重心

令 $p_{\theta,\mathcal C}$ 为在线 RNS 策略 softmax，$p_{\psi,\mathcal C}$ 为冻结曝光
先验。第 $t$ 个联合 epoch 使用 $w_t=1/(t+1)$，求反向 KL 重心：

\[
q_t^*=\arg\min_q
(1-w_t)D_{KL}(q\|p_{\theta,\mathcal C})
+w_tD_{KL}(q\|p_{\psi,\mathcal C}).
\]

**定理 2（加权反向 KL 重心）。** 若两个基础分布在候选集合上严格为正，则唯一解为

\[
q_t^*(j\mid u)\propto
p_{\theta,\mathcal C}(j\mid u)^{1-w_t}
p_{\psi,\mathcal C}(j\mid u)^{w_t}.
\]

**证明。** 加入归一化乘子 $\lambda$，对每个 $q_j$ 求导：

\[
\log q_j+1-(1-w_t)\log p_{\theta,j}-w_t\log p_{\psi,j}+\lambda=0.
\]

指数化并归一化得到结论。目标是 $q$ 的严格凸函数，因此解唯一。∎

候选 softmax 的归一化常数会在乘积中抵消，所以代码直接组合 logits：

\[
b_t(u,j)=(1-w_t)g_\theta(u,j)+w_t a_\psi(u,j).
\]

计数退火不新增超参：可把冻结曝光先验视为一个伪观测，把第 $t$ 轮在线策略视为
$t$ 个适应性观测。第 1 轮 $w_1=1/2$，随后自动把控制权交回在线策略。

### 3.4 曝光校准困难度

令当前 ranker 分数为 $s_\phi(u,j)$。在每行候选内定义

\[
\bar s_u=\frac1{|\mathcal C_u|}\sum_{j\in\mathcal C_u}s_\phi(u,j),
\quad
z(u,j)=\frac{s_\phi(u,j)-\bar s_u}
{\sqrt{|\mathcal C_u|^{-1}\sum_k(s_\phi(u,k)-\bar s_u)^2+10^{-12}}}.
\]

安全困难度为

\[
h_{safe}(u,j)=\sigma(a_\psi(u,j))[z(u,j)]_+.
\]

它有三个直接性质：

1. $0\le h_{safe}\le[z]_+$：曝光证据只能门控难度，不能凭空制造更大难度；
2. 对 ranker 每行分数加常数或乘正数，$h_{safe}$ 不变，避免分数尺度漂移；
3. 低于候选均值的 item 不被额外惩罚，避免“压低容易曝光样本”间接抬高未知假负例。

最终行为分布为

\[
q_{EH,t}(j\mid u)=
\frac{\exp\{b_t(u,j)+h_{safe}(u,j)\}}
{\sum_{k\in\mathcal C_u}\exp\{b_t(u,k)+h_{safe}(u,k)\}}.
\]

**定理 3（安全困难度 Gibbs 最优性）。** 给定基础重心 $q_t^*$，$q_{EH,t}$ 是

\[
\max_q\;\mathbb E_q[h_{safe}]-D_{KL}(q\|q_t^*)
\]

的唯一解。

**证明。** 与定理 2 同样使用归一化乘子，对 $q_j$ 求导得到
$\log q_j=\log q_{t,j}^*+h_j-1-\lambda$。指数化和归一化即为上式；负 KL
使目标严格凹，故解唯一。∎

更一般地令倾斜为 $\eta h$，则

\[
\frac{d}{d\eta}\mathbb E_{q_\eta}[h]
=\operatorname{Var}_{q_\eta}(h)\ge0.
\]

EH-RNS 通过逐行单位 RMS 标准化固定 $\eta=1$，没有用 validation/test 搜索倾斜强度。

### 3.5 联合训练

负样本按 $q_{EH,t}$ 抽取后，ranker 仍使用标准 BPR：

\[
\mathcal L_D=\operatorname{softplus}(-(s_\phi(u,i)-s_\phi(u,j^-)))
+\lambda_D\|\Theta_D\|_2^2.
\]

在线 policy 仍用 RNS 的困难度奖励、精确曝光奖励与多核 MMD 特征奖励做 REINFORCE。
冻结曝光模型不接收联合训练梯度。

采样时的 $h_{safe}$ 在 ranker 更新前计算并缓存；policy 更新时把同一缓存值加回 logits。
由于 policy 和冻结先验在两次计算之间没有更新，REINFORCE 使用的 log-prob 与实际行为
分布完全一致，不会在 ranker step 后用新分数偷偷重算成 off-policy 目标。

## 4. 算法摘要

```text
Input: train clicks D, train exposures E, released BPR-GMF, max epochs T
1. Split E internally into NCE-train and NCE-holdout.
2. Train exposure logit a_psi for five full epochs; restore best holdout-NCE state; freeze psi.
3. Initialize ranker phi and online policy theta from the same released BPR-GMF.
4. For joint epoch t = 1..T:
   a. Set exposure weight w_t = 1 / (t + 1).
   b. For each click (u, i), draw 29 unexposed candidates and 1 exposure anchor.
   c. Compute barycentric logits b_t and cached h_safe using the current ranker.
   d. Sample j- ~ softmax(b_t + h_safe).
   e. Update ranker phi with BPR(u, i, j-).
   f. Update theta with original RNS rewards and the exact cached behavior log-prob.
   g. Evaluate validation; save only validation-AUC best checkpoint.
5. Stop after 11 non-improving validations; evaluate the frozen best checkpoint on test.
6. Without changing the model, run confirmatory paired-user bootstrap for uncertainty.
```

## 5. 实现与正确性证据

核心实现位于：

- `../baselines/reinforcens/src/reinforcens/models.py`：`BarycentricGenerator` 和
  `exposure_calibrated_hard_tilt`；
- `trainer.py`：train-only NCE、候选构造、缓存行为 tilt、RNS 联合训练与早停；
- `checkpoint.py`：ranker、在线 policy、冻结曝光先验和 epoch-dependent weight 的恢复；
- `scripts/ehrns.sh`：完整可复现实验；
- `scripts/paired_bootstrap.py`：冻结模型的同用户配对统计。

回归覆盖包括：

- GMF/MLP 与 legacy NumPy 得分一致；
- KL 重心 logits 与权重正确；
- safe-hard tilt 的平移不变性、常数行退化和曝光门控；
- RNS/EH-RNS smoke training 和 checkpoint round-trip；
- 修改官方 validation/test 文件后，EH-RNS 曝光预训练最终参数逐张量完全相同；
- 配对 bootstrap 的恒等和固定增益情形。

迁移后统一项目的 14 项测试均通过；早期 FlowNS 实现的回归结果已固化在失败记录中。

## 6. 无泄漏与反作弊约束

1. 曝光 NCE 的训练和内部留出都只从 train exposure keys 生成；
2. NCE 负噪声只依赖 train clicks 和 catalog，不读取 validation/test；
3. 联合训练的 click、exact exposure、MMD real exposure 和候选全来自 train；
4. validation 只用于统一的 AUC early stopping；
5. test 不参与方法、epoch、参数或组件选择；越过 validation 门槛后先生成一个冻结点估计，
   再对同一模型做确认性 bootstrap，期间不修改任何模型或决策；
6. 失败候选未越过验证门槛时不读 test，唯一例外 BE-RNS 是其验证先越过门槛后按同一规则读取；
7. EH-RNS 没有使用 test-informed ensemble、checkpoint 拼接或不同指标挑不同 epoch。

## 7. 曝光组件训练充分性

内部留出固定为最多 131,072 个 train exposure，NCE 每个正样本使用 16 个 noise samples。

| Epoch | Train NCE | Holdout NCE | Holdout pairwise AUC |
|---:|---:|---:|---:|
| 1 | 1.133994 | 0.953699 | 0.8992 |
| 2 | 0.929237 | 0.884430 | 0.9040 |
| 3 | 0.859906 | 0.822018 | 0.9084 |
| 4 | 0.802069 | 0.774634 | 0.9123 |
| 5 | 0.758756 | **0.740619** | **0.9155** |

五轮 train/holdout NCE 单调改善，holdout AUC 单调上升。总预训练时间 191.9 秒；最终组件
不是只跑一轮的欠训练占位模型。

## 8. EH-RNS 联合训练轨迹

| Epoch | Validation AUC | Validation NDCG | Hard reward | Exact exposure |
|---:|---:|---:|---:|---:|
| 0 | 0.733783 | 0.655254 | — | — |
| 1 | 0.737480 | 0.662155 | -0.467660 | 0.4451 |
| 3 | 0.743350 | 0.669065 | -0.476011 | 0.4702 |
| 5 | 0.746275 | 0.671013 | -0.487306 | 0.4821 |
| 6 | 0.746732 | **0.671581** | -0.493204 | 0.4864 |
| 8 | 0.747274 | 0.671163 | -0.503770 | 0.4919 |
| 9 | 0.747297 | 0.670816 | -0.508617 | 0.4947 |
| 10 | 0.747218 | 0.670461 | -0.513021 | 0.4961 |
| **11** | **0.747410** | 0.671012 | -0.516880 | 0.4976 |
| 12 | 0.747023 | 0.670028 | -0.521215 | 0.4995 |
| 22 | 0.744072 | 0.665320 | -0.554530 | 0.5109 |

第 11 轮是全局最高 validation AUC；第 22 轮达到 patience 后停止。22 个联合 epoch
总训练时间 2,315.2 秒，最佳轮吞吐约 22,810 examples/s。

## 9. 最终结果

| 方法 | Best epoch | Validation AUC | Validation NDCG | Test AUC | Test NDCG |
|---|---:|---:|---:|---:|---:|
| 完整 RNS | 9 | 0.747364 | 0.670858 | 0.707266 | 0.662166 |
| **EH-RNS** | 11 | **0.747410** | **0.671012** | **0.707490** | **0.662385** |
| EH-RNS − RNS | — | +0.000046 | +0.000154 | **+0.000224** | **+0.000220** |

EH-RNS best checkpoint SHA-256：
`8bb5001676c48f4ea3c46e8c2b496e3d3e7d8825fb782bffe4e17106099d6edc`。

### 同用户配对 bootstrap

对同一 16,015 个 test 用户做 10,000 次有放回配对重采样：

| Metric | Mean delta | 95% percentile CI | 非正尾概率 | Candidate wins / ties |
|---|---:|---:|---:|---:|
| AUC | +0.000224 | **[+0.000070, +0.000375]** | 0.0024 | 8,014 / 469 |
| NDCG | +0.000220 | [-0.000255, +0.000689] | 0.1833 | 8,040 / 114 |

AUC 改善在该用户 bootstrap 下区间为正；NDCG 点估计胜出但区间跨 0。这里报告的是
add-one-smoothed bootstrap 非正尾频率，不将其错误命名为参数假设检验 p-value。

## 10. 失败路线带来的设计收敛

| 路线 | 关键结果 | 淘汰原因 / 对最终方法的贡献 |
|---|---|---|
| FlowFM | 0.694452/0.645819 | 连续端点与离散 item 概率错配 |
| FlowRL | 0.697971/0.652676 | 难度提高但策略熵塌缩；确认需要统一分布目标 |
| GETS | 0.699831/0.653803 | KL tilt 扎实但纯 catalog proposal 曝光覆盖不足 |
| A-GETS | 0.704723/0.658163 | 曝光 anchor 是关键；仍缺 RNS 在线适应 |
| soft-BPR | 0.704460/0.657850 | 假负例软化没有泛化，停止调 soft factor |
| EP-RNS | validation 低于 RNS | 曝光预训练直接覆盖在线 policy，过度曝光 |
| BE-RNS | 0.707426/0.662074 | 固定重心 AUC 略升、NDCG 略降；启发分离先验与 policy |
| CBE-RNS | validation 0.747283/0.671108 | 计数退火改善 NDCG，但未显式增强安全难度 |
| EH-RNS | **0.707490/0.662385** | 计数重心 + 曝光校准困难 tilt 达成双指标点估计胜出 |

完整逐轮失败证据见 `FAILURE_LOG.md`。

## 11. 复现

```bash
cd /home/chen/research/flowns/eh-rns

# 完整基线与最终方法
./scripts/rns.sh
./scripts/ehrns.sh

# 冻结 checkpoint test 评估
../.venv/bin/python ../baselines/reinforcens/train.py evaluate \
  --checkpoint runs/ehrns/best.pt \
  --split test --eval-mode list --list-length 160 \
  --batch-size 1024 --seed 1 --device cuda

# 配对 bootstrap
../.venv/bin/python scripts/paired_bootstrap.py \
  --baseline ../baselines/reinforcens/runs/rns/best.pt \
  --candidate runs/ehrns/best.pt \
  --split test --resamples 10000 --device cuda \
  --output docs/results/eh_rns_paired_bootstrap.json
```

## 12. 限制与声明边界

- 最终训练比较是 seed=1，不是多训练种子均值；结论范围是冻结本地 benchmark。
- 点估计同时超过完整 RNS，但提升约 $2.2\times10^{-4}$，绝对幅度很小。
- NDCG 的用户级配对区间跨 0，不能声称该改善已统计确定。
- 用户 bootstrap 量化 test 用户抽样波动，不包含初始化、训练次序和 GPU 非确定性。
- NCE 后验使用人工等类别先验，是“曝光证据分数”而非线上真实曝光 propensity。
- 候选分布包含经验曝光 anchor；方法定义和理论均明确条件化在有限候选集合上。

因此最准确的最终陈述是：**EH-RNS 在完全匹配的单 seed 本地 ReinforceNS benchmark
上实现 AUC/NDCG 双指标点估计胜出；AUC 的用户配对区间为正，NDCG 的改善仍不确定。**

## 13. 参考工作

- [Reinforced Negative Sampling for Recommendation with Exposure Data](https://www.ijcai.org/proceedings/2019/309)
- [Sampling-Decomposable Generative Adversarial Recommender](https://arxiv.org/abs/2011.00956)
- [Simplify and Robustify Negative Sampling for Implicit Collaborative Filtering](https://proceedings.neurips.cc/paper/2020/hash/0c7119e3a6a2209da6a5b90e5b5b75bd-Abstract.html)
- [On the Theories Behind Hard Negative Sampling for Recommendation](https://arxiv.org/abs/2302.03472)
- [Enhancing Recommender Systems: A Strategy to Mitigate False Negative Impact](https://arxiv.org/abs/2211.13912)
- [Enhanced Bayesian Personalized Ranking for Robust Hard Negative Sampling](https://arxiv.org/abs/2403.19276)
- [Negative Sampling in Recommendation: A Survey and Future Directions](https://arxiv.org/abs/2409.07237)
