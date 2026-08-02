# 生成式负采样失败记录

> 状态：已封存（2026-08-02）。本文主体记录已经实际运行、且未超过 ReinforceNS 的方案；
> 第 11 节只标记失败迭代的成功终点，详细方法见 `FINAL_REPORT.md`。
> 所有正式比较均使用 Zhihu、发布的 BPR-GMF 初始化和 seed=1。早期筛选统一使用
> 5 个联合训练 epoch；进入 ReinforceNS 路线后统一使用最多 400 epoch、validation AUC
> 选择和 patience=10 早停。test 不参与训练、早停或组件选择。

## 1. 当前门槛

| 方法 | Test AUC | Test NDCG | 结论 |
|---|---:|---:|---|
| 发布 BPR-GMF | 0.691997 | 0.646186 | 初始化 |
| ReinforceNS, 5 epochs | 0.705600 | 0.660613 | 早期不充分门槛 |
| ReinforceNS, 完整早停 | 0.707266 | 0.662166 | 冻结基线 |
| FlowFM | 0.694452 | 0.645819 | 失败 |
| FlowRL | 0.697971 | 0.652676 | 失败 |
| GETS, KL=0.20 | 0.699831 | 0.653803 | 失败，但优于 FlowRL |
| GETS, KL=0.15 | 0.699820 | 0.653782 | 失败，硬度微调无效 |
| A-GETS, 2 anchors | 0.703683 | 0.657370 | 失败，显著逼近 RNS |
| A-GETS, 3 anchors | 0.704723 | 0.658163 | 失败，验证集选中的 A-GETS |
| SA-GETS, soft factor=0.2 | 0.704460 | 0.657850 | 失败，soft-BPR 未泛化 |
| EP-RNS | 未读取 | 未读取 | 验证低于完整 RNS，失败 |
| BE-RNS | 0.707426 | 0.662074 | AUC 略高但 NDCG 略低，严格失败 |
| CBE-RNS | 未读取 | 未读取 | 验证 AUC 低于完整 RNS，失败 |
| **EH-RNS** | **0.707490** | **0.662385** | 双指标点估计超过完整 RNS，迭代终点 |

## 2. FlowFM：连续流拟合曝光特征

### 设计

FlowFM 在固定的协同过滤表示空间中学习条件流：

\[
z_t=(1-t)z_0+t x_E,\qquad
\mathcal L_{FM}=\mathbb E\|v_\theta(z_t,t,u)-(x_E-z_0)\|_2^2.
\]

推断时用四步 Euler ODE 得到连续端点，再按端点到候选 item 特征的距离构造离散策略。
曝光监督只来自 train split 的曝光未点击 item。

### 结果与失败原因

- Test AUC/NDCG 为 0.694452/0.645819，只比初始化 AUC 高 0.002455，NDCG 反而低 0.000367。
- exact exposure 为 15.96%，hard reward 为 -0.7725，明显比 RNS 的 -0.4896 更容易。
- 1 个预训练 epoch 后 FM loss 仍为 0.8888；这不足以证明连续生成组件收敛。
- 连续端点到离散 item 的最近距离解码与 FM 目标不一致：较小的速度回归误差不保证正确的 item 概率。
- 四步 ODE 采样仅为 RNS sampler 吞吐的 23.4%，没有换来精度收益。

结论：淘汰“只做 Flow Matching、不直接控制离散负样本难度”的方案。

## 3. FlowRL：在 FlowFM 上追加策略梯度

### 设计

FlowRL 对同一用户做 4 个 rollout，用冻结的 epoch-start ranker 给出硬度奖励，组内标准化后优化：

\[
\mathcal L=-\lambda_{RL}A\log\pi_\theta
+\lambda_{FM}\mathcal L_{FM}-\lambda_H H(\pi_\theta).
\]

ranker 训练时以 0.5 概率使用 Flow negative，以 0.5 概率使用同候选集均匀负例。

### 结果与失败原因

- Test AUC/NDCG 为 0.697971/0.652676，相对 FlowFM 提升 0.003519/0.006857，
  但仍比 RNS 低 0.007628/0.007937。
- exact exposure 升至 45.00%，hard reward 升至 -0.4375，说明 RL 确实把样本推得更难。
- 策略熵降至 0.0514、unique ratio 降至 0.0911，出现严重模式集中。
- 高方差离散 policy gradient、FM 分布目标、hardness 目标同时作用，三者缺少统一的最优分布解释。
- 总训练时间 686.5 秒，高于 RNS 的 592.9 秒；sampler 吞吐仅为 RNS 的 22.4%。

结论：淘汰“FM + policy gradient”组合；后续方法应直接给出目标采样分布的闭式形式。

## 4. GETS：曝光密度比 + KL 信赖域指数倾斜

### 数学设计

GETS 用 train exposure 与 catalog noise 做等先验二分类。人口最优 logit 满足

\[
a^*(u,j)=\log\frac{p_E(j\mid u)}{r(j\mid u)},
\]

所以对从 \(r\) 抽取的 catalog proposals 使用 \(\exp a_\psi\) 自归一化加权，
即可近似学习到的曝光分布。对每个正例定义 BPR 困难度

\[
h_\phi(u,i,j)=\operatorname{softplus}(s_\phi(u,j)-s_\phi(u,i)).
\]

然后解 KL 信赖域：

\[
\max_q\;\mathbb E_q[h_\phi]
\quad\text{s.t.}\quad
D_{KL}(q\|p_\psi)\le\rho.
\]

KKT 条件给出唯一 Gibbs 形式

\[
q_\beta(j\mid u,i)=
\frac{p_\psi(j\mid u)\exp(\beta h_\phi(u,i,j))}
{\sum_k p_\psi(k\mid u)\exp(\beta h_\phi(u,i,k))},
\]

其中 \(\beta\) 由逐行二分求解 KL 约束。并且

\[
\frac{d}{d\beta}\mathbb E_{q_\beta}[h]
=\operatorname{Var}_{q_\beta}(h)\ge 0,
\]

因此 KL 预算具有可解释、单调的硬度含义。训练使用 64 个 catalog proposals，
KL 预算在 5 个联合 epoch 中线性升至上限。

### 曝光组件充分性

曝光模型只用 train exposure；从 train exposure 内部固定留出 131072 对做组件验证。
改变官方 validation/test 文件不会改变其最终参数，已由回归测试覆盖。

| Exposure epoch | Train NCE | Holdout NCE | Holdout pairwise AUC |
|---:|---:|---:|---:|
| 1 | 1.090642 | 0.899653 | 0.9041 |
| 2 | 0.853015 | 0.815288 | 0.9098 |
| 3 | 0.780693 | 0.758367 | 0.9138 |
| 4 | 0.732421 | 0.722115 | 0.9169 |
| 5 | 0.700820 | 0.698867 | 0.9194 |

五轮始终改善，因此没有用未充分训练的曝光模型下结论。

### GETS, KL 上限 0.20

| Joint epoch | 实际 KL | Validation AUC | Validation NDCG | Exact exposure |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.733783 | 0.655254 | — |
| 1 | 0.0404 | 0.735389 | 0.658147 | 0.1348 |
| 2 | 0.0807 | 0.738790 | 0.661093 | 0.1404 |
| 3 | 0.1208 | 0.739046 | 0.661857 | 0.1447 |
| 4 | 0.1608 | **0.740531** | **0.663314** | 0.1478 |
| 5 | 0.2007 | 0.740114 | 0.663187 | 0.1502 |

best epoch=4 的 test AUC/NDCG 为 0.699831/0.653803。

### GETS, KL 上限 0.15

| Joint epoch | 实际 KL | Validation AUC | Validation NDCG | Exact exposure |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.733783 | 0.655254 | — |
| 1 | 0.0305 | 0.735487 | 0.658172 | 0.1325 |
| 2 | 0.0607 | 0.738679 | 0.661005 | 0.1375 |
| 3 | 0.0907 | 0.739141 | 0.661837 | 0.1412 |
| 4 | 0.1208 | **0.740506** | **0.663275** | 0.1439 |
| 5 | 0.1508 | 0.740265 | 0.663171 | 0.1464 |

best epoch=4 的 test AUC/NDCG 为 0.699820/0.653782。与 KL=0.20 的差异低于
0.00003，说明失败不是简单的硬度上限设置问题，停止继续搜索 KL。

### 失败诊断

- GETS 比 FlowRL test AUC/NDCG 高 0.001860/0.001127，闭式采样与充分曝光预训练有效，
  但仍比 RNS 低 0.005768/0.006810。
- GETS 的 exact exposure 只有约 15%，而 RNS 为 48.43%。曝光密度模型能推广到
  “曝光相似” item，但纯均匀 proposal 在有限的 64 个候选中不能稳定覆盖用户真实曝光区。
- KL 从 0.15 改到 0.20 几乎不改变最终结果；继续微调温度或 KL 不符合有限调参约束。

## 5. A-GETS：经验曝光锚点 + 学习曝光密度

A-GETS 保留充分训练的曝光密度与 KL 闭式 tilt，但把 proposal 改成分层集合：
未曝光 catalog item + 少量 train exposure anchor。它不只从曝光集合训练，
也不把曝光锚点直接当作最终负例；最终选择仍由“学习曝光密度 × 经验曝光锚点 × 困难度”决定。

若 proposal 分布写成

\[
r_\pi=(1-\pi)r_0+\pi\hat p_E,
\]

而密度比为 \(d_\psi\approx p_\psi/r_0\)，则有限候选上的基础权重对应

\[
r_\pi d_\psi
=p_\psi\left[(1-\pi)+\pi\frac{\hat p_E}{r_0}\right].
\]

因此曝光锚点不是无解释的候选注入，而是对“学习分布与经验曝光一致区域”的乘积共识增强。

### 2 个锚点

| Joint epoch | 实际 KL | Validation AUC | Validation NDCG | Exact exposure |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.733783 | 0.655254 | — |
| 1 | 0.0410 | 0.735936 | 0.657837 | 0.3764 |
| 2 | 0.0806 | 0.741669 | 0.664720 | 0.3853 |
| 3 | 0.1207 | 0.741702 | 0.664422 | 0.3932 |
| 4 | 0.1607 | **0.743834** | **0.666672** | 0.3946 |
| 5 | 0.2006 | 0.743702 | 0.666604 | 0.3993 |

验证最佳 epoch=4 的 test AUC/NDCG 为 0.703683/0.657370，比 GETS 提升
0.003852/0.003567，但仍低于旧 RNS 门槛 0.001917/0.003243。

### 3 个锚点

根据 2 锚点首轮的选择质量反推，3/64 预计得到约 47% 的真实曝光命中率，接近
RNS 的约 48%；该选择在测试前固定。实际首轮为 47.38%，与估计一致。

| Joint epoch | 实际 KL | Validation AUC | Validation NDCG | Exact exposure |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.733783 | 0.655254 | — |
| 1 | 0.0411 | 0.734443 | 0.656819 | 0.4738 |
| 2 | 0.0806 | 0.742349 | 0.664854 | 0.4810 |
| 3 | 0.1206 | 0.741839 | 0.664101 | 0.4884 |
| 4 | 0.1607 | **0.744464** | **0.666671** | 0.4888 |
| 5 | 0.2006 | 0.743849 | 0.665027 | 0.4910 |

验证 AUC 比 2 锚点高 0.000630，因此按预先固定规则选择 3 锚点。其 test
AUC/NDCG 为 0.704723/0.658163，继续提升 0.001040/0.000793，但仍未超过 RNS。
这说明经验曝光覆盖是 GETS 的主要缺项，但单独匹配 RNS 的曝光率仍不足以复现其
批级 MMD 奖励带来的排序收益。

## 6. 假负样本软化：EC-GETS 与 SA-GETS

### EC-GETS：淘汰不对称软化

EC-GETS 仅对未被训练曝光认证的负样本使用 soft-BPR：

\[
\ell_\gamma(m)=\log(1+e^{-\gamma m}),\qquad
|\partial\ell_\gamma/\partial m|\le\gamma,
\]

并固定 \(\gamma=0.2\)。首轮采样诊断与 3 锚点 A-GETS 完全相同（exact=0.4738、
KL=0.0411），但 validation AUC/NDCG 从初始 0.733783/0.655254 暴跌到
0.719185/0.640067。原因是认证与未认证样本使用不同梯度尺度，破坏了两类负样本的
优化平衡。该方法在首轮即按明显失败规则停止，未读取 test，代码与大检查点已淘汰。

### SA-GETS：统一 soft-BPR 仍未泛化

SA-GETS 按 PDNS 的原始形式对所有负样本统一使用 \(\gamma=0.2\)。当 \(\gamma>0\)
时损失严格随 margin 下降，仍保持正确的成对排序方向；其梯度标量
\(\gamma\sigma(-\gamma m)\) 压缩不同困难度样本的梯度比。

| Joint epoch | 实际 KL | Validation AUC | Validation NDCG | Exact exposure |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.733783 | 0.655254 | — |
| 1 | 0.0411 | 0.736969 | 0.659751 | 0.4738 |
| 2 | 0.0806 | 0.742723 | 0.664947 | 0.4674 |
| 3 | 0.1199 | 0.742397 | 0.664481 | 0.4745 |
| 4 | 0.1597 | **0.744503** | **0.666341** | 0.4754 |
| 5 | 0.1991 | 0.744064 | 0.665810 | 0.4784 |

其验证 AUC 仅比 A-GETS 高 0.000039，test AUC/NDCG 为 0.704460/0.657850，
反而低 0.000263/0.000313。因此验证集上的微小优势没有泛化，停止 soft factor 搜索，
SA-GETS 代码与大检查点淘汰。

## 7. 完整 ReinforceNS：纠正早期门槛

早期 5 epoch 的 RNS 结果不能代表充分训练基线，因此用完全相同数据、发布初始化、
seed=1、最多 400 epoch 和 patience=10 重跑。第 9 轮达到最佳 validation
AUC/NDCG 0.747364/0.670858；连续 11 次不改善后在第 20 轮停止。冻结第 9 轮检查点后
一次性读取 test，得到 **0.707266/0.662166**。此后所有候选都必须同时超过这两个数，
旧门槛 0.705600/0.660613 不再用于宣称胜出。

## 8. EP-RNS：曝光预训练加速但降低峰值

EP-RNS 不再用密度模型替代 RNS，而是用等先验 NCE 对原 RNS 生成器进行训练曝光
预训练，再保留 RNS 的困难度、精确曝光和多核 MMD 奖励做策略梯度训练。若 NCE
noise 为 \(r\)，人口最优 logit 仍满足

\[
g^*(u,j)=\log p_E(j\mid u)-\log r(j\mid u).
\]

因此联合训练从已学习曝光分布的策略开始，同时不新增 RNS 奖励权重、温度或候选数。
曝光预训练的内部留出只来自 train exposure；修改官方 validation/test 后最终生成器
权重逐位相同，已有回归测试覆盖。

五个完整预训练 epoch 的 train NCE 从 1.133994 降到 0.758756，内部留出 NCE 从
0.953699 降到 0.740619，pairwise AUC 从 0.8992 升到 0.9155。联合训练的最佳点提前到
第 6 轮，validation AUC/NDCG 为 0.747053/0.670439，但仍分别低于完整 RNS
0.000312/0.000419，且 exact exposure 已升至 0.5106。第 17 轮早停后仍未恢复。
按预先规则未读取 test。

失败原因：曝光预训练确实加速了早期收敛，但直接把策略初始化到曝光密度导致过度曝光，
削弱了后续困难度奖励可探索的空间；“更快达到较低峰值”不能算改进。

## 9. BE-RNS：固定曝光—策略反向 KL 重心

为避免 EP-RNS 把在线策略覆盖掉，BE-RNS 保留独立的在线 RNS 策略
\(p_\theta\) 和冻结曝光先验 \(p_E\)，使用等权反向 KL 重心：

\[
q^*=\arg\min_q\frac12D_{KL}(q\|p_\theta)
+\frac12D_{KL}(q\|p_E)
\quad\Longrightarrow\quad
q^*(j\mid u)\propto\sqrt{p_\theta(j\mid u)p_E(j\mid u)}.
\]

只有 \(p_\theta\) 接收策略梯度；\(p_E\) 用与 EP-RNS 相同的五轮充分 NCE 训练后冻结。
第 11 轮取得最佳 validation 0.747378/0.670884，第 22 轮早停。该验证 AUC 比 RNS
高 0.000013，因而按规则冻结并一次性读取 test：0.707426/0.662074。相对完整 RNS，
AUC 提高 0.000160，但 NDCG 降低 0.000092，因此没有达到“两项同时超过”的严格目标。

失败原因：固定 1/2 先验权重在后期仍持续限制在线难例策略；它略改善全局成对排序，
但头部排序微降。这个结果接近门槛，值得尝试结构性退火，但不值得事后用 test 调权重。

## 10. CBE-RNS：无超参计数退火仍未越过验证门槛

CBE-RNS 把固定重心改为由一个曝光伪计数和 \(t\) 个在线观测得到的权重
\(w_t=1/(t+1)\)：

\[
q_t(j\mid u)\propto
p_\theta(j\mid u)^{1-w_t}p_E(j\mid u)^{w_t}.
\]

它不新增可调超参；第 1 轮与 BE-RNS 完全相同，之后自动把控制权交还在线策略。
第 6 轮 validation NDCG 达到 0.671470，高于 RNS 轨迹中的最佳 NDCG；但按统一 AUC
规则选择的最佳点是第 8 轮 0.747283/0.671108，AUC 比完整 RNS 低 0.000081。
第 19 轮早停，未读取 test，大检查点已删除，只保留 config、完整 history 和曝光预训练记录。

失败原因：先验退火改善了验证 NDCG，却没有同时改善总体 AUC；仅改变曝光先验随时间的
权重不足以显式提高安全难例的采样概率。

## 11. 失败迭代终点：EH-RNS

EH-RNS 在 CBE-RNS 的基础分布上增加曝光校准的困难度指数倾斜。令曝光 NCE logit
为 \(a_\psi\)，ranker 候选分数的逐用户正标准化部分为
\(z_+(u,j)=[(s_\phi-\bar s)/\operatorname{rms}(s_\phi-\bar s)]_+\)，定义

\[
h_{safe}(u,j)=\sigma(a_\psi(u,j))z_+(u,j),\qquad
q_{EH}(j\mid u)\propto q_t(j\mid u)\exp h_{safe}(u,j).
\]

等先验 NCE 下 \(\sigma(a_\psi)\) 是曝光后验；正部只增强较难候选，不把容易的高曝光
样本额外压低。根据 Gibbs 变分恒等式，该分布唯一最大化
\(\mathbb E_q[h_{safe}]-D_{KL}(q\|q_t)\)，所以没有新增倾斜系数。实现缓存实际采样时
的 tilt 并原样用于 REINFORCE log-prob，避免 ranker 更新后的 off-policy 重算；最终
baseline 14 项测试通过。

五轮曝光预训练后内部留出 NCE/AUC 为 0.740619/0.9155。EH-RNS 第 11 轮取得最佳
validation AUC/NDCG 0.747410/0.671012，越过预先冻结的 RNS validation 门槛；
第 22 轮早停后冻结第 11 轮 checkpoint，并执行冻结 test 点估计：

| 方法 | Test AUC | Test NDCG |
|---|---:|---:|
| 完整 RNS | 0.707266 | 0.662166 |
| **EH-RNS** | **0.707490** | **0.662385** |
| 差值 | **+0.000224** | **+0.000220** |

点估计后未再改变模型或决策；确认性 10,000 次同用户配对 bootstrap 的 AUC 差值 95% CI 为
[+0.000070, +0.000375]；NDCG 差值 CI 为 [-0.000255, +0.000689]。因此 AUC
改善区间为正，NDCG 点估计胜出但统计不确定。EH-RNS 是单 seed 冻结 benchmark 的
双指标点估计胜出，不被夸大成多 seed 普遍结论。

### 实验基础设施中断记录

CBE-RNS 结束后，unattended upgrade 曾把 NVIDIA 用户态库从 580.159.03 升级到
580.173.02，而已加载内核仍为 580.159.03。CPU 误启动在产生任何训练指标前立即停止。
系统重启后版本一致，但 udev 漏建 `/dev/nvidia*`；运行
`sudo /sbin/ub-device-create --verbose` 后 CUDA 恢复。正式 EH-RNS 全程运行在 RTX 4060
CUDA 上，不包含中断前的 CPU 工作。

## 12. 参考依据

- [Reinforced Negative Sampling for Recommendation with Exposure Data](https://www.ijcai.org/proceedings/2019/309)
- [Sampling-Decomposable Generative Adversarial Recommender](https://arxiv.org/abs/2011.00956)
- [Simplify and Robustify Negative Sampling for Implicit Collaborative Filtering](https://proceedings.neurips.cc/paper/2020/hash/0c7119e3a6a2209da6a5b90e5b5b75bd-Abstract.html)
- [On the Theories Behind Hard Negative Sampling for Recommendation](https://arxiv.org/abs/2302.03472)
- [Enhancing Recommender Systems: A Strategy to Mitigate False Negative Impact](https://arxiv.org/abs/2211.13912)
- [Enhanced Bayesian Personalized Ranking for Robust Hard Negative Sampling](https://arxiv.org/abs/2403.19276)
- [Negative Sampling in Recommendation: A Survey and Future Directions](https://arxiv.org/abs/2409.07237)

上述文献均已从论文官网或 arXiv 找到，但由于本机缺少 `verify_papers.py`，按研究流程
统一标记为 `UNVERIFIED (verification helper unavailable)`；未据此虚构 DOI 或实验数据。
