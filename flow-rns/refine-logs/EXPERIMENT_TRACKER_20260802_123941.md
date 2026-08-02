# 新一轮 Flow-RNS 实验跟踪

生成时间：2026-08-02 12:39:41 +08:00。

## 冻结协议

- 数据：仅 `train` 用于参数训练；仅 `validation` 用于 early stopping 和方法推进；在方法与超参数冻结前禁止读取 `test`。
- 统一列表评测：list length 160，evaluation seed 1。
- 参考 RNS：validation NDCG 峰值 `0.671166`（epoch 7）；最终冻结 AUC-best checkpoint 的 validation NDCG 为 `0.670858`。
- Flow-only 进入 RL 的预注册门槛：validation NDCG `>= 0.721166`，即相对 RNS 的 validation-NDCG-best 绝对提高 `>= 0.05`。
- RL 最终门槛：同协议 validation NDCG 严格超过冻结的 Flow-only 模型；满足后才允许一次冻结 test 确认。

## 实验队列

| ID | 方法 | 目的 | 状态 |
|---|---|---|---|
| P0 | Oracle exposure BPR | 检验真实 train exposure 负样本是否能达到 Flow 门槛，为学习 exposure 分布给出经验上界 | completed: negative |
| P1 | Direct continuous conditional flow | 大容量 user-conditioned flow 直接生成连续负 embedding | completed: negative |
| P2 | Soft transport projection flow | 用 top-k 熵正则、同用户 train exposure 支持集降低 continuous-to-catalog gap | completed: negative |
| P3 | Flow-DNS 与鲁棒困难负样本损失 | 将 Flow exposure proposal 注入 29:1 DNS，并检验 Hard-BPR | completed: ordinary BPR best, below gate |
| R1 | Flow policy + group-relative RL | 仅在 P1/P2 达到 `0.721166` 后启动 | gated |

## P1 实现验收

- 连续 Flow ranker 已实现：Flow 端点直接作为 32 维负 embedding 进入 GMF BPR，训练
  过程不执行 catalog ID 映射。
- 发布模型的 item table 被强制冻结；仅更新 user embedding 与 GMF 打分向量，避免
  Flow 坐标系和正样本坐标漂移。另保留同样冻结 item table 的 uniform 对照入口。
- 每用户生成可重复的有限候选池，按固定周期刷新；最近 catalog 距离、最近 train-click
  率、最近 train-exposure 率只作 gap 诊断，不参与连续分支训练。
- 连续 Flow pool 已加入可选 safe-hard 倾斜：困难度由当前 ranker 标准化分数给出，
  train-only 点击/曝光中心距离形成安全支持度；逐用户二分约束倾斜后 ESS 下限，并记录
  相对均匀 Flow pool 的 KL、实际 ESS、选中难度和安全支持度。P1 仍先运行 random
  连续分支，只有失败后才启用该备选。
- 全套测试更新为 30 项并全部通过。
- 文献检索补入 KDD 2025 FlowCF：FM 用于协同过滤本身不构成新颖性；本轮候选贡献
  已收缩为曝光非点击负采样、连续 BPR、安全困难度约束和门控后的 Flow RL 组合。
- 本机缺少论文机器校验脚本与外部交叉模型评审接口；引用保持 machine-unverified，
  novelty/quality jury 不标记为通过。

## P0 配置

- 初始化：发布的 BPR-GMF checkpoint。
- negatives：每个 train click 从该用户的 train-only non-click exposures 采 1 个。
- optimizer：Adam，learning rate `1e-3`，batch size `1024`。
- 最多 50 epochs；validation NDCG early stopping，patience 5。
- GPU：本机 RTX 4060 Laptop 8 GB；预估低于 1 GPU-hour。

## P0 结果

- 10 epochs 后 early stop；总 GPU 训练时间约 320 秒。
- 最佳 epoch：4。
- 最佳 validation AUC/NDCG：`0.744288 / 0.665192`。
- 相对 RNS validation-NDCG-best：`-0.005974`；距 Flow gate：`-0.055974`。
- 结论：原始 train exposure 分布不是足够好的负样本目标。后续 Flow 仍需按用户要求
  充分拟合曝光分布，但下游采样必须显式加入可控难度与 false-negative 安全约束；禁止
  用组件 FM loss 下降替代 ranker NDCG 证据。

## P1 Flow 预训练配置

- 目标：train-only exposed-nonclick item embedding；train 内部固定 holdout 131,072 对。
- 网络：8 个 FiLM residual blocks，width 512，user/time condition 各 128 维，
  `10,501,536` 个可训练参数。
- 目标：rectified conditional FM velocity MSE + `0.25` endpoint reconstruction MSE。
- optimizer：AdamW，learning rate `2e-4`，batch 2048，AMP；最多 50 个完整
  6.55M-exposure epochs，至少 10 epochs。
- 充分性停止：train-only holdout loss 相对累计改善不足 0.5% 持续 5 epochs，且同时
  记录 endpoint MSE、sliced-Wasserstein、均值/方差误差与 16-step Heun 端点。
- 32,768-example profile：52,606 exposures/s，峰值分配显存 364 MiB；估计每个
  全量 epoch 约 125 秒。

## P1 Flow 第一阶段结果与审计

- 第一阶段完成 50 个全量 epochs（327.6M exposure-pair visits）；上限结束时仍刷新
  holdout，故不把它直接标作收敛。
- epoch 50 train/holdout loss：`0.903940 / 0.913175`；相对 epoch 1 holdout
  `1.206928` 下降 24.34%，train–holdout gap `0.009235`。
- 固定 train-only holdout 审计（4,096 对）：conditional C2ST validation/test accuracy
  `0.5018 / 0.5244`；32-step target SWD `0.05118`。
- solver：16→32 step endpoint RMSE `0.02339`，target SWD `0.05212→0.05118`；16-step
  推理误差可接受。4→32 和 8→32 RMSE 分别 `0.19692 / 0.07785`。
- 32-step 端点最近 catalog 标准化距离 mean/p95 `1.571 / 4.444`；最近项属于该用户
  train exposure/click 的比例 `17.58% / 5.08%`，唯一物品率 `95.02%`。这证明直接
  连续训练需要显式监控 false-negative/gap，不能假设 Flow 输出天然落在安全 catalog 上。
- 因 epoch 50 仍改善，现从其最佳权重以新 AdamW、learning rate `5e-5` 做最多 20 个
  全量 fine-tune epochs；train-only 材料性改善阈值收紧为 0.2%，patience 5。官方
  validation/test 仍未读取。

## P1 Flow 第二阶段结果与最终审计

- 从第一阶段 epoch-50 最佳权重 warm-start；使用全新 AdamW、learning rate `5e-5`，
  训练至 epoch 19 后按 train-only holdout 的 0.2% 材料性改善阈值、patience 5 自然早停。
- epoch 19 最佳 train/holdout loss：`0.859113 / 0.869951`。最终 conditional C2ST
  validation/test accuracy `0.50305 / 0.51341`，32-step target SWD `0.05172`。
- 16-step target SWD `0.05313`，16→32 endpoint RMSE `0.03301`；继续用 16-step Heun
  作为下游计算/精度折中。
- 最近 catalog 标准化距离 mean/p95 `1.205 / 4.302`；最近项属于同用户 train
  exposure/click 的比例 `18.07% / 6.25%`，唯一物品率 `94.53%`。Flow 已充分训练，
  但 catalog gap 与 false-negative 风险依旧存在。

## Flow-only 排序结果

所有结果只用官方 validation 做 early stopping；`official_test_used=false`，未读取新方法
test 指标。

| 分支 | item table | 最佳 epoch | validation NDCG | 相对 RNS-best | 距 Flow gate |
|---|---|---:|---:|---:|---:|
| direct continuous | frozen | 0 | `0.655254` | `-0.015912` | `-0.065912` |
| uniform control | frozen | 0 | `0.655254` | `-0.015912` | `-0.065912` |
| soft transport | frozen | 0 | `0.655254` | `-0.015912` | `-0.065912` |
| soft safe-hard | frozen | 0 | `0.655254` | `-0.015912` | `-0.065912` |
| uniform control | trainable | 8 | `0.655300` | `-0.015866` | `-0.065866` |
| soft transport | trainable | 0 | `0.655254` | `-0.015912` | `-0.065912` |
| soft safe-hard | trainable | 0 | `0.655254` | `-0.015912` | `-0.065912` |
| Flow-DNS 29:1 + BPR | trainable | 17 | **`0.664666`** | `-0.006500` | `-0.056500` |
| Flow-DNS 29:1 + Hard-BPR | trainable | 22 | `0.662292` | `-0.008874` | `-0.058874` |

- Soft transport 每次将 256,240 个 Flow endpoints 映射到同用户 train exposure 支持集；
  top-8 exposure-backed rate `98.19%`，ESS ratio 约 `0.557`，加权标准化 gap 约 `0.621`。
  映射本身稳定，但直接/soft BPR 均使 ranker 退化。
- Flow-DNS 中 Flow proposal 仅占 30 个候选的 `3.33%`，却被最高分 DNS 选中
  `42%–51%`，说明 Flow 学到了系统性极难 exposure proposals。普通 BPR 是目前唯一明显
  朝正确方向移动的 Flow-only 分支，但仍低于 RNS，更未达到 `0.721166`。
- Hard-BPR 使用论文示例固定参数 `(a,b,c)=(1,-1,0.8)`，不做 validation 网格搜索；
  28 epochs、patience 6 自然早停。其最好值比普通 BPR Flow-DNS 低 `0.002374`，因此
  “只改变损失以削弱极端假负样本梯度”被否定。
- RL 门槛未通过，R1 继续锁定；禁止用 Flow 拟合 loss、AUC 或低于门槛的 NDCG 解锁。

## 下一个有界机制实验

- 不调 Hard-BPR 参数。对同一个固定 Hard-BPR，候选效用改为其 pairwise loss 对 margin
  的解析梯度幅值：`u(x)=c*sigma(z)*(1-sigma(z))/(sigma(z)+a)`，其中
  `z=c*x+b`、`x=s_pos-s_neg`。
- 当 `a>0` 时该效用在极易和极难两端都趋近 0；固定 `(1,-1,0.8)` 的最优目标 margin
  由解析式给出，不新增温度或 validation 超参数。对 29 个 uniform + 1 个 Flow proposal
  选择最大 `u(x)`，检验“极端 DNS 选择”是否是 Hard-BPR 未奏效的原因。
