# Flow-RNS 研究产物清单

更新时间：2026-08-02；公共 ReinforceNS 依赖位于 `../baselines/reinforcens/`。

## 代码

- `src/flowrns/flow.py`：10.5M 参数 user-conditioned exposure flow 与 Heun/Euler solver。
- `src/flowrns/flow_training.py`：train-only holdout、充分训练规则和分布诊断。
- `src/flowrns/flow_audit.py`：solver 收敛、conditional C2ST 与 catalog-gap 审计。
- `src/flowrns/flow_ranker.py`：连续 BPR、soft transport、safe-hard、Flow-DNS 与 Hard-BPR。
- `src/flowrns/flow_cli.py`、`flow.py`：训练、检查、审计和排序入口。
- `scripts/flow_*.sh`：所有已执行或待执行的固定实验命令。
- `tests/test_flow.py`、`tests/flow_test_data.py`：Flow-RNS 单元与集成测试。

当前验收：16 项 Flow-RNS 测试全部通过。

## 研究记录

- `idea-stage/IDEA_REPORT{,_20260802_130142}.md`：文献、数学方案、门槛和风险。
- `refine-logs/EXPERIMENT_TRACKER{,_20260802_123941}.md`：冻结协议与逐实验结果。
- `.aris/verify-papers/`：论文候选及 machine-unverified 审计状态。

## 实验产物

- `experiments/flow_rns/p0_oracle_exposure/`：真实 train exposure 上界实验。
- `experiments/flow_rns/p1_flow_pretrain_w512_d8_seed1/`：50-epoch 大 Flow 第一阶段。
- `experiments/flow_rns/p1_flow_finetune_lr5e5_seed1/`：自然早停的第二阶段 Flow。
- `experiments/flow_rns/p1_*`、`p2_*`：连续、uniform、soft transport 与 safe-hard 对照。
- `experiments/flow_rns/p3_flow_dns_29_1_seed1/`：当前最佳 Flow-only 排序分支。
- `experiments/flow_rns/p3_flow_dns_hard_bpr_seed1/`：固定论文参数的鲁棒损失实验。

失败分支只保留配置、完整曲线和摘要；两阶段 Flow、普通 Flow-DNS 与 Hard-BPR 分支
保留最佳 checkpoint。所有新方法产物均声明 `official_test_used=false`。

## 保留 checkpoint 与 SHA-256

| 文件 | SHA-256 |
|---|---|
| `experiments/flow_rns/p1_flow_pretrain_w512_d8_seed1/best.pt` | `efebecc7faf42ef6d9d4ce18aca316715f6d9c46accd110c9bd0634c3555fe23` |
| `experiments/flow_rns/p1_flow_finetune_lr5e5_seed1/best.pt` | `2cefdbdc7281fa13afa5fd22c2e3e97a24561349a67257bbcc754ed70c11575c` |
| `experiments/flow_rns/p3_flow_dns_29_1_seed1/best.pt` | `5e62ec7303eff332c59757b34f87b64baa996b77000fce681be391830bacada6` |
| `experiments/flow_rns/p3_flow_dns_hard_bpr_seed1/best.pt` | `facc16ee694fd6513394cbe57bd04d57486231fb0d216db3c930a51a05b05d2a` |

`.cache/`、`__pycache__/`、`.pytest_cache/`、`*.log` 和重复 `last.pt` 不属于保留产物。
