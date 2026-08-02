# Flow-RNS research

本目录保存新一轮研究：用 conditional Flow Matching 学习 train-only 曝光分布，生成
连续或映射到实际物品的负样本，并在严格门槛通过后才允许加入难度感知 RL。

## 项目边界

- 本目录：Flow 网络、训练/审计、连续负样本、soft transport、Flow-DNS、Hard-BPR、
  研究计划、实验记录与 Flow checkpoint。
- `../baselines/reinforcens/`：数据、发布 BPR-GMF 初始化，以及复用的 `reinforcens`
  数据/模型/评测公共组件。
- `../.venv/`：两个项目共用的 Python 环境。

默认路径：

```text
../baselines/reinforcens/data/zhihu/train-valid-test.zip
../baselines/reinforcens/checkpoints/pretrain_model_dis_zhihu.pkl
../baselines/reinforcens/.cache/zhihu.npz
```

## 安装与测试

```bash
cd /home/chen/research/flowns
uv pip install --python .venv/bin/python \
  -e ./baselines/reinforcens -e ./flow-rns
cd flow-rns
../.venv/bin/python -m pytest -q
```

当前验收：Flow-RNS 的 16 项测试与 ReinforceNS 共享实现的 14 项测试分别通过。

## 实验协议

- 参数训练只使用 `train`；`validation` 只用于 early stopping 和方法推进。
- 方法与超参数冻结前禁止读取新方法的 `test` 指标。
- RNS validation-NDCG-best：`0.671166`。
- Flow-only 解锁 RL 的门槛：validation NDCG `>= 0.721166`。
- RL 只有在门槛通过后才启动，并必须严格超过冻结 Flow-only 模型。

当前 Flow-only 最好结果是 Flow-DNS 29:1 + BPR 的 `0.664666`，尚未超过 RNS，
因此 RL 仍锁定。完整结果和下一步见
[`refine-logs/EXPERIMENT_TRACKER.md`](refine-logs/EXPERIMENT_TRACKER.md)。

## 主要入口

```bash
./scripts/flow_pretrain.sh
./scripts/flow_finetune.sh
./scripts/flow_audit.sh
./scripts/flow_dns.sh
./scripts/flow_dns_hard_bpr.sh
```

代码和保留产物见 [`MANIFEST.md`](MANIFEST.md)。
