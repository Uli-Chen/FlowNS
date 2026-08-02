# EH-RNS 与完整 ReinforceNS

协议：Zhihu、发布 BPR-GMF、seed=1、list length 160、batch size 1024、最多 400 个
联合 epoch、validation AUC 选择、patience=10。test 不参与训练或选择。

| 方法 | Best epoch | Validation AUC | Validation NDCG | Test AUC | Test NDCG |
|---|---:|---:|---:|---:|---:|
| RNS | 9 | 0.747364 | 0.670858 | 0.707266 | 0.662166 |
| **EH-RNS** | 11 | **0.747410** | **0.671012** | **0.707490** | **0.662385** |
| 差值 | — | +0.000046 | +0.000154 | **+0.000224** | **+0.000220** |

同一 16,015 个 test 用户做 10,000 次配对 bootstrap：

- AUC 差值 95% CI `[+0.000070, +0.000375]`，非正尾概率 0.0024；
- NDCG 差值 95% CI `[-0.000255, +0.000689]`，非正尾概率 0.1833。

结论：EH-RNS 在冻结本地 benchmark 上双指标点估计超过完整 RNS；AUC 改善的区间为正，
NDCG 改善尚不能排除用户抽样波动。该结果不是多随机种子均值，不能外推为所有训练种子
都稳定胜出。
