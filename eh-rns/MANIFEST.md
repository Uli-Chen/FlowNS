# EH-RNS 专属产物清单

更新时间：2026-08-02。ReinforceNS baseline 已迁移至 `../baselines/reinforcens/`，
Flow-RNS 新一轮研究位于 `../flow-rns/`。

## 本目录内容

- `scripts/ehrns.sh`：EH-RNS 训练入口，调用 baseline 的共享 Trainer。
- `scripts/paired_bootstrap.py`：RNS/EH-RNS 冻结模型的同用户配对 bootstrap。
- `runs/ehrns/best.pt`：最终 EH-RNS checkpoint。
- `runs/ehrns/{config,history}.json`：完整训练协议与曲线。
- `runs/ehrns/generator_pretrain_history.json`：曝光组件 NCE 训练曲线。
- `docs/research/FAILURE_LOG{,_20260802_085130}.md`：方法形成过程的失败记录。
- `docs/research/FINAL_REPORT{,_20260802_085130}.md`：最终方法、推导、协议与结果。
- `docs/results/eh_rns_comparison.{json,md}`：机器可读与精简结果表。
- `docs/results/eh_rns_paired_bootstrap.json`：10,000 次配对 bootstrap。

共享实现、数据、初始化和 RNS checkpoint 的清单见
`../baselines/reinforcens/MANIFEST.md`。

迁移后验收：冻结 EH-RNS checkpoint 使用新 baseline 路径复算 validation，得到
`AUC=0.747410`、`NDCG=0.671012`，与迁移前一致。

## EH-RNS checkpoint

| 文件 | SHA-256 |
|---|---|
| `runs/ehrns/best.pt` | `8bb5001676c48f4ea3c46e8c2b496e3d3e7d8825fb782bffe4e17106099d6edc` |

本目录不保留 baseline 源码副本、数据副本、临时 checkpoint、日志或解释器缓存。
