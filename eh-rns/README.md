# EH-RNS

本目录只保存最终方法 EH-RNS 的专属复现脚本、冻结模型、训练曲线、统计结果与研究报告。
ReinforceNS 公共实现、数据和 RNS baseline 已迁移到
[`../baselines/reinforcens/`](../baselines/reinforcens/)。

EH-RNS 包含三个部分：

1. 用 train exposure 与 uniform catalog noise 做等先验 NCE，学习曝光密度比；
2. 用 `w_t=1/(t+1)` 的反向 KL 重心结合冻结曝光先验和在线 RNS 策略；
3. 用 `sigmoid(exposure_logit) × positive_standardized_ranker_hardness` 对候选分布做
   无额外强度超参的 Gibbs 倾斜。

## 依赖与训练

共用实现位于 `../baselines/reinforcens/src/reinforcens/`，Python 环境位于 `../.venv/`。

```bash
cd /home/chen/research/flowns
uv pip install --python .venv/bin/python -e ./baselines/reinforcens
cd eh-rns
./scripts/ehrns.sh
```

训练脚本使用 baseline 目录中的数据和发布 BPR-GMF 初始化，输出写入本目录
`runs/ehrns/`。

## 冻结评测与配对统计

```bash
../.venv/bin/python ../baselines/reinforcens/train.py evaluate \
  --checkpoint runs/ehrns/best.pt \
  --split test --eval-mode list --list-length 160 --seed 1 --device cuda

../.venv/bin/python scripts/paired_bootstrap.py \
  --baseline ../baselines/reinforcens/runs/rns/best.pt \
  --candidate runs/ehrns/best.pt \
  --split test --resamples 10000 --device cuda \
  --output docs/results/eh_rns_paired_bootstrap.json
```

## 结果

| Model | Best validation AUC/NDCG | Test AUC/NDCG |
|---|---:|---:|
| RNS | 0.747364 / 0.670858 | 0.707266 / 0.662166 |
| EH-RNS | **0.747410 / 0.671012** | **0.707490 / 0.662385** |

详细协议、推导与限制见
[`docs/research/FINAL_REPORT.md`](docs/research/FINAL_REPORT.md)。
