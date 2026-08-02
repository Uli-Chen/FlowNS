# ReinforceNS research workspace

工作区按 baseline、稳定方法和新一轮研究分开保存：

```text
flowns/
├── baselines/
│   └── reinforcens/  # ReinforceNS 公共实现、数据、基线脚本与 RNS 结果
├── eh-rns/           # EH-RNS 专属脚本、模型、曲线、统计结果和报告
├── flow-rns/         # Flow Matching 负采样代码与本轮实验产物
└── .venv/            # 三者共用的 Python 环境
```

安装共享实现与 Flow-RNS：

```bash
cd /home/chen/research/flowns
uv pip install --python .venv/bin/python \
  -e ./baselines/reinforcens -e ./flow-rns
```

入口文档：

- [`baselines/reinforcens/README.md`](baselines/reinforcens/README.md)
- [`eh-rns/README.md`](eh-rns/README.md)
- [`flow-rns/README.md`](flow-rns/README.md)
