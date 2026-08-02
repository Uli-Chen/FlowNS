# ReinforceNS baselines

本目录是共享的 recommendation baseline 项目，包含 BPR、DNS、KBGAN、ReinforceNS
及其曝光先验变体的 PyTorch 实现。EH-RNS 和 Flow-RNS 都依赖这里的数据、GMF、采样、
checkpoint 与评测接口；EH-RNS 所需的扩展分支也在同一 Trainer 中维护，以避免复制。

## 目录

- `src/reinforcens/`：公共实现。
- `data/zhihu/`：原始数据归档。
- `checkpoints/`：发布的 BPR-GMF 初始化。
- `.cache/zhihu.npz`：共享解析缓存，可从原始数据重建。
- `scripts/`：BPR、DNS、KBGAN、RNS 和曝光先验 baseline 入口。
- `runs/rns/`：冻结 RNS checkpoint、配置与完整曲线。
- `tests/`：共享实现的单元与集成测试。

## 安装与验证

```bash
cd /home/chen/research/flowns
uv pip install --python .venv/bin/python -e ./baselines/reinforcens
cd baselines/reinforcens
../../.venv/bin/python -m pytest -q
```

当前验收：14 项共享实现测试全部通过。

运行 RNS：

```bash
./scripts/rns.sh
```

当前冻结 RNS validation NDCG-best 为 `0.671166`；AUC-best checkpoint 的 validation
NDCG 为 `0.670858`。
