# ReinforceNS baseline 清单

更新时间：2026-08-02；本项目位于工作区 `baselines/reinforcens/`。

## 代码与入口

- `src/reinforcens/{models,trainer,config,checkpoint}.py`：模型、训练协议和 checkpoint。
- `src/reinforcens/{data,sampling,metrics,statistics}.py`：数据、负采样、评测和统计。
- `src/reinforcens/{cli,compare}.py`、`train.py`、`compare.py`：命令行入口。
- `scripts/{bpr,dns,kbgan,rns,itempop}.sh`：主要 baseline。
- `scripts/{eprns,berns,cberns}.sh`：曝光先验对照。

共享实现保留 EH-RNS 所需的扩展分支，但 EH-RNS 专属脚本、模型产物和报告仅位于
`../../eh-rns/`。

## 数据、缓存与冻结结果

- `data/zhihu/train-valid-test.zip`
- `.cache/zhihu.npz`（可重建）
- `checkpoints/pretrain_model_dis_zhihu.pkl`
- `runs/rns/{best.pt,config.json,history.json}`

## 测试

- `tests/test_data.py`
- `tests/test_models_metrics.py`
- `tests/test_training.py`
- `tests/test_statistics.py`

当前验收：14 项测试全部通过。

## SHA-256

| 文件 | SHA-256 |
|---|---|
| `checkpoints/pretrain_model_dis_zhihu.pkl` | `09c0f20c782f96a6383b03d75b753c85709d07a7c7d81214049e1453bb705075` |
| `runs/rns/best.pt` | `9f8c2d826b70ba8a6180942af63a42c71b2ebf6fac699f7e3964b2138beb56b8` |
