# Flowns

最小化推荐系统训练项目，只保留两类任务：

- `mf`: general recommender，使用 BPR 矩阵分解。
- `sasrec`: sequential recommender，使用 SASRec 风格 Transformer。

项目改为通过 `uv` 管理环境，通过 `wandb` 记录实验。默认 `wandb.enabled: false`，不会自动联网或要求登录。

## 安装

```bash
uv sync --group dev
```

如果当前沙箱或机器不能写入默认 uv 缓存，可以把缓存放在项目内：

```bash
UV_CACHE_DIR=.uv-cache uv sync --group dev
```

## 训练

```bash
uv run flowns train --config configs/mf_ml100k.yaml
uv run flowns train --config configs/sasrec_ml100k.yaml
```

启用离线 wandb：

```bash
uv run flowns train --config configs/mf_ml100k.yaml --wandb offline
```

启用在线 wandb：

```bash
uv run flowns train --config configs/mf_ml100k.yaml --wandb online
```

## 数据格式

当前数据加载器兼容 RecBole atomic `.inter` 文件，至少需要：

```text
user_id:token	item_id:token	rating:float	timestamp:float
u1	i1	1	1
```

配置里可以使用数据集目录：

```yaml
data:
  data_path: dataset
  dataset: ml-100k
```

也可以直接指定文件：

```yaml
data:
  inter_file: tests/fixtures/ml-mini/ml-mini.inter
```

## 测试

```bash
uv run pytest
```
