# 可复现实验结果

- `eh_rns_comparison.json` / `.md`：最终 RNS 与 EH-RNS 对比及协议。
- `eh_rns_paired_bootstrap.json`：10,000 次同用户配对 bootstrap 原始输出。

最终 checkpoint 只保留在 `runs/{rns,ehrns}/best.pt`；失败方法的中间运行目录不再
重复保留，其指标与淘汰原因已经固化到失败记录。

复现命令见项目 [README](../../README.md)。所有最终模型选择只使用 validation AUC；
候选方法越过冻结验证门槛后才生成 test 点估计；确认性 bootstrap 在模型与所有决策冻结后执行。
