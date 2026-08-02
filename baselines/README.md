# Baselines

本目录集中保存可复用的基线项目：

- [`reinforcens/`](reinforcens/)：BPR、DNS、KBGAN、ReinforceNS 及曝光先验对照的
  共享实现、数据和冻结 RNS 结果。

EH-RNS 与 Flow-RNS 均通过 editable dependency 复用该项目，不在各自目录复制 baseline
源码或数据。
