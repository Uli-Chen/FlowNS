# 实验追踪器

| Run ID | 里程碑 | 用途 | 系统/变体 | 数据集 | 指标 | 优先级 | 状态 | 备注 |
|--------|-------|------|----------|--------|------|-------|------|------|
| R001 | M0 | sanity | RecBole 环境 | ml-100k | - | MUST | TODO | `run_recbole(model='BPR')` |
| R002 | M0 | sanity | 数据下载 | yelp-2018, amazon-books, gowalla-merged | - | MUST | TODO | RecBole 自动下载 |
| R003 | M0 | sanity | Flow model 单测 | synthetic | CFM loss | MUST | TODO | 验证 loss 收敛 |
| R004 | M1 | baseline | LightGCN (Uniform) | yelp-2018 | R@20, N@20 | MUST | TODO | 3 seeds |
| R005 | M1 | baseline | LightGCN (Uniform) | amazon-books | R@20, N@20 | MUST | TODO | 3 seeds |
| R006 | M1 | baseline | LightGCN (Uniform) | gowalla-merged | R@20, N@20 | MUST | TODO | 3 seeds |
| R007 | M1 | baseline | BPR-MF (Uniform) | yelp-2018 | R@20, N@20 | MUST | TODO | 3 seeds |
| R008 | M1 | baseline | BPR-MF (Uniform) | amazon-books | R@20, N@20 | MUST | TODO | 3 seeds |
| R009 | M1 | baseline | BPR-MF (Uniform) | gowalla-merged | R@20, N@20 | MUST | TODO | 3 seeds |
| R010 | M1 | baseline | LightGCN (Pop) | yelp-2018 | R@20, N@20 | MUST | TODO | 3 seeds |
| R011 | M1 | baseline | LightGCN (Pop) | amazon-books | R@20, N@20 | MUST | TODO | 3 seeds |
| R012 | M1 | baseline | LightGCN (Pop) | gowalla-merged | R@20, N@20 | MUST | TODO | 3 seeds |
| R013 | M1 | baseline | LightGCN (DNS) | yelp-2018 | R@20, N@20 | MUST | TODO | dynamic=True, candidate=50 |
| R014 | M1 | baseline | LightGCN (DNS) | amazon-books | R@20, N@20 | MUST | TODO | 3 seeds |
| R015 | M1 | baseline | LightGCN (DNS) | gowalla-merged | R@20, N@20 | MUST | TODO | 3 seeds |
| R016 | M2 | flow-pretrain | Flow Model | yelp-2018 | CFM loss, 生成质量 | MUST | TODO | 需要 R004 嵌入 |
| R017 | M2 | flow-pretrain | Flow Model | amazon-books | CFM loss | MUST | TODO | 需要 R005 嵌入 |
| R018 | M2 | flow-pretrain | Flow Model | gowalla-merged | CFM loss | MUST | TODO | 需要 R006 嵌入 |
| R019 | M3 | main | FlowNS | yelp-2018 | R@20, N@20, W, FN | MUST | TODO | seed=2020 |
| R020 | M3 | main | FlowNS | yelp-2018 | R@20, N@20 | MUST | TODO | seed=2021 |
| R021 | M3 | main | FlowNS | yelp-2018 | R@20, N@20 | MUST | TODO | seed=2022 |
| R022 | M3 | main | FlowNS | amazon-books | R@20, N@20 | MUST | TODO | 3 seeds |
| R023 | M3 | main | FlowNS | amazon-books | R@20, N@20 | MUST | TODO | seed=2021 |
| R024 | M3 | main | FlowNS | amazon-books | R@20, N@20 | MUST | TODO | seed=2022 |
| R025 | M3 | main | FlowNS | gowalla-merged | R@20, N@20 | MUST | TODO | seed=2020 |
| R026 | M3 | main | FlowNS | gowalla-merged | R@20, N@20 | MUST | TODO | seed=2021 |
| R027 | M3 | main | FlowNS | gowalla-merged | R@20, N@20 | MUST | TODO | seed=2022 |
| R028 | M4 | ablation-B2 | FlowNS-Unshaped (R=W) | yelp-2018 | R@20, N@20, FN | MUST | TODO | 3 seeds |
| R029 | M4 | ablation-B2 | FlowNS-Unshaped | yelp-2018 | R@20, N@20, FN | MUST | TODO | seed=2021 |
| R030 | M4 | ablation-B2 | FlowNS-Unshaped | yelp-2018 | R@20, N@20, FN | MUST | TODO | seed=2022 |
| R031 | M4 | ablation-B2 | FlowNS-NoRL | yelp-2018 | R@20, N@20, W dist | MUST | TODO | 3 seeds |
| R032 | M4 | ablation-B2 | FlowNS-NoRL | yelp-2018 | R@20, N@20 | MUST | TODO | seed=2021 |
| R033 | M4 | ablation-B2 | FlowNS-NoRL | yelp-2018 | R@20, N@20 | MUST | TODO | seed=2022 |
| R034 | M4 | ablation-B2 | FlowNS-RawScore | yelp-2018 | R@20, N@20, FN | MUST | TODO | 3 seeds |
| R035 | M4 | ablation-B2 | FlowNS-RawScore | yelp-2018 | R@20, N@20, FN | MUST | TODO | seed=2021 |
| R036 | M4 | ablation-B2 | FlowNS-RawScore | yelp-2018 | R@20, N@20, FN | MUST | TODO | seed=2022 |
| R037 | M4 | ablation-B3 | FlowNS β=0 | yelp-2018 | R@20, N@20, FN, entropy | MUST | TODO | 1 seed |
| R038 | M4 | ablation-B3 | FlowNS β=0.01 | yelp-2018 | R@20, N@20, FN | MUST | TODO | 1 seed |
| R039 | M4 | ablation-B3 | FlowNS β=0.05 | yelp-2018 | R@20, N@20, FN | MUST | TODO | 1 seed |
| R040 | M4 | ablation-B3 | FlowNS β=0.5 | yelp-2018 | R@20, N@20, FN | MUST | TODO | 1 seed |
| R041 | M4 | ablation-B3 | FlowNS β=1.0 | yelp-2018 | R@20, N@20, FN | MUST | TODO | 1 seed |
| R042 | M4 | ablation-B4 | Cond-GMM + shaped reward | yelp-2018 | R@20, N@20, W dist | MUST | TODO | K=10, 3 seeds |
| R043 | M4 | ablation-B4 | Cond-CVAE + shaped reward | yelp-2018 | R@20, N@20, W dist | MUST | TODO | latent=32, 3 seeds |
| R044 | M4 | ablation-B4 | MLP-Gen + REINFORCE | yelp-2018 | R@20, N@20, W dist | MUST | TODO | 参数匹配, 3 seeds |
| R045 | M4 | ablation-B4 | (预留) | - | - | - | TODO | - |
| R046 | M5 | baseline | MixGCF | yelp-2018 | R@20, N@20 | MUST | TODO | 自行实现 |
| R047 | M5 | baseline | MixGCF | amazon-books | R@20, N@20 | MUST | TODO | |
| R048 | M5 | baseline | MixGCF | gowalla-merged | R@20, N@20 | MUST | TODO | |
| R049 | M5 | baseline | AdvInfoNCE | yelp-2018 | R@20, N@20 | MUST | TODO | 自行实现 |
| R050 | M5 | baseline | AdvInfoNCE | amazon-books | R@20, N@20 | MUST | TODO | |
| R051 | M5 | baseline | AdvInfoNCE | gowalla-merged | R@20, N@20 | MUST | TODO | |
| R052 | M5 | baseline | CVAE-NS (baseline) | yelp-2018 | R@20, N@20 | MUST | TODO | 无 shaped reward |
| R053 | M5 | baseline | CVAE-NS (baseline) | amazon-books | R@20, N@20 | MUST | TODO | |
| R054 | M5 | baseline | CVAE-NS (baseline) | gowalla-merged | R@20, N@20 | MUST | TODO | |
| R055 | M6 | theory | 保证紧致性 | yelp-2018 | FN rate vs bound | MUST | TODO | 需 R037-R041 结果 |
| R056 | M6 | theory | 保证紧致性 | amazon-books | FN rate vs bound | MUST | TODO | |
| R057 | M6 | sensitivity | γ=0.5 | yelp-2018 | R@20, N@20, FN | NICE | TODO | |
| R058 | M6 | sensitivity | γ=2 | yelp-2018 | R@20, N@20, FN | NICE | TODO | |
| R059 | M6 | sensitivity | γ=3 | yelp-2018 | R@20, N@20, FN | NICE | TODO | |
| R060 | M6 | sensitivity | γ=5 | yelp-2018 | R@20, N@20, FN | NICE | TODO | |
| R061 | M6 | sensitivity | γ sensitivity on Amazon | amazon-books | R@20, N@20 | NICE | TODO | |
| R062 | M6 | viz | W 分布随 epoch 演变 | yelp-2018 | W histogram per epoch | NICE | TODO | |
| R063 | M6 | viz | 训练曲线 | yelp-2018 | R, KL, Recall vs epoch | NICE | TODO | |
| R064 | M6 | robustness | FlowNS + BPR-MF | yelp-2018 | R@20, N@20 | NICE | TODO | |
| R065 | M6 | robustness | FlowNS + NGCF | yelp-2018 | R@20, N@20 | NICE | TODO | |
