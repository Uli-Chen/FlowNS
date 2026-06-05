# 精炼报告

**问题**: 推荐系统负采样中的硬度-保真度权衡
**初始方法**: Flow Matching + GRPO（reward 设计模糊）
**日期**: 2026-05-19
**轮次**: 2 / 5
**最终得分**: 9.0 / 10
**最终结论**: READY

## Problem Anchor
- **核心问题**: 隐式反馈推荐系统中，不存在同时保证负样本真实性、可控硬度和假负样本防护的数学严密机制
- **必须解决的瓶颈**: 硬度-保真度权衡无形式化解决方案
- **成功标准**: 单一数学严密机制，可证明地生成真实且困难的负样本

## 输出文件
- 评审总结: `refine-logs/REVIEW_SUMMARY.md`
- 最终方案: `refine-logs/FINAL_PROPOSAL.md`

## 分数演进

| 轮次 | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1    | 9                | 9                  | 9                    | 8                 | 9           | 8                | 8               | 8.8     | REVISE  |
| 2    | 9                | 9                  | 9                    | 9                 | 9           | 9                | 9               | 9.0     | READY   |

## 逐轮评审记录

| 轮次 | 主要审稿意见 | 所做修改 | 结果 |
|------|-------------|---------|------|
| 1    | (1) 低维空间 flow matching 必要性未论证 (2) 理论保证与实践脱节风险 (3) 三处可简化 | (1) 添加论证+消融实验 (2) 添加保证紧致性实验 (3) 全部接受 | 全部解决 |
| 2    | 非阻塞性建议（Beta分布联系降级、D-V推导放附录） | N/A（已达 READY） | N/A |

## 最终方案快照
- 完整版: `refine-logs/FINAL_PROPOSAL.md`
- 核心论文:
  1. 从 OPAUC 理论推导 reward: $R = W^a(1-W)^\gamma$，其中 $W$ 为生成负样本对正样本的胜率
  2. 形式化双重假负样本控制: reward shaping（$W=1$ 处零梯度）+ KL anchoring（密度比有界）
  3. 联合假负样本率上界: $\text{FN}(\pi_\theta) \leq \exp(R_{\max}/\beta) \cdot \text{FN}(\pi_{\text{ref}})$
  4. 最优硬度点: $W^* = a/(a+\gamma)$，单超参 $\gamma$ 控制保守程度
  5. 完整 pipeline: Flow Matching → SDE 采样 → GRPO → 联合训练

## 方法演进亮点
1. **最重要的聚焦动作**: 从模糊的"用推荐分数做 reward"到精确的 $R = W^a(1-W)^\gamma$ 含完整数学推导
2. **最重要的机制升级**: 引入胜率 $W$ 作为中间量，建立与 OPAUC 的直接联系
3. **最重要的简化**: 训练 4 阶段→3 阶段，ODE-to-SDE 和 RatioNorm 降级为实现细节

## 推回/漂移日志
| 轮次 | 审稿人说 | 作者回应 | 结果 |
|------|---------|---------|------|
| 1    | 低维空间 flow matching 未必要 | 添加多模态论证 + Claim 4 消融 + hedged framing | 接受（得分 +1） |
| 1    | 理论保证可能脱离实践 | 添加保证紧致性子实验 | 接受（得分 +1） |

## 剩余弱点
1. Flow matching 在 d=64-128 空间的必要性最终依赖实验验证（已设计 Claim 4 消融）
2. 论文写作时需控制数学-新颖性比例（推导放附录）
3. 联合训练阶段的推荐-生成器振荡需工程层面关注

## 审稿原始回复

<details>
<summary>Round 1 Review (8.8/10, REVISE)</summary>

**主要问题:**
- Frontier Leverage (8): 低维嵌入空间为何需要 flow matching？
- Venue Readiness (8): 形式化保证与实际性能脱节风险
- Validation Focus (8): 缺少现代生成式基线

**简化建议:** ODE-to-SDE 降级、RatioNorm 降级、4阶段→3阶段

**漂移警告:** NONE

</details>

<details>
<summary>Round 2 Review (9.0/10, READY)</summary>

**主要评估:**
- Problem Anchor: 已保持（逐字一致）
- 贡献聚焦: 比 Round 1 更锐利（hedged framing 使贡献更稳健）
- 方法简洁性: 比 Round 1 更简单（三项简化已实施）
- 前沿利用: 现已适当

**非阻塞建议:**
1. Beta 分布联系降级为注释
2. Donsker-Varadhan 推导放附录
3. 早期 GRPO 稳定性需关注
4. $W$ 计算可采样正样本子集

</details>

## 下一步
- **READY**: 进入 `/experiment-plan` 制定详细实验路线图，然后 `/run-experiment`
