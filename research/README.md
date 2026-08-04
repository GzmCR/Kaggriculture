# Kaggriculture 研究记录

这里集中保存比赛规则理解、前排 replay 分析、baseline 消融实验和提交版本说明。

- [2026-08-04 前排策略与专家轨迹研究](2026-08-04-top-strategy-research.md)

当前最重要的结论：前排方案的主路线是“高质量完整生产轨迹 + 低频状态路由 + 少量市场/终局修正”，而不是在线强化学习。V012 的替换版已经生成在
`baseline/history/v012_top5_replaced_v18/main.py`，构建脚本是
`experiments/build_v012_submission.py`。
