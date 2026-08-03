# V007：保轨迹式施肥与终局优化

V007 保留根目录当前版本不变，新增三个实验候选：

- `v007a_terminal_safe`：只启用 step 704 之后的本地终局覆盖，以及 step 718 起的强制清仓。
- `v007b_idle_fertilizer`：只允许把 baseline 原本的 `PASS` 替换为当前位置上的 `FERTILIZE`，或在仓库位置拾取 1 个肥料；不加入全局任务队列。
- `v007c_combined`：合并上述两种覆盖。

候选代码位于 `baseline/history/`。它们继续使用 12 只牲畜容量、3 天饲料库存和原有牛羊评分，不使用大模型、外部 API 或 Hamburger 固定动作轨迹。

## 评测

完整矩阵为 4 个版本（含 `current` 对照）× 3 类对手（`starter`、`random`、Hamburger V24）× 6 个 seed × 2 个座位，共 144 局。原始逐局数据在 `raw/`，合并结果由 `experiments/merge_v007_benchmark.py` 生成。

烟雾测试覆盖了 24 局完整 720-step 对局，全部 `DONE`，无 agent 异常和非法 action shape。完整矩阵结果如下：

| 版本 | 平均终局现金 | 相对 current | 最低现金 | 胜率 | Hamburger 平均现金 | p99 最大值 | 晋级 |
|---|---:|---:|---:|---:|---:|---:|---|
| current | 123,918.39 | — | 88,730 | 66.67% | 100,186.42 | 2.43 ms | 对照 |
| v007a_terminal_safe | 123,905.25 | -0.011% | 88,730 | 66.67% | 100,186.42 | 2.24 ms | 否 |
| v007b_idle_fertilizer | 118,918.56 | -4.035% | 82,985 | 66.67% | 95,566.92 | 2.28 ms | 否 |
| v007c_combined | 119,784.97 | -3.336% | 82,985 | 66.67% | 95,566.92 | 2.69 ms | 否 |

四个版本的 144 局均 `DONE`，agent 异常数和非法 action shape 数均为 0。`v007a` 没有带来可测的现金提升；`v007b/c` 虽然分别执行了 83/72 次 `FERTILIZE`，但同时使完整矩阵中的 `WATER`、`PLANT`、`HARVEST` 数量下降，最终现金反而减少。三者均不替换根目录 `main.py`。

逐局结果见 `v007_results.csv`，汇总见 `v007_summary.csv`，门槛判定见 `v007_gate.json`。`field_item_counts` 额外记录了带物品参数的动作；本轮 `PICKUP:FERTILIZER` 为 0，施肥来自原本已携带的肥料。

## 晋级规则

只有满足全部条件的候选才可替换根目录 `main.py`：平均现金至少提升 0.5%、最低现金达到当前版本最低值的 97%、总体胜率不下降、全部对局 DONE、无异常、p99 单回合耗时低于 1000ms，并且对 Hamburger 的平均现金不低于当前版本的 95%。否则保留当前版本。
