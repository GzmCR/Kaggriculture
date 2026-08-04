# V006：借鉴 Hamburger 的通用机制实验

## 目的

本实验保留根目录 `main.py` 作为当前控制组，只把 Hamburger 的可解释机制抽象成动态规则：

- premium crop 的条件式 `FERTILIZE` 任务和自动 `PICKUP FERTILIZER`；
- step 704 之后的空闲单位终局清算动作；
- step 710 之后按预期收入重排并强制出售库存；
- 13 个牲畜容量、4 天小麦饲料缓冲和原有动态牛羊评分。

Hamburger notebook 中的 720 回合固定轨迹只在 benchmark 中作为 `hamburger` 对手载入，没有复制到任何候选代码中。

## 文件

- `baseline/history/v006a_fertilizer_terminal/main.py`：施肥 + 终局清算。
- `baseline/history/v006b_livestock_wheat/main.py`：13 只牲畜 + 4 天小麦库存。
- `baseline/history/v006c_combined/main.py`：两者合并。
- `experiments/run_v006_benchmark.py`：逐局评测和运行时/动作统计。
- `experiments/merge_v006_benchmark.py`：合并并汇总分批结果。
- `v006_results.csv`：144 场逐局结果。
- `v006_summary.csv`：按候选/对手汇总结果。
- `v006_ablation_summary.csv`：仅扩容、仅增加小麦缓冲的诊断消融。
- `v006_gate.json`：按计划阈值自动判定的晋级报告。
- `raw/`：每个候选的原始 36 场结果。

## 评测矩阵

每个策略运行 `6 seeds × 2 seats × 3 opponents = 36` 场完整 720 steps：

`seeds = 17, 42, 2026, 217, 317, 733`

对手为 `starter`、`random`、Hamburger V24。当前版本也作为控制组运行，因此总计 144 场。

## 完整结果

| 策略 | 对手 | 平均终局现金 | 最低现金 | 胜/平/负 | 全部 DONE | p99 单回合最大值 |
|---|---:|---:|---:|---:|---:|---:|
| current | starter | 136,735.7 | 132,023 | 12/0/0 | 是 | 1.71 ms |
| current | random | 133,979.0 | 124,981 | 12/0/0 | 是 | 2.10 ms |
| current | Hamburger | 100,186.4 | 88,730 | 0/0/12 | 是 | 2.14 ms |
| v006a | starter | 93,578.8 | 88,814 | 12/0/0 | 是 | 1.67 ms |
| v006a | random | 93,716.0 | 81,295 | 12/0/0 | 是 | 1.88 ms |
| v006a | Hamburger | 73,107.3 | 65,775 | 0/0/12 | 是 | 1.97 ms |
| v006b | starter | 136,423.3 | 126,299 | 12/0/0 | 是 | 1.68 ms |
| v006b | random | 131,879.1 | 113,384 | 12/0/0 | 是 | 2.07 ms |
| v006b | Hamburger | 95,803.9 | 76,565 | 0/0/12 | 是 | 2.05 ms |
| v006c | starter | 106,820.3 | 96,519 | 12/0/0 | 是 | 1.75 ms |
| v006c | random | 103,415.3 | 94,081 | 12/0/0 | 是 | 2.02 ms |
| v006c | Hamburger | 76,294.2 | 57,647 | 0/0/12 | 是 | 2.06 ms |

跨全部 36 场的平均终局现金为：current `123,633.7`，v006a `86,800.7`，v006b `121,368.8`，v006c `95,509.9`。所有策略均无 agent exception、无非法 action shape；p99 远低于 1000 ms。

## 结论

本轮没有候选满足晋级门槛，因此根目录 `main.py` 保持不变。

- v006a 证明了施肥规则会扰动当前的全局移动/任务分配，虽然触发了 222 次 `FERTILIZE`，但现金明显下降。
- v006b 是最接近控制组的候选；13 只牲畜带来更多生产，但 4 天小麦缓冲的饲料成本和 Hamburger 对手下的产能差距使综合平均现金仍下降约 1.8%，且最差现金门槛失败。
- v006c 同时引入两个扰动，回撤最大，不能晋级。

独立消融也没有形成可晋级版本：仅扩容（13 只、3 天小麦）在 starter 上平均 `137,178.1`，仅增加小麦缓冲（12 只、4 天小麦）为 `135,714.9`；两者在 Hamburger 对手上分别为 `98,838.7` 和 `98,284.5`，对照组为 `100,186.4`。完整诊断数据见 `v006_ablation_summary.csv`。

自动 gate 报告见 `v006_gate.json`；三个正式候选的 `pass` 均为 `false`，所以没有执行 promotion。

动作计数、逐局现金差和运行时明细见 `v006_results.csv`；候选代码仍保留在 `baseline/history/v006*`，便于继续做更窄的调度消融。
