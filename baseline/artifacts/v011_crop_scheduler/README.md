# V011 高价值作物维护与单位调度实验

## 实验范围

V011 基于根目录 baseline 创建三个独立候选，根目录
`/Users/guoziming/Desktop/比赛/Kaggriculture/main.py` 未修改：

- `v011a_water_guard`：高价值作物浇水保护；
- `v011b_harvest_storage`：高价值作物收获阈值与仓库协同；
- `v011c_priority_scheduler`：两者合并。

高价值作物为 `MELON`、`STRAWBERRY`、`TOMATO`。候选不引入新作物配比、牲畜扩容、肥料策略或市场逻辑。

最终矩阵为 4 个候选（含 control）× 5 个对手 × 6 个 seed × 2 个 seat，共 240 局：

- 对手：root `baseline`、`starter`、`random`、Hamburger、frontier；
- seeds：17、42、2026、217、317、733；
- 每局 720 steps。

## 结果摘要

| 候选 | 平均现金（全部对手） | 最低现金 | 胜/平/负 | 胜率 | baseline 平均现金差 | Hamburger 平均现金差 | frontier 平均现金差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 114946.9 | 85845 | 28/4/28 | 46.7% | 0 | -48241 | -43398 |
| v011a_water_guard | 112818.5 | 65029 | 31/0/29 | 51.7% | -2727 | -48226 | -43161 |
| v011b_harvest_storage | 113259.1 | 81050 | 27/1/32 | 45.0% | -1182 | -50071 | -44688 |
| v011c_priority_scheduler | 112457.6 | 64865 | 29/0/31 | 48.3% | -3111 | -49294 | -44862 |

工程稳定性全部通过：240/240 局 `DONE`，异常 0，非法 action shape 0；候选最大 p99 单回合耗时为 1.663ms，低于 1000ms 门槛。

## 行为指标（全部对手加权平均）

| 候选 | 高价值 WATER | 高价值 HARVEST | 日终未浇水 | weeds | PASS | 移动 | FEED | CARE | DROP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 585.6 | 70.3 | 9.81 | 1.57 | 1630.1 | 4289.2 | 257.6 | 236.1 | 80.1 |
| v011a_water_guard | 579.8 | 71.8 | 9.74 | 1.49 | 1531.2 | 4395.6 | 254.7 | 230.6 | 80.6 |
| v011b_harvest_storage | 582.2 | 70.4 | 9.68 | 1.61 | 1628.0 | 4304.4 | 257.6 | 234.8 | 79.2 |
| v011c_priority_scheduler | 578.7 | 71.9 | 9.55 | 1.51 | 1537.1 | 4398.8 | 254.9 | 230.8 | 79.9 |

高价值作物单独归因来自行动发生前单位所在 tile；`WATER` 和 `HARVEST` 不在高价值作物 tile 上时不会计入该列。

## 结论

本轮没有候选满足晋级条件，因此不替换 root `main.py`。

主要结论：

1. A/C 能略微降低 PASS 和日终未浇水，但移动量增加约 2.5%，FEED/CARE 减少，且高价值 WATER 实际没有超过 control。
2. B 对现金和路线最稳，但实际高价值 HARVEST 只从约 70.3 提升到 70.4，远低于目标 270～300/局。
3. A/C 的 `unassigned_urgent_jobs` 很高，说明当前“每回合重复生成维护 job + 仅替换 PASS”仍有大量任务没有被可达单位接手；该字段是累计诊断计数，不是去重后的植物数量。
4. 当前瓶颈不是简单提高任务优先级，而是高价值作物任务的空间聚集、单位路线和原有 FEED/CARE/收获任务之间没有形成稳定的 ownership。继续加权抢占会改善局部维护，却损失生产动作和现金。

## 产物

- 候选代码：
  - `baseline/history/v011a_water_guard/main.py`
  - `baseline/history/v011b_harvest_storage/main.py`
  - `baseline/history/v011c_priority_scheduler/main.py`
- 评测脚本：`experiments/run_v011_benchmark.py`
- 合成状态单元测试：`experiments/test_v011_scheduler.py`
- 原始局级结果：`v011_raw.csv`
- 分组汇总：`v011_summary.csv`、`v011_summary.json`

复现实验：

```bash
/opt/anaconda3/envs/kaggriculture/bin/python experiments/test_v011_scheduler.py
/opt/anaconda3/envs/kaggriculture/bin/python experiments/run_v011_benchmark.py --out baseline/artifacts/v011_crop_scheduler
```
