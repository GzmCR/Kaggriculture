# V019 replay 分数段与公开风格分析

输入目录：`/Users/guoziming/Desktop/比赛/kaggriculture/log/2026-08-05`。

## 数据处理

- 31 个 JSON 文件，按 `EpisodeId` 去重后得到 29 个唯一 replay。
- 完全重复的标签组：`2882/2924 -> 90121273`、`2907/2972 -> 90130811`。
- 分数只作为 episode-level 标签；replay 没有可靠的“分数对应 TeamName/seat”映射，运行时不读取文件名、TeamName 或分数。
- 使用 `step + 1` 对齐 replay action，并按环境的双方 SELL lockstep 顺序重建实际成交。
- 使用安装环境中的 `market_price`、商品库存、town/shop 消耗和市场订单上限；原始 SELL 请求量不当作成交量。

分层标签：L1 `[1600, 2000)`、L2 `[2000, 2400)`、L3 `[2400, 2800)`、L4 `[2800, 10000)`。

校验结果：29 个 replay 全部解析成功，0 个价格 mismatch。输出中的 `transactions_by_turn.csv` 同时保存 requested、filled、revenue、weighted_price 和 floor_units。

## 主要输出

- `replay_manifest.csv`：episode、分数标签、重复信息、seed、TeamNames。
- `daily_public_features.csv`：按日公开 farm、土地、hands、作物、动物和 premium pipeline。
- `side_profiles.csv`：每个 episode/seat 的最终结构和主风格。
- `transactions_by_turn.csv`：按商品、回合、seat 的实际成交重建。
- `analysis_summary.json`：汇总和校验信息。

## 观察结论

大多数样本属于 `standard_converged`；少量样本在第 10 天左右即可识别为 `reduced_ne_only`，其特征是没有 SW、动物和作物规模明显低于 8 牛 5 羊的标准结构。14+ hands 属于少量 `high_worker_maintenance`，没有显示出稳定优势。

市场侧不能用请求 SELL 数量判断收益：请求量远大于实际成交量。应比较每件商品的实际成交价格和成交时点，尤其关注 MILK、STRAWBERRY 的集中销售。

分析脚本：[`experiments/v019_replay_analysis.py`](/Users/guoziming/Desktop/比赛/Kaggriculture/experiments/v019_replay_analysis.py)。
