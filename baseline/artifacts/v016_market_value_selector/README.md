# V016：完整市场专家价值选择器

V016 基于 V012 的固定农场路线，只重做每日市场专家选择。每个候选在当天边界展开五个完整市场专家未来 24 回合的订单，估计销售收入、购买/雇佣/扩地成本、饲料与现金流风险，以及 premium 商品的公开供给碰撞风险。

候选文件：

- `v016a_value_only.py`：只使用未来净现金评分；
- `v016b_collision_hedged.py`：加入 MELON、WOOL、STRAWBERRY、MILK 的公开供给碰撞惩罚；
- `v016c_aggressive_value.py`：提高当前销售收益权重，降低滞留奖励和碰撞惩罚。

三者都叠加 V015a 的价格冲击保护，但不改变 V012 的 farmer/hands、种植、浇水、收获、喂养、购买路线。`baseline/history/v016_market_value_selector/main.py` 是 `v016b_collision_hedged` 的自包含副本，未通过 gate 前不替换根目录 `main.py`。

## 本地检查

```bash
cd "/Users/guoziming/Desktop/比赛/Kaggriculture"
/opt/anaconda3/envs/kaggriculture/bin/python experiments/test_v016_market_selector.py
/opt/anaconda3/envs/kaggriculture/bin/python experiments/run_v016_market_value_search.py --stage smoke
```

重新生成自包含候选：

```bash
python experiments/build_v016_market_selector.py
```

完整反事实和配对矩阵：

```bash
/opt/anaconda3/envs/kaggriculture/bin/python experiments/run_v016_market_value_search.py --stage all
```

默认矩阵为 5 个候选（V012、V015a、V016a/b/c）× 5 个对手 × 6 个 seed × 2 个 seat。结果写入本目录的 `replay_counterfactual_*.csv`、`matrix_*.csv` 和 `gate_report.json`。

候选只有在反事实不退化、本地平均现金至少提升 1%、最差现金达到 V012 的 90%、胜率不下降、对 Hamburger/frontier 不明显退化、无新增错误/非法动作且 p99 小于 1000ms 时才允许进入提交测试。根目录 `main.py` 始终保持不变。

## 本轮结果

已完成 9 局输局反事实和 300 局本地矩阵。V012 control 的平均终局现金为 `133675.28`，胜负为 `51/0/9`，最低现金为 `101598`。

| 版本 | 平均现金 | 最低现金 | W/T/L | 输局反事实平均变化 | 结论 |
|---|---:|---:|---:|---:|---|
| V016a value_only | 133679.88 | 101376 | 55/0/5 | +472 | 不晋级，提升仅 0.003% |
| V016b collision_hedged | 133960.62 | 102114 | 45/0/15 | -381 | 不晋级，胜率和反事实退化 |
| V016c aggressive_value | 133679.88 | 101376 | 55/0/5 | +472 | 不晋级，与 V016a 选择相同 |

V016a/c 在 60 局中全部选择 `automatylicza`，说明当前“未来 24 回合静态现金估计”没有提供足够区分度；V016b 全部选择 `mohit`，碰撞惩罚过强。三个候选均保留用于研究，但没有替换 V012/V015a，也没有写入根目录 `main.py`。
