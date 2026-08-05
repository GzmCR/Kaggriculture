# V017：商品级市场 Rollout 控制器

V017 以 V012 的固定 farmer/hands 和完整市场专家计划为销售骨架，只对当前已有的 MELON、STRAWBERRY、MILK、WOOL SELL 数量进行短期 rollout 调整，随后继续使用 V015a 的价格冲击保护。非终局不新增原计划中不存在的 premium 卖单，根目录 `main.py` 不变。

候选：

- `v017a_curve_only.py`：真实价格曲线和城镇消耗，不使用对手供给；
- `v017b_opponent_aware.py`：加入对手公开作物、牛羊和成熟度供给估计；
- `v017c_robust_quota.py`：LOW/NORMAL/HIGH 三种供给场景、风险惩罚和 50% 销售配额。

每个当前 premium 商品订单评估 0%、25%、50%、75%、100% 五个数量候选，窗口为未来 8 回合。现金低于安全线或当前订单包含 HIRE、BUY_LAND、BUY_ANIMAL、BUY_PRODUCT 时，控制器跳过调整，避免间接破坏 V012 的雇佣和饲料现金流。

## 本地运行

```bash
cd "/Users/guoziming/Desktop/比赛/Kaggriculture"
/opt/anaconda3/envs/kaggriculture/bin/python experiments/test_v017_market_rollout.py
/opt/anaconda3/envs/kaggriculture/bin/python experiments/run_v017_market_rollout.py --stage smoke
```

重新生成自包含候选：

```bash
python experiments/build_v017_market_rollout.py
```

完整 9 局反事实和 300 局矩阵：

```bash
/opt/anaconda3/envs/kaggriculture/bin/python experiments/run_v017_market_rollout.py --stage all
```

结果写入本目录的 `replay_counterfactual_*.csv`、`matrix_*.csv` 和 `gate_report.json`。主要 control 是 V015a，V012 作为辅助 control。只有平均现金提升至少 0.5%、最差现金不低于 V015a 的 95%、胜率和对 Hamburger/frontier 不退化、9 局反事实平均不下降、field/non-premium 不变量通过且 p99 小于 1000ms，候选才允许进入提交测试。

## 本轮结果

已完成 9 局平台输局反事实和 300 局本地矩阵。V015a control 的平均终局现金为 `133885.93`，最低现金 `101552`，W/T/L 为 `56/0/4`。

| 版本 | 平均现金 | 最低现金 | W/T/L | 反事实相对 V015a | 结论 |
|---|---:|---:|---:|---:|---|
| V017a curve_only | 133605.68 | 102999 | 43/0/17 | -932.33，0/9 不低于 | 不晋级 |
| V017b opponent_aware | 133986.78 | 102945 | 54/0/6 | -126.56，5/9 不低于 | 不晋级 |
| V017c robust_quota | 133706.15 | 102650 | 51/0/9 | -144.56，2/9 不低于 | 不晋级 |

三个候选均满足 `DONE`、无新增异常/非法动作、field 路线不变、非 premium 订单不变和 p99 小于 1000ms，但没有同时满足现金、胜率和反事实门槛。V017b 是最接近的版本，说明对手公开供给信息有价值；当前 8 回合价格 rollout 仍不足以稳定替代 V015a。根目录 `main.py` 未替换。
