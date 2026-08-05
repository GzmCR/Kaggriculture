# V018：波次级市场计划 + 每日 MPC

V018 是一个只改市场销售数量的实验版本，根目录 `main.py` 未修改。

流水线固定为：

```text
V012 farmer/hands 与生产路线
    -> V018 premium SELL 数量控制
    -> V015a 价格冲击保护与终局清算
```

控制器只处理 `MELON`、`STRAWBERRY`、`MILK`、`WOOL` 已经存在的 `SELL` 订单：

- `v018a_fixed_wave.py`：按可解释的销售波次和最低价格比例限额；
- `v018b_daily_mpc.py`：每天边界用 72 回合短期 rollout，评估 0/25/50/75/100% 当前卖单比例；
- `v018c_robust_mpc.py`：使用低/正常/高三种对手供给场景，并对最差场景做风险惩罚。

未来回合只作为参考计划，当前比例不会错误地乘到整个未来 horizon；下一天重新根据实际观察结果规划。现金不足或当前订单包含关键支出时，控制器保持原动作。

## 本地测试

```bash
cd "/Users/guoziming/Desktop/比赛/Kaggriculture"
PYTHON=/opt/anaconda3/envs/kaggriculture/bin/python

$PYTHON experiments/build_v018_market_wave.py
$PYTHON experiments/test_v018_market_wave.py
$PYTHON experiments/run_v018_market_wave.py --stage smoke
$PYTHON experiments/run_v018_market_wave.py --stage replay
```

完整矩阵命令：

```bash
$PYTHON experiments/run_v018_market_wave.py --stage matrix
```

## 9 局输局反事实结果

以 `control_v015a` 为主要对照，45 场运行全部完成，错误数和非法动作数为 0：

| 版本 | 平均现金差 | 相对 V015a | 不低于 V015a 的局数 | 最大 p99 |
|---|---:|---:|---:|---:|
| V015a control | -1,026.0 | — | — | 0.145 ms |
| V018a fixed wave | -1,820.0 | -794.0 | 0/9 | 0.203 ms |
| V018b daily MPC | -3,623.1 | -2,597.1 | 0/9 | 4.765 ms |
| V018c robust MPC | -3,452.6 | -2,426.6 | 0/9 | 12.851 ms |

因此三个候选均未通过 replay gate，也没有替换根目录 `main.py`。这次实验说明：在当前 V012 架构下，单纯延迟已有 premium 销售会造成现金流和后续动态决策联动损失；下一步需要把现金流、生产路线和销售计划作为联合状态建模，而不是继续加大销售限额控制。

详细逐局数据在：

- `replay_counterfactual_raw.csv`；
- `replay_counterfactual_summary.csv`；
- `gate_report.json`。

