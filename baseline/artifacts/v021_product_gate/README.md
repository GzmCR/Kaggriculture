# V021：分商品安全门控与胜率保护

V021 基于 V020/V012，只调整已有 premium SELL 数量，保持 farmer、hands、田地动作、生产路线和非 premium 市场订单不变。

## 候选

- `v021a_safety_patch.py`：V020 conservative 的一单位安全延迟，增加终局冻结和恢复预算。
- `v021b_product_gate.py`：MILK、WOOL 使用不同价格冲击阈值，STRAWBERRY/MELON 默认 passthrough。
- `v021c_win_guard.py`：在 V021b 上加入公开现金差保护，落后超过 1,000 时不延迟卖单。

所有候选都是 self-contained agent；对应压缩包为 `submission_v021_*.tar.gz`。默认 `submission.tar.gz` 使用 `win_guard`。

## 本地验证

不变量测试和 starter 720 回合 smoke test 均通过：

- `V021 product-gate invariants: PASS (6 tests)`；
- 三个候选均 `DONE`；
- farmer/hands 不变；
- 不新增 premium SELL；
- step 648 后完全沿用 V012；
- 无 agent error 或非法 action shape。

本地 replay 目录实际只有 9 局可用 JSON；计划中的第 10 局不在当前工作区，因此报告按 9 局记录，没有伪造样本。

## 评测结果

矩阵使用 8 个对手、6 个 seeds、两个 seat，共 384 局（control + 3 个候选，每个版本 96 局）。同一矩阵中的 V012 control：

- 平均现金：145,868.85；
- 最低现金：101,598；
- 胜率：77/96 = 80.21%。

| 候选 | 平均现金 | 胜场 | 最低现金 | Replay 平均 margin 差 |
|---|---:|---:|---:|---:|
| V021a | 145,792.09 | 77/96 | 101,613 | +5.44 |
| V021b | 145,612.83 | 77/96 | 101,604 | +5.44 |
| V021c | 145,654.28 | 77/96 | 101,604 | 0.00 |

三个候选均保住了总体胜率、V012/V18 合计胜场、最低现金和 replay gate，但平均现金均低于同一 control，因此全部未通过晋级条件。

主要退化来自 `random` 对手：在没有强市场碰撞时，延迟 premium 销售会损失即时现金；V021c 的现金保护能够保胜率，但不能恢复平均现金优势。

结论：V021 的状态安全修复有效，但“分商品延迟卖出”当前不适合作为 active main。根目录 `main.py` 保持不变，V012/V019b 继续作为后续对照。

结果文件：`matrix_summary.csv`、`replay_counterfactual_summary.csv`、`gate_report.json`。
