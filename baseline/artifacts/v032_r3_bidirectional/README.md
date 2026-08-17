# V032-R3：真实库存约束下的多商品双向销售时机

R3 固定 V27 `order_only` 路线，只对 `MILK`、`STRAWBERRY`、`MELON`、`WOOL`
做单事件市场诊断。R3 的提前销售不会从未来销售事件借货：在原销售回合
`t` 前 `H1/H2/H3` 的回合，只有观察到的 shed 余量才能用于新增 SELL；原回合
再扣回相同数量。延后仍沿用 V032-R2 的当前事件到下一同商品事件转移，并
保留现金流、仓库和偿还安全门。

所有结果都是离线 paired diagnostics，不替换根目录 `main.py`。

Runner 已对同一 `(opponent, seed, seat)` 缓存 capture，对同一商品缓存
control 对局；只有真正通过库存检查的 candidate 才再次运行完整环境。运行时
会打印 `capture_runs`、`control_runs`、`candidate_runs` 和 cache hit 数量，并
把它们写入与 JSONL 同名的 `_run_stats.json`。

## 快速 smoke

先从一个有机会产生 STRAWBERRY 提前库存的晚期窗口开始：

```bash
cd "/Users/guoziming/Desktop/比赛/Kaggriculture"
PYTHONPATH=experiments python experiments/run_v032_r3_bidirectional.py \
  --mode advance \
  --opponents v27_current \
  --seeds 17 \
  --seats 0 1 \
  --items STRAWBERRY \
  --min-step 575 \
  --max-events 1 \
  --output baseline/artifacts/v032_r3_bidirectional/advance_smoke.jsonl
```

检查结果：

```bash
PYTHONPATH=experiments python experiments/analyze_v032_r3_bidirectional.py \
  baseline/artifacts/v032_r3_bidirectional/advance_smoke.jsonl
```

联合提前/延后 smoke：

```bash
PYTHONPATH=experiments python experiments/run_v032_r3_bidirectional.py \
  --mode combined \
  --opponents v27_current \
  --seeds 17 \
  --seats 0 1 \
  --items STRAWBERRY \
  --min-step 575 \
  --max-events 1 \
  --output baseline/artifacts/v032_r3_bidirectional/combined_smoke.jsonl
```

## 第一阶段：提前单独验证

```bash
PYTHONPATH=experiments python experiments/run_v032_r3_bidirectional.py \
  --mode advance \
  --opponents v27_current v13_r3 adaptive_replay frontier_current \
  --seeds 17 42 2026 \
  --seats 0 1 \
  --items MILK STRAWBERRY MELON WOOL \
  --output baseline/artifacts/v032_r3_bidirectional/advance_phase1.jsonl
```

如果只想做快速筛选，可以加 `--max-events 1 --progress-every 25`；前者每个
商品/控制类型只取一个事件，后者每 25 行打印一次进度。`--progress-every 0`
可以关闭进度输出。

## 第二阶段：提前与延后

```bash
PYTHONPATH=experiments python experiments/run_v032_r3_bidirectional.py \
  --mode combined \
  --opponents v27_current v13_r3 adaptive_replay frontier_current \
  --seeds 17 42 2026 217 317 733 \
  --seats 0 1 \
  --items MILK STRAWBERRY MELON WOOL \
  --output baseline/artifacts/v032_r3_bidirectional/combined_phase2.jsonl
```

## 结果字段

每行对应一个固定 opponent/seed/seat、商品、事件和转移数量。重点字段：

- `available_extra_inventory`：提前事件开始回合中，扣除原有同商品 SELL 后的真实 shed 余量；
- `predicted_local_margin_delta`：按 R2 官方价格曲线和 lockstep 市场模拟的局部 margin；
- `actual_interval_margin_delta`：固定对手 action tape 的真实环境，从事件起点到终点的目标商品收入差；
- `actual_final_margin_delta`：完整 720 回合终局 margin 差；
- `safe`：订单、库存、现金、非目标动作和终点偿还等硬门通过；
- `farmer_hands_action_diff`、`non_target_market_action_diff`：应为 0。

没有真实 shed 余量的 ADVANCE 候选会记录为 `SKIPPED/no_warehouse_inventory`，不会
被当成可训练样本；这正是 R3 用来避免“虚空提前卖”的关键约束。没有实际 SELL
事件的商品会自然跳过。R3 先用于验证某商品/某 horizon 是否在多个对手和 seed
上稳定有效，再考虑接入运行时控制器。

## 已完成阶段二结果（2026-08-16）

阶段二使用 `v27_current`、`v13_r3`、`adaptive_replay`、`frontier_current`，
6 个 seeds、两个 seat、4 个商品，共 `35,376` 条事件记录。结果文件为：

```text
combined_phase2_cached.jsonl
combined_phase2_cached_summary.json
combined_phase2_cached_run_stats.json
```

| 指标 | 结果 |
| --- | ---: |
| evaluated rows | 7,760 |
| safe rows | 7,005 |
| skipped rows | 27,616 |
| safe candidate mean final margin | -23.02 |
| local prediction MAE | 0.72 |
| local prediction sign accuracy | 99.4% |

绝大多数提前候选因为没有真实 shed 余量而跳过。唯一有效的提前样本是
`STRAWBERRY` 的 15 条 1-unit 候选，平均最终差约 `+12.4`，不足以支持通用
抢跑策略。延后候选总体为负，不能直接替换 V27 control；少数长间隔
`MILK/STRAWBERRY/WOOL` 事件只作为后续状态门控的候选，不作为已验证结论。

阶段二是 paired counterfactual 诊断，不是完整运行时策略 benchmark；每个
事件行并非独立样本，最终晋级仍必须使用完整 720 回合对局。

中断后可使用以下命令续跑，不会重复已有 JSONL 行：

```powershell
$env:PYTHONPATH="experiments"
D:\kg311\python.exe experiments\run_v032_r3_bidirectional.py `
  --mode combined `
  --opponents v27_current v13_r3 adaptive_replay frontier_current `
  --seeds 17 42 2026 217 317 733 `
  --seats 0 1 `
  --items MILK STRAWBERRY MELON WOOL `
  --resume --skip-unsafe-candidates --flush-every 100 `
  --output baseline/artifacts/v032_r3_bidirectional/combined_phase2_cached.jsonl
```
