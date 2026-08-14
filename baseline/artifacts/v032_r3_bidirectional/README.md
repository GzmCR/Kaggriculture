# V032-R3：真实库存约束下的多商品双向销售时机

R3 固定 V27 `order_only` 路线，只对 `MILK`、`STRAWBERRY`、`MELON`、`WOOL`
做单事件市场诊断。R3 的提前销售不会从未来销售事件借货：在原销售回合
`t` 前 `H1/H2/H3` 的回合，只有观察到的 shed 余量才能用于新增 SELL；原回合
再扣回相同数量。延后仍沿用 V032-R2 的当前事件到下一同商品事件转移，并
保留现金流、仓库和偿还安全门。

所有结果都是离线 paired diagnostics，不替换根目录 `main.py`。

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
