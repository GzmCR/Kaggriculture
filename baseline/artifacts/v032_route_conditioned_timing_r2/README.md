# V032-R2：安全现金流下的区间销售收益估计

R2 是 V032 的离线研究版本，当前不替换根目录 `main.py`。

它以 V27 `order_only` 为路线 control，只改变某个目标商品当前 SELL 和下一次同商品 SELL 之间的延后数量。目标商品第一轮为 `MILK` 和 `STRAWBERRY`。

## 估计逻辑

- 使用 Kaggriculture 1.32.6 的官方价格函数；
- 每单位 lockstep 处理双方市场队列；
- 遵守每回合最多 10 个订单；
- 市场处理后执行 town center/shop 消耗；
- 只统计目标商品的我方收入、对手收入和局部 margin；
- 不预加载未来作物库存；
- 现金流和仓库安全门不通过时回退 control。

## 运行

```bash
PYTHONPATH=experiments python experiments/run_v032_r2_interval.py \
  --output baseline/artifacts/v032_route_conditioned_timing_r2/intervals.jsonl

PYTHONPATH=experiments python experiments/analyze_v032_r2_interval.py \
  baseline/artifacts/v032_route_conditioned_timing_r2/intervals.jsonl \
  --output baseline/artifacts/v032_route_conditioned_timing_r2/summary.json
```

默认使用 V27、V13-R3、adaptive replay、Frontier，seeds 为 `17,42,2026,217,317,733`，双方 seat，延后比例为 1%、25%、50%，每种商品最多取前两个相邻销售区间。

## 解释

`predicted_local_margin_delta` 只回答当前商品在当前销售区间内的局部收益变化；`actual_final_margin_delta` 仍用于检查更远期市场和路线外部性。安全门过于保守导致大多数事件回退是可接受结果，不应为了提高干预率而放宽现金或库存约束。
