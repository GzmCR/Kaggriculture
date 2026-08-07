# V025：14-hands route + V022c market graft

## 实验设计

以 V024a 的 14-hands 完整路线为唯一生产底座，只替换 market lane：

- farmer、hands、PLANT、WATER、HARVEST、FEED、CARE、FERTILIZE、BUY、HIRE、BUY_LAND 全部沿用 V024a；
- 复用 V022c 的公开 farm distance、mirror probability、opponent exposure 和 ranked SELL overlay；
- 不加入 WEED recovery、V024 order-memory 或新的 SELL 数量；
- 所有 market action 继续限制为最多 10 个订单，并裁剪非法 SELL 数量。

候选：

- `v025a_route14_v022c_market`：V022c 原始 gate；
- `v025b_route14_v022c_open_market`：step 24 后始终启用 V022c SELL 排序，用于判断 gate 是否过于保守；
- `v025c_route14_v022c_mirror_market`：仅在严格 mirror latch 时启用。

## 结果

### V022c 配对矩阵：6 seeds × 2 seats

| 候选 | 平均现金 | 平均 margin | W/T/L |
|---|---:|---:|---:|
| V025a gate | 99,072 | -21,261 | 0/0/12 |
| V025b open | 102,688 | -4,098 | 0/0/12 |
| V025c mirror | 99,072 | -21,261 | 0/0/12 |

V025b 相比 V024a 的平均 margin 提升约 17,163，说明 V022c 市场排序层确实能弥补 route14 的部分差距；但尚未转化为对 V022c 的胜场。

### root baseline 配对

V025b 为 `12-0`，平均现金约 `146,589`，平均 margin 约 `+42,736`，没有新增 error，p99 约 `0.26ms`。

### 固定 ours 反事实：19 episode × 2 seats

| 候选 | 平均现金 | 平均 margin | W/T/L |
|---|---:|---:|---:|
| V024a control | 121,068 | -4,050 | 10/0/28 |
| V025b open | 121,138 | -2,378 | 11/0/27 |

38/38 个 seat 均无 agent error；V025b 比 V024a 多恢复 1 个胜场，平均 margin 进一步改善。

### Top10 future holdout：最新 7 episode × 2 seats

V025b 平均现金为 `118,912`，平均 margin 为 `-22,507`，0 胜 / 14 负。它比 V024a 在该 holdout 上更差，因此当前不能把 always-on overlay 视为稳健晋级版本。

## 结论

V025 证明了“14-hands 生产路线 + V022c 市场排序”是有效的组合方向，且 V025b 是当前最强的局部候选；但 always-on 市场干预对 future holdout 有明显过拟合迹象。暂不替换 root `main.py`，也不直接将 V025b 作为最终提交。

下一轮应在 V025b 的基础上加入更严格的商品级安全门控，例如只在 premium 商品公开供给高、价格未处于低位且当前排序确实改变时执行，而不是继续扩大市场干预范围。
