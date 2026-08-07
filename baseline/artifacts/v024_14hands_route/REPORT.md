# V024 实验报告

## 路线冻结

从 `log/2026-08-07/top10` 去重得到 44 个 episode、88 个 seat 样本，按 episode 时间顺序切分为 70% fit、15% option validation、15% future holdout。fit 窗口选择出的完整 route medoid 为：

| 指标 | medoid |
|---|---:|
| hands | 14 |
| NE / SW | day 7 / day 11 |
| PLANT | 158 |
| WATER | 922 |
| HARVEST | 374 |
| FEED / FERTILIZE | 314 / 80 |
| WHEAT / STRAWBERRY / MELON | 92 / 42 / 24 |
| COW / SHEEP | 8 / 6 |

## 已生成候选

- `v024a_route14_control`：14-hands 完整路线、动态 hands 对齐、SELL 合法裁剪和终局清算。
- `v024b_route14_weed`：a + actor-local `DIG → retry`，最多 8 回合追赶，失败 suppression。
- `v024c_route14_order_memory`：b + 只重排已有 premium SELL 的高置信度公共状态记忆。
- `v024d_route14_strict_r3`：c + 严格镜像门控的 R3，8 回合商品冷却和下一回合偿还。

四个候选都生成了 `submission.tar.gz`，压缩包根目录包含 `main.py`。root `main.py` 未修改。

## 测试结果

- 结构测试：4/4 通过。
- starter smoke：四候选双 seat 均 `DONE`，无 agent error，p99 小于 1 ms。
- 固定 `ours` 反事实：V022c control 的 38 个 seat 全部落后；V024a 平均 margin 约 `-4,050`，相对 control 平均改善约 `+98,729`，38/38 个 seat margin 改善，并得到 10 胜 / 28 负。该测试使用固定对手动作，不能直接等价为平台胜率。
- 与 root baseline 配对：V022c 为 8 胜 4 负；V024a 为 12 胜 0 负，平均现金约 `146,638`，平均 margin 约 `+42,841`。
- 与 V022c 配对：V024a/b/c/d 均为 0 胜 12 负；V024a 平均现金约 `99,072`，平均 margin 约 `-21,261`。这说明 V022c 的市场销售层仍然是更强对手，route14 生产层尚未足够单独晋级。
- 最新 Top10 外部 holdout（7 episodes × 2 seat）：V024a 与 V022c 都为 0 胜；V024a 平均现金 `120,927`，V022c `111,397`，平均 margin 略低约 `269`。

## 当前结论

V024 已验证了计划中的主要判断：当前 V022c 的生产结构确实落后于 14-hands 高产路线，route14 能显著提高现金和对 root baseline 的胜率；但 V022c 的完整市场专家/销售时机仍能在直接配对中压过 route14。WEED recovery 和 order-memory 在这批测试中没有额外收益，strict R3 也只偶发触发。

因此当前不替换 root `main.py`，也不把 V024a 直接作为最终提交。下一步应以 `v024a_route14_control` 为新的 production control，嫁接 V022c 的市场专家选择或做 route14 专用销售计划，再重新对 V022c、Hamburger 和 Frontier 做胜率测试。
