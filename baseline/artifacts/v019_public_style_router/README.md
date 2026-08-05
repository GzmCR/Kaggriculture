# V019：公开状态风格门控实验

V019 保持 V012 的 farmer/hands、土地、种植、浇水、收获、喂养和购买路线，只在日边界根据对手公开 farm 状态选择已有完整市场专家。

运行时不读取 replay 文件名、分数、TeamName 或对手私有库存。风格分类使用对手可见的土地解锁、hands、作物、牛羊、成熟度、premium pipeline，以及共享市场价格。

## 候选

- `v019a_price_priority.py`：只替换已有的 MILK/STRAWBERRY SELL 槽位，MELON、WOOL 和其他订单沿用 V012。
- `v019b_public_style.py`：按公开风格选择一个完整市场专家；这是收益最高但市场改动最宽的版本。
- `v019c_weak_counter.py`：只有识别到弱路线时才切换完整专家，其他情况完全回退 V012。
- `../history/v019_public_style_router/main.py`：研究历史版本，不是 root `main.py`。

校准后的默认映射为：

```text
standard_converged      -> navazsh_fathi
reduced_ne_only         -> mohit
high_worker_maintenance -> manual_player
premium_concentrated    -> navazsh_fathi
```

该映射来自 348 局固定对手专家校准，并在反事实评估中采用 leave-one-episode-out；不代表使用了测试 replay 的文件名或分数。

## 评估

- 29 个唯一 replay × 2 个 seat = 58 个固定对手目标。
- 专家校准：58 ×（V012 control + 5 个固定专家）= 348 局。
- 反事实：58 ×（V012、V018、V019a/b/c）= 290 局。
- 每局 720 steps；所有候选均 DONE，agent error 为 0，field action 与 V012 一致，p99 小于 1000ms。

| 版本 | 平均终局现金 | 相对 V012 | 胜/平/负 | 不低于 control 的目标比例 |
|---|---:|---:|---:|---:|
| V012 | 131,879 | — | 14/0/44 | — |
| V018 | 131,936 | +57 | 13/0/45 | 46.6% |
| V019a | 132,809 | +930 | 12/0/46 | 89.7% |
| V019b | 133,248 | +1,369 | 13/0/45 | 86.2% |
| V019c | 131,883 | +4 | 14/0/44 | 100% |

V019a、V019b、V019c 都通过本阶段研究 gate：平均现金不下降、最低现金不低于 control 的 95%、至少 60% 目标不下降、四个分数层平均不低于 control 的 95%、无新增 invalid shape、field 不变且全部 DONE。

需要注意：V019a/b 的胜场数没有超过 V012；V019b 的优势主要体现为终局现金和分数层平均现金，而不是 W/L 增长。因此当前只保留为实验候选，不自动替换 root `main.py`。完整逐局数据见 `calibration_raw.csv`、`counterfactual_raw.csv`，汇总见 `counterfactual_summary.csv` 和 `gate_report.json`。

脚本：[`experiments/v019_style_router.py`](/Users/guoziming/Desktop/比赛/Kaggriculture/experiments/v019_style_router.py)、[`experiments/run_v019_style_counterfactual.py`](/Users/guoziming/Desktop/比赛/Kaggriculture/experiments/run_v019_style_counterfactual.py)。
