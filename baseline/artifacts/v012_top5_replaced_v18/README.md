# V012：用五条当前 replay 替换 v18 专家库

这个实验保留原始 v18 的运行时结构：

- farmer/hands 的 board route 固定为 V012 中表现最好的 `automatylicza`；
- 每天边界从五条当前 replay 中选择一个完整 market expert；
- 使用 v18 原有的 29 维公开状态距离、stay bonus 和低频切换机制；
- 原始 `40-53-top-10-future-holdout-v18-closed-loop.ipynb` 不修改。

五条 market expert 为：`mohit`、`automatylicza`、`manual player`、`navazsh fathi`、
`Lucien de Rubempre`。运行脚本会从 replay 中读取它们的完整动作和日级状态原型。

```bash
/opt/anaconda3/envs/kaggriculture/bin/python experiments/run_v012_replaced_v18.py
```

本地 gate 先验为 `automatylicza=0.75`，其他四条为 `0.0`。这个先验来自上一轮
五条完整路线对 v18 的结果；零先验消融会在 Day 0 错选 `navazsh_fathi`，只取得
2/12 胜，因此不采用。

结果文件：

- `experts.json`：board route 和五条 market expert；
- `v012_replaced_raw.csv`：逐局结果和 market route history；
- `v012_replaced_summary.csv/json`：对原始 v18 的汇总。

## 对战结果

使用 seeds `17, 42, 2026, 217, 317, 733`，双方 seat 各一局，共 12 局：

| 版本 | 平均现金 | 平均现金差 | W/T/L | 胜率 | 最低现金 |
|---|---:|---:|---:|---:|---:|
| 五专家替换版 | 125,111 | +404 | 10/0/2 | 83.3% | 103,410 |

所有对局均为 `DONE`，agent error 为 0；market route 最多切换 2 次，且只在日边界切换。
这说明五条 replay 中真正有用的是 `automatylicza` 作为主路线，其他专家目前更适合作为
后期市场候选，而不是平权专家。

## 提交包

已生成自包含提交文件，不依赖本地 notebook 或 `log/2026-08-04`：

- `baseline/history/v012_top5_replaced_v18/main.py`
- `baseline/artifacts/v012_top5_replaced_v18/submission.tar.gz`

推荐提交 tar 包：

```bash
kaggle competitions submit kaggriculture \
  -f baseline/artifacts/v012_top5_replaced_v18/submission.tar.gz \
  -m "V012 top5 replay experts replacing v18"
```

也可以直接提交单文件：

```bash
kaggle competitions submit kaggriculture \
  -f baseline/history/v012_top5_replaced_v18/main.py \
  -m "V012 top5 replay experts replacing v18"
```
