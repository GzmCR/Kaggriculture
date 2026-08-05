# V014：宏观参数搜索

V014 不枚举 720 回合的逐步动作，而是搜索 V012 动态规划器实际会读取的宏观参数：

- farm hands 数量；
- 牛羊目标数量；
- 初始小麦、甜瓜、胡萝卜和草莓规模；
- NE/SW 解锁时间；
- 动物分阶段进入时间；
- 饲料缓冲、现金储备和持续作物收获阈值。

候选使用 V012 的 observation-driven executor（`use_fixed_schedule=False`），并与固定路线
V012 control 在相同 seed、seat 和对手下配对比较。未通过 gate 的候选不会替换
`main.py`。

## 推荐运行

```bash
/opt/anaconda3/envs/kaggriculture/bin/python -u experiments/run_v014_macro_search.py \
  --stage all \
  --candidate-count 96 \
  --top-k 8 \
  --workers 6 \
  --calibration-seeds 17,42 \
  --final-seeds 17,42,2026,217,317,733 \
  --calibration-seats 0 \
  --final-seats 0,1 \
  --calibration-opponents v012,baseline \
  --final-opponents v012,v18,hamburger,frontier,baseline \
  --write-submission
```

结果会写入本目录：

- `candidate_configs.json`：可复现候选参数；
- `calibration_raw.csv` / `calibration_ranking.json`：校准阶段；
- `final_raw.csv` / `final_summary.csv`：最终矩阵；
- `gate_report.json`：是否出现可替换 V012 的候选；
- `submission.tar.gz`：只有显式使用 `--write-submission` 且候选通过 gate 才生成。

注意：V012 固定路线本身不会读取 `hands`、`cows` 等动态规划参数，所以 V014
必须使用动态 executor。若动态 executor 的上限明显低于 V012，结果应作为架构上限诊断，
而不是直接提交。
