# V012：五位选手代表轨迹 vs v18

本实验从 `log/2026-08-04` 的 15 份 replay 中保留 5 位选手各一条完整 720 回合轨迹，与 notebook 中的 v18 agent 对战。

选择规则：

1. 按该选手在已有 replay 中的平均终局现金选出前五位；
2. 同一选手有多份 replay 时，选择其 field + market 动作序列距离自身中位轨迹最近的一份；
3. 轨迹作为完整 bundle 使用，不把不同选手的 farmer/hands/market 动作逐回合拼接；
4. replay 只用于本地实验，根目录 `main.py` 不改变。

Replay 的 JSON 序列化中，`steps[i + 1].action` 才是从第 `i` 个 observation
推进到下一个状态的可执行动作；脚本已处理这一回合偏移，并用原始 seed + 双方
固定 replay 动作复现了五份源对局的终局现金。

当前五条代表轨迹：

| 选手 | replay | 原始 seat |
|---|---:|---:|
| mohit | 89817349 | 0 |
| automatylicza | 89830916 | 0 |
| manual player | 89820316 | 0 |
| navazsh fathi | 89830910 | 0 |
| Lucien de Rubempre | 89822684 | 1 |

运行：

```bash
python experiments/run_v012_top5_vs_v18.py
```

结果写入本目录：

- `selection.json`：选中的五条 replay；
- `v012_raw.csv`：逐局现金、胜负、DONE、耗时和轨迹修复次数；
- `v012_summary.csv/json`：汇总结果。

这个版本先测试五条完整路线本身，不包含状态路由。这样可以先判断哪位选手的完整生产结构最适合作为我们的 field route，再决定是否实现低频专家路由。

## 首轮结果

使用 seeds `17, 42, 2026, 217, 317, 733`，双方 seat 各一局，共 60 局；对手是
notebook 中解码出的 v18 agent。

| 代表轨迹 | 平均现金 | 平均现金差 | W/T/L | 最低现金 |
|---|---:|---:|---:|---:|
| automatylicza | 125,262 | +382 | 9/0/3 | 103,410 |
| mohit | 124,143 | -1,738 | 1/0/11 | 102,407 |
| navazsh fathi | 123,135 | -5,231 | 1/0/11 | 100,282 |
| manual player | 122,740 | -4,966 | 1/0/11 | 100,585 |
| Lucien de Rubempre | 121,950 | -5,853 | 1/0/11 | 100,107 |

所有 60 局均为 `DONE`，没有环境错误。由于 replay 的 action/observation
序列化方式，运行日志中每条路线有 21 个 hand-list mismatch；脚本保留原始
动作，环境按实际存在的 hands 处理尾部动作。当前最值得继续研究的是
`automatylicza` 路线；其余四条暂不做平均混合。
