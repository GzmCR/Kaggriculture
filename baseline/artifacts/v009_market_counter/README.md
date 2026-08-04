# V009：公共元策略市场增强与反制

V009 保留根目录 `main.py` 不变，实验两个独立候选：

- `v009a_market_memory`：记录跨回合价格和市场库存，只调整高价值商品的卖出数量、顺序和少量提前销售。
- `v009b_public_meta_counter`：识别对手是否接近 12 手、三象限、8 牛 5 羊的公共元策略，确认后最多增加或提前一笔高价值商品卖单。

两个候选都保持 baseline 的田地动作、移动、浇水、喂养、种植和终局现场逻辑不变，不使用固定 replay 轨迹或外部 API。

运行完整矩阵：

```bash
python experiments/run_v009_benchmark.py
```

默认使用 6 个 seeds、两个 seat，以及 `baseline`、`starter`、`random`、Hamburger、frontier 五类当前可用对手。计划中的 builder notebook 当前不在仓库，因此没有伪造该组结果；补入文件后可通过 `--opponents baseline starter random hamburger builder frontier` 加入。现有本目录中的 144 局结果是加入 baseline 默认项之前生成的旧矩阵；新的运行会额外包含 baseline 对照。

通过门槛前不替换根目录 `main.py`。原始结果写入本目录的 `v009_raw.csv`、`v009_summary.csv` 和 `v009_summary.json`。

## 本轮结果

本轮实际完成 144 局：control、A、B 各 48 局，覆盖 starter、random、Hamburger、frontier；builder 因 notebook 缺失未运行。三者均为 100% `DONE`，错误数和非法动作数均为 0，所有候选的单回合 p99 都低于 2ms。

在可复现的三个对手（starter、Hamburger、frontier）上，36 局加权平均终局现金为：

| 版本 | 平均现金 | 相对 control | 最低现金 | 胜场 |
|---|---:|---:|---:|---:|
| control | 111,369.33 | — | 85,845 | 12/36 |
| v009a | 111,683.14 | +0.282% | 85,356 | 12/36 |
| v009b | 111,366.83 | -0.002% | 85,827 | 12/36 |

A 在 frontier 上提升约 0.92%，但整体未达到 +0.5% 晋级门槛；B 基本不改变 baseline，在识别到 Hamburger/frontier 风格时确实进入 `PUBLIC_META`，但实际只进行了极少量市场调整。两个候选均不 promotion，根目录 `main.py` 保持不变。

`random` 是 Kaggle 环境内置的非确定性 agent，每次调用会创建新的随机源，因此本轮 random 数字只作为探索性参考，不能与 control 做严格配对比较。后续若要把随机对手纳入晋级判定，应增加独立重复轮次或使用固定随机对手副本。
