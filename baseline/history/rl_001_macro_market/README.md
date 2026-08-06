# RL-001 高层市场策略选择器

RL-001 固定 V022c 的 farmer/hands 路线，只在每 48 回合边界选择一个市场 overlay。第 672 回合起关闭 overlay，恢复 V022c 的终局市场逻辑。

四个动作分别是：V022c 原始市场、V13-R3 风格的一回合 premium 提前销售、conditional order-only 重排、Frontier 风格的延迟重排。selector 使用 NumPy Double Q-learning；提交文件内嵌权重，不依赖外部文件或 GPU。

## 文件

- `main.py`：零权重 control 构建，行为退化为 V022c。
- `baseline/artifacts/rl_001_macro_market/training/weights.json`：本轮 32 局 pilot 权重，仅用于实验。
- `baseline/artifacts/rl_001_macro_market/evaluation/holdout_summary.json`：pilot 的 16 局 smoke 结果。
- `baseline/artifacts/rl_001_macro_market/ablation/`：四种固定策略的 ablation 结果。
- `replay_training_report.json`：使用 `log/2026-08-06` 后的 replay warm-start 结果；未通过 gate。
- `replay_training/`：replay 校准训练的权重、manifest 和 holdout 日志。

## 运行

```powershell
& 'D:\kg311\python.exe' experiments\test_rl_001.py
& 'D:\kg311\python.exe' experiments\run_rl_001.py --mode train --episodes 1000 --replay-dir log\replays
& 'D:\kg311\python.exe' experiments\run_rl_001.py --mode train --episodes 1000 --replay-dir log\2026-08-06
& 'D:\kg311\python.exe' experiments\run_rl_001.py --mode train --episodes 1000
& 'D:\kg311\python.exe' experiments\run_rl_001.py --mode evaluate --episodes 300 --weights baseline\artifacts\rl_001_macro_market\training\weights.json
& 'D:\kg311\python.exe' experiments\build_rl_001.py --weights baseline\artifacts\rl_001_macro_market\training\weights.json
```

完整 holdout 通过 gate 之前，不替换根目录 `main.py`，也不把 pilot 权重当作正式提交。
