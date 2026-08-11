# RL-010 MILK 双向销售时机实验

固定 V27 `order_only` 路线，只修改已有 MILK SELL 的数量，并在下一次
MILK 事件精确偿还。执行顺序为：

```text
V27 路线 + WEED 层
→ RL-010 MILK 数量转移
→ V27 原有 SELL price-impact 重排
→ 合法性与终局保护
```

候选目录：

- `rl010a_delay_only`：只允许 DELAY；
- `rl010b_bidirectional_no_opp`：提前/延后，但屏蔽对手特征；
- `rl010c_bidirectional_opp`：完整双向版本，使用公开 MILK 供给管线。

当前目录中的模型是 smoke 产物，支持样本不足时会自动回退 CONTROL，
不应直接作为最终提交。

## 训练数据

训练和验证 seed 已在入口中固定分开：

```bash
/opt/anaconda3/envs/kaggriculture/bin/python experiments/build_rl_010_data.py \
  --split train \
  --output baseline/artifacts/rl_010_milk_bidirectional/data_train

/opt/anaconda3/envs/kaggriculture/bin/python experiments/build_rl_010_data.py \
  --split validation \
  --output baseline/artifacts/rl_010_milk_bidirectional/data_validation
```

## 拟合与评测

```bash
/opt/anaconda3/envs/kaggriculture/bin/python experiments/fit_rl_010_bandit.py \
  --samples baseline/artifacts/rl_010_milk_bidirectional/data_train/samples.jsonl \
  --output baseline/artifacts/rl_010_milk_bidirectional

/opt/anaconda3/envs/kaggriculture/bin/python experiments/run_rl_010_benchmark.py \
  --output baseline/artifacts/rl_010_milk_bidirectional/benchmark_validation
```

只做合法性和 720 回合 smoke：

```bash
/opt/anaconda3/envs/kaggriculture/bin/python experiments/test_rl_010_timing.py \
  --smoke --variant rl010c_bidirectional_opp
```

只有训练后验证集同时满足胜率、现金、偿还失败率和耗时门槛，才考虑使用
对应目录内的 `submission.tar.gz`。
