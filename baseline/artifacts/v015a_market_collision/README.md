# V015a：共享市场碰撞反应层

V015a 保留 V012 的 farmer/hands、作物、牲畜、雇佣、土地、购买和非 premium 市场订单，只在运行时对 `MELON`、`STRAWBERRY`、`MILK`、`WOOL` 的卖单做保守 overlay。

核心行为：记录价格和市场库存；识别单回合至少 20% 下跌、两回合累计至少 30% 下跌或近期中位价 70% 以下的拥堵；每种商品每回合最多延迟一个卖出单位；价格恢复或终局时通过已有卖单/清算卖出；市场订单不超过 10 个。输局 replay 仅作为固定对手的反事实诊断，不进入提交代码。

## 快速检查

```bash
cd "/Users/guoziming/Desktop/比赛/Kaggriculture"
/opt/anaconda3/envs/kaggriculture/bin/python -m py_compile \
  experiments/v015a_market_overlay.py \
  baseline/history/v015a_market_collision/main.py \
  experiments/run_v015a_market_collision.py \
  experiments/test_v015a_market_collision.py
/opt/anaconda3/envs/kaggriculture/bin/python experiments/test_v015a_market_collision.py
```

## 评测命令

先跑 9 局平台输局的反事实：

```bash
/opt/anaconda3/envs/kaggriculture/bin/python \
  experiments/run_v015a_market_collision.py --stage replay --variants default sensitive conservative
```

跑完整本地矩阵（V012 control + V015a，对战 V012、root baseline、v18、Hamburger、frontier）：

```bash
/opt/anaconda3/envs/kaggriculture/bin/python \
  experiments/run_v015a_market_collision.py --stage matrix --variants default \
  --opponents v012 baseline v18 hamburger frontier \
  --seeds 17 42 2026 217 317 733
```

结果写入本目录：`replay_counterfactual_raw.csv`、`replay_counterfactual_summary.csv`、`matrix_raw.csv`、`matrix_summary.csv` 和 `gate_report.json`。

只有反事实不恶化且矩阵通过全部门槛时，才使用 `--write-submission` 生成本目录下独立的 `main.py`；根目录 `main.py` 与 V012 源文件不会被替换。

## 本轮结果

默认阈值在 9 局平台输局反事实中 9/9 改善，平均现金差改善 `+621.8`；120 局本地矩阵中，V015a 平均终局现金 `133885.9`，V012 control 为 `133675.3`，提升约 `+0.158%`。全部对局 `DONE`，没有新增错误或非法 action shape，field/hands 与非 premium 订单改动数均为 0，p99 单回合耗时低于 1 秒。

因此已生成可独立提交的 [main.py](main.py)。`baseline/history/v015a_market_collision/main.py` 现在也同步为同一份自包含提交代码，方便直接选取；此前的本地路径适配器不应上传到 Kaggle。根目录的 [main.py](/Users/guoziming/Desktop/比赛/Kaggriculture/main.py) 未替换。

推荐从仓库根目录提交：

```bash
kaggle competitions submit kaggriculture \
  -f baseline/artifacts/v015a_market_collision/main.py \
  -m "V015a shared market collision overlay"
```
