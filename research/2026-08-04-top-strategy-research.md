# Kaggriculture 前排方案与专家轨迹研究

日期：2026-08-04  
研究范围：`log/2026-08-04` 的 15 份 replay、Frontier/Hamburger/Soil/v18 notebook、V006～V012 本地实验。

## 结论摘要

当前比赛的有效主路线不是在线强化学习，而是：

```text
高质量完整生产轨迹
        +
稳定的 farmer/hands 调度
        +
低频状态路由
        +
完整市场策略专家
        +
少量终局和市场修正
```

前排方案已经在宏观生产结构上高度收敛。我们当前 baseline 的主要问题不是缺少更多局部优先级，而是缺少一条同等级的完整生产路线。V011 的调度补丁只能减少一部分 `PASS`，却不能解决雇佣、种植、动物维护、路线和现金流之间的长期耦合。

## 1. 比赛中最重要的机制

### 1.1 并行动作经济

每个 farmer/hand 每回合独立行动，hands 数量直接决定每天能完成多少维护、种植、收获和仓库搬运。因此前排先投资 hands，再扩大生产面积。

从 replay 观察到的共同节奏是：

```text
Day 0：4 hands
Day 2：5 hands
Day 4：6 hands
Day 7：8 hands
Day 8：10 hands
Day 9：11 hands
Day 10：12 hands
```

### 1.2 土地和生产分阶段扩张

前排普遍只解锁 NW、NE、SW，避免购买 SE。第三象限通常在 Day 11 左右解锁，与第二波草莓、动物扩张同步。

### 1.3 作物波次

- 早期使用 MELON 形成一次性现金爆发；
- 中期转向 STRAWBERRY 的持续产出；
- WHEAT 主要服务动物饲料和终局现金流；
- Day 26～27 左右将衰退地块转成快速 WHEAT，作为终局资产重置。

### 1.4 动物维护是固定负载

常见结构是 8 COW + 6 SHEEP。稳定后每天约有 14 个 `FEED` 和 `CARE` 维护任务，不能让普通种植任务长期抢占这些动作。

### 1.5 仓库是缓冲池

前排不是把 shed 维持为空，而是长期保持约 80～100 的高利用率，通过周期性 SELL 避免爆仓。仓库容量、DROP、销售顺序和市场价格共同构成短期现金流系统。

### 1.6 高价值商品的市场拥堵

MELON、STRAWBERRY、MILK、WOOL 的价格对供给更加敏感。多个顶级农场同时销售时，策略差异往往集中在 100 个左右的市场回合，而不是 field route。

## 2. 前排 30 天的共同趋势

以下是 15 份 replay 中 30 个 player-game 轨迹的中位趋势：

| 时点 | 峰值 hands | 解锁象限 | MELON | STRAWBERRY | WHEAT | COW | SHEEP | 活跃植物 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day 1 | 4 | 1 | 7 | 0 | 10 | 3 | 1 | 17 |
| Day 7 | 8 | 1 | 10 | 2 | 6 | 4 | 2 | 18 |
| Day 8 | 10 | 2 | 11 | 6 | 10 | 5 | 2 | 27 |
| Day 10 | 12 | 2 | 11 | 17 | 7 | 6 | 6 | 35 |
| Day 11 | 12 | 3 | 4 | 21 | 8 | 6 | 6 | 33 |
| Day 12 | 12 | 3 | 6 | 27 | 6 | 8 | 6 | 39 |
| Day 18 | 12 | 3 | 9 | 40 | 5 | 8 | 6 | 54 |
| Day 24 | 12 | 3 | 6 | 38 | 0 | 8 | 6 | 44 |
| Day 27 | 12 | 3 | 2 | 22 | 30 | 8 | 6 | 54 |
| Day 29 | 12 | 3 | 0 | 12 | 24 | 8 | 5 | 36 |

核心结构可以概括为：

```text
快速补齐并行能力
→ MELON + WHEAT 开局
→ STRAWBERRY + 牛羊中期生产
→ 收获和清理旧作物
→ 终局 WHEAT reset + liquidation
```

## 3. 前排 notebook 的实际方法

### Frontier

Frontier 不是实时规划器，而是可见状态轨迹路由器：

1. 保存多条高分 replay；
2. 提取自己的现金、hands、土地、作物、动物、shed、价格等公开特征；
3. 每 24 回合计算当前状态与候选 replay 快照的距离；
4. 选择最接近的轨迹并返回对应动作。

它的关键是候选轨迹质量和状态空间收敛，而不是复杂模型。

### Hamburger / Soil

这类方案通常是：

- 固定 720 回合 field route；
- 只有在杂草导致固定动作失效时做 `DIG` 修复；
- 716 回合后做终局 PLACE/SELL；
- 确认双方拓扑和现金高度相似后，提前增加一笔高价值商品卖单。

固定路线被保留得非常严格，因为重新分配 farmer/hands 很容易破坏后续现金流和位置连续性。

### v18 Closed Loop

v18 的实际结构是：

```text
seat-specific board route
        +
4 个完整 market expert
        +
29 维公开状态
        +
每天一次选择 + stay bonus
```

market expert 不只是 SELL 排序，还包含 SELL 时间/数量、BUY、HIRE 和 BUY_LAND。v18 notebook 报告的 40/53 是 counterfactual replay holdout，不是官方 LB 分数；其固定版和闭环版都为 40/53，说明主要收益来自刷新完整市场策略，动态 gate 本身没有单独增加胜利。

因此这些方案在运行时基本不是强化学习。最多可以说使用了离线结果搜索、轨迹挖掘或轻量的 outcome-trained gate；没有在线训练、LLM 调用或 PPO/DQN 推理。

## 4. 我们的实验结论

### V011：调度补丁没有解决架构上限

V011 尝试高价值作物浇水、收获和仓库协同。结果显示：

- `PASS` 下降一部分，但主要转化成额外移动；
- `FEED`/`CARE` 有被任务竞争挤压的风险；
- 高价值作物收获次数没有接近前排水平；
- 每回合重新建全局任务池，破坏了稳定的单位职责和路线。

结论：当前 baseline 不是继续添加 urgency 规则就能达到前排水平，需要先替换底层完整生产路线。

### V012：五条代表轨迹 vs v18

从日志中按观察到的平均终局现金选出五位选手，并为每位选取一条接近自身中位轨迹的 replay：

| 选手 | replay |
|---|---:|
| mohit | 89817349 |
| automatylicza | 89830916 |
| manual player | 89820316 |
| navazsh fathi | 89830910 |
| Lucien de Rubempre | 89822684 |

五条完整 bundle 分别与原始 v18 对战 12 局：

| 路线 | 平均现金差 | W/T/L | 胜率 |
|---|---:|---:|---:|
| automatylicza | +382 | 9/0/3 | 75.0% |
| mohit | -1,738 | 1/0/11 | 8.3% |
| manual player | -4,966 | 1/0/11 | 8.3% |
| navazsh fathi | -5,231 | 1/0/11 | 8.3% |
| Lucien de Rubempre | -5,853 | 1/0/11 | 8.3% |

这说明五个方案不能平权混合，`automatylicza` 的完整路线明显最适合作为 board backbone。

### V012：用五条 replay 替换 v18 experts

替换版采用：

- board route 固定为 `automatylicza`；
- 五条 replay 作为 market experts；
- 每天边界选择一次；
- `automatylicza` 使用 0.75 outcome prior，其余专家为 0；
- 保留 v18 的 29 维距离和 stay bonus。

结果：

| 版本 | 平均现金差 | W/T/L | 胜率 | 最低现金 |
|---|---:|---:|---:|---:|
| 五专家替换版 vs 原始 v18 | +404 | 10/0/2 | 83.3% | 103,410 |

零先验消融只有 2/12 胜，因为 Day 0 由于原型近似并列，错误选择了 `navazsh fathi`。这说明低频 gate 必须有离线结果先验，否则“最近状态”在开局没有足够区分度。

## 5. 提交版本与复现

最终自包含提交版本：

- `baseline/history/v012_top5_replaced_v18/main.py`
- `baseline/artifacts/v012_top5_replaced_v18/submission.tar.gz`
- 构建脚本：`experiments/build_v012_submission.py`

生成版本不依赖本地 notebook 或 replay 日志，压缩包内部根目录只有 `main.py`。提交命令：

```bash
kaggle competitions submit kaggriculture \
  -f baseline/artifacts/v012_top5_replaced_v18/submission.tar.gz \
  -m "V012 top5 replay experts replacing v18"
```

根目录 `main.py` 保持原 baseline，不在本研究中自动 promotion。

## 6. 后续建议

1. 先提交 V012 自包含版本，观察官方对局和真实对手反馈；
2. 继续以 `automatylicza` 为 board backbone，不再让五条 field route 平权切换；
3. 分析 V012 与 v18 的差异集中在哪些市场回合、商品和订单数量；
4. 只有当某个 market expert 在 holdout 上稳定增加胜场时，才保留动态切换；
5. 后续再研究更细的 hand ownership、终局 WHEAT reset 和市场订单容量，不回到每回合全局任务池架构。

## 7. 研究边界

日志没有包含官方 leaderboard 排名字段，因此“五位选手”的选择是基于本地 15 份 replay 的观察现金，不等同于官方最终排名。V012 的 10/12 也是本地 open-loop 对战结果，不能替代 Kaggle 官方评测；真正的晋级标准仍然是官方 W/T/L、现金和跨提交稳定性。
