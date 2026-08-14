# V032-R3 研究候选

本目录保留 R3 的研究说明；实际 paired 环境评测由
`experiments/run_v032_r3_bidirectional.py` 生成。当前没有把未验证的
提前/延后规则包装成提交版 `main.py`，因此不会误替换 V27 control。

R3 只有在多个 seed、两个 seat 和多个对手上同时通过真实库存、现金流、
仓库容量和订单守恒检查后，才会生成运行时提交候选。
