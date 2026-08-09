# RL-004 Trade Timing

RL-004 keeps the V022c/V22 farmer and hands route unchanged. It only delays one
existing premium SELL unit to the next same-product route opportunity when the
event has enough paired counterfactual support and the state-aware model has a
positive lower confidence bound.

## Current Artifact

- Training data: `data_train_augmented/samples.jsonl`
- Samples: 180 paired event interventions
- Supported event support: 12 independent `(seed, seat)` episodes per event
- Supported products: MILK and STRAWBERRY
- Model: NumPy ridge regression with an event intercept and 29 observable features
- Submission: `submission.tar.gz` containing only `main.py`
- Root `main.py`: unchanged

The event-level training gate disables the high-variance `MILK 432->450` and
`MILK 480->504` events. Unsupported events use the V22 control action.

## Validation

Strict holdout against V22, 6 seeds and 2 seats, 36 games total:

| candidate | mean cash | minimum cash | wins | ties | losses | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V22 control | 124893.7 | 97889 | 0 | 12 | 0 | 0.240 |
| RL-003 | 124987.3 | 98017 | 12 | 0 | 0 | 0.384 |
| RL-004 | 124994.3 | 97974 | 12 | 0 | 0 | 0.422 |

All 36 games were `DONE`; agent errors and invalid actions were zero. RL-004
farmer and hands action diffs were zero. Its maximum market action diff was 14
calls in a game, with no market-order overflow.

The 19-game RL-003 losslog regression set produced zero errors. It no longer
selects one fixed delay schedule: selected events vary with observed state, and
unsupported events fall back to control.

The 144-game, 3-seed pool check was stable in engineering terms: all games
completed, zero errors or invalid actions, and p99 below 0.71 ms. It was not a
universal performance win: the random opponent was slightly better with V22,
and one of the three-seed V22 self-play pairs was unfavorable. Keep RL-004 as
an experimental submission candidate; do not replace the formal baseline
without fresh online confirmation.

## Reproduce

```powershell
D:\kg311\python.exe experiments\build_rl_004_trade_timing.py --samples baseline\artifacts\rl_004_trade_timing\data_train_augmented\samples.jsonl
D:\kg311\python.exe experiments\run_rl_004_benchmark.py --candidates v22 rl003 rl004 --opponents v22 --seeds 811 919 1021 1123 1229 1337 --output baseline\artifacts\rl_004_trade_timing\benchmark_holdout_v4
D:\kg311\python.exe experiments\run_rl_004_losslog.py --output baseline\artifacts\rl_004_trade_timing\losslog_holdout_final
```
