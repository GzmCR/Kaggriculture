# RL-005 Multi-Opponent Trade Timing

RL-005 keeps the V22 farmer, hands, production, and base market route. It
learns only whether to delay one existing premium SELL unit. Training uses
paired counterfactual games against the local 2026-08-09 opponent pool:

```text
V22 control vs opponent
V22 + one-event delay vs the same opponent, seed, and seat
```

Opponent names are not model inputs. The model only receives observable market,
farm, inventory, and public opponent features.

## Data

- 324 paired samples
- 54 control games
- 9 executable notebook agents
- 6 event groups
- 54 independent `(seed, seat, opponent)` samples per event
- All collection games `DONE`

The updated rank notebook contributes the embedded `C45` strategy as
`rank_c45`. The rank notebook itself is an evaluator, not an agent. Several
other files are byte-identical wrappers and are retained for source tracking.

The five MILK events have positive samples only and remain eligible. The
STRAWBERRY event has a `-3` sample, so the existing training gate disables it
and falls back to V22.

## Holdout

Seeds `811, 919, 1021`, two seats, 9 opponents, 108 games total:

| opponent group | V22 mean cash | RL-005 mean cash | improvement |
| --- | ---: | ---: | ---: |
| adaptive replay wrappers | 181686.7 | 181726.8 | +40.2 |
| frontier soil | 124540.7 | 124577.0 | +36.3 |
| high score pipeline | 124024.0 | 124058.7 | +34.7 |
| top meta | 117966.0 | 118040.0 | +74.0 |

All 9 opponent groups had 6/6 RL-005 wins. Agent errors and invalid actions
were zero. Farmer and hands action diffs were zero. The largest p99 latency
was below `0.48 ms`.

## Artifact

- `main.py` is self-contained and CPU-only.
- `submission.tar.gz` contains only root-level `main.py`.
- Root project `main.py` is unchanged.

## Reproduce

```powershell
D:\kg311\python.exe experiments\run_rl_005_multi_opponent_data.py --output baseline\artifacts\rl_005_multi_opponent\data_train --seed 17 --seed 217 --seed 733 --event MILK:215 --event MILK:260 --event MILK:288 --event MILK:336 --event MILK:452 --event STRAWBERRY:480
D:\kg311\python.exe experiments\build_rl_005_multi_opponent.py --samples baseline\artifacts\rl_005_multi_opponent\data_train\samples.jsonl
D:\kg311\python.exe experiments\run_rl_005_benchmark.py --candidates v22 rl005 --seeds 811 919 1021 --output baseline\artifacts\rl_005_multi_opponent\benchmark_holdout
```
