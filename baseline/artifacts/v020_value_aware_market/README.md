# V020 value-aware market strategy

V020 is a market-only experiment built from the clean V012 strategy. It keeps
the V012 farmer and farm-hand route, crop mix, and non-premium market orders.
It adjusts only `MELON`, `STRAWBERRY`, `MILK`, and `WOOL` sell quantities when
the current market shows a supply or price shock.

The controller uses a 72-turn market rollout with town demand, public farm
supply estimates, carried and shed inventory value, cash reserves, and
opponent-supply scenarios. It keeps the original V012 terminal schedule and
does not remove non-premium orders when the market order list is full.

## Variants

- `v020_balanced.py`: MILK-only intervention.
- `v020_conservative.py`: MILK and WOOL intervention with a larger minimum
  improvement threshold. This was the strongest replay candidate.
- `v020_sensitive.py`: all premium products. It was rejected because it is
  unstable on the fixed replay set.

## Verification

Build the self-contained submission with:

```powershell
D:\kg311\python.exe experiments/build_v020_value_aware.py
```

Run the local evaluation with:

```powershell
D:\kg311\python.exe experiments/run_v020_value_aware.py --stage replay --variants conservative
D:\kg311\python.exe experiments/run_v020_value_aware.py --stage matrix --variants conservative --opponents v012 v015a baseline v18 hamburger frontier starter random --seeds 17 42 2026 217 317 733
```

The generated archive is `submission.tar.gz`, with `main.py` at its root.

## Gate result

The 10 replay counterfactuals and 192 full local games all completed without
agent errors or invalid action shapes. Conservative V020 improved mean final
cash from `145348.83` to `145700.59` and the replay mean by `131.7`, while
raising the minimum final cash from `101598` to `102645`.

It did not pass the final gate because the overall win rate fell from `77/192`
to `69/192`. The root `main.py` and V012 were therefore left unchanged, and
this candidate has not been uploaded to Kaggle or GitHub.
