# RL-010 cross-graft validation

Each candidate preserves the complete source mechanism from one notebook and
replaces only its frozen route table with an independently archived route.
The builder does not append the V031 shared market overlay.

Core matrix: six mechanisms × six archived routes. The current manifest has
36 candidates. Optional replay routes are added with
`--replay name=path:seat` and are kept for validation only.

## Rebuild and test

```bash
python experiments/build_cross_graft_validation.py
python experiments/test_cross_graft_validation.py
```

For a replay route, pass the action source explicitly. The builder extracts
`steps[t+1][seat]["action"]`, stores its route hash and macro statistics, and
does not put the replay path into the generated runtime agent:

```bash
python experiments/build_cross_graft_validation.py \
  --replay top10_14=/absolute/path/replay.json:0
```

The first-stage matrix can be started with the three seed smoke set:

```bash
python experiments/run_cross_graft_validation.py \
  --output baseline/artifacts/rl_010_milk_bidirectional/cross_graft/benchmark_core
```

To compare RL-010 with a cross-grafted mechanism, first build the RL-010
artifact, then use the same runner. `rl010` is loaded from the standalone
artifact and the other names are candidate directories in this folder:

```bash
python experiments/run_cross_graft_validation.py \
  --candidate rl010 \
  --opponent v27_x_v22 \
  --opponent adaptive_x_v27 \
  --seed 17 --seed 42 --seed 2026 \
  --output baseline/artifacts/rl_010_milk_bidirectional/cross_graft/rl010_validation
```

Each row records the mechanism and route hashes, macro route label, actual
MILK sales, timing, and field/hands/market differences from the injected pure
route. A cross-graft may legitimately have different market actions because
those changes belong to mechanism A; field and hands differences are the
important invariants for separating route effects from market effects.
