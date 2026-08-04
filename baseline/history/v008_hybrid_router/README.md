# V008 Hybrid Router

This candidate keeps the current rule-based agent as a fallback and embeds the
frontier public trajectory data as an optional 24-turn macro policy.

At each 24-turn boundary it computes the nearest frontier state. The frontier
action is used only when the state distance is below the selected threshold and
the current hand count is compatible. Urgent watering, feeding, or terminal
work can fall back to the current planner for the individual turn.

The root `main.py` remains the active submission until the V008 benchmark gate
passes on tuning and holdout seeds.

## Result

The tuning winner was `v008_frontier`, equivalent to the 100% distance
threshold candidate with emergency current-policy fallbacks. On tuning seeds
`17, 42, 2026` it reached mean cash `146748.47` versus current `112972.47`
(`+29.90%`). On holdout seeds `217, 317, 733` it reached mean cash `138557.63`
versus current `115231.33` (`+20.24%`). All evaluated games finished `DONE`
with zero agent errors and zero invalid action shapes, and measured p99 agent
time stayed below `3 ms`.

The candidate is **not promoted**: its holdout minimum cash was `79383`, only
`90.73%` of current's `87490`, below the required `97%`. This folder is an
experiment archive; the active root `main.py` is unchanged.

- `main.py` — selected tuning candidate, retained for reproducibility only.
- `selection.json` — threshold selection and gate report.
- `debug_local.py` — matching local debug runner snapshot.

The candidate also normalizes route indices defensively because Kaggle may
deliver persisted values as `Struct` objects; malformed route state now falls
back to the current policy instead of raising inside `TRACES[...]`.
