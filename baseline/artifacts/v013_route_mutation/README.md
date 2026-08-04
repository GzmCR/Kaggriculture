# V013 Local Route Mutation Search

V013 searches farmer/hands routes while using V012 as the market policy.
The runner is:

```text
D:\kg311\python.exe experiments\run_v013_route_search.py --stage calibration
D:\kg311\python.exe experiments\run_v013_route_search.py --stage final
```

The original `log/2026-08-04` replay directory was not present in this
workspace. The run therefore recovered five routes from the V012 embedded
payload and two public notebook traces. The source and fallback details are in
`source_manifest.json`; recovered aliases are marked in the candidate names.

The calibration matrix completed successfully for 24 candidates. The final
matrix completed 504 games, all with `DONE` and zero agent exceptions. No
candidate passed the holdout gate, so no V013 submission package was generated
and V012 remains the recommended submission.

Important result from `gate_report.json`:

- V012 control holdout mean: `149223.12`
- Best V012/automatylicza route variant: `149109.17` (`-0.08%`)
- Soil day-10 crossover: `147363.19` (`-1.25%`)
- Kaito builder trace: `20223.69` (`-86.45%`)

Hand-list mismatches are recorded diagnostically because the Kaggriculture
environment silently ignores missing or extra hand actions.
