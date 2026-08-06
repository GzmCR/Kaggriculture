# V023 quick validation report

Generated on 2026-08-06.  The 1500～2500 folder was used only as an external
holdout: 11 unique episodes, 22 fixed opponent-seat cases.

| Candidate | Mean cash | Min cash | W/T/L | DONE | Errors | Invalid | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| V012 control | 127,790.5 | 82,642 | 8/0/14 | 22/22 | 0 | 0 | 0.178 |
| V023a high output | 132,070.6 | 98,589 | 14/0/8 | 22/22 | 0 | 0 | 0.250 |
| V023b stable 12-hands | 19,741.5 | 14,988 | 1/0/21 | 22/22 | 0 | 0 | 0.305 |
| V023c high-hands proxy | 50,146.5 | 44,278 | 1/0/21 | 22/22 | 0 | 0 | 0.253 |
| V023d portfolio, corrected bootstrap | 132,070.6 | 98,589 | 14/0/8 | 22/22 | 0 | 0 | 0.242 |

The portfolio row uses the second run in
`recheck_portfolio2/replay_counterfactual_summary.csv`; the first portfolio
run exposed and is not used because its majority-voted bootstrap mixed
incompatible coordinate routes.  The builder now uses exact consensus only
and otherwise keeps the high-output medoid bootstrap.

This is not a promotion decision.  The stable and high-hands routes are
research candidates only: they complete legally but do not generalize to this
external holdout.  The full local matrix remains runnable with
`experiments/run_v023_benchmark.py`.
