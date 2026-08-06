# V022 fresh-route experiment

This artifact contains four independent candidates derived from the two
supplied public notebooks.

- `v022a_weed_recovery`: clean V012 route plus actor-local visible-WEED
  `DIG -> retry -> at most 8 turns of catch-up`.  Market orders are untouched.
- `v022b_fresh_medoid`: anonymous complete route extracted from the
  159/160 notebook, with no runtime tactical overlay.
- `v022c_medoid_recovery`: the same anonymous route plus local WEED recovery.
- `v022d_medoid_recovery_tactical`: the 144/150 notebook artifact, containing
  its fit-only medoid, balanced market hazard, one-turn half-quantity
  preemption, public-similarity gate and bounded WEED recovery.

The notebook payloads are used only at build time.  Runtime code uses no
notebook, replay, team name, score, seed lookup, network or external API.
Each candidate has its own `main.py` and `submission.tar.gz` with `main.py` at
the archive root.  The root repository `main.py` is not modified.

The notebook scores are local/replay research results, not official Kaggle
leaderboard guarantees.  Candidates must pass both fixed-replay leave-one-out
checks and fresh closed-loop games before any promotion is considered.

## Local matrix result

The completed matrix contains 420 games: 5 candidate/control policies × 7
opponents × 6 seeds × 2 seats.  Every game reached `DONE`; all candidates had
zero agent errors and zero invalid action shapes.

| Candidate | Mean cash | Minimum cash | W/T/L | Gate |
|---|---:|---:|---:|---|
| V012 control | 149,462 | 101,598 | 75/0/9 | control |
| V022a WEED recovery | 148,720 | 101,598 | 78/1/5 | not promoted: mean cash lower |
| V022b fresh medoid | 142,163 | 82,780 | 78/0/6 | not promoted: mean/min cash lower |
| V022c medoid + recovery | 142,199 | 61,994 | 80/0/4 | not promoted: mean/min cash lower |
| V022d medoid + tactical memory | 146,399 | 106,473 | 84/0/0 | not promoted: mean cash lower |

V022d is the strongest win-rate candidate in this local matrix, but it does
not meet the cash gate, so root `main.py` remains unchanged.  The machine
readable details are in `full_matrix_v2/matrix_summary.csv` and
`full_matrix_v2/gate_report.json`.
