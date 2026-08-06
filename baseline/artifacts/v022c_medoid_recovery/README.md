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
