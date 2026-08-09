# V027 Product-Level Sell-Wave Shift

## Scope

V027 keeps the embedded v22 route, WEED recovery, and price-impact SELL
ordering. It changes only quantities inside existing MELON or STRAWBERRY
SELL orders and records an exact future-order deduction. The root `main.py`
and the v22 control were not modified.

The benchmark below is the 192-game development matrix: four candidates,
eight opponents, three seeds, and two seats. It is used as a screening matrix
because the candidate behavior already failed the promotion gate.

## Result

| Candidate | Mean cash | Min cash | W/T/L | Action diff | Shifted MELON | Gate |
|---|---:|---:|---:|---:|---:|---|
| v22 control | 128,993.69 | 77,933 | 43/4/1 | 0 | 0 | control |
| V027a MELON | 129,247.65 | 77,867 | 42/0/6 | market only | 84 units | reject |
| V027b mirror | 128,614.96 | 77,933 | 43/4/1 | 0 | 0 | reject, never activated |
| V027c products | 129,637.10 | 77,933 | 43/4/1 | 0 | 0 | reject, never activated |

The apparent V027c mean improvement is not a policy improvement: all 48 games
had zero action differences from v22, so the small difference comes from
independent random-opponent trajectories. Its cash ratio is also just below
the 0.5 percent promotion threshold.

## Causal Observation

V027a shifted MELON in only two route windows:

- step 258 -> 260: 1 unit, 48/48 games;
- step 522 -> 524: 6 units, 6 games.

The shift was quantity-conserving, but it reduced cash against every fixed
opponent in the development matrix. The paired delta versus v22 was -75
against v22, -70.7 against v022c, -35.7 against V13-R3, -37.0 against V21.1,
-57.7 against V012, -62.3 against V024a, and -56.7 against V025b. The first
window is therefore not a useful collision-avoidance opportunity even when the
price and inventory gates pass.

V027b and V027c checked the public state after the daily hiring window. None
of the 48 games satisfied both day-8 and day-12 mirror checks, so both safely
returned the exact v22 action. The synthetic gate test does latch and release
as designed, so this is a data/threshold result rather than an untested state
machine.

All 192 games reached `DONE`; agent errors and invalid market shapes were zero.
The maximum observed p99 agent time was 1.299 ms, far below the 1000 ms limit.

## Decision

V027 is not promoted and no V027 submission is recommended. Keep v22 as the
active control. The generated candidate files and diagnostics remain for
research; `submission_ready` is false.

Raw diagnostics are in `matrix/matrix_raw.jsonl`; summaries are in
`matrix/matrix_summary.json` and `matrix/matrix_summary.csv`.
