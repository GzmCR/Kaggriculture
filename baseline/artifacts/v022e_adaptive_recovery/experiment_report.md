# V022e adaptive WEED recovery report

V022e was evaluated against the unchanged V022c medoid-recovery route.  The
matrix covered 192 games: 2 candidates × 8 opponents × 6 seeds × 2 seats.
The plan text said 168 games, but the listed opponent set contains eight
opponents, which produces 192 games.

| Candidate | Games | Mean cash | Min cash | W/T/L | DONE | Errors | Invalid | Max p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V022c control | 96 | 125,072.3 | 1,020 | 80/0/16 | 96/96 | 0 | 0 | 0.286 |
| V022e adaptive | 96 | 123,048.1 | 5,314 | 78/0/18 | 96/96 | 0 | 0 | 0.306 |

V022e therefore does not pass the promotion gate: mean cash is about 1.62%
below V022c and it loses two additional games.  It does pass legality,
completion, minimum-cash, and latency checks.

Recovery totals for V022e across its 96 games:

- visible WEED repairs: 314;
- first retry successes: 237;
- second retry successes: 12;
- early releases: 42;
- catch-up actions: 1,001;
- abandoned transactions after two failures: 37.

The adaptive logic is useful diagnostically: it confirms that most successful
retries still need catch-up, while early release is relatively rare.  The
additional losses suggest that the second-failure suppression/return-to-current
route path is too aggressive for some open-loop farm states.  V022c remains the
recommended control and was not replaced.

Raw data and gate output are in `full_matrix/`.
