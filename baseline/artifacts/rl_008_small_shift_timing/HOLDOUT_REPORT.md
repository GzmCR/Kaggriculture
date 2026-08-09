# RL-008 Holdout Report

## Data

- Training counterfactuals: 4,536 rows from 7 source-deduplicated opponents, 6 seeds, 2 seats, and 6 route events.
- Applied and repaid training shifts: 3,276 / 3,276.
- Holdout benchmark: 210 complete games from 7 opponents, seeds `811`, `919`, `1021`, and both seats.

## Model Fit

All nine action heads met the 12-group support threshold. The six small-preemption heads were negative on training data, while delay heads were mildly positive:

| Action family | Mean cash delta | Positive rate |
|---|---:|---:|
| PREEMPT_H1/H2/H3, 1 unit | about -74 to -75 | 7.1% |
| PREEMPT_H1/H2/H3, 25% | about -126 to -130 | 7.7% |
| DELAY_1 | +8.0 | 82.1% |
| DELAY_25 | +11.5 | 83.3% |
| DELAY_50 | +20.1 | 84.8% |

The LCB gate rejected the preemption heads and, on the holdout routes, also rejected all dynamic choices. This is a valid conservative result: the packaged candidates fall back to the V022c control when uncertainty is too high.

## Holdout Summary

| Candidate | Mean cash | Minimum cash | Games | DONE | Errors | Invalid | Max p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V022c control | 156,182.48 | 100,606 | 42 | 42 | 0 | 0 | 0.2703 ms |
| Gated preempt | 156,199.86 | 100,787 | 42 | 42 | 0 | 0 | 0.8299 ms |
| Ungated preempt | 156,199.86 | 100,787 | 42 | 42 | 0 | 0 | 1.2012 ms |
| Gated bidirectional | 156,202.86 | 100,632 | 42 | 42 | 0 | 0 | 0.9335 ms |
| Ungated bidirectional | 156,202.86 | 100,632 | 42 | 42 | 0 | 0 | 1.0155 ms |

All candidates won all 42 holdout games against the selected opponent pool. The gated bidirectional version is the strongest technical candidate, but it is not promoted as a submission because the runtime log shows zero accepted non-CONTROL decisions.
