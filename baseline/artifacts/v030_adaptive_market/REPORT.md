# V030 Adaptive Market Experiment

## Conclusion

V030 is implemented and passes local full-game stability checks, but it does not
pass the return gate. Keep V029 as the control and leave the repository root
`main.py` unchanged.

- Keep for further research: `v030a_milk_momentum_gate`.
- Keep as a negative ablation: `v030b_cross_product_guard`.
- Do not submit either V030 variant yet.

## Full comparison

Command:

```text
D:\kg311\python.exe experiments\run_v030_benchmark.py --candidates v22 v029a v030a v030b --opponents v22 --seeds 17 42 2026 217 317 733 --output baseline\artifacts\v030_adaptive_market\benchmark_v22_6seed_conservative
```

| candidate | mean cash | min cash | win rate | p99 ms | DONE/errors/invalid |
|---|---:|---:|---:|---:|---|
| v029a | 121049.25 | 104024 | 11/12 | 0.4834 | 12/0/0 |
| v030a | 121044.42 | 104011 | 11/12 | 0.4795 | 12/0/0 |
| v030b | 120894.42 | 103840 | 1/12 | 0.5503 | 12/0/0 |
| v22 | 120927.58 | 103843 | 2/12 | 0.3049 | 12/0/0 |

V030a is 4.83 cash below V029a on average. The difference is small, but it
does not satisfy the requirement that the average return must not decrease.
V030b is about 154.83 cash below V029a and has a much lower win rate.

## Badcase regression

The seven downloaded V029 loss replays are used by the fixed-trace stress test:

- `baseline/artifacts/v030_adaptive_market/badcase_replay_7_conservative/matrix.csv`
- `baseline/artifacts/v030_adaptive_market/badcase_replay_7_conservative/matrix.json`

All 28 fixed-trace games finished with `DONE`. V030a and V030b had zero agent
errors and zero pending failures. The replay opponent does not recompute its
actions after the candidate changes the shared market state, so this test is
for diagnosis and regression only, not proof of online return.

V030a detects some MILK declines and cancels the original delay, but its gain
against the badcase traces is not stable. V030b often makes the result worse by
moving MELON/STRAWBERRY future orders into a lower-price window.

## Stability

The starter two-seat 720-step smoke test passed for both candidates:

- DONE: 4/4
- agent errors: 0
- invalid action shape: 0
- farmer/hands action diff: 0
- only market actions changed
- p99 action time: about 0.56 ms

The generated files both end with the public `agent` callable and can be used as
single-file Kaggle submission entries.

## Next step

Keep badcases as a regression set, but do not hard-code replay seed, opponent
identity, or replay-specific turns into the submitted policy. The next useful
experiment is a conservative classifier for whether to cancel the MILK delay,
calibrated with more non-loss replays. The cross-product early-sale rule should
remain out of the main candidate line for now.
