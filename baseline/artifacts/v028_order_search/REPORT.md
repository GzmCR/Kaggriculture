# V028 order search report

## Scope

V028 starts from the embedded 44-46 v22 route. The farmer route, hands route,
SELL quantities, SELL products, non-SELL order positions, and terminal logic
are unchanged. The overlay only permutes existing premium SELL orders in their
existing slots.

## Verification

- experiments/test_v028_order_search.py: passed for all three candidates.
- Kaggle-style callable check: the last callable is agent.
- Smoke: 8 complete games against v22, all DONE, zero agent errors, zero
  invalid market shapes.
- Smoke action diff: farmer 0, hands 0, market 0 after the per-product
  inventory fix.
- v22/v028a development check: 8 complete games against v022c and v13-r3,
  all DONE, zero errors/invalid actions, and v028a was action-identical to
  v22 in all games.

## Result

No candidate passed the activation threshold in the tested games:

- v028a margin: 50 coins;
- v028b margin: 100 coins;
- v028c robust margin: 50 coins under v22 and reversed premium-slot shadows.

The candidates therefore fell back to v22 on every turn. This is a useful
negative result: under the public v22 shadow, the existing price-impact order
already dominates every tested premium-slot permutation by the required
safety margin. No submission is promoted and the root main.py is unchanged.

## Artifacts

- v028a_marginal_order/submission.tar.gz
- v028b_safe_order/submission.tar.gz
- v028c_robust_order/submission.tar.gz
- smoke_fixed/matrix_summary.json
- dev_seed17_small/matrix_summary.json
- v22_3seed/matrix_summary.json
- gate_report.json
