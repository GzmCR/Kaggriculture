# RL-003 Trade Timing

RL-003 keeps the v22 farmer and hands route unchanged. It applies a linear,
item-specific contextual bandit to existing premium SELL events and can delay
one unit to the next same-product SELL event. The action space is:

- `0`: v22 control
- `1`: delay one MILK unit
- `2`: delay one WOOL unit
- `3`: delay one STRAWBERRY unit
- `4`: delay one MELON unit

The selector has 40 bounded features, a maximum of 8 delayed events, and
never creates a new product order or changes non-premium orders.

## Calibration

The initial fit used 118 paired cash-delta samples from the V029 quantity
counterfactuals. The item models had 90 MILK samples and 28 STRAWBERRY
samples; there were not enough WOOL or MELON samples to activate them.

The threshold sweep used seeds `17, 42, 2026, 217, 317, 733`, both seats,
against v22:

| threshold | mean cash | mean margin | wins |
| ---: | ---: | ---: | ---: |
| 0 | 121005.6 | 27.8 | 10/12 |
| 5 | 121056.9 | 121.2 | 11/12 |
| 7.5 | 121041.3 | 118.3 | 11/12 |
| 10 | 121016.6 | 98.2 | 11/12 |
| 12 | 120979.8 | 46.3 | 11/12 |

The deployed threshold is `5`. It consistently selects eight MILK timing
events on the current v22 route. This is slightly above the V029 control
schedule in the same 6-seed comparison (`121049.3` mean cash), but it is not
yet evidence of general state-dependent learning.

## Verification

- v22 comparison: 12 games, 11 wins, 1 loss, all `DONE`, errors `0`, invalid
  action shape `0`, p99 `0.55 ms`.
- Strong-pool check: 36 games, including v22, v022c, v13-R3, v21-1,
  Hamburger, and Frontier; all `DONE`, errors `0`, invalid action shape `0`.
- Against v022c, v13-R3, v21-1, Hamburger, and Frontier: 30 wins in 30
  games on the three-seed check.
- Farmer and hands action differences remain `0`; only existing market SELL
  quantities are changed.

## Limitation

The current counterfactual samples mostly contain route/time features, so the
learned policy is effectively a fixed MILK schedule. The next useful RL step
is to collect paired interventions together with the actual observation at
each event, then train a conservative gate on price, market inventory,
opponent public supply, and current-vs-future order size.

The candidate submission is `submission.tar.gz`; the root `main.py` was not
modified.
