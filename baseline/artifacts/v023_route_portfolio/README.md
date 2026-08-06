# V023 route portfolio

V023 is a route-refresh experiment built offline from the 2026-08-06 Top10
replays.  The Top10 folder is deduplicated by EpisodeId and clustered into
three anonymous macro-route medoids:

- `v023a_high_output_14hands`: high-output 14-hands route;
- `v023b_stable_12hands`: stable 12-hands route;
- `v023c_high_hands_15hands`: 15-hands research route with a low-confidence
  penalty in the portfolio selector;
- `v023d_early_portfolio`: common bootstrap for 48 turns, then one selection
  from the three routes using the agent's own public farm state.  The choice
  is locked through the end of the season.

Every generated submission is self-contained with `main.py` at archive root.
Runtime code does not read replay files, notebooks, team names, scores,
episode ids, seeds, or network resources.  It preserves selected field and
market actions, clips illegal SELL quantities, aligns hands dynamically, and
uses actor-local visible-WEED recovery plus terminal liquidation safeguards.

The `1500～2500` replay folder is held out by the benchmark and is not used to
build the runtime routes.  No candidate is promoted to the repository root by
this builder.
