# Agent History

Each version directory contains a runnable `main.py` snapshot and the
debugger used with that snapshot. The root `main.py` is the active version.

## Versions

- `v000_baseline`: scenario-aware baseline before the livestock scoring experiment.
- `v001_soft_livestock_score`: experimental season-economics livestock score; retained for comparison but not active.
- `v002_own_supply_sales`: experimental own-supply-aware sales forecast; no meaningful score change.
- `v003_crop_wheat_mix`: experimental replacement of two strawberry slots with wheat slots; slightly lower average score.
- `v004_dynamic_strawberry_mix`: conservative opponent-supply-aware strawberry adjustment; no score change.
- `v005_early_strawberry_price_mix`: opening-price strawberry adjustment; no score change.
