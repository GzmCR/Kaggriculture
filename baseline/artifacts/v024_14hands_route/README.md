# V024 14-hands route

The four candidates use one anonymous complete route medoid selected from the chronological 70% fit window of the 2026-08-07 Top10 data. The middle 15% is option validation and the newest 15% is future holdout. The 1500-2500 and `ours` folders are not used for fitting.

- `v024a_route14_control`: route plus legal market clipping and terminal liquidation.
- `v024b_route14_weed`: adds one actor-local DIG/retry recovery.
- `v024c_route14_order_memory`: adds high-confidence order-only memory.
- `v024d_route14_strict_r3`: adds the strict, cooldown-limited R3 front-run gate.

All submissions are self-contained and contain `main.py` at archive root. No candidate changes the repository root `main.py`.
