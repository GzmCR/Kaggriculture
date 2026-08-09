# V028: v22 premium SELL order search

V028 keeps the embedded v22 farmer/hands route and its existing price-impact
market layer. It only permutes the contents of already-existing premium SELL
slots: MELON, STRAWBERRY, MILK, and WOOL.

The runtime simulates the environment's per-order, per-unit lockstep market
against a public v22 route shadow. A permutation is used only when its
predicted cash gain exceeds the candidate safety margin. Quantities, products,
non-SELL slots, BUY/HIRE/BUY_LAND orders, and all field actions are unchanged.

Candidates:

- v028a_marginal_order: 50 coin predicted-gain margin.
- v028b_safe_order: 100 coin predicted-gain margin.
- v028c_robust_order: 50 coin margin under both the v22 shadow and a reversed
  premium-slot stress shadow.

All candidates stop changing order slots at step 672 and fall back to v22 for
the terminal clear-out. These are experimental candidates; the root main.py
and the v22 control are not modified.
