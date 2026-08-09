# V027: product-level sell-wave shift on the v22 route

V027 keeps the 44-46 v22 route, WEED recovery, and price-impact SELL ordering.
It only moves a bounded quantity from a future existing premium SELL into the
current existing SELL of the same product.  The future order receives an exact
ledger deduction, so total product sales and all farmer/hands actions remain
unchanged.

Candidates:

- `v027a_melon_ratio`: MELON only, up to 25 percent / 6 units, with an
  eight-turn cooldown and market-price/inventory gates.
- `v027b_mirror_gated`: the same MELON rule, enabled only after the public
  opponent matches the v22-like structure on both day 8 and day 12.
- `v027c_product_specific`: the gated MELON rule plus a 12.5 percent
  STRAWBERRY ablation.  At most one product is adjusted per turn.

Dynamic quantity changes stop before step 648.  No pending deduction may be
scheduled at or after step 672, where every candidate returns the v22 action.
The root `main.py` and the v22 control are not modified.
