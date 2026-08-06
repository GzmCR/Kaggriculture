# V022e adaptive WEED recovery

V022e keeps the same complete anonymous route and market actions as V022c.
Only the actor-local visible-WEED transaction is changed:

`DIG -> retry -> observe tile -> release when safe, otherwise bounded catch-up`.

The retry is confirmed from the next observation.  A failed retry gets one
additional `DIG -> retry` attempt; a second failure suppresses the same actor
and tile briefly so the agent cannot loop forever.  Other actors and market
orders are copied unchanged.  The root `main.py` and V022c archive are not
modified by this builder.
