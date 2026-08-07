# V026: v22 route with V022c-family local recovery

V026 keeps the self-contained 14-hands high-output route and official
price-impact SELL ordering from the 44-46 v22 artifact.  It grafts only the
V022f single-retry actor-local visible-WEED recovery:

`DIG -> one retry -> observe -> early release or bounded catch-up`.

The second retry, mirror gate, opponent exposure market ranking, and old V022c
15-hands route are not included.  V026a changes only recovery.  V026b adds a
last-resort legality guard that clips an existing SELL quantity only when it is
greater than the currently visible shed-plus-carried inventory.

Both candidates are experimental and do not replace the repository root
`main.py`.
