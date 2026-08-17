# V032 route-conditioned timing

V032 uses V27 order-only as the default route. The optional 8C4S candidate is extracted offline from the latest high-score notebook and is not selected automatically.

Runtime order: timing transfer -> V27 price-impact SELL reorder -> legality/hand alignment. Unknown or weakly matched opponents are a strict V27 order-only fallback.

Profiles are anonymous offline data. The submission does not read replay files, notebooks, names, scores, seeds or the network.
