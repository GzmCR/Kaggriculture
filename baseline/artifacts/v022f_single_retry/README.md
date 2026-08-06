# V022f single-retry WEED recovery

V022f is an ablation of V022e. It keeps the same complete route and market
actions, but removes the second `DIG -> retry` attempt. After the first retry
fails, the actor is suppressed briefly and returns to the current route.
V022c and V022e remain unchanged; this candidate is experimental.
