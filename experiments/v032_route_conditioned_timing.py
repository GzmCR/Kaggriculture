"""Runtime overlay source for V032.

This file is appended to a frozen V27 route by the offline builder.  It is
deliberately self-contained at runtime: replay files, notebooks, identities,
scores, seeds and network access are not needed or consulted.

The overlay is conservative.  It first applies a one-event advance/delay
proposal, then runs the V27 price-impact reorder on the resulting quantities.
If route confidence, market evidence, inventory, storage, repayment or the
short market rollout is uncertain, it returns the V27 order-only action.
"""

V032_PREMIUM = ("MILK", "STRAWBERRY", "WOOL", "MELON")
V032_MAX_ORDERS = 10
V032_CUTOFF = 648
V032_TERMINAL_CUTOFF = 672
V032_MAX_TRANSFER = 30
V032_MIN_GAIN = 10.0
V032_ROUTE_CHECKPOINTS = (96, 144, 192, 240, 288)

V032_STATS = {}
V032_STATE = {
    0: {"last_step": -1, "pending": None, "route_hits": 0, "market_hits": 0,
        "known": False, "profile": None, "market_prev": None, "own_sell_prev": {},
        "last_checkpoint": -1, "market_observations": 0},
    1: {"last_step": -1, "pending": None, "route_hits": 0, "market_hits": 0,
        "known": False, "profile": None, "market_prev": None, "own_sell_prev": {},
        "last_checkpoint": -1, "market_observations": 0},
}


def _v032_stat(name, amount=1):
    V032_STATS[name] = V032_STATS.get(name, 0) + amount


def _v032_seat(obs):
    return 1 if int(_v031_get(obs, "player", 0) or 0) == 1 else 0


def _v032_reset(obs, step):
    state = V032_STATE[_v032_seat(obs)]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "pending": None, "route_hits": 0,
                      "market_hits": 0, "known": False, "profile": None,
                      "market_prev": None, "own_sell_prev": {},
                      "last_checkpoint": -1, "market_observations": 0})
    state["last_step"] = step
    return state


def _v032_copy(action):
    return _v031_copy_action(action)


def _v032_is_sell(order, item=None):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return False
    if str(order[0]).upper() != "SELL":
        return False
    return item is None or str(order[1]).upper() == str(item).upper()


def _v032_qty(action, item):
    return sum(max(0, _v031_int(order[2])) for order in action.get("market", [])
               if _v032_is_sell(order, item))


def _v032_visible_inventory(obs, item):
    private = _v031_get(obs, "private", {}) or {}
    total = _v031_int(_v031_get(_v031_get(private, "shed", {}) or {}, item, 0))
    for inventory in _v031_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            total += _v031_int(inventory.get(item, 0))
    return max(0, total)


def _v032_used_storage(obs):
    private = _v031_get(obs, "private", {}) or {}
    used = sum(max(0, _v031_int(value)) for value in
               (_v031_get(private, "shed", {}) or {}).values())
    # Seeds are a separate slot.  inventories are carried items and therefore
    # count toward the shed capacity once dropped; retain a reserve for them.
    for inventory in _v031_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            used += sum(max(0, _v031_int(value)) for value in inventory.values())
    return max(0, used)


def _v032_add_or_merge(action, item, quantity):
    quantity = max(0, _v031_int(quantity))
    if quantity <= 0:
        return False
    for order in action.get("market", []) or []:
        if _v032_is_sell(order, item):
            order[2] = max(0, _v031_int(order[2])) + quantity
            return True
    if len(action.get("market", []) or []) >= V032_MAX_ORDERS:
        return False
    action.setdefault("market", []).append(["SELL", str(item).upper(), quantity])
    return True


def _v032_reduce(action, item, quantity):
    remaining = max(0, _v031_int(quantity))
    if remaining <= 0:
        return True
    for order in action.get("market", []) or []:
        if not _v032_is_sell(order, item):
            continue
        current = max(0, _v031_int(order[2]))
        take = min(current, remaining)
        order[2] = current - take
        remaining -= take
        if remaining <= 0:
            return True
    return False


def _v032_signature(farm):
    tiles = _v031_get(farm, "tiles", []) or []
    counts = {"COW": 0, "SHEEP": 0, "GOOSE": 0, "PASTURE": 0,
              "COOP": 0, "WHEAT": 0, "STRAWBERRY": 0, "MELON": 0,
              "TOMATO": 0, "CARROT": 0, "WEED": 0}
    for row in tiles:
        if not isinstance(row, list):
            continue
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "")).upper()
            if kind == "PLANT":
                key = str(tile.get("crop", "")).upper()
            elif kind in ("COOP", "PASTURE"):
                key = str(tile.get("animal", "")).upper() or kind
                counts[kind] += 1
            else:
                key = kind
            if key in counts:
                counts[key] += 1
    unlocked = _v031_get(farm, "unlocked_quadrants", []) or []
    hands = len(_v031_get(farm, "hands", []) or [])
    return {
        "hands": hands,
        "unlocked": sorted(str(x) for x in unlocked),
        "counts": counts,
    }


def _v032_signature_distance(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 10**6
    distance = abs(int(left.get("hands", 0)) - int(right.get("hands", 0))) * 2
    distance += 2 * len(set(left.get("unlocked", [])) ^ set(right.get("unlocked", [])))
    lc, rc = left.get("counts", {}), right.get("counts", {})
    for key in set(lc) | set(rc):
        distance += abs(int(lc.get(key, 0)) - int(rc.get(key, 0)))
    return distance


def _v032_profiles():
    value = globals().get("V032_PROFILES", [])
    return value if isinstance(value, list) else []


def _v032_profile_key(profile):
    """Group duplicate route samples before applying confidence margins."""
    key = profile.get("route_key") if isinstance(profile, dict) else None
    if key:
        return str(key)
    checkpoints = profile.get("checkpoints", {}) if isinstance(profile, dict) else {}
    try:
        return repr(sorted((str(k), checkpoints[k]) for k in checkpoints))
    except Exception:
        return repr(checkpoints)


def _v032_market_observation(obs, state, step):
    market = _v031_get(obs, "market", {}) or {}
    inventory = _v031_get(market, "inventory", {}) or {}
    current = {item: _v031_int(inventory.get(item, 10000), 10000)
               for item in V032_PREMIUM}
    previous = state.get("market_prev")
    own_sell = {}
    if previous is not None:
        for item in V032_PREMIUM:
            delta = current[item] - int(previous.get(item, current[item]))
            own = int(state.get("own_sell_prev", {}).get(item, 0))
            # Positive residual inventory growth is the observable proxy for
            # opponent supply.  Town consumption is intentionally not guessed
            # here; profile matching uses a tolerant band.
            own_sell[item] = max(0, delta - own)
    state["market_prev"] = current
    return own_sell


def _v032_match_profile(obs, state, step):
    farms = list(_v031_get(obs, "farms", []) or [])
    opponent = 1 - _v032_seat(obs)
    if opponent >= len(farms):
        return None
    checkpoint = max((x for x in V032_ROUTE_CHECKPOINTS if x <= step), default=-1)
    if checkpoint < 0 or checkpoint == state.get("last_checkpoint"):
        return state.get("profile")
    state["last_checkpoint"] = checkpoint
    current = _v032_signature(farms[opponent])
    # Several training replays can be the same public route with different
    # market tapes.  Confidence must compare distinct route families, not
    # duplicate samples from one family.
    grouped = {}
    for profile in _v032_profiles():
        checkpoints = profile.get("checkpoints", {}) if isinstance(profile, dict) else {}
        expected = checkpoints.get(str(checkpoint), checkpoints.get(checkpoint))
        if expected is None:
            continue
        distance = _v032_signature_distance(current, expected)
        # For the confidence margin, profiles that predict the same public
        # signature at this checkpoint are one hypothesis, even if they
        # diverge later or have different offline market tapes.
        key = repr(expected)
        previous = grouped.get(key)
        if previous is None or distance < previous[0]:
            grouped[key] = (distance, profile)
    rows = list(grouped.values())
    if not rows:
        return state.get("profile")
    rows.sort(key=lambda row: row[0])
    best_distance, best = rows[0]
    second_distance = rows[1][0] if len(rows) > 1 else best_distance + 4
    # A deterministic profile from the same route should be close; allowing a
    # modest distance handles harmless WEED and timing differences.
    route_hit = best_distance <= int(best.get("route_distance", 8))
    best_confidence = max(0.0, 1.0 - float(best_distance) / 20.0)
    second_confidence = max(0.0, 1.0 - float(second_distance) / 20.0)
    confidence_margin = best_confidence - second_confidence
    if route_hit and best_confidence >= 0.70 and confidence_margin >= 0.15:
        state["route_hits"] = int(state.get("route_hits", 0)) + 1
        state["profile"] = best
    elif state.get("known"):
        state["route_hits"] = max(0, int(state.get("route_hits", 0)) - 1)

    # Market evidence is counted from actual observed supply residuals.  The
    # profile stores broad expected bands rather than exact action traces.
    observed = state.get("last_market_supply", {})
    expected_market = best.get("market_bands", {}) if isinstance(best, dict) else {}
    if observed and expected_market:
        matches = 0
        for item in V032_PREMIUM:
            band = expected_market.get(item, {})
            value = float(observed.get(item, 0))
            low, high = float(band.get("low", 0)), float(band.get("high", 10**9))
            matches += int(low <= value <= high)
        if matches >= 2:
            state["market_hits"] = int(state.get("market_hits", 0)) + 1
    # A zero residual is still an observation: it tells us that no positive
    # opponent supply was visible at this checkpoint.  Requiring a positive
    # residual here made the third confidence gate impossible for many valid
    # routes, because town consumption and our own sells often cancel the
    # inventory delta exactly.
    if observed:
        state["market_observations"] = int(state.get("market_observations", 0)) + 1
    state["known"] = bool(int(state.get("route_hits", 0)) >= 2 and
                           int(state.get("market_hits", 0)) >= 2 and
                           int(state.get("market_observations", 0)) >= 3)
    if not state["known"] and int(state.get("route_hits", 0)) <= 0:
        state["profile"] = None
    return state.get("profile")


def _v032_future_event(step, item):
    for target in range(int(step) + 1, min(V032_CUTOFF, len(_ACTIONS))):
        quantity = 0
        for order in _ACTIONS[target].get("market", []) or []:
            if _v032_is_sell(order, item):
                quantity += max(0, _v031_int(order[2]))
        if quantity > 0:
            return target, quantity
    return None, 0


def _v032_price(item, inventory):
    try:
        return float(_market_price(item, max(0, int(inventory))))
    except Exception:
        return 1.0


def _v032_event_revenue(item, inventory, quantity):
    total = 0.0
    inventory = int(inventory)
    for _ in range(max(0, int(quantity))):
        total += _v032_price(item, inventory)
        inventory += 1
    return total, inventory


def _v032_expected_gain(obs, item, now_q, future_q, due, transfer, mode, profile):
    market = _v031_get(obs, "market", {}) or {}
    inventory = _v031_get(market, "inventory", {}) or {}
    base_inventory = _v031_int(inventory.get(item, 10000), 10000)
    current_price = float((_v031_get(market, "prices", {}) or {}).get(item, 1) or 1)
    supplies = (profile or {}).get("supply_forecast", {}).get(item, {})
    gains = []
    for multiplier in (0.75, 1.0, 1.25):
        inv = base_inventory
        control = 0.0
        candidate = 0.0
        control_now, inv = _v032_event_revenue(item, inv, now_q)
        control += control_now
        candidate_now_q = now_q + transfer if mode == "advance" else now_q - transfer
        candidate_now, candidate_inv = _v032_event_revenue(item, base_inventory, candidate_now_q)
        candidate += candidate_now
        predicted = 0.0
        for raw_step in range(int(obs.get("step", 0) or 0) + 1, int(due) + 1):
            bucket = str(raw_step)
            predicted += float(supplies.get(bucket, supplies.get("default", 0))) * multiplier
        inv += int(round(predicted))
        candidate_inv += int(round(predicted))
        # Town/shop consumption makes waiting less harmful; use a small
        # conservative drawdown rather than pretending the market is static.
        inv = max(0, inv - max(0, int(due - int(obs.get("step", 0) or 0)) // 12))
        candidate_inv = max(0, candidate_inv - max(0, int(due - int(obs.get("step", 0) or 0)) // 12))
        control_future, _ = _v032_event_revenue(item, inv, future_q)
        if mode == "advance":
            candidate_future, _ = _v032_event_revenue(item, candidate_inv, future_q - transfer)
        else:
            candidate_future, _ = _v032_event_revenue(item, candidate_inv, future_q + transfer)
        control += control_future
        candidate += candidate_future
        # Penalize carrying stock and uncertain prices.  Advance gets a small
        # penalty for consuming the current inventory buffer.
        storage_penalty = 0.0
        if mode == "delay":
            storage_penalty = transfer * max(1.0, current_price * 0.03)
        elif mode == "advance":
            storage_penalty = transfer * max(0.5, current_price * 0.01)
        gains.append(candidate - control - storage_penalty)
    worst = min(gains)
    # A positive mean is not sufficient: the candidate must remain positive
    # in LOW, NORMAL and HIGH opponent-supply scenarios.
    if worst <= 0:
        return worst
    return worst * 0.6 + sum(gains) / len(gains) * 0.4


def _v032_apply_pending(action, state, step):
    pending = state.get("pending")
    if not pending or int(pending.get("due", -1)) != int(step):
        return action, True
    trial = _v032_copy(action)
    item, quantity = pending["item"], int(pending["quantity"])
    ok = _v032_reduce(trial, item, quantity) if pending["mode"] == "advance" else _v032_add_or_merge(trial, item, quantity)
    if not ok:
        _v032_stat("repayment_failures")
        state["pending"] = None
        return action, False
    _v032_stat("repayment_successes")
    state["pending"] = None
    return trial, True


def _v032_choose_timing(obs, action, state, step, profile):
    if not profile or not state.get("known") or step < 120 or step >= V032_CUTOFF:
        return action
    if state.get("pending"):
        return action
    if _v032_used_storage(obs) > 90:
        _v032_stat("storage_blocked")
        return action
    for item in V032_PREMIUM:
        now_q = _v032_qty(action, item)
        due, future_q = _v032_future_event(step, item)
        if due is None or future_q <= 0:
            continue
        available = _v032_visible_inventory(obs, item)
        candidates = []
        ratios = (1.0, 0.25, 0.50)
        # ADVANCE is allowed only when actual visible stock covers current plus
        # the transferred units.  This prevents selling future production.
        for ratio in ratios:
            transfer = min(V032_MAX_TRANSFER, future_q, max(0, int(round(future_q * ratio))))
            if transfer <= 0 or available < now_q + transfer:
                continue
            if due >= V032_CUTOFF:
                continue
            gain = _v032_expected_gain(obs, item, now_q, future_q, due, transfer, "advance", profile)
            if gain > V032_MIN_GAIN:
                candidates.append((gain, "advance", transfer, due))
        # DELAY is conservative: retain a ten-unit storage reserve and require
        # an existing current SELL, so it never creates a new sale from thin air.
        for ratio in ratios:
            transfer = min(V032_MAX_TRANSFER, now_q, max(0, int(round(now_q * ratio))))
            if transfer <= 0 or now_q <= 0 or future_q <= 0:
                continue
            if _v032_used_storage(obs) + transfer > 90:
                continue
            gain = _v032_expected_gain(obs, item, now_q, future_q, due, transfer, "delay", profile)
            if gain > V032_MIN_GAIN:
                candidates.append((gain, "delay", transfer, due))
        if not candidates:
            continue
        gain, mode, transfer, due = max(candidates, key=lambda value: value[0])
        trial = _v032_copy(action)
        if mode == "advance":
            if not _v032_add_or_merge(trial, item, transfer):
                continue
        else:
            if not _v032_reduce(trial, item, transfer):
                continue
        state["pending"] = {"item": item, "quantity": transfer,
                             "due": due, "mode": mode}
        _v032_stat(f"{mode}_accepted")
        _v032_stat(f"{item}_{mode}_units", transfer)
        return trial
    return action


def _v032_agent(obs, config=None):
    del config
    step = _v031_step(obs)
    state = _v032_reset(obs, step)
    action = _v032_copy(_ACTIONS[step])
    action = _v031_weed_action(obs, action, step)
    action, _ = _v032_apply_pending(action, state, step)
    if (not globals().get("V032_DISABLE_TIMING", False) and
            step < V032_TERMINAL_CUTOFF):
        supply = _v032_market_observation(obs, state, step)
        state["last_market_supply"] = supply
        profile = _v032_match_profile(obs, state, step)
        action = _v032_choose_timing(obs, action, state, step, profile)
    else:
        state["known"] = False
    # The user's requested ordering: timing first, then the original V27
    # price-impact ranking over the final quantities.
    action = _v031_reorder_existing(obs, action)
    action = _v031_align_hands(action, obs)
    state["own_sell_prev"] = {
        item: _v032_qty(action, item) for item in V032_PREMIUM
    }
    if len(action.get("market", []) or []) > V032_MAX_ORDERS:
        action["market"] = action["market"][:V032_MAX_ORDERS]
        _v032_stat("market_overflow_guard")
    return action


def agent(obs, config=None):
    try:
        return _v032_agent(obs, config)
    except Exception:
        _v032_stat("runtime_errors")
        return _v031_agent(obs, config)
