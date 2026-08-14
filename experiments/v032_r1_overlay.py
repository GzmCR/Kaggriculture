"""V032-R1 runtime overlay.

This source is appended after V032 and the dependency-free market rollout.
It deliberately owns a separate state machine, so the old V032 timing layer
cannot fire twice.  With an empty calibration payload it is a strict
order-only control; a fitted payload unlocks only supported, risk-checked
events.
"""

V032_R1_PREMIUM = ("MILK", "STRAWBERRY", "WOOL", "MELON")
V032_R1_CUTOFF = 648
V032_R1_TERMINAL_CUTOFF = 672
V032_R1_MAX_ORDERS = 10
V032_R1_MAX_TRANSFER = 30
V032_R1_MIN_GAIN = 10.0
V032_R1_MIN_SUPPORT = 24
V032_R1_CALIBRATION = {}
V032_R1_DISABLE_TIMING = False
V032_R1_OFFLINE_OPPONENT_SCHEDULE = None

V032_R1_STATS = {}
_V032_R1_STATE = {
    0: {"last_step": -1, "pending": None, "event_count": 0},
    1: {"last_step": -1, "pending": None, "event_count": 0},
}


def _v032_r1_stat(name, amount=1):
    V032_R1_STATS[name] = V032_R1_STATS.get(name, 0) + amount


def _v032_r1_reset(obs, step):
    seat = _v032_seat(obs)
    state = _V032_R1_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "pending": None, "event_count": 0})
    state["last_step"] = step
    return state


def _v032_r1_shed_inventory(obs, item):
    """Only shed stock can be sold by the Kaggriculture market engine."""
    private = _v031_get(obs, "private", {}) or {}
    shed = _v031_get(private, "shed", {}) or {}
    return max(0, _v031_int(shed.get(str(item).upper(), 0)))


def _v032_r1_qty(action, item):
    return sum(max(0, _v031_int(order[2])) for order in action.get("market", []) or []
               if _v031_is_sell(order, item))


def _v032_r1_remove_zero(action):
    action["market"] = r1_clean_zero_sells(action.get("market", []) or [])
    return action


def _v032_r1_add(action, item, quantity):
    quantity = max(0, _v031_int(quantity))
    if quantity <= 0:
        return False
    for order in action.get("market", []) or []:
        if _v031_is_sell(order, item):
            order[2] = max(0, _v031_int(order[2])) + quantity
            return True
    if len(action.get("market", []) or []) >= V032_R1_MAX_ORDERS:
        return False
    action.setdefault("market", []).append(["SELL", str(item).upper(), quantity])
    return True


def _v032_r1_reduce(action, item, quantity):
    remaining = max(0, _v031_int(quantity))
    if remaining <= 0:
        return True
    total = _v032_r1_qty(action, item)
    # A zeroed order must not be removed: market processing is lockstep by
    # order index, so deleting it shifts the opponent's queue and can change
    # unrelated product prices.  R1 rejects the transfer instead.
    if total <= remaining:
        return False
    for order in action.get("market", []) or []:
        if not _v031_is_sell(order, item):
            continue
        current = max(0, _v031_int(order[2]))
        take = min(current, remaining)
        order[2] = current - take
        remaining -= take
        if remaining <= 0:
            return True
    return False


def _v032_r1_future_events(step, item, count=2):
    events = []
    for target in range(int(step) + 1, min(V032_R1_CUTOFF, len(_ACTIONS))):
        quantity = sum(max(0, _v031_int(order[2])) for order in _ACTIONS[target].get("market", []) or []
                        if _v031_is_sell(order, item))
        if quantity > 0:
            events.append((target, quantity))
            if len(events) >= int(count):
                break
    return events


def _v032_r1_private_shed(obs):
    private = _v031_get(obs, "private", {}) or {}
    return {str(k).upper(): max(0, _v031_int(v)) for k, v in
            (_v031_get(private, "shed", {}) or {}).items()}


def _v032_r1_shed_for_window(obs, schedule, start_step, end_step):
    shed = _v032_r1_private_shed(obs)
    # Future harvest is not available at the current event.  For the rollout
    # only, future route SELLs are deposited into the hypothetical shed at the
    # start of the window; the real-time gate separately checks current shed
    # legality.  This lets the rollout price later events without making an
    # impossible current sale legal.
    for target in range(int(start_step) + 1, int(end_step) + 1):
        for order in schedule.get(target, []) or []:
            if _v031_is_sell(order):
                item = str(order[1]).upper()
                shed[item] = shed.get(item, 0) + max(0, _v031_int(order[2]))
    return shed


def _v032_r1_profile_orders(profile, item, start_step, end_step, multiplier):
    schedule = {}
    forecast = (profile or {}).get("supply_forecast", {}).get(item, {})
    for target in range(int(start_step), int(end_step) + 1):
        raw = forecast.get(str(target), forecast.get(target, 0))
        quantity = max(0, int(float(raw or 0) * float(multiplier) + 0.5))
        if quantity:
            schedule[target] = [["SELL", item, quantity]]
    return schedule


def _v032_r1_opponent_orders(profile, item, start_step, end_step, multiplier):
    """Use a captured opponent tape only in offline diagnostics.

    The submitted agent never sets this variable, so production continues to
    use the embedded anonymous supply forecast.  Offline calibration can set
    it to the control opponent's observed market queues to measure residuals
    against the same market sequence.
    """
    tape = globals().get("V032_R1_OFFLINE_OPPONENT_SCHEDULE")
    if not isinstance(tape, dict):
        return _v032_r1_profile_orders(profile, item, start_step, end_step, multiplier)
    schedule = {}
    for target in range(int(start_step), int(end_step) + 1):
        orders = tape.get(target, tape.get(str(target), [])) or []
        schedule[target] = [list(order) for order in orders if isinstance(order, (list, tuple))]
    return schedule


def _v032_r1_route_schedule(start_step, end_step, item, due, transfer, mode, current_action):
    schedule = {}
    for target in range(int(start_step), int(end_step) + 1):
        if target == int(start_step):
            schedule[target] = _v031_copy_action(current_action).get("market", [])
        elif target < len(_ACTIONS):
            schedule[target] = [list(order) for order in _ACTIONS[target].get("market", []) or []]
    if due in schedule and int(transfer) > 0:
        if mode == "advance":
            due_quantity = sum(max(0, _v031_int(order[2])) for order in schedule[due]
                               if _v031_is_sell(order, item))
            if int(transfer) >= due_quantity:
                return None
        adjusted = {"market": [list(order) for order in schedule[due]]}
        ok = (_v032_r1_add(adjusted, item, transfer) if mode == "delay"
              else _v032_r1_reduce(adjusted, item, transfer))
        if not ok:
            return None
        schedule[due] = adjusted["market"]
    return schedule


def _v032_r1_rollout_one(obs, control_action, candidate_action, profile, item,
                         due2, first_due, transfer, mode, multiplier, config):
    step = _v031_step(obs)
    control_schedule = _v032_r1_route_schedule(step, due2, item, first_due, 0, "delay", control_action)
    candidate_schedule = _v032_r1_route_schedule(step, due2, item, first_due, transfer, mode, candidate_action)
    if control_schedule is None or candidate_schedule is None:
        return None
    control_opp = _v032_r1_opponent_orders(profile, item, step, due2, multiplier)
    candidate_opp = _v032_r1_opponent_orders(profile, item, step, due2, multiplier)
    control_shed = _v032_r1_shed_for_window(obs, control_schedule, step, due2)
    candidate_shed = _v032_r1_shed_for_window(obs, candidate_schedule, step, due2)
    opponent_shed = {}
    for schedule in (control_opp, candidate_opp):
        for orders in schedule.values():
            for order in orders:
                if _v031_is_sell(order):
                    key = str(order[1]).upper()
                    opponent_shed[key] = opponent_shed.get(key, 0) + max(0, _v031_int(order[2]))
    market = _v031_get(obs, "market", {}) or {}
    inventory = {str(k).upper(): _v031_int(v) for k, v in (_v031_get(market, "inventory", {}) or {}).items()}
    farms = list(_v031_get(obs, "farms", []) or [])
    seat = _v032_seat(obs)
    own_money = float(_v031_get(farms[seat], "money", 0) if seat < len(farms) else 0)
    opp_money = float(_v031_get(farms[1 - seat], "money", 0) if 1 - seat < len(farms) else 0)
    shops = _v031_get(_v031_get(obs, "town", {}) or {}, "unlocked_shops", []) or []
    cfg = config or {}
    kwargs = {
        "shops": shops,
        "shop_interval": _v031_int(_v031_get(cfg, "townShopSellInterval", 4), 4),
        "center_interval": _v031_int(_v031_get(cfg, "townCenterSellInterval", 24), 24),
        "max_orders": _v031_int(_v031_get(cfg, "maxMarketOrdersPerTurn", 10), 10),
        "shed_capacity": _v031_int(_v031_get(cfg, "shedCapacity", 100), 100),
    }
    if seat == 0:
        control_own_schedule, control_opponent_schedule = control_schedule, control_opp
        candidate_own_schedule, candidate_opponent_schedule = candidate_schedule, candidate_opp
        start_money = [own_money, opp_money]
        start_sheds = [control_shed, opponent_shed]
        candidate_sheds = [candidate_shed, opponent_shed]
    else:
        # The rollout helper treats its first schedule as player 0.  Swap the
        # schedules and private states when our controlled seat is player 1.
        control_own_schedule, control_opponent_schedule = control_opp, control_schedule
        candidate_own_schedule, candidate_opponent_schedule = candidate_opp, candidate_schedule
        start_money = [opp_money, own_money]
        start_sheds = [opponent_shed, control_shed]
        candidate_sheds = [opponent_shed, candidate_shed]
    control = r1_simulate_window(inventory, start_money, start_sheds,
                                 control_own_schedule, control_opponent_schedule,
                                 step, due2, **kwargs)
    candidate = r1_simulate_window(inventory, start_money, candidate_sheds,
                                   candidate_own_schedule, candidate_opponent_schedule,
                                   step, due2, **kwargs)
    idx = seat
    control_margin = control["money"][idx] - control["money"][1 - idx]
    candidate_margin = candidate["money"][idx] - candidate["money"][1 - idx]
    return float(candidate_margin - control_margin)


def _v032_r1_calibration(item, mode, features):
    payload = V032_R1_CALIBRATION if isinstance(V032_R1_CALIBRATION, dict) else {}
    rows = payload.get("items", {}) if isinstance(payload, dict) else {}
    model = rows.get(f"{item}:{mode}") or rows.get(str(item)) or payload.get("global", {})
    if not isinstance(model, dict):
        return None
    support = int(model.get("support_groups", model.get("support", 0)) or 0)
    if support < V032_R1_MIN_SUPPORT:
        return None
    correction = float(model.get("median_residual", 0.0) or 0.0)
    coefficients = model.get("coefficients", []) or []
    if coefficients and len(coefficients) == len(features):
        correction += sum(float(a) * float(b) for a, b in zip(coefficients, features))
    return correction, support


def _v032_r1_expected_gain(obs, control_action, candidate_action, profile, item,
                           first_due, second_due, transfer, mode, config):
    if second_due is None:
        return None
    scenario_values = _v032_r1_raw_gain(
        obs, control_action, candidate_action, profile, item,
        first_due, second_due, transfer, mode, config,
    )
    if scenario_values is None:
        return None
    scenario_values = list(scenario_values)
    step = _v031_step(obs)
    price = _v031_item_price(obs, item)
    market = _v031_get(obs, "market", {}) or {}
    inv = _v031_int(_v031_get(_v031_get(market, "inventory", {}) or {}, item, 10000), 10000)
    features = [float(price) / 100.0, float(inv - 10000) / 1000.0,
                float(transfer) / 10.0, float(first_due - step) / 24.0,
                float(second_due - first_due) / 24.0]
    calibrated = _v032_r1_calibration(item, mode, features)
    if calibrated is None:
        _v032_r1_stat("calibration_fallback")
        return None
    correction, support = calibrated
    adjusted = [value + correction for value in scenario_values]
    worst = min(adjusted)
    mean = sum(adjusted) / len(adjusted)
    _v032_r1_stat("rollout_evaluations")
    V032_R1_STATS["last_raw_gain"] = scenario_values[1]
    V032_R1_STATS["last_calibrated_gain"] = worst * 0.6 + mean * 0.4
    V032_R1_STATS["last_support"] = support
    if worst <= 0 or worst < V032_R1_MIN_GAIN:
        return None
    return worst * 0.6 + mean * 0.4


def _v032_r1_raw_gain(obs, control_action, candidate_action, profile, item,
                      first_due, second_due, transfer, mode, config):
    """Return LOW/NORMAL/HIGH margin deltas before residual calibration."""
    if second_due is None:
        return None
    scenario_values = []
    for multiplier in (0.75, 1.0, 1.25):
        value = _v032_r1_rollout_one(
            obs, control_action, candidate_action, profile, item,
            second_due, first_due, transfer, mode, multiplier, config,
        )
        if value is None:
            return None
        scenario_values.append(float(value))
    return scenario_values


def _v032_r1_apply_pending(action, state, step):
    pending = state.get("pending")
    if not pending or int(pending.get("due", -1)) != int(step):
        return action
    trial = _v031_copy_action(action)
    item, quantity = str(pending["item"]).upper(), int(pending["quantity"])
    ok = (_v032_r1_reduce(trial, item, quantity) if pending["mode"] == "advance"
          else _v032_r1_add(trial, item, quantity))
    if not ok:
        _v032_r1_stat("repayment_failures")
    else:
        _v032_r1_stat("repayment_successes")
    state["pending"] = None
    return trial if ok else action


def _v032_r1_choose(obs, action, state, step, profile, config):
    if V032_R1_DISABLE_TIMING or not profile or not state.get("known"):
        return action
    if step < 120 or step >= V032_R1_CUTOFF or state.get("pending"):
        return action
    for item in V032_R1_PREMIUM:
        now_q = _v032_r1_qty(action, item)
        events = _v032_r1_future_events(step, item, 2)
        if now_q <= 0 or len(events) < 2:
            continue
        first_due, future_q = events[0]
        second_due, _ = events[1]
        if first_due >= V032_R1_CUTOFF or second_due >= V032_R1_CUTOFF:
            continue
        available = _v032_r1_shed_inventory(obs, item)
        if available < now_q:
            _v032_r1_stat("shed_blocked")
            continue
        candidates = []
        for mode in ("advance", "delay"):
            for ratio in (1.0, 0.25, 0.50):
                source = future_q if mode == "advance" else now_q
                transfer = min(V032_R1_MAX_TRANSFER, source, int(source * ratio + 0.5))
                if transfer <= 0:
                    continue
                if mode == "advance" and available < now_q + transfer:
                    continue
                if mode == "delay" and _v032_r1_used_storage(obs) + transfer > 90:
                    continue
                trial = _v031_copy_action(action)
                ok = (_v032_r1_add(trial, item, transfer) if mode == "advance"
                      else _v032_r1_reduce(trial, item, transfer))
                if not ok:
                    continue
                gain = _v032_r1_expected_gain(obs, action, trial, profile, item,
                                              first_due, second_due, transfer, mode, config)
                if gain is not None and gain > V032_R1_MIN_GAIN:
                    candidates.append((gain, mode, transfer, first_due))
        if candidates:
            gain, mode, transfer, due = max(candidates, key=lambda row: row[0])
            trial = _v031_copy_action(action)
            ok = (_v032_r1_add(trial, item, transfer) if mode == "advance"
                  else _v032_r1_reduce(trial, item, transfer))
            if ok:
                state["pending"] = {"item": item, "quantity": transfer,
                                     "due": due, "mode": mode}
                _v032_r1_stat(f"{mode}_accepted")
                _v032_r1_stat(f"{item}_{mode}_units", transfer)
                return trial
    return action


def _v032_r1_used_storage(obs):
    private = _v031_get(obs, "private", {}) or {}
    used = sum(max(0, _v031_int(v)) for v in (_v031_get(private, "shed", {}) or {}).values())
    for inventory in _v031_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            used += sum(max(0, _v031_int(v)) for v in inventory.values())
    return used


def _v032_r1_agent(obs, config=None):
    step = _v031_step(obs)
    state = _v032_r1_reset(obs, step)
    _v031_reset_state(obs, step)
    action = _v031_copy_action(_ACTIONS[step])
    action = _v031_weed_action(obs, action, step)
    action = _v032_r1_apply_pending(action, state, step)
    if step < V032_R1_TERMINAL_CUTOFF:
        supply = _v032_market_observation(obs, _v032_reset(obs, step), step)
        r0 = _v032_reset(obs, step)
        r0["last_market_supply"] = supply
        profile = _v032_match_profile(obs, r0, step)
        state["known"] = bool(profile and r0.get("known"))
        action = _v032_r1_choose(obs, action, state, step, profile, config)
    action = _v031_reorder_existing(obs, _v032_r1_remove_zero(action))
    action = _v031_align_hands(action, obs)
    old_state = _v032_reset(obs, step)
    old_state["own_sell_prev"] = {item: _v032_qty(action, item) for item in V032_PREMIUM}
    if len(action.get("market", []) or []) > V032_R1_MAX_ORDERS:
        _v032_r1_stat("market_overflow_guard")
        action["market"] = action["market"][:V032_R1_MAX_ORDERS]
    return action


def agent(obs, config=None):
    try:
        return _v032_r1_agent(obs, config)
    except Exception:
        _v032_r1_stat("runtime_errors")
        return _v031_agent(obs, config)
