"""RL-010: conservative bidirectional MILK timing for a frozen V27 route.

The module is deliberately self-contained so a fitted policy can be appended
to the V27 ``order_only`` source and submitted as one main.py.  It only edits
the quantity of an existing MILK SELL event and repays that quantity at the
next same-product route event.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict

import numpy as np


RL010_ITEM = "MILK"
RL010_ACTIONS = ("ADVANCE_25", "ADVANCE_50", "CONTROL", "DELAY_25", "DELAY_50")
RL010_RATIOS = {
    "ADVANCE_25": 0.25,
    "ADVANCE_50": 0.50,
    "DELAY_25": 0.25,
    "DELAY_50": 0.50,
}
RL010_CUTOFF = 648
RL010_DELAY_CUTOFF = 624
RL010_MIN_GAP = 1
RL010_MAX_GAP = 72
RL010_DELAY_MAX_GAP = 24
RL010_SHED_CAPACITY = 100
RL010_SHED_RESERVE = 10
# The first repaired training/runtime slice is deliberately restricted to the
# stable same-day MILK window found in V27 replays (455 -> 456).  The route
# parser remains general so later experiments can opt into other events.
RL010_EVENT_STEPS = (455,)
RL010_MAX_INTERVENTIONS = 8
RL010_MIN_SUPPORT = 24
RL010_MIN_EXPECTED_DELTA = 5.0
RL010_LCB_Z = 1.5
RL010_BAD_UCB = 0.10
# Intercept + 40 normalized state features.  Keep this explicit because the
# feature vector is serialized with the fitted ridge models.
RL010_FEATURE_DIM = 41
RL010_OPPONENT_FEATURE_INDICES = (19, 20, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40)


def rl010_get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def rl010_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def rl010_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rl010_step(obs):
    value = rl010_get(obs, "step", None)
    if value is not None:
        return max(0, rl010_int(value))
    return max(
        0,
        rl010_int(rl010_get(obs, "day", 0)) * 24
        + rl010_int(rl010_get(obs, "hour", 0)),
    )


def rl010_normalize_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(value or ["PASS"]) for value in action.get("hands", []) or []],
        "market": [
            list(value)
            for value in action.get("market", []) or []
            if isinstance(value, (list, tuple))
        ],
    }


def rl010_align_hands(action, obs):
    action = rl010_normalize_action(action)
    farms = list(rl010_get(obs, "farms", []) or [])
    seat = rl010_int(rl010_get(obs, "player", 0))
    expected = 0
    if 0 <= seat < len(farms):
        expected = len(rl010_get(farms[seat], "hands", []) or [])
    action["hands"].extend([["PASS"] for _ in range(max(0, expected - len(action["hands"])) )])
    action["hands"] = action["hands"][:expected]
    return action


def rl010_route_opportunities(actions, allowed_current_steps=None):
    """Find adjacent same-item route SELL events before the cutoff."""
    events = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            if str(order[0]).upper() != "SELL" or str(order[1]).upper() != RL010_ITEM:
                continue
            quantity = max(0, rl010_int(order[2]))
            if quantity:
                events[RL010_ITEM][int(step)] = events[RL010_ITEM].get(int(step), 0) + quantity
    result = []
    for item, rows in events.items():
        ordered = sorted(rows.items())
        for index, (current_step, current_quantity) in enumerate(ordered[:-1]):
            future_step, future_quantity = ordered[index + 1]
            gap = int(future_step) - int(current_step)
            if current_step >= RL010_CUTOFF or future_step >= RL010_CUTOFF:
                continue
            if RL010_MIN_GAP <= gap <= RL010_MAX_GAP and (
                allowed_current_steps is None
                or int(current_step) in {int(value) for value in allowed_current_steps}
            ):
                result.append({
                    "item": item,
                    "current_step": int(current_step),
                    "future_step": int(future_step),
                    "current_quantity": int(current_quantity),
                    "future_quantity": int(future_quantity),
                    "gap": int(gap),
                })
    return sorted(result, key=lambda row: (row["current_step"], row["item"]))


def rl010_opportunity_key(opportunity):
    return "{}|{}|{}".format(
        str(opportunity["item"]).upper(),
        int(opportunity["current_step"]),
        int(opportunity["future_step"]),
    )


class RL010History:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_step = -1
        self.prices = {RL010_ITEM: []}
        self.inventories = {RL010_ITEM: []}

    def observe(self, obs):
        step = rl010_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        market = rl010_get(obs, "market", {}) or {}
        prices = rl010_get(market, "prices", {}) or {}
        inventory = rl010_get(market, "inventory", {}) or {}
        self.prices[RL010_ITEM].append((step, rl010_float(prices.get(RL010_ITEM, 0))))
        self.inventories[RL010_ITEM].append((step, rl010_float(inventory.get(RL010_ITEM, 10000))))
        self.prices[RL010_ITEM] = self.prices[RL010_ITEM][-96:]
        self.inventories[RL010_ITEM] = self.inventories[RL010_ITEM][-96:]
        self.last_step = step

    @staticmethod
    def _lagged(rows, step, lag):
        target = int(step) - int(lag)
        values = [value for seen_step, value in rows if seen_step <= target]
        return values[-1] if values else None

    def market_values(self, step):
        prices = self.prices[RL010_ITEM]
        inventories = self.inventories[RL010_ITEM]
        current_price = prices[-1][1] if prices else 0.0
        current_inventory = inventories[-1][1] if inventories else 10000.0
        p6 = self._lagged(prices, step, 6)
        p12 = self._lagged(prices, step, 12)
        p24 = self._lagged(prices, step, 24)
        i6 = self._lagged(inventories, step, 6)
        i12 = self._lagged(inventories, step, 12)
        i24 = self._lagged(inventories, step, 24)
        return (
            current_price,
            current_inventory,
            0.0 if p6 is None else current_price - p6,
            0.0 if p12 is None else current_price - p12,
            0.0 if p24 is None else current_price - p24,
            0.0 if i6 is None else current_inventory - i6,
            0.0 if i12 is None else current_inventory - i12,
            0.0 if i24 is None else current_inventory - i24,
        )


def rl010_tile_rows(farm):
    for row in rl010_get(farm, "tiles", []) or []:
        if isinstance(row, list):
            for tile in row:
                yield tile


def rl010_milk_pipeline(farm):
    cows = 0
    pastures = 0
    fed_cows = 0
    unfed_cows = 0
    yield_units = 0
    pending_bonus = 0
    for tile in rl010_tile_rows(farm):
        if not isinstance(tile, dict):
            continue
        kind = str(tile.get("kind", "")).upper()
        animal = str(tile.get("animal", "")).upper()
        if kind == "PASTURE":
            pastures += 1
        if animal != "COW":
            continue
        cows += 1
        if tile.get("fed_today"):
            fed_cows += 1
        else:
            unfed_cows += 1
        yield_units += max(0, rl010_int(tile.get("yield_units", 0)))
        pending_bonus += max(0, rl010_int(tile.get("pending_care_bonus", 0)))
    return {
        "cows": cows,
        "pastures": pastures,
        "fed_cows": fed_cows,
        "unfed_cows": unfed_cows,
        "yield_units": yield_units,
        "pending_bonus": pending_bonus,
    }


def rl010_private_inventory(obs, item=RL010_ITEM):
    private = rl010_get(obs, "private", {}) or {}
    shed = rl010_get(private, "shed", {}) or {}
    total = max(0, rl010_int(shed.get(item, 0)))
    for inventory in rl010_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            total += max(0, rl010_int(inventory.get(item, 0)))
    return total


def rl010_shed_inventory(obs, item=RL010_ITEM):
    """Return only inventory that the market can actually sell this turn.

    Carried inventory is intentionally excluded.  Kaggriculture processes
    farmer/hand actions before the market, but a harvested item is still in a
    unit's inventory and cannot be sold until a later DROP/PLACE puts it in the
    shed.
    """
    private = rl010_get(obs, "private", {}) or {}
    shed = rl010_get(private, "shed", {}) or {}
    return max(0, rl010_int(shed.get(item, 0))) if isinstance(shed, dict) else 0


def rl010_shed_total(obs):
    private = rl010_get(obs, "private", {}) or {}
    shed = rl010_get(private, "shed", {}) or {}
    return sum(max(0, rl010_int(value)) for value in shed.values()) if isinstance(shed, dict) else 0


def _rl010_position(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return rl010_int(value[0]), rl010_int(value[1])
    return None


def _rl010_config_int(config, key, default):
    value = rl010_get(config, key, default)
    return max(1, rl010_int(value, default))


def _rl010_shed_adjacent(position, board_size):
    if position is None:
        return False
    half = max(1, int(board_size) // 2)
    return position in {
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    }


def rl010_market_shed_state(obs, action, item=RL010_ITEM, config=None):
    """Simulate the player's pre-market shed after this turn's unit actions.

    This is intentionally a small mirror of the environment's DROP/PICKUP/
    PLACE handling.  It does not count HARVEST as market inventory: harvested
    produce remains carried until a later turn or an explicit shed drop.
    """
    private = rl010_get(obs, "private", {}) or {}
    raw_shed = rl010_get(private, "shed", {}) or {}
    shed = {
        str(key): max(0, rl010_int(value))
        for key, value in raw_shed.items()
    } if isinstance(raw_shed, dict) else {}
    raw_inventories = list(rl010_get(private, "inventories", []) or [])
    inventories = [
        {
            str(key): max(0, rl010_int(value))
            for key, value in (inventory.items() if isinstance(inventory, dict) else [])
        }
        for inventory in raw_inventories
    ]
    farms = list(rl010_get(obs, "farms", []) or [])
    seat = rl010_int(rl010_get(obs, "player", 0))
    mine = farms[seat] if 0 <= seat < len(farms) else {}
    positions = [_rl010_position(rl010_get(mine, "farmer", None))]
    positions.extend(_rl010_position(value) for value in (rl010_get(mine, "hands", []) or []))
    actions = [list(action.get("farmer") or ["PASS"])]
    actions.extend(list(value or ["PASS"]) for value in action.get("hands", []) or [])
    board_size = _rl010_config_int(config, "boardSize", 10)
    capacity = _rl010_config_int(config, "shedCapacity", RL010_SHED_CAPACITY)

    def shed_total():
        return sum(max(0, rl010_int(value)) for value in shed.values())

    def add_to_shed(name, amount):
        amount = max(0, rl010_int(amount))
        if amount <= 0:
            return 0
        room = max(0, capacity - shed_total())
        accepted = min(amount, room)
        if accepted:
            shed[name] = shed.get(name, 0) + accepted
        return accepted

    for index, unit_action in enumerate(actions):
        if index >= len(inventories):
            inventories.append({})
        inventory = inventories[index]
        position = positions[index] if index < len(positions) else None
        if not _rl010_shed_adjacent(position, board_size) or not unit_action:
            continue
        op = str(unit_action[0]).upper()
        if op == "DROP":
            # The environment clears the whole unit inventory, including
            # overflow.  Preserve that behavior when modelling the shed cap.
            for name, amount in list(inventory.items()):
                add_to_shed(name, amount)
            inventory.clear()
        elif op == "PLACE" and len(unit_action) >= 2:
            name = str(unit_action[1])
            amount = max(0, rl010_int(unit_action[2], 1)) if len(unit_action) >= 3 else 1
            held = min(amount, max(0, rl010_int(inventory.get(name, 0))))
            accepted = add_to_shed(name, held)
            if held:
                inventory[name] = max(0, rl010_int(inventory.get(name, 0)) - held)
                if inventory[name] <= 0:
                    inventory.pop(name, None)
            # If the shed is full, PLACE keeps the unaccepted carried item;
            # this is conservative for sellability and matches the no-overflow
            # path used by our routes.
            _ = accepted
        elif op == "PICKUP" and len(unit_action) >= 2:
            name = str(unit_action[1])
            amount = max(0, rl010_int(unit_action[2], 1)) if len(unit_action) >= 3 else 1
            taken = min(amount, max(0, rl010_int(shed.get(name, 0))))
            if taken:
                shed[name] = max(0, shed.get(name, 0) - taken)
                inventory[name] = inventory.get(name, 0) + taken
                if shed[name] <= 0:
                    shed.pop(name, None)

    current_sell = rl010_sell_quantity(action, item)
    return {
        "shed_before": rl010_shed_inventory(obs, item),
        "shed_after_actions": max(0, rl010_int(shed.get(item, 0))),
        "shed_total_after_actions": shed_total(),
        "current_sell": current_sell,
        "capacity": capacity,
        "sellable_extra": max(0, rl010_int(shed.get(item, 0)) - current_sell),
    }


def rl010_delay_guard(obs, action, opportunity, units, config=None):
    """Safety checks specific to holding a sold unit for the next event."""
    units = max(0, rl010_int(units))
    step = rl010_step(obs)
    future_step = rl010_int(opportunity.get("future_step", -1))
    current = rl010_sell_quantity(action)
    if units <= 0:
        return False, "zero_units"
    if step >= RL010_DELAY_CUTOFF:
        return False, "delay_cutoff"
    if future_step <= step:
        return False, "future_not_after_current"
    if future_step - step > RL010_DELAY_MAX_GAP:
        return False, "delay_gap"
    # A one-turn 23:00 -> 00:00 transition is the stable V27 MILK window
    # used by the first repaired model.  Longer day-boundary transfers are
    # rejected because a full day can add unmodelled production/overflow.
    crosses_day = step // 24 != future_step // 24
    if crosses_day and not (future_step - step == 1 and step % 24 == 23):
        return False, "day_boundary"
    if current <= units:
        return False, "current_order_too_small"
    state = rl010_market_shed_state(obs, action, config=config)
    if state["shed_after_actions"] < current:
        return False, "current_inventory_short"
    # After selling q-u this turn, u units remain.  Keep a reserve for
    # harvest/production surprises and avoid intentionally filling the shed.
    held_total = state["shed_total_after_actions"] - current + units
    if held_total > state["capacity"] - RL010_SHED_RESERVE:
        return False, "shed_headroom"
    return True, "safe"


def rl010_features(obs, opportunity, history, base_action=None):
    step = rl010_int(opportunity["current_step"])
    future_step = rl010_int(opportunity["future_step"])
    current_quantity = max(0, rl010_int(opportunity["current_quantity"]))
    future_quantity = max(0, rl010_int(opportunity["future_quantity"]))
    gap = max(0, future_step - step)
    price, market_inventory, p6, p12, p24, i6, i12, i24 = history.market_values(step)

    farms = list(rl010_get(obs, "farms", []) or [])
    seat = rl010_int(rl010_get(obs, "player", 0))
    mine = farms[seat] if 0 <= seat < len(farms) else {}
    other = farms[1 - seat] if len(farms) > 1 and seat in (0, 1) else {}
    mine_pipe = rl010_milk_pipeline(mine)
    other_pipe = rl010_milk_pipeline(other)
    mine_money = rl010_float(rl010_get(mine, "money", 0))
    other_money = rl010_float(rl010_get(other, "money", 0))
    shops = len(rl010_get(rl010_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    market_orders = len((base_action or {}).get("market", []) or [])
    market_state = rl010_market_shed_state(obs, base_action or {"farmer": ["PASS"], "hands": [], "market": []})
    sellable_milk = market_state["shed_after_actions"]
    shed_total = market_state["shed_total_after_actions"]

    # All continuous features are clipped to keep the fitted policy stable
    # when a new opponent has an unusually large public farm.
    values = [
        1.0,
        step / 720.0,
        (step // 24) / 30.0,
        (step % 24) / 24.0,
        min(1.0, current_quantity / 32.0),
        min(1.0, future_quantity / 32.0),
        min(1.0, gap / 72.0),
        min(2.0, price / 300.0),
        min(2.0, market_inventory / 10000.0),
        max(-2.0, min(2.0, p6 / 300.0)),
        max(-2.0, min(2.0, p12 / 300.0)),
        max(-2.0, min(2.0, p24 / 300.0)),
        max(-2.0, min(2.0, i6 / 10000.0)),
        max(-2.0, min(2.0, i12 / 10000.0)),
        max(-2.0, min(2.0, i24 / 10000.0)),
        min(2.0, sellable_milk / 100.0),
        min(1.0, shed_total / 100.0),
        min(1.0, max(0.0, (100.0 - shed_total) / 100.0)),
        min(2.0, mine_money / 100000.0),
        min(2.0, other_money / 100000.0),
        max(-2.0, min(2.0, (mine_money - other_money) / 100000.0)),
        min(1.0, market_orders / 10.0),
        min(1.0, shops / 8.0),
        min(1.0, len(rl010_get(mine, "hands", []) or []) / 20.0),
        min(1.0, len(rl010_get(mine, "unlocked_quadrants", []) or []) / 4.0),
        min(1.0, mine_pipe["cows"] / 20.0),
        min(1.0, mine_pipe["pastures"] / 10.0),
        min(1.0, mine_pipe["fed_cows"] / 20.0),
        min(2.0, mine_pipe["yield_units"] / 50.0),
        min(1.0, mine_pipe["pending_bonus"] / 20.0),
        min(1.0, other_pipe["cows"] / 20.0),
        min(1.0, other_pipe["pastures"] / 10.0),
        min(1.0, other_pipe["fed_cows"] / 20.0),
        min(1.0, other_pipe["unfed_cows"] / 20.0),
        min(2.0, other_pipe["yield_units"] / 50.0),
        min(1.0, other_pipe["pending_bonus"] / 20.0),
        min(1.0, len(rl010_get(other, "hands", []) or []) / 20.0),
        min(1.0, len(rl010_get(other, "unlocked_quadrants", []) or []) / 4.0),
        min(2.0, (other_pipe["yield_units"] + other_pipe["pending_bonus"]) / 50.0),
        min(2.0, (other_pipe["yield_units"] + other_pipe["pending_bonus"] * 2) / 50.0),
        min(2.0, (other_pipe["yield_units"] + other_pipe["pending_bonus"] * 3) / 50.0),
    ]
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (RL010_FEATURE_DIM,):
        raise AssertionError(f"RL010 feature size {array.size} != {RL010_FEATURE_DIM}")
    return array


def rl010_round_half_up(value):
    return int(float(value) + 0.5)


def rl010_mask_features(features, include_opponent=True):
    """Mask opponent-derived columns for the no-opponent ablation."""
    array = np.asarray(features, dtype=np.float64).copy()
    if array.shape != (RL010_FEATURE_DIM,):
        raise ValueError(f"RL010 feature shape {array.shape} != {(RL010_FEATURE_DIM,)}")
    if not include_opponent:
        array[list(RL010_OPPONENT_FEATURE_INDICES)] = 0.0
    return array


def rl010_sell_quantity(action, item=RL010_ITEM):
    return sum(
        max(0, rl010_int(order[2]))
        for order in action.get("market", []) or []
        if isinstance(order, (list, tuple))
        and len(order) >= 3
        and str(order[0]).upper() == "SELL"
        and str(order[1]).upper() == item
    )


def rl010_adjust_sell(action, delta, item=RL010_ITEM):
    """Adjust existing item SELL rows without creating a new order."""
    delta = rl010_int(delta)
    if delta == 0:
        return True
    rows = [
        (index, order)
        for index, order in enumerate(action.get("market", []) or [])
        if isinstance(order, list)
        and len(order) >= 3
        and str(order[0]).upper() == "SELL"
        and str(order[1]).upper() == str(item).upper()
    ]
    if not rows:
        return False
    if delta > 0:
        rows[0][1][2] = max(0, rl010_int(rows[0][1][2])) + delta
        return True
    remaining = -delta
    for _, order in rows:
        current = max(0, rl010_int(order[2]))
        take = min(current, remaining)
        order[2] = current - take
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        return False
    action["market"] = [
        order for order in action.get("market", []) or []
        if not (
            isinstance(order, list)
            and len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == str(item).upper()
            and rl010_int(order[2]) <= 0
        )
    ]
    return True


class RL010Policy:
    def __init__(self, payload=None):
        payload = payload or {}
        self.feature_dim = rl010_int(payload.get("feature_dim", RL010_FEATURE_DIM))
        self.min_support = rl010_int(payload.get("min_support", RL010_MIN_SUPPORT))
        self.min_expected_delta = rl010_float(
            payload.get("min_expected_delta", RL010_MIN_EXPECTED_DELTA)
        )
        self.lcb_z = rl010_float(payload.get("lcb_z", RL010_LCB_Z))
        self.bad_ucb = rl010_float(payload.get("bad_ucb", RL010_BAD_UCB))
        self.models = dict(payload.get("models", {}))
        default_actions = [name for name in RL010_ACTIONS if name != "CONTROL"]
        self.allowed_actions = tuple(
            str(name).upper()
            for name in payload.get("allowed_actions", default_actions)
            if str(name).upper() in RL010_RATIOS
        )
        self.include_opponent_features = bool(payload.get("include_opponent_features", True))
        if self.feature_dim != RL010_FEATURE_DIM:
            raise ValueError("RL010 feature dimension mismatch")

    @staticmethod
    def _predict_model(model, features, uncertainty_floor=1.0):
        if not model:
            return None
        mean = np.asarray(model.get("mean", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        beta = np.asarray(model.get("beta", []), dtype=np.float64)
        if len(mean) != RL010_FEATURE_DIM or len(scale) != RL010_FEATURE_DIM or len(beta) != RL010_FEATURE_DIM:
            return None
        x = (np.asarray(features, dtype=np.float64) - mean) / np.maximum(scale, 1e-9)
        prediction = float(model.get("intercept", 0.0) + x @ beta)
        uncertainty = max(float(uncertainty_floor), rl010_float(model.get("uncertainty", 1.0)))
        return prediction, uncertainty

    def predict(self, event_key, action_name, features):
        features = rl010_mask_features(features, self.include_opponent_features)
        model = self.models.get(f"{event_key}|{action_name}")
        if not model or rl010_int(model.get("support", 0)) < self.min_support:
            return None
        margin = self._predict_model(model.get("margin"), features, 1.0)
        bad = self._predict_model(model.get("bad"), features, 0.01)
        if margin is None or bad is None:
            return None
        prediction, uncertainty = margin
        bad_prediction, bad_uncertainty = bad
        return {
            "prediction": prediction,
            "lcb": prediction - self.lcb_z * uncertainty,
            "bad_prediction": max(0.0, min(1.0, bad_prediction)),
            "bad_ucb": max(0.0, min(1.0, bad_prediction + self.lcb_z * bad_uncertainty)),
            "uncertainty": uncertainty,
            "support": rl010_int(model.get("support", 0)),
            "train_mean_delta": rl010_float(model.get("train_mean_delta", -math.inf)),
        }


class RL010Runtime:
    def __init__(self, payload=None, opportunities=None):
        self.policy = RL010Policy(payload)
        self.opportunities = {
            int(row["current_step"]): row
            for row in opportunities or []
            if str(row.get("item", "")).upper() == RL010_ITEM
        }
        self.history = RL010History()
        self.pending = None
        self.last_step = -1
        self.interventions = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.advance_units = 0
        self.delay_units = 0
        self.repayment_successes = 0
        self.repayment_failures = 0
        self.fallbacks = 0
        self.errors = 0
        self.decisions = []
        self.last_route_action = None
        self.last_adjusted_action = None
        self.last_final_action = None
        self.consumed_event_steps = set()
        self.last_repayment_step = None

    def reset(self):
        self.history.reset()
        self.pending = None
        self.last_step = -1
        self.interventions = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.advance_units = 0
        self.delay_units = 0
        self.repayment_successes = 0
        self.repayment_failures = 0
        self.fallbacks = 0
        self.errors = 0
        self.decisions = []
        self.last_route_action = None
        self.last_adjusted_action = None
        self.last_final_action = None
        self.consumed_event_steps = set()
        self.last_repayment_step = None

    def _apply_pending(self, action, obs, step, config=None):
        pending = self.pending
        if not pending or int(pending.get("due_step", -1)) != int(step):
            return action, True, 0, False
        trial = copy.deepcopy(action)
        delta = int(pending["delta"])
        ok = rl010_adjust_sell(trial, delta)
        if ok:
            # A positive repayment (DELAY) consumes extra shed inventory.  A
            # negative repayment (ADVANCE) only reduces the future order.
            state = rl010_market_shed_state(obs, trial, config=config)
            ok = state["shed_after_actions"] >= rl010_sell_quantity(trial)
        if not ok:
            self.repayment_failures += 1
            self.pending = None
            self.fallbacks += 1
            self.consumed_event_steps.add(int(step))
            self.last_repayment_step = int(step)
            return action, False, 0, True
        self.pending = None
        self.repayment_successes += 1
        self.consumed_event_steps.add(int(step))
        self.last_repayment_step = int(step)
        return trial, True, abs(delta), True

    def _legal_units(self, action, obs, opportunity, action_name):
        current = rl010_sell_quantity(action)
        if current <= 0:
            return 0
        if action_name.startswith("ADVANCE"):
            ratio = RL010_RATIOS[action_name]
            desired = rl010_round_half_up(opportunity["future_quantity"] * ratio)
            state = rl010_market_shed_state(obs, action)
            return max(0, min(desired, opportunity["future_quantity"], state["sellable_extra"]))
        if action_name.startswith("DELAY"):
            ratio = RL010_RATIOS[action_name]
            desired = rl010_round_half_up(opportunity["current_quantity"] * ratio)
            return max(0, min(desired, current))
        return 0

    def _safe(self, obs, action, opportunity, action_name, units):
        step = rl010_step(obs)
        if action_name == "CONTROL" or units <= 0:
            return False, "control_or_zero"
        if step >= RL010_CUTOFF:
            return False, "terminal_cutoff"
        if rl010_float(rl010_get(rl010_get(obs, "market", {}) or {}, "prices", {}).get(RL010_ITEM, 0)) <= 1:
            return False, "price_floor"
        if len(action.get("market", []) or []) > 10:
            return False, "market_overflow"
        if self.pending is not None:
            return False, "pending_debt"
        farms = list(rl010_get(obs, "farms", []) or [])
        seat = rl010_int(rl010_get(obs, "player", 0))
        mine = other = 0.0
        if len(farms) == 2 and seat in (0, 1):
            mine = rl010_float(rl010_get(farms[seat], "money", 0))
            other = rl010_float(rl010_get(farms[1 - seat], "money", 0))
        if action_name.startswith("DELAY") and len(farms) == 2 and mine + 1000 < other:
            return False, "cash_lag"
        if action_name.startswith("ADVANCE"):
            state = rl010_market_shed_state(obs, action)
            if rl010_sell_quantity(action) + units > state["shed_after_actions"]:
                return False, "inventory_short"
        if action_name.startswith("DELAY"):
            allowed, reason = rl010_delay_guard(obs, action, opportunity, units)
            if not allowed:
                return False, reason
        if int(opportunity["future_step"]) <= step:
            return False, "future_not_after_current"
        return True, "safe"

    def act(self, obs, base_action, config=None):
        step = rl010_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl010_align_hands(base_action, obs)
        self.last_route_action = copy.deepcopy(action)
        action, repayment_ok, repayment_units, repayment_consumed = self._apply_pending(
            action, obs, step, config=config
        )
        # A repayment turn is a transaction boundary.  Do not immediately
        # open a second intervention on the same observation/event.
        if repayment_consumed:
            self.last_adjusted_action = copy.deepcopy(action)
            self.last_step = step
            return action
        if step >= RL010_CUTOFF:
            self.last_adjusted_action = copy.deepcopy(action)
            self.last_step = step
            return action

        opportunity = self.opportunities.get(step)
        if opportunity is None or self.interventions >= RL010_MAX_INTERVENTIONS or self.pending is not None:
            self.last_adjusted_action = copy.deepcopy(action)
            self.last_step = step
            return action

        event_key = rl010_opportunity_key(opportunity)
        features = rl010_features(obs, opportunity, self.history, action)
        scored = []
        for action_name in self.policy.allowed_actions:
            prediction = self.policy.predict(event_key, action_name, features)
            units = self._legal_units(action, obs, opportunity, action_name)
            safe, safe_reason = self._safe(obs, action, opportunity, action_name, units)
            if prediction is None:
                continue
            prediction = dict(prediction)
            prediction.update({"action": action_name, "units": units, "safe": safe, "safe_reason": safe_reason})
            if (
                safe
                and prediction["lcb"] > self.policy.min_expected_delta
                and prediction["bad_ucb"] <= self.policy.bad_ucb
            ):
                scored.append(prediction)
        selected = max(scored, key=lambda row: (row["lcb"], row["prediction"])) if scored else None
        decision = {
            "step": step,
            "event": event_key,
            "selected": selected["action"] if selected else "CONTROL",
            "units": int(selected["units"] if selected else 0),
            "candidates": [
                {key: value for key, value in row.items() if key not in {"safe"}}
                for row in scored
            ],
        }
        self.decisions.append(decision)
        if selected is None:
            self.fallbacks += 1
            self.last_adjusted_action = copy.deepcopy(action)
            self.last_step = step
            return action

        units = int(selected["units"])
        delta = units if selected["action"].startswith("ADVANCE") else -units
        trial = copy.deepcopy(action)
        if not rl010_adjust_sell(trial, delta):
            self.fallbacks += 1
            self.last_adjusted_action = copy.deepcopy(action)
            self.last_step = step
            return action
        self.pending = {
            "item": RL010_ITEM,
            "delta": -delta,
            "due_step": int(opportunity["future_step"]),
            "source_step": step,
            "action": selected["action"],
            "units": units,
        }
        self.interventions += 1
        self.changed_calls += 1
        self.changed_units += units
        if delta > 0:
            self.advance_units += units
        else:
            self.delay_units += units
        self.last_step = step
        self.last_adjusted_action = copy.deepcopy(trial)
        return trial

    def record_final_action(self, action):
        self.last_final_action = copy.deepcopy(action)


def rl010_fit_models(
    samples,
    ridge=8.0,
    min_support=RL010_MIN_SUPPORT,
    allowed_actions=None,
    include_opponent_features=True,
):
    allowed_actions = tuple(
        str(name).upper()
        for name in (
            allowed_actions
            or [name for name in RL010_ACTIONS if name != "CONTROL"]
        )
        if str(name).upper() in RL010_RATIOS
    )
    grouped = defaultdict(list)
    for row in samples:
        action_name = str(row.get("action", "")).upper()
        if action_name not in allowed_actions:
            continue
        key = f"{rl010_opportunity_key(row)}|{action_name}"
        grouped[key].append(row)

    def fit_head(matrix, target, uncertainty_floor=1.0):
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        mean[0] = 0.0
        scale[0] = 1.0
        scale = np.where(scale < 1e-9, 1.0, scale)
        normalized = (matrix - mean) / scale
        design = np.column_stack((np.ones(len(matrix)), normalized))
        penalty = np.eye(RL010_FEATURE_DIM + 1, dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        residual = target - design @ coefficients
        return {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "beta": coefficients[1:].tolist(),
            "intercept": float(coefficients[0]),
            "uncertainty": max(float(uncertainty_floor), float(np.std(residual)) * 1.25),
        }

    models = {}
    report = {"groups": {}, "skipped_groups": {}}
    for key, rows in sorted(grouped.items()):
        support = len({
            (row.get("opponent_source_sha256", row.get("opponent", "unknown")), row.get("seed"), row.get("seat"))
            for row in rows
        })
        if support < int(min_support):
            report["skipped_groups"][key] = {"rows": len(rows), "support": support}
            continue
        matrix = np.asarray(
            [
                rl010_mask_features(row["features"], include_opponent_features)
                for row in rows
            ],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[1] != RL010_FEATURE_DIM:
            raise ValueError(f"invalid feature matrix for {key}: {matrix.shape}")
        target = np.asarray([
            rl010_float(row.get("margin_delta", row.get("cash_delta", 0.0))) for row in rows
        ], dtype=np.float64)
        bad_target = np.asarray([1.0 if row.get("bad_outcome") else 0.0 for row in rows], dtype=np.float64)
        event_key, action_name = key.rsplit("|", 1)
        model = {
            "support": support,
            "rows": len(rows),
            "margin": fit_head(matrix, target, 1.0),
            "bad": fit_head(matrix, bad_target, 0.01),
            "train_mean_delta": float(target.mean()),
            "train_min_delta": float(target.min()),
            "train_positive_rate": float(np.mean(target > 0)),
            "train_bad_rate": float(bad_target.mean()),
        }
        models[key] = model
        report["groups"][key] = {
            "rows": len(rows),
            "support": support,
            "mean_delta": float(target.mean()),
            "median_delta": float(np.median(target)),
            "min_delta": float(target.min()),
            "max_delta": float(target.max()),
            "positive_rate": float(np.mean(target > 0)),
            "bad_rate": float(bad_target.mean()),
            "action": action_name,
            "event": event_key,
        }
    deltas = [rl010_float(row.get("margin_delta", row.get("cash_delta", 0.0))) for row in samples]
    report.update({
        "samples": len(samples),
        "models": len(models),
        "mean_margin_delta": float(np.mean(deltas)) if deltas else 0.0,
        "min_support": int(min_support),
        "actions": list(RL010_ACTIONS),
        "allowed_actions": list(allowed_actions),
        "include_opponent_features": bool(include_opponent_features),
    })
    return {
        "version": "rl010",
        "feature_dim": RL010_FEATURE_DIM,
        "min_support": int(min_support),
        "min_expected_delta": RL010_MIN_EXPECTED_DELTA,
        "lcb_z": RL010_LCB_Z,
        "bad_ucb": RL010_BAD_UCB,
        "allowed_actions": list(allowed_actions),
        "include_opponent_features": bool(include_opponent_features),
        "models": models,
    }, report


def rl010_load_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


"""v27 current-meta midgame reset for Kaggriculture.

Both seats use one coherent fit-only public route selected after the HIRE4
opening became the dominant Top-30 prior. Runtime feedback remains limited to
actor-local WEED repair and ordering route-existing SELL slots by official
price impact plus bounded current Town demand. Opponent identity is unused.
"""
import base64
import copy
import json
import math
import zlib


_LEGACY_ACTIONS = json.loads(zlib.decompress(base64.b85decode(
    (
    'c-rk<O>Y}nlKd|^^I(#)Z0}8NbEbt+TZSwzG20Lt4a_VSSj--J_qN#ozOqEJij|R(k@;Rxvd1@CCad1}%Z!YS{Plm&{{8nq{_*!e'
    '&i>`svrm_wKcC$%&i>=~|N7g1Km6h0<3E1?<3IoYKM$XOJ^T6UcJuJR^uteI{`%YH$E#m1ug?}|?{Btei>3MV=bty5PiKqs{eOJk'
    'Y(6~vdHeI`^6qT$dh+LAHrF>FM}Piwd-LJT`@8WE?*DIb)QhY4fBEuh^!`JCem&c6KHohy^zdQV=h4p&?HhOBd&jO3$8Y&~b9?vm'
    '<3oo}_C33w()a9|sXqIsFIU$eetY=m-'
    'IuQuLLNN%rr!GN%lDhZAkiV(ee>%q96kTxKR(_aX4ZMnpT>)Vz2^9fM{|97x4HG6|Nb%<pr<e3aoP7^|I*QOcVA-TGTCJ4aYNG!Q'
    ')^!^JPs^-eM0SX4^Q(4M4m|d_|G?Ab^{K^Bb-'
    '2goQH*Hhodroqt^N3&@_LCQ_GG+%ls(=(lCG0xK!qG|64E|PaUW|Zdh;AKh>URhqudWVBK$84f}^|E;}v)Wi&dkfu|3N$00i>ybi'
    '*Z_WtJjdh`D7w?A!e@2;+|{_U}u_C877{)KA`HG@20f6JvB3f>wvG#H&^v-'
    'f+q=LA(YfBnGt@sl4vc|kupJ`+E0uD`l&qn+~Pkzo(e_-'
    'GgRDgWtUg~TV1Z~j|9Yf(GOj6ZZdG_b?V`{Y?O=|{`$FkFh2hJy1Swq0qVf0y7k#y>a16dv+>`=IkMfx*Y4RB7PU-cK!pk=ImswH'
    '@F>6NUjcEs)0-Oq(;nVFQ_GSvX3~5EY(r7$N&xb%ek}@c@-'
    '?i+`5iR<F9FJMS39Tu%P{`R?{|`_tz3_OEA)b@4KseCU2D_PQRQ=b~)AGWYK3Xr@|wBDrD<092N*RQ=wtjkCuZ9+76ZYI^-P-'
    '4np?qZe_H4j9-'
    'oJ3?R*5!M;|l8S{iERWLjhK9NLcPBH`Gd(mT#M%oJOt9&)wFid_KouJ~0o_`zz8??i^N=>XpvmJUXW}$m`s42J<u121KIst_+id*'
    'jqKoD@(qC8a<?X)~E-'
    ')~d<eDgh4ha(nJQM`dDo*m$#g>}0JK(kD{2Y^@yWEG?NgW@)jXiK1{m#emnrsK+o*!;yWkPh1+=fHtS&~9wl|Fy}SNHzpe)DJy*W'
    '7%E+@yQ|=WTSQx~DX~{#O~}paF6~HbU%z#qMcrDYY9M&ut+f2lE8x0wIU_c0+t>dxWsQqwKFlb+qF{*kc1k<E-'
    '|<+6u#cdAO3EKD2G3>0^7qIu0sw0;D_PinEwNiz}|8Xf4+w>t=mS6}aTY4`~Y17~;{1X5cwNg>@cOP>gl(nwMo9j2)6w*aghp2>n'
    'SPhv)*L<nO;k_<=z{4A<y2@WAuz4uG_fPEc&4ZU*QPq!SGNwleO_kV)7c_Az)Ngb(^~d-'
    'Ew+2gJS_Jjt7z>)lBWPVn{K{d>50KAgq2okbhatBD)B>KTOIKQVJ~=r$?$LJ!Y5EH?eg#Az8GYhba7QIOzed)^F4Hc2c{<*KwXqk'
    '=1db9=-'
    'PcnN)Z+%rqMX_OSR6hN~oq9BhyUgMRh#Zlx4&YJAfcRHi8HSV*3Au2h<(}Le5Kg|mCp!cdiU6+>XAQ4D!^UaUBY)WAMIiurr!yfs'
    'S)ipi*=?E+^-J-=%3zpaMWq}Tr9Q&r`XArN&n=`haI+Kj|uR>EJubr-hQv{ycS9%lG<eh-^9J~*%=#6S002-'
    'B9vtSAM#&g2tQCP?ToQv>A=3dlsSQ25?&O0=W9r#rCNSqjlcwyM6zs4TaGN+vTRG^eet^|Q_?Sm(?!@^1W(uFCtN4o|+_F!;-'
    'xcc+tONBSagg<2Cqp!CFp~1cYZx^qTN`#n!yo=ca({A?Zyu`C{1bc8~db&*u?5gyPDRYO);1R@t<q9HWhic_IR4d))m0X$OzP4l3'
    'l2YruZhU?p23EKDSN9##b5-{#<47tgWY`6=Ce$nrP>3Th7!3$nShYpL{T<Uv$Y+{)aR^?IG3o@`nnZfAX5-'
    'W>>J&W~ZarrdGWmH5Wfhplt~Z=}S1^l?wRN<l71|uI|AEOmyv>Zg>pRDbhs7sC8}cw2ETJ^<LbQd$b2eN=z_s&F1M1;sl0wNz<+C'
    '4Mtm7~o%6h7vPGIyAC!_~cb9zHB0zzzyiOf`Z7$VeQ9^>GgVi-@Y+Kc-'
    '<&jW}I^|k&_X!QSbb^VvGqkv;3%U9M9c+(n3X)^Iwjs?U`eC8iS*1iCsOZGaZkYr3%ffXQ`$2ohsJC^{p(cW~GHm$REBtovDo3R7'
    'Qr9c*moWS@5uW(8x%a)wxbqF-iK@qHbHZHxK5NbmX3%ame8loIvX@m{~h;mfYVF?vH42{JYzS%M^IV4=PO}#w41DI}PJ$BeAYzUB'
    '35iLP851rEExpy2aLBrlk3ftHS1jwZmgR@O=EpaEutFul2(R>&`n)yTha;D(2pMgD3-'
    'apa0nJGDJ5$NB{_u9E*470q5QM4|c=)yp0?CfI9HV9o9WyeC~Z#p<g;LOt<BpbU0Ar<zeF2sFD1WS8k(0HFx-'
    'kK+1kD{$>hY$`z*7`4U=xR8Y?V9!8`~-H#C~WMCwT-OC@X*|pV>eaq4q{<qm)pEUkibMm^)1poQsma=g~j~xo+jvkfV$i601MMe)'
    'd2*n3dXEj0Cr1&)<gYyQoLX$yc%fs2Nq9wG5|9eb(YT0WF#uu%&<>@aE;7o-'
    '%WLxP8nBXGRZ_a5^oklycrhGE*AwlK2Dy~CQeP+nJE{Z_2P3Q-n!ffg)9l=-'
    '5pAfNXU{Iv26NnR0Y&Jh?#<rNs{pYA%5va4<TftILAr<oUC&2wh*S0VL@<U9=3uHG~g~p%YHm#7w-'
    'CgcY^gzLWz@af`MWezBp2it2H`$xuxThF2?`&v<Ud@Rt{<bnD%+hLB9gHOoJwpih+#_QM<Qj@g#N$%8~>t0juO|Qf7f3ju!$RWcy'
    'TKnWDW=h&HFYk<h2?x5$)kgfo`81ax*kB`W*NEx|ZA9ciFPLgguOI}OR}3BHL0WC*@dXhci=JZ~VG@@pxF+cZLHs3{pRE!(o0Nvd'
    'IPCx8wZCV+0Ra-Rr}HUmYXJHDi(btyE8GIl4h65V;%9n@hR9J%o?fSykY^RaknIfI9-'
    'TRXS8WsCM2SXV_w)sO@%P4Obws7)fRFvZMF95?cb@J5Bx%w{*<M<Uh{Z+F|U;n(}zJDV6lIrik<s*O;HYUOqV<G-'
    '{C_e7|GC)tA6S(Eq=il0gg-r^7inQ2VLT?y<j3QH#fEN<P?V!Cxci!$VNL<FWjZ0jWB!)oVL?%uiGo?z8IW3wvx&?gw07A@#k-'
    '<Weu*)1t`NKD-aaXuve$em;R+woDka{p(kJI`<L72!%krRNTZ9R`=f(kmk%$&MA+H78FKPzbiq<?f@21hF{H9N<y%KI~y-'
    '?Iv6zXsR84cf1j`5E9&R)qG?xzpm_Zre#p!071g8B&ZUP9}yAvYGNeVyxjipC-'
    't%=GzB_h_Uj256cD~7%tC#LFz+H$wANpDTl!!jiHEF;Owl;9?l3tE_)pw3?<*!mb_M4AW#vk1MQE&B%28jUm9-'
    '(fmCTvPBdCUQMQ8Ka3URLqg_1Nj@!knhU*`HbbL-nl+I><INJF$V;wt;^O>Y?zxm043GHroOys~2&66JE?Z;{W;2t<{(CH&O|=RA'
    'fpZiTCh?z7(rmW}_I7;-+!3)#n*8+`B8_<6h3lhrquTJp^4@>@r?Bb_$irf{TwI(?wIO-'
    'T5(fj0t0^o|y{M%~nk8x5;01sNnQB_^a*f#zY#3A}D*U1m%sm36RTk&)0=NAhr+-'
    'Vt(T7(cXzIEcB{JgR72lZt7|T%82@H4|$Et?>E+SEc<<!zs;E&(T?23$o@Cedq!qlz*B8P*7J>*CMs5nJvu&qq#(ViiQ(X!+hEE)'
    '*i8*VCRx-EP<_-0@g*w>{JoF)AMRy6q9$Hk8cuMj(w{23v!#ZMEnl^6Y#Jbvg!SpWuM|b?&E>66Qltq0CS-'
    'r%NB%(WUAJ4)oZbhW<0Jo2W#D4(6^Z6zKt`cYwSy|rJVfV(#{xXm`B0LMK+?J7$jwbuq;l+iav$xncW$&^F?GKg6vDJ&8XaqhvWN'
    '^1`uD=m0UB99AfTmbSmQv7CYaPOsS;*Q1FT#{(eB-#L3F`#4s0&(y+;^6Fit~Pq=5nH_6@+R82HD5&LBdaN^kmz-4?Jo1-'
    'b2KSUSy)QVC*0wXy48v#VLa#L4#C=qd3-WfJL$oZ<(MiObDD1zCP{VEMtTc+;ms?E&hi@y6l__DU-%~DrCsqkv!7~i^-'
    'bn5VE)48V_eYS`2S%Vi%LxGOhJePwi%OV5MDa-H$!L~V^>QvSi4_f9O9ff!y-'
    'M6JQ!G=*y?wm|%4`^%!i=kyR*#&^Zyp*1bEtwuNZd`Xpa@R9q0-'
    '2W>@JJ5>4NYeI$0F@jp&SM|?*P2WfILj+jf4QHY!^5L8v$4(L8MAo36(30Mc64P80*5J#<Z_$I#4uXUE)=0)w}Qj97JMx`64xi^T'
    'N>Ep6F;Os2Q6B6ezN(oz!Tg{&y~kcN0#Q3T?F%*-A-'
    'GxN(rC6m=@~b68n~SL``ugh2Tg5Ge5iTP3uLteb3OAx)04UEC7QRbY<I<*SaPq|0*~COs~kG|bI66*<jIn2$ZpnyZ@%$M)f~LJxQ'
    'C{!gfp4uADps482qH6k}Xb;$WQTv1;<5fV<Y#{}p9B?OMOGFLqA$$C?%AXE=9C&VfB6jv=6$CKI8-XJpw$Gq<dFb`;YQ7Er-'
    '4KJzOyv8bKS#Ui5lF(d{s!t2C4(c4B_#t+1DyTv$Q9nwuro|N4c_3t@)gS;zqgRfW<2C7SG)8c!sL)6$duWGdYdj>5x>hWPBE4d='
    'aafa}tzsIY&zT8O!unZ-fk<3B&SYl7V@;udEnl6T#DxU~0B=0F;JVUT0+c1yjU*yeGmbd~2yIf%BVFt%DkW=o3s@{7Lp{gXKteo#'
    'o1R`|OlXU2e?c;Xi7#%IYS3=1D3~QlJ1mw=+s~d%MGR}0v=`4N7!ROaDRdH{e#t58aSt_6qEk96%}rFoACb4!Qd9}=QwaR!NP<t0'
    'uy;;!!MB%*Jq5jHMz>nj!{w}lPN}rnYSo?}Q_QGU1fT-'
    'vWxIJk(PSVfB#`^IV!*ymAPw-1(A@2zOx;CCK$mPnOR2459N_)B#vElhk%Zk-Ezm8c98lM}fv6)z8HI%@(by9_GDb&WCydB4>sys'
    'Ol3<EmJqoZ@28-'
    'G8KG4Ujy_ZNND!+zu5(=seia!Y68BP_sG++l~AcQUTKmwA{aAq1aVrL|lxjIyu44E}gdCm=n=B;Be@+~{~>Ji(O9HA2|B)u<LV`N'
    '&HZ#S`_E21~7#Z+R>*EUx~@rq-'
    '}G7Myy^9&KpHU=UO3SEPLzbBbvWhpf8zK1OMML#EYZk2CVyxtHoDyrF#P~E~4+iv<0CK|=7vN&;fdWXI0bJE>}a9X5qtUGx?phL2'
    'xcP_S2p>-'
    '5<K_f&|%(^Va3A1YYI8Q)vpn0*Mt_u5(S}@D}&E=!Fr1j@@_wzh|V0g#1)8}AeG57>boGMr~lM+Ao1X2Qt$=F{PWktN~1)rXuCZ)'
    '1cj3-'
    'ZJuq|rIf{URTYC9H3)!b8R<Ylr=g)79SPf#cql0BLM3k0NSy7OqCNi~&@)dv<0I38cb@lipE8pIcC9cw5`Ae;w?>)h0`Kx)Ez>5P'
    'Oc4p_r{D+=4JL`Epgof0djwcV8JaqTF?T*`-VDwV?S1#SR<6qYrm^CwkvPr&aezpSll_cxJ-GD}3>7t|}|pU{)d&Ejy9N^%kGIm&'
    '}8&`e6WVMd++z(CTRS*f|tsRU2Ul4ivqfT_fOF&phbmzYCywlBp?F$<n2-_A~eS-'
    'L`!iJneFvh0xyyB`%L(YGjGC}pKMn5yEmFvW=a95yK+a$*Flu%*?Yucc*&J|Y?9e301}whP3DLKXp)3(QrD!(mDxtpJ~t#q#g;ND'
    '5Cqq6f*rt^MAQ9cikPBaBuqV~gg;kqou=(FgK5#B2!ka`L&2717ShJxNYvayHSzwcE@VV2F`6cpC-'
    'A6^qLWX*g9KRcZcpvxa8X1pKAiyab`nB6kd!&(cUK_j_h~1avyA*Ggk*Qn({E3G6OSZEIQ+qygg2Y3E8eEGL}OObu3A(1K7xYRa1'
    'eXcPOB;oh;ls+?|Te8uOKLHp;VJb`SA&PqKq<w&IEi172OAHC;gq^2~Ax13*`D8HpkF%1r#y)J6C%n=XGSKnZs5({uwDl}WZWTHa'
    '2c6H{<KJ-W)(4BeWG!BeSr9yVZTA|;ZRmDZmhKz&C8JAfttSyNi4k!EO`sO1@8Qm)6zQB-'
    '6bDuPWtE<Ed$)}FDDAYEjl_n~Dg4Dc8nlQjWw|E@W1W?>T$Luap^Ugb=`iRHWD!&B&G{ig22!gdi47c4JFNK=1>?(`F!VqqRO1}O'
    '{6^pkein;igvv)~C=4=dxOr}IGr08u$twN2jVo3sqB9g(5qTW4nunZ#B20B5PqXKp`)IZ9J9Vt1yF_Gvn`MH(KIo?XnOE}dLzh7H'
    'zlBq|vUFD?Cpg+@=aJrobP4i^eAn2Qv$csG?uws+sQTTOcg^~7RW3awJh2$}yw@a>E$S3zhD}nS9_e3)i_Dp#>QJtGze{#(L3+iq'
    '2fzsffCfYtCFrPyzQ|IA+sjco4rLzZv!6(1zQG0j_`5>Z5k@F5l)-'
    'TTMv)vDK6OX<iC=VgneA*ufz^!U9=fU81sDNA>e2j5_N9@3^+SE?s<51lRR&SWE3e!rhb*R`!u$>&9>qYD9gNEoOoA@_lT3@GfPK'
    '>d}3pi0JPttw@GL|<<tE3F=DUA9hWT5f4AIUE-pkkDu(;H`>Fr2iZz7)5*5~+jyLAAn3r9A1NAsb0Ynni4ivQXStQ#v$!dr#|Pu7'
    'nFS1P)8H-'
    'gV&3;GyMKSCt?s<wvW!CqC=e@{aLJG6z!Vru_8f>iWZP4|dYs7oVgiCbf|cARUMH0|#X}U0=T6>};vDu9tOTKnM17q2)PFEM?<)>'
    '|V)E9&!)Ez<2-*lUa(G>yt7xft9<AtQ4LYjsBNIPn8Pr%S2cqaOtBW0b)h8gf>(n@dlobhxHzXf&k>R-v-'
    '2DGI;?(*N(i*SFzy>*M(ok4?FVYDu{~>&dMXeYwIv)An!4f=F65k=as*fpfXjC3`z}gPj-vznRO`|7661%oeB#M*>-'
    'v@?~{~jvrY9hAdb8xJ5VbjmloSvpIeY`x%v*scTF_8b^oN(6ynjydE=N-'
    'oH*g*OPxxM5&57bsMJLcra(|*1;rxDSwVfL&NtJGQtQ&#eo{)x;bSF8!<z4cPDN=Kag$k5yOQ9Rwv7Y^%nwNh6<A0fTDLT5<^}5#'
    '=1O&*26-'
    '+yWmC##N>oT~tFEjeqS?|do;Q$dNTt|hP!CDcj?L8gAz6n(14zZL%K8K2mtTPS>_fH+64eBGS}<LQEX$K%8I_y<>9C7*-'
    'x7E`Eo&yHHyt^L%M$qrUyg!?yXgAnwihCE8C55GQxR9NVgBVa9_vxbRU!f3A{f*13TARnit`DS&BMx8{({VEN^5f6r8SsxEkQd@Q'
    'MlaGy8b7~e4EKZe_eIZ;nObE^4u|9{-'
    'kA|n2|^R(#i3+#+6tUN+J<ZMn4~xZ>F%6TDUW!1<q583Jte@H&Pq{ea*yaPglK)4td@>S1n@3p-%X`5uHGhX`><dkd-'
    'R&*{c+S;+9M$Nn&oq#$^{o201vq!f<}TP3==5833_~VWI)WcD50qD+tiSR+4{DmVYMHrLMJyB_F@Q>enl37=*Bb@rGe>dcG(*Tk<'
    '}$>!K!NMI}`%5sZSLh7v=Q*#zoPehu0w6PBP{K03wPqBb7CS4tjYLbiPiQh!q7{y6w97EV{smqk5%L{18t(hF8DVy}Q7kYmLcZJ('
    ';dlqXpM@s+rXX0);6sCWw5^Z^T?0E~5%Fq}M6qoS24ze<si*wOnI)hiPKU;=PUB4IWSE8507PQ~Qq1wxob?bvxra-'
    '^6t@AOMW*p1P_f{~?CSJXN3fV~v8H>d8U>uUuMVh15-'
    '_~miOl|4+KWVpT;<guxOo)0gsixu@~>Tj<EGBF#>BLImqQFCXEQI9|~8kzi8PnblidQZQtsTZ9%VFVMHqFkzIE=9g}B2R=ZhUnUt'
    'G_R;8bQ-A93hr^tk#G(cTA{7BN_$+pTlpLntuFAR!UgGD^ch93t&kC-oiXX?0x<WON=}HlZF(r<N@`BAzOq`!kcscMBrKwh#L^-'
    'nLsiTbXkq=KPyR{{Rg|+z)iU2t^fAL^#NN3k(^Tu&NeiK(A*)BL)9H#+2Vx#M-%`Om0YI7Igaw%}W8?YVeL_)R!q;l-'
    'N{GZxJhWnoca*Ma1ZybB%Xe$4S%GG})G$Si%d7K}6jed%&Hx{Wd9o>Z)DB`|m{0Q5OHZ~1$Wx!YnO%uu3!<Ej9TdRyBGW{A&YNB='
    'G$^%~VZ%KMR%r6pORmE+D(Lqt*vHK3;*ApG=bl)f!n9hKD-!kQQHioTwCGw>V<q;5$eR*d>0a=sBt)?kDow>)r9_G5-'
    '%?lF*$N^pIwdNuu^1sJed$k}k)!9qGFFTf@$3N6RWeVI=vQR)s7fjQp*zND;jAUGN?bppSw2}&AlbQ-'
    'N3%#uRi?5jR;)l4Zwi~QTo_K+0CC+LMS)y#5b8S{Vy1NM=c?VQXI|@eCMgqgVL|4oGXjyXkJ^K=#uH`}u>wE)5%-'
    '}IJCHE7K*S=@X(k&yhuzRsPH;bwLdGiA6_V5v^eT!)$~6T85=!fag$GYSY0Z{)Ru~%$%E}`y^;3vZi6r?ls+yNWXN(|6BXJ0{HWH'
    '|z{c>je_;oc#N>;;1X4Q0Q-FzQrK!z>=GW5zIL$ALlIp&upibGj3lz-YqCPyX7NG0DGlja#@bF`r~l*d!ZvuP?il0_xeOn%||IyC'
    '=_g$V_E=0->jqS>Ev)SFCKYgAq^v{(X082$#l;r@YoALGn1cn!+d06Lvm$fufel}!{_)|FLQ2wZk&Xw7z!+*j-tRrO$<3nf-'
    '7d@@B5NGVt)Y6PiT&y7h#O0`ndtoWj`YFxaCNiy;rxp;ZdRiaC>wI7P2D595=lsGK+O{&TTuKuy%l-mQj6fz-'
    '~g~cQohASfcK`xe2wk>A9M&^+pxM7R~UK1)21YoQmg4NQHCOnlC!cuEH!&}>U2=;JQ>O58HO5`OuZ3H57bWAPEA3KD;Bxo5$cnpG'
    'QoZjW{1~&FNJt^hG^x0hnlvY?k35>#Pkn;2$fICfzPUragkn)<=30arM#3mt_$@wlK=o5R?Bt=5@C@Lkf+SH7e<$Qy5Wh`C%xJyU'
    '4?4$skX#z!5WTIQeCn*=GUM`s4ueltY^fr)$(jxQ~_!eaW?ZOU->L}^7ZmI5#4owm`Ox(2;b!TO9I)=3Ut*NS^ZD(A-DRF_6>P-'
    'Z(wr1n-VnJkMA~%VIJ9-'
    'f@BuqIb@#t%z0uKr$zI7H!o7#8qBF?F0QGlX;y|spI_9&n#ZpnF1Br8Q1=xuDXXEKJvGH6VV(Nm3ys=&OfR)O*O09K#Z79vxV!B{'
    '`^=CK;xOvo$L*<d&9MZ&Xp%TW&^E+IH&d(;S=)TBsQ7Ll^O#%}m|XO~z?bs~*yg%tod<oq?w8ZsgwFEg9S!$rzXi>wb=Xc2||(9s'
    'iAX`~621z<jwEMT6kqEOsl&6Kk#`HkZ{EPd*yP)}TqF{xZ_PF~d;oh(45XEN~Bl_K0FgbVfbbLJDo%a^IpDssR<ibjG;O*!h67R4'
    'N}z7wc$ymG!J_klLE%4D*Y6RTX+97ru4Q_1lv`HZ$f!1RPdjB=e=QrfprDpC=-'
    '*rqDk)Qa>r+<+9A&{9Wv8RRs{7(O{H3B1uXA2LIiYZxavw3Q=cp+yoyCI!U(sybvFyj8b>;*5eC!dihJURB4bOl`dky~+ISfhYj0'
    'd=p>$DbJ)SqWq%~LL;Wv60?P0Q}?w|b^`i}xeD>bY^3oY-Ubyw)wIBfO;f2#$i>E3L*ik}#@wlS$Gu)=fyTo^E1w6v?NnNI+xUo<'
    'q$q)yk&2_~=<55p5t2jvX%OC7!J^((WO!EVG(_2@lsD?14fREoNNtVUo@kWyOa<>z(0Wz#15!;B0%`mtmrm8<#;cGlRr+Ug<d~{n'
    'Fc^%TQ49UTWgq#9kqbqCLw}rFaOmU_6bazBbYL1d=2WGC73?_pD;gvnM>BTd{Kb8((VV3hq9iDT@v(=MPMoLu07eqHh2+EZ+Z3k0'
    '5IC&6#Q3*DHxkW)POe)fm3Cs4>4aTlrt<Amv#5im^r5X+>kC!-v2XbU*vq)NZys88|NntQ0?7'
    )
)).decode("utf-8"))
_REBALANCE_ACTIONS = _LEGACY_ACTIONS
_PRICE_FLOOR = 1
_DEMAND_ALPHA = 0.25
_MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _regime(configuration):
    interval = int(_get(configuration, "townCenterSellInterval", 12) or 12)
    return "rebalance" if interval >= 24 else "legacy"


def _copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align_hands(action, obs):
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(actions, step, actor):
    trace = actions[min(max(int(step), 0), len(actions) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair_action(obs, action, actions, step):
    action = _align_hands(action, obs)
    seat = _seat(obs)
    game = _WEED_STATE[seat]
    if step == 0 or step < game.get("last_step", -1):
        game = {"last_step": step, "active": {}}
        _WEED_STATE[seat] = game
    game["last_step"] = step
    farm = _farm(obs, seat)
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - transaction["start"]
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(actions, step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    raise ValueError(name)


def _market_price(item, inventory):
    base, equilibrium, scale, below_func, below_target, above_func, above_target = (
        _MARKET_PARAMS[item]
    )
    if inventory < equilibrium:
        amplitude = below_target * base / _shape(below_func, scale)
        price = base + amplitude * _shape(below_func, equilibrium - inventory)
    else:
        amplitude = above_target * base / _shape(above_func, scale)
        price = base - amplitude * _shape(above_func, inventory - equilibrium)
    return max(_PRICE_FLOOR, int(round(price)))


def _is_sell(order):
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in _MARKET_PARAMS
    )


def _impact_score(obs, order):
    if not _is_sell(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_quote = float(
        _get(prices, item, _market_price(item, current_inventory)) or 0
    )
    later_quote = float(_market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def _demand_per_day(obs, configuration, item):
    town = _get(obs, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(
        1, int(_get(configuration, "townShopSellInterval", 4) or 4)
    )
    demand = 0.0
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (
                2 if len(products) == 1 else 1
            )
    regime = _regime(configuration)
    if item != "FERTILIZER":
        center_default = 24 if regime == "rebalance" else 12
        center_interval = max(
            1,
            int(
                _get(configuration, "townCenterSellInterval", center_default)
                or center_default
            ),
        )
        day = int(_get(obs, "day", int(_get(obs, "step", 0) or 0) // 24) or 0)
        multiplier = (
            1
            if regime == "rebalance"
            else (4 if day >= 20 else 2 if day >= 10 else 1)
        )
        demand += (turns_per_day / center_interval) * multiplier
    return demand


def _order_score(obs, configuration, order):
    score = _impact_score(obs, order)
    if _regime(configuration) != "rebalance" or score <= 0 or not _is_sell(order):
        return score
    item = str(order[1])
    quantity = max(0, int(order[2]))
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    demand = max(0.25, _demand_per_day(obs, configuration, item))
    excess = max(0.0, current_inventory + quantity - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return score * (1.0 + _DEMAND_ALPHA * urgency)


def _rank_sell_slots(obs, action, configuration):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_order_score(obs, configuration, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def agent(obs, configuration=None):
    try:
        actions = (
            _REBALANCE_ACTIONS
            if _regime(configuration) == "rebalance"
            else _LEGACY_ACTIONS
        )
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(actions) - 1)
        action = _weed_repair_action(
            obs, _copy_action(actions[step]), actions, step
        )
        return _align_hands(_rank_sell_slots(obs, action, configuration), obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs, configuration=None):
    return agent(obs, configuration)
# Normalize route payload names for the shared V031 overlay.
_ACTIONS = _LEGACY_ACTIONS

"""V031 runtime overlay for a frozen complete route.

The builder concatenates this module after one decoded route source.  The
route source contributes only ``_ACTIONS`` and the official market constants;
this overlay owns the runtime market controller so adaptive/V27 market logic
is not applied twice.
"""

import copy as _v031_copy_module
from collections import Counter as _V031Counter


V031_PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
V031_CUTOFF = 648
V031_MAX_ORDERS = 10
V031_MAX_BATCH = 30
V031_ROUTE_LENGTH = 719
V031_CONTROLLER = "raw"
V031_ROUTE_NAME = "unknown"

_V031_STATE = {
    0: {"last_step": -1, "pending": None, "stats": {}},
    1: {"last_step": -1, "pending": None, "stats": {}},
}
V031_STATS = {}


def _v031_get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _v031_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _v031_copy_action(action):
    action = _v031_copy_module.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(x or ["PASS"]) for x in (action.get("hands") or [])],
        "market": [list(x) for x in (action.get("market") or []) if isinstance(x, list)],
    }


def _v031_seat(obs):
    return 1 if _v031_int(_v031_get(obs, "player", 0)) == 1 else 0


def _v031_farm(obs):
    farms = list(_v031_get(obs, "farms", []) or [])
    seat = _v031_seat(obs)
    return farms[seat] if seat < len(farms) else {}


def _v031_align_hands(action, obs):
    action = _v031_copy_action(action)
    expected = len(_v031_get(_v031_farm(obs), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(x or ["PASS"]) for x in hands[:expected]]
    return action


def _v031_step(obs):
    return min(max(0, _v031_int(_v031_get(obs, "step", 0))), len(_ACTIONS) - 1)


def _v031_stat(name, amount=1):
    V031_STATS[name] = V031_STATS.get(name, 0) + amount


def _v031_reset_state(obs, step):
    state = _V031_STATE[_v031_seat(obs)]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "pending": None, "stats": {}})
    state["last_step"] = step
    return state


def _v031_tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_v031_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, KeyError, TypeError, ValueError):
        return "LOCKED"


def _v031_actor_trace(step, actor):
    trace = _ACTIONS[min(max(int(step), 0), len(_ACTIONS) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _v031_weed_action(obs, action, step):
    """Common actor-local DIG/retry/catch-up layer for all three routes."""
    action = _v031_align_hands(action, obs)
    state = _V031_STATE[_v031_seat(obs)]
    active = state.setdefault("weed", {})
    farm = _v031_farm(obs)
    positions = [_v031_get(farm, "farmer"), *list(_v031_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]

    for actor, tx in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - int(tx.get("start", step))
        if age == 1:
            unit_actions[index] = list(tx.get("intended") or ["PASS"])
        elif 2 <= age <= 9:
            unit_actions[index] = _v031_actor_trace(step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if str(intended[0]).upper() not in ("PLANT", "BUILD_PASTURE"):
            continue
        tile = _v031_tile_at(farm, position)
        if isinstance(tile, dict) and str(tile.get("kind", "")).upper() == "WEED":
            active[actor] = {"start": step, "intended": list(intended)}
            unit_actions[index] = ["DIG"]
            _v031_stat("weed_repairs")

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _v031_align_hands(action, obs)


def _v031_route_action_only(obs):
    """Pure route action used by tests/benchmark for field-action diffs."""
    action = _v031_copy_action(_ACTIONS[_v031_step(obs)])
    return _v031_align_hands(action, obs)


def _v031_is_sell(order, item=None):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return False
    if str(order[0]).upper() != "SELL":
        return False
    if item is None:
        return True
    return str(order[1]).upper() == str(item).upper()


def _v031_item_price(obs, item):
    market = _v031_get(obs, "market", {}) or {}
    prices = _v031_get(market, "prices", {}) or {}
    return float(_v031_get(prices, item, 0) or 0)


def _v031_market_inventory(obs, item):
    market = _v031_get(obs, "market", {}) or {}
    inventory = _v031_get(market, "inventory", {}) or {}
    return _v031_int(_v031_get(inventory, item, 10000), 10000)


def _v031_impact_score(obs, order):
    if not _v031_is_sell(order):
        return float("-inf")
    item = str(order[1]).upper()
    quantity = max(0, _v031_int(order[2]))
    current = _v031_item_price(obs, item)
    try:
        later = float(_market_price(item, _v031_market_inventory(obs, item) + quantity))
    except Exception:
        later = current
    return float(quantity) * max(0.0, current - later)


def _v031_reorder_existing(obs, action):
    action = _v031_copy_action(action)
    market = list(action.get("market") or [])
    sell_rows = [
        (_v031_impact_score(obs, order), -index, list(order))
        for index, order in enumerate(market)
        if _v031_is_sell(order)
    ]
    if len(sell_rows) < 2:
        return action
    sell_rows.sort(reverse=True)
    ranked = iter(row[2] for row in sell_rows)
    action["market"] = [next(ranked) if _v031_is_sell(order) else order for order in market]
    _v031_stat("reorder_calls")
    return action


def _v031_visible_inventory(obs, item):
    private = _v031_get(obs, "private", {}) or {}
    total = _v031_int(_v031_get(_v031_get(private, "shed", {}) or {}, item, 0))
    for inventory in list(_v031_get(private, "inventories", []) or []):
        total += _v031_int(_v031_get(inventory or {}, item, 0))
    return max(0, total)


def _v031_current_sell_quantity(action, item):
    return sum(
        max(0, _v031_int(order[2]))
        for order in action.get("market", []) or []
        if _v031_is_sell(order, item)
    )


def _v031_future_sells(step, horizon):
    target = int(step) + int(horizon)
    if target < 0 or target >= len(_ACTIONS):
        return {}
    result = {}
    for order in (_ACTIONS[target].get("market") or []):
        if _v031_is_sell(order) and str(order[1]).upper() in V031_PREMIUM:
            item = str(order[1]).upper()
            result[item] = result.get(item, 0) + max(0, _v031_int(order[2]))
    return result


def _v031_public_signature(farm):
    counts = _V031Counter()
    for row in _v031_get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            for field in ("crop", "animal", "kind"):
                value = str(tile.get(field, "")).upper()
                if value:
                    counts[value] += 1
                    break
    return (
        len(_v031_get(farm, "hands", []) or []),
        len(_v031_get(farm, "unlocked_quadrants", []) or []),
        tuple(sorted(counts.items())),
    )


def _v031_clone_distance(obs):
    farms = list(_v031_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10**9
    left, right = _v031_public_signature(farms[0]), _v031_public_signature(farms[1])
    left_counts, right_counts = dict(left[2]), dict(right[2])
    keys = set(left_counts) | set(right_counts)
    return (
        abs(left[0] - right[0])
        + 3 * abs(left[1] - right[1])
        + sum(abs(left_counts.get(key, 0) - right_counts.get(key, 0)) for key in keys)
    )


def _v031_append_or_merge(action, item, quantity):
    item = str(item).upper()
    quantity = max(0, _v031_int(quantity))
    if quantity <= 0:
        return False
    for order in action.get("market", []) or []:
        if _v031_is_sell(order, item):
            order[2] = max(0, _v031_int(order[2])) + quantity
            return True
    if len(action.get("market", []) or []) >= V031_MAX_ORDERS:
        return False
    action.setdefault("market", []).append(["SELL", item, quantity])
    return True


def _v031_reduce_sell(action, item, quantity):
    remaining = max(0, _v031_int(quantity))
    if remaining <= 0:
        return True
    for index, order in enumerate(list(action.get("market", []) or [])):
        if not _v031_is_sell(order, item):
            continue
        current = max(0, _v031_int(order[2]))
        reduction = min(current, remaining)
        current -= reduction
        remaining -= reduction
        if current <= 0:
            action["market"].pop(index)
        else:
            action["market"][index][2] = current
        if remaining <= 0:
            return True
    return False


def _v031_repay(obs, action, state, step):
    pending = state.get("pending")
    if not pending or int(pending.get("due_step", -1)) != int(step):
        return action, True
    trial = _v031_copy_action(action)
    if not _v031_reduce_sell(trial, pending["item"], pending["quantity"]):
        _v031_stat("repayment_failures")
        state["pending"] = None
        return action, False
    state["pending"] = None
    _v031_stat("repayment_successes")
    return trial, True


def _v031_preempt(obs, action, state, step):
    if V031_CONTROLLER not in ("preempt_h3_h2_h1", "combined"):
        return action, False
    if step < 120 or step >= V031_CUTOFF or state.get("pending"):
        return action, False
    distance = _v031_clone_distance(obs)
    thresholds = ((3, 3), (2, 4), (1, 6))
    prices = _v031_get(_v031_get(obs, "market", {}) or {}, "prices", {}) or {}

    for horizon, max_distance in thresholds:
        if distance > max_distance:
            continue
        future = _v031_future_sells(step, horizon)
        if not future:
            continue
        candidates = []
        for item, future_quantity in future.items():
            current_price = float(_v031_get(prices, item, 0) or 0)
            if current_price <= 1 or future_quantity <= 0:
                continue
            available = _v031_visible_inventory(obs, item)
            available -= _v031_current_sell_quantity(action, item)
            quantity = min(
                max(1, (future_quantity + 1) // 2),
                future_quantity,
                max(0, available),
                V031_MAX_BATCH,
            )
            if quantity <= 0:
                continue
            try:
                after_price = float(
                    _market_price(item, _v031_market_inventory(obs, item) + quantity)
                )
            except Exception:
                after_price = current_price
            candidates.append((current_price - after_price, current_price, item, quantity))
        if not candidates:
            continue
        _, _, item, quantity = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        trial = _v031_copy_action(action)
        if not _v031_append_or_merge(trial, item, quantity):
            _v031_stat("preempt_blocked_market_full")
            continue
        state["pending"] = {
            "item": item,
            "quantity": quantity,
            "due_step": step + horizon,
            "horizon": horizon,
        }
        _v031_stat("preempt_calls")
        _v031_stat(f"preempt_h{horizon}")
        _v031_stat("preempt_units", quantity)
        return trial, True
    return action, False


def _v031_agent(obs, config=None):
    step = _v031_step(obs)
    state = _v031_reset_state(obs, step)
    action = _v031_copy_action(_ACTIONS[step])
    action = _v031_weed_action(obs, action, step)
    action, _ = _v031_repay(obs, action, state, step)

    if V031_CONTROLLER in ("order_only", "combined"):
        action = _v031_reorder_existing(obs, action)

    if V031_CONTROLLER in ("preempt_h3_h2_h1", "combined"):
        action, _ = _v031_preempt(obs, action, state, step)

    if len(action.get("market", []) or []) > V031_MAX_ORDERS:
        _v031_stat("market_overflow_guard")
        action["market"] = action["market"][:V031_MAX_ORDERS]
    return _v031_align_hands(action, obs)


def agent(obs, config=None):
    try:
        return _v031_agent(obs, config)
    except Exception:
        _v031_stat("runtime_errors")
        farm = _v031_farm(obs)
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_v031_get(farm, "hands", []) or [])],
            "market": [],
        }
# V031 generated candidate: route=v27, controller=order_only
V031_ROUTE_NAME = 'v27'
V031_CONTROLLER = 'order_only'

# RL-010: MILK timing overlay.  The route and WEED layer are prepared first,
# then the RL quantity transfer is applied, and only then V27's existing
# price-impact order ranking is run.
RL010_PAYLOAD = {'version': 'rl010', 'feature_dim': 41, 'min_support': 24, 'min_expected_delta': 5.0, 'lcb_z': 1.5, 'bad_ucb': 0.1, 'allowed_actions': ['ADVANCE_25', 'ADVANCE_50', 'DELAY_25', 'DELAY_50'], 'include_opponent_features': True, 'models': {}, 'variant': 'rl010c_bidirectional_opp'}
_RL010_OPPORTUNITIES = rl010_route_opportunities(_ACTIONS, RL010_EVENT_STEPS)
_RL010_RUNTIME = RL010Runtime(payload=RL010_PAYLOAD, opportunities=_RL010_OPPORTUNITIES)

def _rl010_route_action(obs, config=None):
    step = _v031_step(obs)
    action = _v031_copy_action(_ACTIONS[step])
    action = _v031_weed_action(obs, action, step)
    return _v031_align_hands(action, obs)

def agent(obs, config=None):
    try:
        route_action = _rl010_route_action(obs, config)
        adjusted_action = _RL010_RUNTIME.act(obs, route_action, config=config)
        final_action = _v031_reorder_existing(obs, adjusted_action)
        final_action = _v031_align_hands(final_action, obs)
        _RL010_RUNTIME.record_final_action(final_action)
        return final_action
    except Exception:
        _RL010_RUNTIME.errors += 1
        return _v031_agent(obs, config)
