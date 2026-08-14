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
RL010_DELAY_MAX_GAP = 72
RL010_SHED_CAPACITY = 100
RL010_SHED_RESERVE = 10
# Empty means “all route MILK events”.  Different seeds can have different
# carried/shed inventory at a given route step, so the live safety checks—not
# a fixed step such as 455—must decide whether an event is executable.
RL010_EVENT_STEPS = ()
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
    """Find adjacent same-item route SELL events before the cutoff.

    The result is a set of potential events.  Actual inventory legality is
    checked from the live observation immediately before changing an order.
    """
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
            if current_step >= RL010_CUTOFF:
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

    def _legal_units(self, action, obs, opportunity, action_name, config=None):
        current = rl010_sell_quantity(action)
        if current <= 0:
            return 0
        if action_name.startswith("ADVANCE"):
            ratio = RL010_RATIOS[action_name]
            desired = rl010_round_half_up(opportunity["future_quantity"] * ratio)
            state = rl010_market_shed_state(obs, action, config=config)
            return max(0, min(desired, opportunity["future_quantity"], state["sellable_extra"]))
        if action_name.startswith("DELAY"):
            ratio = RL010_RATIOS[action_name]
            desired = rl010_round_half_up(opportunity["current_quantity"] * ratio)
            # Delay must be repayable from the next event's own shed state.
            # The current order may be reduced only when the held units fit
            # before the future SELL; the final transaction check below still
            # validates the live future observation.
            return max(0, min(desired, current - 1))
        return 0

    def _safe(self, obs, action, opportunity, action_name, units, config=None):
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
            state = rl010_market_shed_state(obs, action, config=config)
            if rl010_sell_quantity(action) + units > state["shed_after_actions"]:
                return False, "inventory_short"
        if action_name.startswith("DELAY"):
            allowed, reason = rl010_delay_guard(
                obs, action, opportunity, units, config=config
            )
            if not allowed:
                return False, reason
        if int(opportunity["future_step"]) <= step:
            return False, "future_not_after_current"
        # A transfer whose repayment lands in the terminal protection window
        # can change liquidation rather than just the intended MILK event.
        # Keep those events as CONTROL; only create debts that are repayable
        # before the cutoff.
        if int(opportunity["future_step"]) >= RL010_CUTOFF:
            return False, "future_terminal_window"
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
            units = self._legal_units(
                action, obs, opportunity, action_name, config=config
            )
            safe, safe_reason = self._safe(
                obs, action, opportunity, action_name, units, config=config
            )
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
RL010_PAYLOAD = {'version': 'rl010', 'feature_dim': 41, 'min_support': 24, 'min_expected_delta': 5.0, 'lcb_z': 1.5, 'bad_ucb': 0.1, 'allowed_actions': ['ADVANCE_25', 'ADVANCE_50', 'DELAY_25', 'DELAY_50'], 'include_opponent_features': False, 'models': {'MILK|216|264|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.0, 0.0, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.478839213598466, -125.7608337434667, -9.38463861023191, 14.345443626427645, -106.50995146559445, 32.16804337043512, -17.11557531209942, 63.847775599205036, 7.726456916077134e-29, 1.5949766690178356e-28, 0.0, 105.00351290455355, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7808094976605164, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -65.05000000003935, 'uncertainty': 398.5532507391957}, 'bad': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.0, 0.0, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02327479307377139, 0.09778565856156266, 0.006879451285141208, -0.010679394766183254, 0.08200624330287275, -0.02380222734624441, 0.010792730578141422, -0.038831303612101326, -5.831562431135282e-32, -1.3734914860609732e-31, 0.0, -0.09110025714555604, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0019393575340298464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.05000000000003051, 'uncertainty': 0.2294366959722735}, 'train_mean_delta': -65.05, 'train_min_delta': -2840.0, 'train_positive_rate': 0.4166666666666667, 'train_bad_rate': 0.05}, 'MILK|216|264|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.0, 0.0, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 48.65523926424503, -212.63522226066289, -20.859118463400613, 18.854995595626438, -185.95431769538826, 52.77008881094876, -27.72923311540502, 101.88398763159556, 1.1241620132386979e-28, 2.696726267995709e-28, 0.0, 144.78498627692716, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8777809795231098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -116.08333333339921, 'uncertainty': 338.5017652618712}, 'bad': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.0, 0.0, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02327479307377139, 0.09778565856156266, 0.006879451285141208, -0.010679394766183254, 0.08200624330287275, -0.02380222734624441, 0.010792730578141422, -0.038831303612101326, -5.831562431135282e-32, -1.3734914860609732e-31, 0.0, -0.09110025714555604, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0019393575340298464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.05000000000003051, 'uncertainty': 0.2294366959722735}, 'train_mean_delta': -116.08333333333333, 'train_min_delta': -1874.0, 'train_positive_rate': 0.4166666666666667, 'train_bad_rate': 0.05}, 'MILK|264|311|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0, 0.0, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -3.0425174312583674e-29, -3.0425174312583674e-29, 0.0, 0.0, 0.0, 6.715168505347116e-29, 13.706731162871774, -4.236710167184994, -4.661862154249715, 11.082824212670527, -9.034480562906918, -1.4569691139241294, -4.494001326945977, 13.051996130317049, 3.9927167726299775e-30, 0.0, 0.0, 0.34401678252101414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -14.800000000000715, 'uncertainty': 16.714838203782044}, 'bad': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0, 0.0, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 1.2965825807140323e-31, 1.2965825807140323e-31, 0.0, 0.0, 0.0, -3.575188943013193e-31, 0.011571139594830722, 0.011019439035798943, 0.012461566012247672, -0.010997891890965615, 0.01119514898084614, -0.01388468318904374, 0.026595619044519483, -0.003966164965029789, -1.5540440341438585e-32, 0.0, 0.0, -0.030568181040327935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.01666666666666853, 'uncertainty': 0.15262242357989056}, 'train_mean_delta': -14.8, 'train_min_delta': -58.0, 'train_positive_rate': 0.38333333333333336, 'train_bad_rate': 0.016666666666666666}, 'MILK|264|311|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0, 0.0, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 3.7872681877801896e-29, 3.7872681877801896e-29, 0.0, 0.0, 0.0, 2.147191558169935e-29, 25.673938686545355, -11.30963150777953, 5.686446372989382, 12.315955059457734, -27.209995076694774, -5.567230653814444, -3.617537464325712, 17.909502029453343, 3.5537183144871613e-31, 0.0, 0.0, 6.121990972589493, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -34.43333333333523, 'uncertainty': 48.038204861373764}, 'bad': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0, 0.0, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 1.2417543567098949e-31, 1.2417543567098949e-31, 0.0, 0.0, 0.0, -2.6137144316744444e-31, 0.009246767594404001, 0.0043807028713992915, 0.012397681033897757, -0.012209869341555919, 0.009643981253292645, -0.012731893699556988, 0.017826051326324484, -0.0011653655248085588, -2.1242381008295887e-32, 0.0, 0.0, 0.002771409545149136, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.008333333333334048, 'uncertainty': 0.11241463204475369}, 'train_mean_delta': -34.43333333333333, 'train_min_delta': -298.0, 'train_positive_rate': 0.4166666666666667, 'train_bad_rate': 0.008333333333333333}, 'MILK|311|336|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.0, 0.0, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.2165392820818512e-30, 0.0, 2.627932032178959e-29, 0.0, 0.0, 1.2165392820818512e-30, 1.0953530050692395, 0.24999707340424443, -1.6701277094252318, 1.492423825944096, 1.428895870216979, -1.8704229391595495, -1.3538240810065125, -0.8109320072564152, -1.9833852997426596e-30, 9.188163693575195e-31, 0.0, -0.36786551321166877, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -2.8666666666667013, 'uncertainty': 12.834037195506143}, 'bad': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.0, 0.0, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -5.109806087110763e-31, 0.0, 9.642317001513399e-31, 0.0, 0.0, 5.109806087110763e-31, -0.01697136730279186, 0.031127321344357903, 0.0779286003974248, 0.0438090613308312, 0.021368347392226636, 0.05049684720070917, 0.01843359885725492, -0.014416210313044667, 1.4579221302820643e-31, 2.673919334137973e-31, 0.0, -0.05403350541531382, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0416666666666621, 'uncertainty': 0.21002637520692097}, 'train_mean_delta': -2.8666666666666667, 'train_min_delta': -25.0, 'train_positive_rate': 0.5166666666666667, 'train_bad_rate': 0.041666666666666664}, 'MILK|311|336|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.0, 0.0, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 3.31368492775644e-30, 0.0, 3.831527712655822e-29, 0.0, 0.0, -3.31368492775644e-30, 1.3139813501974227, -1.2907494915499402, -1.716729511256674, 1.8665451669648592, 1.9282746993199698, -2.8641538237885094, -4.366345385109263, -1.6677468950917058, -2.804278343769907e-30, 4.391835618669937e-31, 0.0, 0.24737005074685892, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -6.116666666666467, 'uncertainty': 27.5606569527007}, 'bad': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.0, 0.0, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -5.109806087110763e-31, 0.0, 9.642317001513399e-31, 0.0, 0.0, 5.109806087110763e-31, -0.01697136730279186, 0.031127321344357903, 0.0779286003974248, 0.0438090613308312, 0.021368347392226636, 0.05049684720070917, 0.01843359885725492, -0.014416210313044667, 1.4579221302820643e-31, 2.673919334137973e-31, 0.0, -0.05403350541531382, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0416666666666621, 'uncertainty': 0.21002637520692097}, 'train_mean_delta': -6.116666666666666, 'train_min_delta': -59.0, 'train_positive_rate': 0.55, 'train_bad_rate': 0.041666666666666664}, 'MILK|336|360|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.0, 0.0, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 6.927731259140099e-29, 6.927731259140099e-29, 0.0, 0.0, 0.0, 0.0, -9.492538630758075, -1.4520600792930995, 26.327068884764124, -23.121311355434344, -12.155089566216981, -30.045542387512704, -6.222749376816766, -8.447420894584745, -4.987150892954296e-29, 2.7710925036560392e-28, -3.9897207143634368e-28, -7.895418387190977, 0.0, 0.0, 1.4249501369031185e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.087962749401619e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -49.366666666666376, 'uncertainty': 34.87559705489184}, 'bad': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.0, 0.0, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -9.953600396307731e-32, -9.953600396307731e-32, 0.0, 0.0, 0.0, 0.0, 0.00862818969398939, 0.01977409649768267, -0.031200501338761562, 0.012608412508406735, 0.03734871926194874, 0.026874227944916024, 0.0006729101882574557, -0.009192111473728252, 7.715590873714904e-32, -3.981440158523093e-31, 6.172472698971923e-31, -0.052667467688715136, 0.0, 0.0, -1.5780141875802509e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.644819315148341e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.04166666666666316, 'uncertainty': 0.23236805806285848}, 'train_mean_delta': -49.36666666666667, 'train_min_delta': -149.0, 'train_positive_rate': 0.1, 'train_bad_rate': 0.041666666666666664}, 'MILK|336|360|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.0, 0.0, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 1.1436035622706272e-28, 1.1436035622706272e-28, 0.0, 0.0, 0.0, 0.0, -10.53924564049216, -6.564282184808526, 40.27678524966426, -29.93986413340789, -29.638006381495813, -46.96347085812789, -4.441018670340218, -11.04443637593127, -8.131511003202965e-29, 4.574414249082508e-28, -6.505208802562372e-28, -14.4595821556225, 0.0, 0.0, 2.186110399521999e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.703421953495286e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -81.98333333333213, 'uncertainty': 47.41780162984342}, 'bad': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.0, 0.0, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.3041129130950811e-31, -1.3041129130950811e-31, 0.0, 0.0, 0.0, 0.0, 0.008739364509349486, 0.029505955224501182, -0.04483357882851968, 0.004959642129104799, 0.04603441588690123, 0.03318608202983486, -0.008943745586105138, -0.01400846059035235, 9.926738834630816e-32, -5.2164516523803245e-31, 7.941391067704653e-31, -0.06929744477011192, 0.0, 0.0, -1.5669415847415259e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5798109687224575e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.04999999999999475, 'uncertainty': 0.2471364964850396}, 'train_mean_delta': -81.98333333333333, 'train_min_delta': -223.0, 'train_positive_rate': 0.06666666666666667, 'train_bad_rate': 0.05}, 'MILK|360|377|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.0, 0.0, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.538417492634394e-29, 9.143777190392544, -9.889763492182322, 15.277762994999367, -7.682124143759898, -15.899254748591874, -28.17130194152617, 0.07579550288947504, 28.604605385672492, -8.468274140397596e-30, 0.19208659050970714, -0.19208659050971022, 3.7401930482362076, 0.0, 0.0, -1.3937124045287585e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2242307100262028e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -8.866666666665216, 'uncertainty': 27.537705653814292}, 'bad': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.0, 0.0, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9608737903521128e-31, -0.0020598500299432564, -0.003890980118614411, -0.05133189273259069, 0.006417049849411548, -0.08057702771447865, 0.05784605198670394, -0.03265292784766648, -0.1616879931911651, 7.087872514844666e-32, -0.009135236268098878, 0.009135236268098715, -0.052823983062048135, 0.0, 0.0, -4.7690077051288485e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1308716813398348e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.06666666666665969, 'uncertainty': 0.2747695950826605}, 'train_mean_delta': -8.866666666666667, 'train_min_delta': -105.0, 'train_positive_rate': 0.5666666666666667, 'train_bad_rate': 0.06666666666666667}, 'MILK|360|377|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.0, 0.0, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.296249445821642e-29, 19.86399310297973, -24.92918489952507, 39.02393001856645, -20.07135440031637, -36.90197992070119, -74.66774165234149, 0.43357083192695184, 73.6787436687574, -2.2358566072166858e-29, 0.7283872293758897, -0.7283872293759249, 10.115235353052945, 0.0, 0.0, -3.4815569390672976e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.752574122788134e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -27.049999999996142, 'uncertainty': 74.86753889548845}, 'bad': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.0, 0.0, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6778157894225517e-31, 0.01895322755941669, -0.023314857114509025, -0.03772485918863144, 0.003989704823304186, -0.0699027136169101, 0.11097494685765724, -0.030977930008263696, -0.16925215866933171, 6.190148792784821e-32, -0.008148251804152604, 0.008148251804152504, -0.04288316349719827, 0.0, 0.0, -5.951472574093483e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1166227019146258e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.07499999999999644, 'uncertainty': 0.29198822365654664}, 'train_mean_delta': -27.05, 'train_min_delta': -295.0, 'train_positive_rate': 0.5666666666666667, 'train_bad_rate': 0.075}, 'MILK|377|381|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -3.5640729378867075e-29, 0.0, -1.7820364689433538e-29, 0.0, 0.0, -3.2640075476284767e-31, -4.610683713880229, -5.950797645887253, -0.9115767965522037, -3.715365860122435, 3.2516277598404697, -4.510247175354004, 1.4151425096403898, 3.4383436636542517, -8.194256645328827e-30, 0.9463850555475413, -0.9463850555475483, 0.8354306970666828, 0.0, 0.0, 0.0, 0.0, 8.094438518874247e-30, 0.0, 0.0, 0.0, 0.0, -2.0013020921503438e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -9.516666666666785, 'uncertainty': 22.841714878914367}, 'bad': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 5.879939974313686e-31, 0.0, 2.939969987156843e-31, 0.0, 0.0, -1.1324229025883201e-32, 0.018921856410951256, -0.06394334661386841, -0.02477408252689792, 0.0051744322052237526, -0.1349426174297984, -0.024414333213770997, -0.07675452305777083, 0.0067101934805389825, 1.1734427608930865e-31, -0.0043484134897350726, 0.004348413489735134, -0.04262548251021992, 0.0, 0.0, 0.0, 0.0, -2.4360700759097626e-31, 0.0, 0.0, 0.0, 0.0, -1.9845616241114768e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.05833333333332037, 'uncertainty': 0.25392214476875835}, 'train_mean_delta': -9.516666666666667, 'train_min_delta': -55.0, 'train_positive_rate': 0.38333333333333336, 'train_bad_rate': 0.058333333333333334}, 'MILK|377|381|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -6.827202855554056e-29, 0.0, -3.413601427777028e-29, 0.0, 0.0, -1.687728761220423e-31, -8.66526365532071, -10.760420326768628, -1.0480480574428823, -7.97280976163181, 8.029125118517992, -8.559903200945357, 3.634874212726667, 5.252712816107547, -1.571784632643743e-29, 1.995095774899785, -1.9950957748997837, 1.9775182060574648, 0.0, 0.0, 0.0, 0.0, 1.7317989595886838e-29, 0.0, 0.0, 0.0, 0.0, -2.1251411832232674e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -19.14999999999998, 'uncertainty': 47.832288390562624}, 'bad': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 6.563138106386223e-31, 0.0, 3.2815690531931117e-31, 0.0, 0.0, -1.0624675978419942e-32, 0.029318598803784793, -0.06791356813229096, -0.025111966218897067, 0.009748172217172448, -0.15953245025003637, -0.021918978157493642, -0.08406462934437159, 0.009276403987258834, 1.3758222291419468e-31, -0.006256714352932099, 0.006256714352932094, -0.05749222448769212, 0.0, 0.0, 0.0, 0.0, -2.7157197928454906e-31, 0.0, 0.0, 0.0, 0.0, -2.6464296595194105e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.06666666666665161, 'uncertainty': 0.2596806332397566}, 'train_mean_delta': -19.15, 'train_min_delta': -151.0, 'train_positive_rate': 0.38333333333333336, 'train_bad_rate': 0.06666666666666667}, 'MILK|381|406|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.0210166756105085e-29, 0.0, 0.0, 0.0, 0.0, 9.898128649145347e-30, -0.13013071968400422, -20.323763030458046, 10.282529134577894, -1.6997313069961562, -16.201069222354093, -9.141944239687767, 0.30325310933026495, 10.224252423408767, 2.3499867853055902e-30, 0.17859359421401788, -0.17859359421393617, -7.029309286809506, 0.0, 0.0, 0.0, 0.0, -1.6909737945214936e-29, 0.0, 0.0, 0.0, 0.0, 9.39994714122236e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -35.48333333333537, 'uncertainty': 29.903426323547478}, 'bad': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 1.137227985518776e-31, 0.0, 0.0, 0.0, 0.0, -2.805834784790089e-31, 0.015075911192346046, -0.10718638266409068, -0.0142193495784702, 0.013999668148699273, -0.022507889937614798, 0.07304448654191945, 0.0074223325724139914, 0.07779852848262661, 9.104621271922273e-33, 0.0005716857561832169, -0.000571685756183583, -0.03406089750530183, 0.0, 0.0, 0.0, 0.0, 2.925536323170326e-31, 0.0, 0.0, 0.0, 0.0, 3.6418485087689087e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.05833333333332274, 'uncertainty': 0.22376406260376305}, 'train_mean_delta': -35.483333333333334, 'train_min_delta': -91.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.058333333333333334}, 'MILK|381|406|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -3.554559766824123e-29, 0.0, 0.0, 0.0, 0.0, 4.169043907123252e-29, -0.3778376174926344, -32.84095829419952, 25.113202137186043, -3.248896704821871, -25.082371711393776, -22.290037321771152, 0.37546347391562956, 15.239029820877981, 4.247733717401925e-30, 0.01789960155227201, -0.01789960155209995, -16.150404563152133, 0.0, 0.0, 0.0, 0.0, -4.350919758939983e-29, 0.0, 0.0, 0.0, 0.0, 1.69909348696077e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -76.05000000000342, 'uncertainty': 36.46111153636336}, 'bad': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.0, 0.0, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 1.4252537907881752e-31, 0.0, 0.0, 0.0, 0.0, -3.2096692470288794e-31, 0.017674208408248522, -0.12190069098623782, -0.02392543757064235, 0.012802169457977925, -0.038343517371284795, 0.0789158861967345, 0.006599001560118306, 0.08779054432823048, 1.006112644281002e-32, 0.0012527863397101635, -0.001252786339710541, -0.0282285067695364, 0.0, 0.0, 0.0, 0.0, 3.0401126888713765e-31, 0.0, 0.0, 0.0, 0.0, 4.0244505771240077e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.06666666666665488, 'uncertainty': 0.22436754824059688}, 'train_mean_delta': -76.05, 'train_min_delta': -178.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.06666666666666667}, 'MILK|406|408|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.0, 0.0, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 1.0135631405980236e-29, 9.109746803060525e-29, 0.0, 0.0, -2.208502202316735e-30, -26.010399507046362, 2.1692238925594265, -16.346796951058252, -18.37561995242106, 2.3435566146276914, -7.247915360611907, -19.70477060998205, 0.7451975333460334, -1.0135631405980041e-29, 1.8002130494353397, -1.800213049435425, 6.508872896998175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4992724885027826e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -33.116666666667406, 'uncertainty': 31.852303742852037}, 'bad': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.0, 0.0, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 5.408128345597397e-32, -3.409153811953574e-31, 0.0, 0.0, 6.59896617808053e-32, 0.005927423691415831, -0.01988510883455907, -0.01186714757872799, 0.02383711338129534, -0.03329650017039157, 0.049241519873220396, 0.03464467483337204, -0.16937335288309252, -5.408128345597416e-32, -0.0008133622649298238, 0.0008133622649298049, 0.025872997754500186, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0035047031874228e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.04166666666666717, 'uncertainty': 0.20074821010752558}, 'train_mean_delta': -33.11666666666667, 'train_min_delta': -122.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.041666666666666664}, 'MILK|406|408|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.0, 0.0, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 3.3537469242685203e-29, 3.124121192782479e-28, 0.0, 0.0, -6.226189261494918e-30, -75.22142396310468, -5.780041539853624, -56.960223434041204, -42.843832891913195, 2.1029982587174914, -3.5635329690160815, -59.62103371698224, 1.4227083524201873, -3.353746924268473e-29, 4.178687534991038, -4.178687534991057, 16.950477780528303, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.7123245406683015e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -109.98333333333494, 'uncertainty': 101.67084459123316}, 'bad': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.0, 0.0, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 5.408128345597397e-32, -3.409153811953574e-31, 0.0, 0.0, 6.59896617808053e-32, 0.005927423691415831, -0.01988510883455907, -0.01186714757872799, 0.02383711338129534, -0.03329650017039157, 0.049241519873220396, 0.03464467483337204, -0.16937335288309252, -5.408128345597416e-32, -0.0008133622649298238, 0.0008133622649298049, 0.025872997754500186, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0035047031874228e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.04166666666666717, 'uncertainty': 0.20074821010752558}, 'train_mean_delta': -109.98333333333333, 'train_min_delta': -415.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.041666666666666664}, 'MILK|408|432|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.0, 0.0, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -6.2852127554221955e-30, -6.2852127554221955e-30, 0.0, 0.0, 0.0, 0.0, -9.86042703096327, 0.07480345178541847, -6.589949441586745, 12.228535039563955, 0.6968940066747906, 15.20744946885938, -3.7459690392149554, -17.836162968984787, -1.891284343481715e-30, -3.900548301196616, 3.900548301196788, -3.581827191238512, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0084169250210779e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -6.699999999999424, 'uncertainty': 43.10329364768694}, 'bad': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.0, 0.0, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -6.7, 'train_min_delta': -90.0, 'train_positive_rate': 0.18333333333333332, 'train_bad_rate': 0.0}, 'MILK|408|432|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.0, 0.0, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 6.72059717353914e-29, 6.72059717353914e-29, 0.0, 0.0, 0.0, 0.0, -3.875228037952903, -7.811690532222949, -1.9606680121315028, -5.4958874489811, -1.4498865988164904, -2.6775462271796533, -0.4350976412239556, -30.910475501404292, -6.011221103890391e-30, -8.350348453895837, 8.350348453896286, -7.236525040754475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.6628822566836804e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -12.133333333328673, 'uncertainty': 67.78260015981002}, 'bad': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.0, 0.0, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 3.439421998423694e-32, 3.439421998423694e-32, 0.0, 0.0, 0.0, 0.0, 0.0012617052880414669, 0.0026027395315476954, -0.00311247219395868, -0.03758649900755245, 0.005482988378666958, -0.007825306335414105, 0.020460418120547316, -0.026911884510355154, -3.628802561322319e-33, 0.006804928311685804, -0.00680492831168608, 0.014647858549510149, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.473004250218835e-34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.008333333333331038, 'uncertainty': 0.1021342755149516}, 'train_mean_delta': -12.133333333333333, 'train_min_delta': -174.0, 'train_positive_rate': 0.21666666666666667, 'train_bad_rate': 0.008333333333333333}, 'MILK|432|455|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.6694439317736844e-28, -1.9718464822906623, -2.038020205800927, 13.497351849769199, -15.397033438781666, 13.635826377994935, 5.023738887071173, 11.511162131287518, -12.339068195076024, -3.336804914717095e-29, 2.793927439517845, -2.79392743951793, 1.6209133405584153, 0.0, 0.0, 3.6747639035676213e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -15.183333333332316, 'uncertainty': 23.311445773664104}, 'bad': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -15.183333333333334, 'train_min_delta': -99.0, 'train_positive_rate': 0.16666666666666666, 'train_bad_rate': 0.0}, 'MILK|432|455|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.911928116635052e-28, -5.91205264612423, -6.822475537114321, 36.12541713902257, -39.664640418268164, 30.81041967135065, 12.424994912761834, 29.16796898992722, -29.119241898937034, -8.639910145793791e-29, 7.591198750564504, -7.591198750564544, 2.945240984981534, 0.0, 0.0, 9.461350113626889e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -40.38333333333023, 'uncertainty': 61.37339214417605}, 'bad': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.155307966658074e-32, 0.013012055744342346, 0.010511439702226653, 0.0025823138187151214, 0.013635939953760291, 0.010290431151788765, -0.004188000653613535, 0.014478811569576693, 0.005988529478914657, 5.1941349583228504e-33, 0.0072447094934804624, -0.007244709493480709, 0.01191253583235244, 0.0, 0.0, -2.774254138481625e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.008333333333331857, 'uncertainty': 0.11024346690245596}, 'train_mean_delta': -40.38333333333333, 'train_min_delta': -241.0, 'train_positive_rate': 0.18333333333333332, 'train_bad_rate': 0.008333333333333333}, 'MILK|455|456|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.0, 0.0, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.0777072632521517e-30, 0.0, -2.513072060269439e-31, 0.0, 0.0, -2.282488241003754e-31, -0.4386347248812261, 1.0241952199171585, -0.639244900930942, -1.2371652040825327, 1.5976147376543055, -0.9004961750119302, 0.5383653937216818, 0.9986935896198514, -1.5706700376683746e-32, -0.08889554442707853, 0.08889554442706112, 0.25766406292498173, 0.0, 0.0, 0.0, 0.0, -1.0777072632521531e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -1.1833333333334606, 'uncertainty': 5.887683030184567}, 'bad': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.0, 0.0, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -2.4053476505780617e-31, 0.0, 4.438011409496626e-32, 0.0, 0.0, -1.0731357639956435e-32, -0.007422903704731168, -0.023333450379584175, -0.01922363396515797, -0.06495843207235395, -0.024843489585280692, -0.02907724764393956, -0.08554950763553558, -0.03068128875059921, 2.773757130935421e-33, -0.009623694121680505, 0.009623694121679809, 0.008207353707697318, 0.0, 0.0, 0.0, 0.0, -2.4053476505780626e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.016666666666669654, 'uncertainty': 0.1254038231340243}, 'train_mean_delta': -1.1833333333333333, 'train_min_delta': -34.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.016666666666666666}, 'MILK|455|456|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.0, 0.0, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -5.866485258641314e-29, 0.0, -2.4390604511397877e-29, 0.0, 0.0, -4.330578726863822e-30, -11.634995129231376, 1.9860715834939444, -20.752307314079367, -13.708875414648872, -12.762551167773294, -23.040142912615256, -11.684992810161114, -11.447880244850884, -1.5244127819623649e-30, -0.0374995301747224, 0.037499530174479954, -5.603230605870045, 0.0, 0.0, 0.0, 0.0, -5.866485258641317e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -11.066666666666915, 'uncertainty': 25.88511571440445}, 'bad': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.0, 0.0, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 8.775073557199819e-32, 0.0, 2.764491270791926e-31, 0.0, 0.0, -1.5526856019967406e-33, 0.008055607598174908, -0.015857697098651674, -0.029905728059923538, 0.012606308911339504, -0.02409117060145899, -0.016066365757947253, -0.08266231586512435, -0.020624284542131183, 1.7278070442449578e-32, -0.01502160915314574, 0.01502160915314574, -0.01934152134865391, 0.0, 0.0, 0.0, 0.0, 8.775073557199813e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.041666666666668746, 'uncertainty': 0.21794528091535761}, 'train_mean_delta': -11.066666666666666, 'train_min_delta': -136.0, 'train_positive_rate': 0.1, 'train_bad_rate': 0.041666666666666664}, 'MILK|456|480|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 4.4630385397209126e-30, 4.4630385397209126e-30, 0.0, 0.0, 0.0, 0.0, -0.3584111405187896, 2.308398998744465, -3.460297844969549, 6.834253238861572, 6.448886396941303, 3.6461272292926132, -5.079099288773598, -3.6615712885979472, -7.402027307037731e-31, -0.33804242200961776, 0.33804242200965495, 1.8814634749351629, 0.0, 0.0, 1.0554847512300631e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -9.316666666667412, 'uncertainty': 14.80814516962755}, 'bad': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -9.316666666666666, 'train_min_delta': -111.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.0}, 'MILK|456|480|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 6.815696464670641e-30, 6.815696464670641e-30, 0.0, 0.0, 0.0, 0.0, -2.0073831130098294, 4.211514266204851, -4.039356573484814, 14.271971573720132, 10.047820126046805, 6.685065235195675, -9.150424047025822, -4.951732191370791, -1.2465077505975732e-30, -0.2450954784257068, 0.2450954784257704, 3.557372644676681, 0.0, 0.0, 1.489869515959235e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -17.350000000001405, 'uncertainty': 18.651253639366384}, 'bad': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.0, 0.0, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -6.497318984322598e-32, -6.497318984322598e-32, 0.0, 0.0, 0.0, 0.0, -0.017592477142379385, -0.04360885702909881, 0.028450261392293733, 0.023939493760528743, 0.008935951435021754, 0.038443106800336094, 0.010748764004261018, 0.02163350288922345, 6.025938223521271e-33, -0.00037517247115826855, 0.0003751724711580918, -7.514659026947753e-05, 0.0, 0.0, -9.42530728029881e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.025000000000014573, 'uncertainty': 0.1829503853833567}, 'train_mean_delta': -17.35, 'train_min_delta': -122.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.025}, 'MILK|480|502|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.0, 0.0, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.378341900395687e-28, 201.92205097314508, 73.2207551262431, -5.051802334215153, 45.55244595780422, 156.65188892953714, -88.3973991036862, 124.73789659320981, 147.12339623639662, 0.0, 7.46475620598358, -7.464756205983534, 137.53985967440948, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.6548341940494385e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 91.51666666664025, 'uncertainty': 321.56474704627664}, 'bad': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.0, 0.0, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2184508422985234e-31, 0.007673340263612343, 0.00623870335772527, -0.007872604067202143, 0.02960999816792167, 0.007895962694035209, -0.004516211228429019, -0.0066466033300843835, 0.02495501602728358, 0.0, -0.0063695397104006615, 0.006369539710400652, 0.0060880944099030495, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.755935198312217e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.008333333333331359, 'uncertainty': 0.10706619967902285}, 'train_mean_delta': 91.51666666666667, 'train_min_delta': -131.0, 'train_positive_rate': 0.2833333333333333, 'train_bad_rate': 0.008333333333333333}, 'MILK|480|502|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.0, 0.0, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4590235229892806e-28, 219.86684088692317, 64.73499866407533, -11.552724843135032, 26.136977466696386, 136.77876131005837, -95.53708030892649, 119.21115031849133, 161.86711139331783, 0.0, 12.949377232840217, -12.949377232839577, 133.50690348295478, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2996119306654212e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 110.06666666664304, 'uncertainty': 318.8631930824509}, 'bad': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.0, 0.0, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.670454147746215e-31, 0.013556276301512767, 0.016350246684102326, 0.0011738929795272269, 0.05065587104207961, 0.04093853962322908, -0.0055728922233018036, 0.009712759168307453, 0.015461418300061269, 0.0, 0.0013758699419094023, -0.001375869941909198, -0.011490407479114841, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.6099142551504745e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.024999999999994145, 'uncertainty': 0.17160955887797422}, 'train_mean_delta': 110.06666666666666, 'train_min_delta': -176.0, 'train_positive_rate': 0.3333333333333333, 'train_bad_rate': 0.025}, 'MILK|502|504|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.0, 0.0, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.0455065962715281e-29, 0.0, -5.335291040830216e-29, 0.0, 0.0, -1.452817433414285e-30, -2.9037194105735504, 0.4216690764604916, -8.92004107150303, -2.6317334306629245, 2.121537189483776, -1.64347177610143, -1.6434717761015385, 0.9589404225839725, 1.6315284371957032e-30, 0.8109631307209281, -0.8109631307209642, 1.638428160358902, 0.0, 0.0, 0.0, 0.0, -2.6104454995131257e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -11.61666666666658, 'uncertainty': 16.336318270347807}, 'bad': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.0, 0.0, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -3.617475689324303e-32, 0.0, -1.0648934350512845e-31, 0.0, 0.0, -1.1731312073784394e-32, -0.008341119032505987, -0.03457633464975516, -0.03027805349412302, -0.016145955714465924, 0.017372493651923306, -0.02212268300135766, -0.022122683001358515, 0.019386254318683918, -4.780553912932749e-33, -0.008115750631160795, 0.008115750631161524, -0.002235355463389451, 0.0, 0.0, 0.0, 0.0, 7.648886260692397e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.01666666666666561, 'uncertainty': 0.1399005352356968}, 'train_mean_delta': -11.616666666666667, 'train_min_delta': -52.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.016666666666666666}, 'MILK|502|504|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.0, 0.0, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -2.1630918638775297e-30, 0.0, 1.3803883447562698e-29, 0.0, 0.0, -1.1559420062203572e-29, -10.952518160941448, 3.147760431983348, -23.811986629731567, -28.40975772079142, 6.2281702382892306, -11.990347741352284, -11.99034774135279, -0.6846145734248371, -6.446654382788305e-30, 3.4677534616896306, -3.4677534616895818, -3.71351846604429, 0.0, 0.0, 0.0, 0.0, 1.031464701246129e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -43.44999999999963, 'uncertainty': 30.80905760140599}, 'bad': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.0, 0.0, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 1.405703227418193e-32, 0.0, 1.7611288985947614e-31, 0.0, 0.0, -2.805969785446612e-32, -0.04088176621849238, -0.015140194537713451, -0.0181794046178947, -0.013453695333247853, -0.01730264287592738, -0.05330131260055963, -0.05330131260056074, 0.031537794149753245, -2.4021208144019003e-32, -0.009797372591986528, 0.009797372591987197, -0.036942275891341164, 0.0, 0.0, 0.0, 0.0, 3.8433933030430404e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.041666666666665554, 'uncertainty': 0.23009092140411114}, 'train_mean_delta': -43.45, 'train_min_delta': -204.0, 'train_positive_rate': 0.08333333333333333, 'train_bad_rate': 0.041666666666666664}, 'MILK|504|518|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.0, 0.0, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 8.978050488955185e-29, 8.978050488955185e-29, 0.0, 0.0, 0.0, -2.2445126222387962e-29, 1.8248252905985218, 0.8049665479913761, -9.240209915996115, 3.360092263534317, -1.902427417528226, 0.6813510879076824, -3.2513209010478574, -3.463910941021588, -1.10623654629086e-30, -0.4031638490550433, 0.4031638490550801, 2.5240095792636543, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.84989237032688e-30, -1.6861503825177867e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 8.749999999999568, 'uncertainty': 8.196722256486474}, 'bad': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.0, 0.0, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 8.75, 'train_min_delta': 0.0, 'train_positive_rate': 0.5166666666666667, 'train_bad_rate': 0.0}, 'MILK|504|518|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.0, 0.0, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 1.7403091784677137e-28, 1.7403091784677137e-28, 0.0, 0.0, 0.0, -4.3507729461692843e-29, 4.40102751093395, 1.087385220770298, -17.733824847807686, 5.956499139913819, -5.478431071937544, 0.09121736915714308, -6.0684022505184725, -6.617651736887737, -2.145492329890612e-30, -0.7072505902061937, 0.7072505902062581, 4.15523488284111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7163938639124895e-29, -3.7394347609977716e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 16.349999999999394, 'uncertainty': 13.93891325306149}, 'bad': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.0, 0.0, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 16.35, 'train_min_delta': 0.0, 'train_positive_rate': 0.5333333333333333, 'train_bad_rate': 0.0}, 'MILK|518|524|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.0, 0.0, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, -9.024727488256179e-30, 0.0, 0.0, 0.0, 0.0, 1.1857189130990295, -1.3499034204206686, 1.8205339698140457, -2.2282614289297773, -7.710069716113945, 2.2656816103256827, 1.3340358019490686, -6.283273270402162, 2.4474076642692048e-30, -0.19703685744597121, 0.197036857445973, -1.7351599179519408, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.8948153285384095e-30, 1.0213243897999001e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -10.949999999999296, 'uncertainty': 23.32136778767772}, 'bad': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.0, 0.0, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 1.979868429421304e-32, 0.0, 0.0, 0.0, 0.0, -0.013292562244406529, 8.029770567659868e-05, 0.008998285735398279, 0.007329045895082693, 0.0068386660737104425, -0.006560137568710385, -0.023098535308825458, 0.0021175064949915294, -1.1751381569572519e-32, 1.602335075521483e-05, -1.6023350755044522e-05, -0.024867167704943263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.3502763139145043e-32, 2.01906216441644e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.01666666666666672, 'uncertainty': 0.1538485883429109}, 'train_mean_delta': -10.95, 'train_min_delta': -41.0, 'train_positive_rate': 0.25, 'train_bad_rate': 0.016666666666666666}, 'MILK|518|524|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.0, 0.0, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, -1.792648494598763e-29, 0.0, 0.0, 0.0, 0.0, 2.8740729029148246, -2.1388633942732005, 3.302046250993481, -2.4176868826394884, -12.486881150001375, 4.344827540762273, 2.9531980583761204, -10.08018854536929, 4.3875124690700696e-30, -0.3490660039176871, 0.3490660039177421, -2.192043791397376, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.775024938140139e-30, 1.8221112949546505e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -16.883333333332214, 'uncertainty': 35.9063291050518}, 'bad': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.0, 0.0, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 3.0592669238892617e-31, 0.0, 0.0, 0.0, 0.0, 0.008405084142332112, -0.046410225946614876, -0.020747807092938417, -0.04354741900382503, 0.10121273281976138, -0.051787484037339994, -0.05715151837922789, 0.13366423771084537, -5.26262862279416e-32, -0.004704212204452917, 0.004704212204452064, -0.028950335392179952, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0525257245588317e-31, -3.010143493400235e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.03333333333335773, 'uncertainty': 0.18920296602679543}, 'train_mean_delta': -16.883333333333333, 'train_min_delta': -62.0, 'train_positive_rate': 0.25, 'train_bad_rate': 0.03333333333333333}, 'MILK|524|527|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.0, 0.0, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.8263966421139333e-29, 2.107107279943357e-29, 5.577745612891528e-30, 0.0, 0.0, 0.0, -0.895481788035727, -0.23888202597453048, 1.1566663131229684, -0.8802601314401173, -0.19005462572027193, 0.7204654302992047, 0.15181632553083096, -0.6299588852303047, -1.3905648044936142e-31, 0.028814028968187994, -0.02881402896817882, 0.0022792299169738708, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.127357459453205e-31, 0.0, 0.0, -2.781129608987228e-31, 7.869732196983194e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -0.9833333333331976, 'uncertainty': 3.5210462088296826}, 'bad': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.0, 0.0, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -0.9833333333333333, 'train_min_delta': -9.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.0}, 'MILK|524|527|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.0, 0.0, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.775350978125469e-29, 3.1841784764738387e-29, 1.2687328265941909e-29, 0.0, 0.0, 0.0, -1.0769702805554693, -0.8356300022355964, 3.446721758846061, -3.2747364466751554, 0.24881373519562322, 0.7597042144517769, 1.534530867784544, -1.293671794486412, 1.2918712207452042e-30, 0.2106866037543566, -0.21068660375432813, -0.2590757257357874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.1583393547787574e-31, 0.0, 0.0, 2.5837424414904084e-30, 3.876662029271792e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -3.449999999999541, 'uncertainty': 4.16012139941483}, 'bad': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.0, 0.0, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.1562747309608842e-30, 1.2418728226182427e-30, 5.485132136610004e-31, 0.0, 0.0, 0.0, 0.009587679798350202, -0.03525371625535404, 0.026119180638508988, 0.04304242749201275, 0.11500472909233238, 0.06858244052994975, 0.008285651205198221, 0.13121091300052906, -2.656558376827253e-32, -0.005069773762106152, 0.0050697737621056066, -0.027552039549294964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.1812641359246804e-31, 0.0, 0.0, -5.313116753654506e-32, 2.218515864267653e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.03333333333335521, 'uncertainty': 0.18539590061679603}, 'train_mean_delta': -3.45, 'train_min_delta': -18.0, 'train_positive_rate': 0.016666666666666666, 'train_bad_rate': 0.03333333333333333}, 'MILK|527|552|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.0, 0.0, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -4.139372688634048e-29, -2.4761631827502452e-29, 2.4761631827502452e-29, 0.0, 0.0, -6.318756240140309e-30, -4.6115088852783055, -0.8443259347801343, 2.514603733004246, -0.12402237494183384, -4.25637358250835, 3.4032821078276925, 2.185009467168649, -3.1248985978997377, -1.3204042291453878e-30, -0.1736132582360475, 0.17361325823602372, -0.7656203684382783, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.994697663856712e-29, 0.0, -1.238081591375124e-29, 0.0, -1.263751248028063e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.9500000000001642, 'uncertainty': 14.117209072654022}, 'bad': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.0, 0.0, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 0.95, 'train_min_delta': -32.0, 'train_positive_rate': 0.23333333333333334, 'train_bad_rate': 0.0}, 'MILK|527|552|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.0, 0.0, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -8.013949116006931e-29, -1.3144969406511864e-29, 1.3144969406511864e-29, 0.0, 0.0, -2.729393054677306e-29, -6.89385879354282, -1.7464236233134296, 5.741156485064653, -0.49224426675864763, -9.001969879581212, 4.266798222666586, 3.1076960884876335, -3.875802204385251, -1.3425418604230054e-30, -0.7650465141599805, 0.7650465141599789, -0.35537362445479137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.222857742978995e-29, 0.0, -6.57248470325601e-30, 0.0, -5.458786109354615e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -0.03333333333299494, 'uncertainty': 24.66149919040305}, 'bad': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.0, 0.0, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -0.03333333333333333, 'train_min_delta': -52.0, 'train_positive_rate': 0.23333333333333334, 'train_bad_rate': 0.0}, 'MILK|552|571|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.0, 0.0, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 4.022172209696411e-30, 4.022172209696411e-30, 0.0, 0.0, 0.0, 3.4443682069440836e-31, 6.23994265737031, -1.1905161206389732, -2.9393817324800446, -2.5219505633935695, 7.920435278228662, 2.8428031738451214, 3.9245350268553554, 0.7216718914630282, 5.02771526212052e-31, -0.14801054551524825, 0.14801054551538262, -0.10689361498297689, 0.0, 0.0, -1.2021268770115363e-29, 0.0, 0.0, 0.0, -6.0106343850576815e-30, 0.0, 0.0, -2.052322602835934e-30, -8.044344419392832e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 3.5666666666666975, 'uncertainty': 11.42923431122011}, 'bad': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.0, 0.0, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 3.566666666666667, 'train_min_delta': -29.0, 'train_positive_rate': 0.3333333333333333, 'train_bad_rate': 0.0}, 'MILK|552|571|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.0, 0.0, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -6.649455476718338e-30, -6.649455476718338e-30, 0.0, 0.0, 0.0, 7.411868088287357e-30, 25.13101133550932, -4.779024353625879, -3.9070474707905123, -1.0813242255507889, 13.882521262070059, 7.871343547925217, 5.95001665336777, 14.983191768533562, -8.311819345897976e-31, -2.5571425723561347, 2.5571425723564767, -0.576532285587782, 0.0, 0.0, -4.306417592339984e-29, 0.0, 0.0, 0.0, -2.153208796169992e-29, 0.0, 0.0, 6.802164403581455e-31, 1.3298910953436762e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -4.933333333333219, 'uncertainty': 42.3033431689328}, 'bad': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.0, 0.0, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 6.662313832883925e-32, 6.662313832883925e-32, 0.0, 0.0, 0.0, -2.700912193246499e-32, -0.009591428862699198, -7.562020967899515e-05, 0.00916763824779495, -0.01960179549102678, 0.033369502734049594, -0.00648069587431651, 0.010508678009932996, -0.027650113801979154, 8.327892291104917e-33, 0.0023906034666652844, -0.00239060346666489, -0.017462079232661396, 0.0, 0.0, -1.6195801514300214e-32, 0.0, 0.0, 0.0, -8.097900757150107e-33, 0.0, 0.0, -1.930752188107953e-32, -1.3324627665767867e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.01666666666666664, 'uncertainty': 0.15163478683279194}, 'train_mean_delta': -4.933333333333334, 'train_min_delta': -74.0, 'train_positive_rate': 0.31666666666666665, 'train_bad_rate': 0.016666666666666666}, 'MILK|571|597|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.0, 0.0, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 5.725760779861508e-30, -6.203723341632778e-30, -5.725760779861508e-30, 0.0, 0.0, 1.3728379120405968e-30, 3.4912920528310543, -5.101049232455705, 1.0878224557981255, -5.649245046633595, -0.3401400382297153, 0.8183779090539904, 2.6826194376239263, 0.7929565991113493, 1.1426758795948313e-30, 2.34006664185814, -2.3400666418581437, -5.507213716708804, 0.0, 0.0, 0.0, 0.0, 1.2407446683265444e-29, 0.0, 1.028383014291882e-29, 0.0, 0.0, 2.2853517591896623e-30, 1.2407446683265444e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -5.991666666666354, 'uncertainty': 27.87001395089441}, 'bad': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.0, 0.0, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -5.991666666666666, 'train_min_delta': -60.0, 'train_positive_rate': 0.2, 'train_bad_rate': 0.0}, 'MILK|571|597|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.0, 0.0, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 6.679653034246072e-30, -2.593691468628066e-30, -6.679653034246072e-30, 0.0, 0.0, -4.121068876799999e-31, 8.161408925693532, -8.15962591296926, -0.1699986665821584, -7.574905997393703, -0.040692454000794194, 1.9170907879296326, 6.8780807644200666, 2.0140637337201373, 5.704239354911614e-30, 2.7821065384360444, -2.782106538436063, -3.882578079508466, 0.0, 0.0, 0.0, 0.0, 5.1873829372560196e-30, 0.0, 7.925443082126561e-30, 0.0, 0.0, 1.1408478709823228e-29, 5.1873829372560196e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -14.24166666666611, 'uncertainty': 45.432561091935135}, 'bad': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.0, 0.0, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -4.3414904379335475e-32, 6.983710315681291e-33, 4.3414904379335475e-32, 0.0, 0.0, 1.3330621289457242e-32, 0.001069299506120764, -0.0014835121144855525, -0.0033664925752430585, 0.004081786918139075, -0.006869248117088291, -0.009073756649895903, -0.005111028259986242, 0.005643158263069388, -8.441189083721773e-33, 0.007694148245441935, -0.00769414824544208, -0.018546357124661992, 0.0, 0.0, 0.0, 0.0, -1.3967420631363015e-32, 0.0, 2.819613157053362e-32, 0.0, 0.0, -1.6882378167443547e-32, -1.3967420631363015e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.00833333333333301, 'uncertainty': 0.11022647851777773}, 'train_mean_delta': -14.241666666666667, 'train_min_delta': -137.0, 'train_positive_rate': 0.15, 'train_bad_rate': 0.008333333333333333}, 'MILK|597|600|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.0, 0.0, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -7.906339809704749e-29, 0.0, 0.0, 0.0, 0.0, 0.0, -9.145493190760824, -7.837223545372758, -10.968486197525694, -31.230183217642896, -1.889113881356596, -2.170607435327839, 13.832540264671914, -20.584519080185242, -4.153261564813846e-29, 7.979692949708318, -7.97969294970833, -7.44392734652882, 0.0, 0.0, 0.0, 0.0, -4.675138687339983e-28, 0.0, 1.2342754558098924e-28, 0.0, 0.0, 0.0, 1.581267961940933e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -41.075000000000735, 'uncertainty': 60.55271120810784}, 'bad': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.0, 0.0, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.0260722697985267e-31, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01925675034798047, -0.020124563230049013, 0.010160363839276048, -0.038515248730622766, 0.010814687058745503, -0.016280618209436597, 0.04426696792269696, -0.10733321233215716, -7.137335843615933e-32, -0.01624407085286696, 0.016244070852866607, 0.008726310946160658, 0.0, 0.0, 0.0, 0.0, -9.863071040752804e-31, 0.0, 4.5503872631445607e-32, 0.0, 0.0, 0.0, 2.0521445395970845e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.03333333333333665, 'uncertainty': 0.19411679260952358}, 'train_mean_delta': -41.075, 'train_min_delta': -198.0, 'train_positive_rate': 0.008333333333333333, 'train_bad_rate': 0.03333333333333333}, 'MILK|597|600|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.0, 0.0, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 6.113734862654601e-29, 0.0, 0.0, 0.0, 0.0, 0.0, -3.2795549831794415, -13.96452128163835, -10.170403059508297, -44.11666748044021, -30.84932571533815, -5.211207257476875, 19.627136026713828, -9.864731883450515, -3.0617140736451847e-29, 15.699790830154745, -15.699790830155294, -12.101377710253029, 0.0, 0.0, 0.0, 0.0, -3.999017778816521e-28, 0.0, 1.695699688165957e-28, 0.0, 0.0, 0.0, -1.2227469725309518e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -78.26666666666821, 'uncertainty': 86.78210219422228}, 'bad': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.0, 0.0, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -1.5639491337952506e-31, 0.0, 0.0, 0.0, 0.0, 0.0, -0.021364836151956997, -0.021088769606239198, 0.004490396604247681, -0.03939380748992824, 0.028754757915840447, -0.008803077816574183, 0.04692432594279013, -0.11814990370225338, -8.729818897285779e-32, -0.011618840548032342, 0.01161884054803196, 0.001578830184642129, 0.0, 0.0, 0.0, 0.0, -1.1291052253371323e-30, 0.0, 6.06557422423148e-32, 0.0, 0.0, 0.0, 3.1278982675905196e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.041666666666669384, 'uncertainty': 0.21750195465361039}, 'train_mean_delta': -78.26666666666667, 'train_min_delta': -424.0, 'train_positive_rate': 0.025, 'train_bad_rate': 0.041666666666666664}, 'MILK|600|620|DELAY_25': {'support': 114, 'rows': 114, 'margin': {'mean': [0.0, 0.8333333333333325, 0.8333333333333325, 0.0, 0.09375, 0.34375, 0.27777777777777835, 0.12114035087719308, 1.0055578947368407, -0.06055555555555559, -0.041023391812865496, -0.005409356725146196, 0.0009421052631578953, 0.0005947368421052636, 6.140350877193017e-06, 0.029999999999999933, 0.9057894736842121, 0.09421052631578936, 0.6139526315789475, 0.0, 0.0, 0.9000000000000015, 1.0, 0.0, 0.75, 0.45000000000000073, 1.0, 0.0, 0.32000000000000023, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2410049739230031, 0.004486566591324987, 0.04152764331375287, 0.027784071309206994, 0.022913475485954925, 0.0005675269120818952, 0.0003738941836569323, 0.00047021553246176154, 1.0, 0.009070362073481098, 0.009070362073481093, 0.11762724713480882, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.6326817461328543e-29, 2.7591390369667352, -2.526427247217422, 3.3719020801771578, -8.576397033521046, -3.4213550001070345, -5.737609153571508, 2.4986894725567836, -0.5028632584400773, -4.532282824589152e-31, 0.16529086185152284, -0.1652908618514901, -1.1331978465869874, 0.0, 0.0, 2.1073444233210504e-29, 0.0, 0.0, 0.0, 1.0536722116605252e-29, 0.0, 0.0, 0.0, -2.970185220382239e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -1.8070175438587868, 'uncertainty': 12.88126230125307}, 'bad': {'mean': [0.0, 0.8333333333333325, 0.8333333333333325, 0.0, 0.09375, 0.34375, 0.27777777777777835, 0.12114035087719308, 1.0055578947368407, -0.06055555555555559, -0.041023391812865496, -0.005409356725146196, 0.0009421052631578953, 0.0005947368421052636, 6.140350877193017e-06, 0.029999999999999933, 0.9057894736842121, 0.09421052631578936, 0.6139526315789475, 0.0, 0.0, 0.9000000000000015, 1.0, 0.0, 0.75, 0.45000000000000073, 1.0, 0.0, 0.32000000000000023, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2410049739230031, 0.004486566591324987, 0.04152764331375287, 0.027784071309206994, 0.022913475485954925, 0.0005675269120818952, 0.0003738941836569323, 0.00047021553246176154, 1.0, 0.009070362073481098, 0.009070362073481093, 0.11762724713480882, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4840091517636272e-32, 0.0016195998252819632, 0.008900520873882932, -0.012773020678657546, -0.023066296716049324, 0.057350596240542145, -0.0005709999222245869, 0.011242558089847894, -0.008164769320378866, 8.570376209843477e-34, -0.012888137181133635, 0.01288813718113413, 0.014285965453267194, 0.0, 0.0, -7.62625974853161e-32, 0.0, 0.0, 0.0, -3.813129874265805e-32, 0.0, 0.0, 0.0, -1.6013194861875437e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.017543859649117463, 'uncertainty': 0.14850057190553678}, 'train_mean_delta': -1.8070175438596492, 'train_min_delta': -53.0, 'train_positive_rate': 0.12280701754385964, 'train_bad_rate': 0.017543859649122806}, 'MILK|600|620|DELAY_50': {'support': 81, 'rows': 81, 'margin': {'mean': [0.0, 0.833333333333334, 0.833333333333334, 0.0, 0.09375, 0.34375, 0.27777777777777823, 0.10588477366255139, 1.0059370370370357, -0.0653086419753087, -0.04753086419753085, -0.013086419753086428, 0.0009950617283950617, 0.0006987654320987658, 0.00016666666666666663, 0.029999999999999985, 0.8999999999999997, 0.09999999999999984, 0.6265604938271605, 0.0, 0.0, 0.8999999999999997, 1.0, 0.0, 0.75, 0.44999999999999984, 1.0, 0.0, 0.3200000000000002, 0.700000000000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2114839278608835, 0.0037151157516302357, 0.04088116465480159, 0.024943160503555113, 0.018553875711481028, 0.0005708778673231972, 0.0002795471766808699, 0.0002434322477800738, 1.0, 1.0, 1.0, 0.12025273573440227, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 2.2604687391355292e-29, 2.2604687391355292e-29, 0.0, 0.0, 0.0, 0.0, 5.305410821674604, -3.449694341471809, 6.197181230641602, -19.830068296615583, -5.616366243224001, -14.500794469825099, 3.8587243023741467, -9.376822792613648, 0.0, -1.1302343695677643e-29, -5.651171847838822e-30, 0.37345755339439757, 0.0, 0.0, -1.1302343695677643e-29, 0.0, 0.0, 0.0, -5.651171847838822e-30, 0.0, 0.0, 5.651171847838822e-30, 3.283338294786983e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.3950617283962992, 'uncertainty': 16.110871557398944}, 'bad': {'mean': [0.0, 0.833333333333334, 0.833333333333334, 0.0, 0.09375, 0.34375, 0.27777777777777823, 0.10588477366255139, 1.0059370370370357, -0.0653086419753087, -0.04753086419753085, -0.013086419753086428, 0.0009950617283950617, 0.0006987654320987658, 0.00016666666666666663, 0.029999999999999985, 0.8999999999999997, 0.09999999999999984, 0.6265604938271605, 0.0, 0.0, 0.8999999999999997, 1.0, 0.0, 0.75, 0.44999999999999984, 1.0, 0.0, 0.3200000000000002, 0.700000000000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2114839278608835, 0.0037151157516302357, 0.04088116465480159, 0.024943160503555113, 0.018553875711481028, 0.0005708778673231972, 0.0002795471766808699, 0.0002434322477800738, 1.0, 1.0, 1.0, 0.12025273573440227, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -8.507648063602136e-33, -8.507648063602136e-33, 0.0, 0.0, 0.0, 0.0, 0.004548636921220655, 0.01703988201574741, -0.02509067451692857, -0.0393399504366903, 0.07497779617720504, -0.005804069172023814, 0.014458331372486167, -0.023005029290757123, 0.0, 4.2538240318010826e-33, 2.1269120159005413e-33, 0.035684500687342476, 0.0, 0.0, 4.2538240318010826e-33, 0.0, 0.0, 0.0, 2.1269120159005413e-33, 0.0, 0.0, -2.1269120159005413e-33, 1.2998007027623359e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.02469135802468522, 'uncertainty': 0.157766064442045}, 'train_mean_delta': 0.3950617283950617, 'train_min_delta': -41.0, 'train_positive_rate': 0.1728395061728395, 'train_bad_rate': 0.024691358024691357}, 'MILK|620|624|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.0, 0.0, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -2.3004003452733854e-29, -2.7205051917478585e-30, -2.7205051917478585e-30, 0.0, 0.0, 1.4377502157958659e-30, -15.930244831029624, -11.139872044423326, -2.4066755108159694, -8.90027321110155, 4.781477443552891, -1.9293956888608612, 0.7571846765666296, 3.3230255728366473, -3.400631489684532e-31, 1.9428030141526746, -1.9428030141527537, -3.8987223439853484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.627693850402643e-29, 0.0, -2.627693850402643e-29, 0.0, -2.777297669004733e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -12.683333333333373, 'uncertainty': 24.02021286106279}, 'bad': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.0, 0.0, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, 9.501278833008991e-32, 6.752484223703073e-32, 6.752484223703073e-32, 0.0, 0.0, -5.9382992706306195e-33, -0.0061674234875879926, -0.00014629957689679963, 0.012199101569525327, 0.02638643433903694, -0.006806860064940443, 0.0009068189395624381, -0.003565571596872697, 0.01373465516598421, 8.440605279628805e-33, -0.0023258166314461936, 0.002325816631446131, 0.005190174341833158, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3042935226831242e-32, 0.0, -1.3042935226831242e-32, 0.0, -8.174341128636601e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.016666666666666503, 'uncertainty': 0.15263844776256755}, 'train_mean_delta': -12.683333333333334, 'train_min_delta': -108.0, 'train_positive_rate': 0.05, 'train_bad_rate': 0.016666666666666666}, 'MILK|620|624|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.0, 0.0, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -9.719599072004113e-29, 2.9796517193670525e-30, 2.9796517193670525e-30, 0.0, 0.0, 6.074749420002571e-30, -27.677824775066682, -22.90087780846425, -5.893597164286787, -28.942563734316657, 6.1170356095292675, -3.554228327194988, 1.1802823582016415, 3.967340028791217, 3.724564649209446e-31, 4.024955558394364, -4.024955558394825, -13.025283444493425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.3343079018610797e-29, 0.0, -4.3343079018610797e-29, 0.0, -8.159952562771376e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': -36.85833333333344, 'uncertainty': 52.9739254705422}, 'bad': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.0, 0.0, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'beta': [0.0, -3.349794627700315e-31, 2.1468380625248248e-31, 2.1468380625248248e-31, 0.0, 0.0, 2.0936216423126968e-32, -0.010186741548593956, -0.031920900536628743, -0.010134719603551157, -0.03866320621502248, 0.030772396710263442, -0.029044281670096095, -0.07350372160434512, 0.062204645020906925, 2.6835475781560216e-32, -0.0076457226932542185, 0.007645722693253703, 0.00285954751934275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.018543016365542e-31, 0.0, -3.018543016365542e-31, 0.0, -4.590366613833079e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.03333333333333221, 'uncertainty': 0.19979584538628076}, 'train_mean_delta': -36.858333333333334, 'train_min_delta': -229.0, 'train_positive_rate': 0.09166666666666666, 'train_bad_rate': 0.03333333333333333}}, 'variant': 'rl010b_bidirectional_no_opp'}
_RL010_EVENT_FILTER = RL010_EVENT_STEPS or None
_RL010_OPPORTUNITIES = rl010_route_opportunities(_ACTIONS, _RL010_EVENT_FILTER)
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
