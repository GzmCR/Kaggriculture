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
RL010_PAYLOAD = {'version': 'rl010', 'feature_dim': 41, 'min_support': 24, 'min_expected_delta': 5.0, 'lcb_z': 1.5, 'bad_ucb': 0.1, 'allowed_actions': ['ADVANCE_25', 'ADVANCE_50', 'DELAY_25', 'DELAY_50'], 'include_opponent_features': True, 'models': {'MILK|216|264|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.007859333333333331, 0.0005125000000000009, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.2800000000000001, 1.0, 0.0, 0.2800000000000001, 0.0, 0.5795833333333336, 0.0, 0.5, 0.23183333333333264, 0.4636666666666653, 0.6954999999999991], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 0.007245243236465947, 0.007753140143408563, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 0.04582575694955843, 1.0, 1.0, 0.04582575694955843, 1.0, 0.11133579712842681, 1.0, 1.0, 0.044534318851370734, 0.08906863770274147, 0.13360295655411245], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9598512856803031, -86.34960206343627, -1.541666262595287, 11.345860494973454, -101.76421457958371, 0.23799031570051107, -15.625837289502874, 8.268221845964005, 4.243323372725217e-29, 1.3150937857150634e-28, 0.0, 61.08504453462122, 56.530982560941546, -24.896181212273294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.304541650880607, -27.921112446400215, 0.0, 0.0, -27.921112446400155, 0.0, -19.302111926927378, 0.0, 0.0, -19.302111926930827, -19.30211192693083, -19.3021119269288], 'intercept': -65.05000000002661, 'uncertainty': 375.9068161735291}, 'bad': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.007859333333333331, 0.0005125000000000009, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.2800000000000001, 1.0, 0.0, 0.2800000000000001, 0.0, 0.5795833333333336, 0.0, 0.5, 0.23183333333333264, 0.4636666666666653, 0.6954999999999991], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 0.007245243236465947, 0.007753140143408563, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 0.04582575694955843, 1.0, 1.0, 0.04582575694955843, 1.0, 0.11133579712842681, 1.0, 1.0, 0.044534318851370734, 0.08906863770274147, 0.13360295655411245], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.003738399877193836, 0.06999038752268004, 0.0028924354888215558, -0.007642247790315176, 0.07542276268830368, -0.010333431654340393, 0.003732118095706375, -0.00673655497787704, -1.8730112144904333e-32, -5.401546526577098e-32, 0.0, -0.0433309595427844, -0.034093533521700616, 0.012046749090284307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.004024364414595763, 0.024670239948368068, 0.0, 0.0, 0.024670239948367992, 0.0, 0.02055081094125363, 0.0, 0.0, 0.02055081094125829, 0.02055081094125829, 0.020550810941255764], 'intercept': 0.050000000000021444, 'uncertainty': 0.18557832313087744}, 'train_mean_delta': -65.05, 'train_min_delta': -2840.0, 'train_positive_rate': 0.4166666666666667, 'train_bad_rate': 0.05}, 'MILK|216|264|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.007859333333333331, 0.0005125000000000009, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.2800000000000001, 1.0, 0.0, 0.2800000000000001, 0.0, 0.5795833333333336, 0.0, 0.5, 0.23183333333333264, 0.4636666666666653, 0.6954999999999991], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 0.007245243236465947, 0.007753140143408563, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 0.04582575694955843, 1.0, 1.0, 0.04582575694955843, 1.0, 0.11133579712842681, 1.0, 1.0, 0.044534318851370734, 0.08906863770274147, 0.13360295655411245], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.525614111443433, -157.77969282882592, -10.382680402523354, 13.834262829108841, -176.23190084939569, 9.92350639839396, -22.673330554002664, 19.03477716194504, 6.85601513152464e-29, 2.2488952588478305e-28, 0.0, 86.22522811609707, 76.81076604939794, -32.35195106092486, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.43005113543273177, -47.047450358253066, 0.0, 0.0, -47.047450358252995, 0.0, -21.138298584669844, 0.0, 0.0, -21.138298584685423, -21.138298584685426, -21.13829858467062], 'intercept': -116.08333333338167, 'uncertainty': 290.4680714004212}, 'bad': {'mean': [0.0, 0.3000000000000001, 0.3000000000000001, 0.0, 0.1875, 0.09375, 0.6666666666666667, 0.5989444444444444, 0.9987700000000006, -0.013611111111111109, -0.011555555555555557, -0.04627777777777776, 0.00020666666666666666, 0.00013999999999999996, 0.0004799999999999997, 0.05999999999999985, 0.3599999999999996, 0.6400000000000005, 0.00837183333333333, 0.007859333333333331, 0.0005125000000000009, 1.0, 0.375, 0.0, 0.5, 0.39999999999999913, 1.0, 0.0, 0.0, 0.8433333333333318, 0.2800000000000001, 1.0, 0.0, 0.2800000000000001, 0.0, 0.5795833333333336, 0.0, 0.5, 0.23183333333333264, 0.4636666666666653, 0.6954999999999991], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09486538545762994, 0.0020032723229755456, 0.02808249531665368, 0.028902560867358415, 0.06429414945639288, 0.0004020226638166338, 0.00043289721643826717, 0.0008340263784797217, 1.0, 1.0, 1.0, 0.0035451744158629043, 0.007245243236465947, 0.007753140143408563, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.016996731711975945, 0.04582575694955843, 1.0, 1.0, 0.04582575694955843, 1.0, 0.11133579712842681, 1.0, 1.0, 0.044534318851370734, 0.08906863770274147, 0.13360295655411245], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.003738399877193836, 0.06999038752268004, 0.0028924354888215558, -0.007642247790315176, 0.07542276268830368, -0.010333431654340393, 0.003732118095706375, -0.00673655497787704, -1.8730112144904333e-32, -5.401546526577098e-32, 0.0, -0.0433309595427844, -0.034093533521700616, 0.012046749090284307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.004024364414595763, 0.024670239948368068, 0.0, 0.0, 0.024670239948367992, 0.0, 0.02055081094125363, 0.0, 0.0, 0.02055081094125829, 0.02055081094125829, 0.020550810941255764], 'intercept': 0.050000000000021444, 'uncertainty': 0.18557832313087744}, 'train_mean_delta': -116.08333333333333, 'train_min_delta': -1874.0, 'train_positive_rate': 0.4166666666666667, 'train_bad_rate': 0.05}, 'MILK|264|311|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0883281666666666, -0.01831566666666666, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.3399999999999999, 1.0, 0.0, 0.3399999999999999, 0.05399999999999998, 0.8199999999999998, 0.0, 0.6, 0.4038333333333325, 0.7536666666666672, 1.1034999999999995], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 0.05351054642129489, 0.05268626482511566, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.04898979485566356, 1.0, 1.0, 0.04898979485566356, 0.08248636250920513, 0.07141428428542858, 1.0, 0.12247448713915889, 0.09836313107844598, 0.17045005785338463, 0.2548288641421925], 'beta': [0.0, 4.844788874229768e-31, 4.844788874229768e-31, 0.0, 0.0, 0.0, 3.363566805008947e-30, 1.7708254535574688, -1.7425977733344853, 2.8434133525638168, 4.563508524408118, -4.326530093375496, -2.933772841871299, -3.7005176413946543, -0.4971393864667226, -3.750690155910234e-30, 0.0, 0.0, 0.9073101799365577, 6.232596960459596, -6.1412760729868445, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5544490575678095, 0.0, 0.0, 0.5544490575678125, -1.5354352977495715, 0.9431152359742165, 0.0, -4.658790210310627, 1.4112997336556097, 2.371909983817193, 2.6282914594129303], 'intercept': -14.800000000000313, 'uncertainty': 12.687053772363328}, 'bad': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0883281666666666, -0.01831566666666666, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.3399999999999999, 1.0, 0.0, 0.3399999999999999, 0.05399999999999998, 0.8199999999999998, 0.0, 0.6, 0.4038333333333325, 0.7536666666666672, 1.1034999999999995], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 0.05351054642129489, 0.05268626482511566, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.04898979485566356, 1.0, 1.0, 0.04898979485566356, 0.08248636250920513, 0.07141428428542858, 1.0, 0.12247448713915889, 0.09836313107844598, 0.17045005785338463, 0.2548288641421925], 'beta': [0.0, -2.3120858385696078e-33, -2.3120858385696078e-33, 0.0, 0.0, 0.0, -2.035852337791157e-31, 0.00039245374039818717, 0.008019607526971602, 0.008404508250271538, 0.0017961969190361591, 0.0029883375523203147, -0.0025643084777413725, 0.003033367327794418, 0.011922155186295197, -1.6242697029950244e-32, 0.0, 0.0, -0.02071613596961988, -0.004898116485995913, 0.0006632815215082585, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0033144197903841204, 0.0, 0.0, 0.0033144197903841426, -0.006350707659513007, 0.008197619352441675, 0.0, -0.00980546954844638, 0.009641848169297794, 0.014201529499655226, 0.015276529554197551], 'intercept': 0.01666666666666791, 'uncertainty': 0.14372637056910503}, 'train_mean_delta': -14.8, 'train_min_delta': -58.0, 'train_positive_rate': 0.38333333333333336, 'train_bad_rate': 0.016666666666666666}, 'MILK|264|311|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0883281666666666, -0.01831566666666666, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.3399999999999999, 1.0, 0.0, 0.3399999999999999, 0.05399999999999998, 0.8199999999999998, 0.0, 0.6, 0.4038333333333325, 0.7536666666666672, 1.1034999999999995], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 0.05351054642129489, 0.05268626482511566, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.04898979485566356, 1.0, 1.0, 0.04898979485566356, 0.08248636250920513, 0.07141428428542858, 1.0, 0.12247448713915889, 0.09836313107844598, 0.17045005785338463, 0.2548288641421925], 'beta': [0.0, 5.70766285927173e-29, 5.70766285927173e-29, 0.0, 0.0, 0.0, -4.402448545960242e-29, 6.407873421252467, -8.174814520945969, 15.950381972171387, 4.564705536776299, -21.318230704825098, -5.244783962987502, -5.815443040208596, -0.08245995338106653, -1.0228886866386651e-29, 0.0, 0.0, 8.310879823444633, 10.088830006779096, -8.517000672116234, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1437132363416758, 0.0, 0.0, 1.1437132363416798, -3.3842741090350033, 3.0114310785668095, 0.0, -7.314179051081123, 3.4165991154650674, 5.581055508238974, 6.14732199651305], 'intercept': -34.43333333333474, 'uncertainty': 42.57033424822243}, 'bad': {'mean': [0.0, 0.36666666666666703, 0.36666666666666703, 0.0, 0.09375, 0.09375, 0.6527777777777771, 0.5558888888888888, 0.9991800000000004, -0.003111111111111111, 0.0002777777777777776, 0.006388888888888892, 3.833333333333334e-05, -6.5e-05, -0.00020000000000000025, 0.029999999999999926, 0.39999999999999913, 0.6000000000000002, 0.07001249999999998, 0.0883281666666666, -0.01831566666666666, 1.0, 0.375, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.0, 1.0, 0.3399999999999999, 1.0, 0.0, 0.3399999999999999, 0.05399999999999998, 0.8199999999999998, 0.0, 0.6, 0.4038333333333325, 0.7536666666666672, 1.1034999999999995], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.11646548792042737, 0.0025342849090029267, 0.008093588378433027, 0.00895030863665371, 0.023040155499303796, 0.00014034679270372456, 0.0002197157254271982, 0.0004438468204234431, 1.0, 1.0, 1.0, 0.010965127089246768, 0.05351054642129489, 0.05268626482511566, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.04898979485566356, 1.0, 1.0, 0.04898979485566356, 0.08248636250920513, 0.07141428428542858, 1.0, 0.12247448713915889, 0.09836313107844598, 0.17045005785338463, 0.2548288641421925], 'beta': [0.0, -1.772591416561457e-32, -1.772591416561457e-32, 0.0, 0.0, 0.0, -8.560712172253407e-32, 0.0023554346333831038, 0.002117617998085282, 0.008525796017691553, -0.005890405283855849, 0.007155753725365968, 0.001458823382800027, 0.006361374977690496, 0.003168313252403076, -9.988726123942633e-33, 0.0, 0.0, 0.008771285588636091, 0.008305537475464619, -0.006609988173865756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002369978541105931, 0.0, 0.0, 0.00023699785411059352, -0.002849454911381174, 0.007932187660113284, 0.0, 0.011753424753718409, 0.004861638706125901, 0.006990042692810244, 0.0074744294131653954], 'intercept': 0.008333333333333588, 'uncertainty': 0.10791982116121256}, 'train_mean_delta': -34.43333333333333, 'train_min_delta': -298.0, 'train_positive_rate': 0.4166666666666667, 'train_bad_rate': 0.008333333333333333}, 'MILK|311|336|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.12153233333333335, -0.034366, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.9299999999999987, 0.4400000000000009, 0.75, 0.4058333333333344, 0.8116666666666688, 1.1960000000000013], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 0.022934074312157354, 0.0191622865023984, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05999999999999999, 0.11357816691600556, 1.0, 0.11303084633064651, 0.22606169266129303, 0.27608694282779844], 'beta': [0.0, -3.8816017576561566e-29, 0.0, 2.6373162643296423e-28, 0.0, 0.0, 3.8816017576561566e-29, -1.0589956158160307, 1.6707644538615678, -1.071530347667942, 1.7523447584428502, 0.7003748204490069, -4.167268355138874, -2.610151285941958, -2.184404685757857, 1.5936031785169544e-30, 2.850071844435937e-29, 0.0, -2.5740573134830225, 2.5438090707943406, -4.510459565442996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.509935412902975, 0.9021382542115007, 0.0, -1.7052809369684687, -1.7052809369684743, -1.2075787044840929], 'intercept': -2.8666666666670433, 'uncertainty': 9.171529996280489}, 'bad': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.12153233333333335, -0.034366, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.9299999999999987, 0.4400000000000009, 0.75, 0.4058333333333344, 0.8116666666666688, 1.1960000000000013], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 0.022934074312157354, 0.0191622865023984, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05999999999999999, 0.11357816691600556, 1.0, 0.11303084633064651, 0.22606169266129303, 0.27608694282779844], 'beta': [0.0, -7.001651318243499e-32, 0.0, -2.3310710203970404e-30, 0.0, 0.0, 7.001651318243499e-32, 0.0010987430832219947, 0.027824101770838276, 0.0649097384192919, 0.04069098238950349, 0.023900626504384685, 0.05536009223424231, 0.027651261833019677, 0.011700498314999733, 5.961797592463748e-32, 1.4347233842262445e-32, 0.0, -0.02006641333524169, -0.009627718919046662, 9.48258509626627e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.030512885849919475, 0.0040316385878220035, 0.0, 0.04092804351507804, 0.040928043515078186, 0.03414270129358434], 'intercept': 0.04166666666666414, 'uncertainty': 0.16962287130568593}, 'train_mean_delta': -2.8666666666666667, 'train_min_delta': -25.0, 'train_positive_rate': 0.5166666666666667, 'train_bad_rate': 0.041666666666666664}, 'MILK|311|336|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.12153233333333335, -0.034366, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.9299999999999987, 0.4400000000000009, 0.75, 0.4058333333333344, 0.8116666666666688, 1.1960000000000013], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 0.022934074312157354, 0.0191622865023984, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05999999999999999, 0.11357816691600556, 1.0, 0.11303084633064651, 0.22606169266129303, 0.27608694282779844], 'beta': [0.0, -7.229979804106795e-29, 0.0, 5.396290285287228e-28, 0.0, 0.0, 7.229979804106795e-29, -3.3958457367170136, 1.5574255642690993, -0.23793811517792188, 2.5476798306285957, 0.3054847162426159, -7.859654686689082, -7.110510402399986, -4.636740250662296, 5.122495518831793e-30, 5.448752964310546e-29, 0.0, -4.525303284301017, 5.200336990354372, -8.801129707238891, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.333019073536208, 1.7391772933098084, 0.0, -3.6850067473873027, -3.6850067473873156, -2.854935976808794], 'intercept': -6.116666666667141, 'uncertainty': 20.139031219478305}, 'bad': {'mean': [0.0, 0.4319444444444451, 0.39999999999999913, 0.9583333333333318, 0.09375, 0.1875, 0.3472222222222216, 0.5400000000000004, 0.9991899999999995, -0.0045555555555555575, -0.00522222222222222, -0.01016666666666667, 6.0000000000000015e-05, 1.0000000000000008e-05, 1.6985834035606236e-19, 0.029999999999999926, 0.2699999999999997, 0.7300000000000001, 0.08716633333333328, 0.12153233333333335, -0.034366, 0.19999999999999957, 0.5, 0.5, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 1.0, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.9299999999999987, 0.4400000000000009, 0.75, 0.4058333333333344, 0.8116666666666688, 1.1960000000000013], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14894070408499266, 0.003315855847288884, 0.011831637854540156, 0.020650232414319982, 0.03855455490216042, 0.00018275666882497065, 0.0003477067730142741, 0.0006723094525588642, 1.0, 1.0, 1.0, 0.01091304845382607, 0.022934074312157354, 0.0191622865023984, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05999999999999999, 0.11357816691600556, 1.0, 0.11303084633064651, 0.22606169266129303, 0.27608694282779844], 'beta': [0.0, -7.001651318243499e-32, 0.0, -2.3310710203970404e-30, 0.0, 0.0, 7.001651318243499e-32, 0.0010987430832219947, 0.027824101770838276, 0.0649097384192919, 0.04069098238950349, 0.023900626504384685, 0.05536009223424231, 0.027651261833019677, 0.011700498314999733, 5.961797592463748e-32, 1.4347233842262445e-32, 0.0, -0.02006641333524169, -0.009627718919046662, 9.48258509626627e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.030512885849919475, 0.0040316385878220035, 0.0, 0.04092804351507804, 0.040928043515078186, 0.03414270129358434], 'intercept': 0.04166666666666414, 'uncertainty': 0.16962287130568593}, 'train_mean_delta': -6.116666666666666, 'train_min_delta': -59.0, 'train_positive_rate': 0.55, 'train_bad_rate': 0.041666666666666664}, 'MILK|336|360|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.1307306666666666, -0.0008176666666666633, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.14399999999999982, 1.0, 0.0, 0.75, 0.6700000000000003, 1.1960000000000008, 1.673999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 0.03605628243300866, 0.03220251104943354, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029393876913398138, 1.0, 1.0, 1.0, 0.11392980294900883, 0.22553048574416715, 0.263066531508666], 'beta': [0.0, 1.1409480191735785e-28, 1.1409480191735785e-28, 0.0, 0.0, 0.0, 0.0, 0.11446536942674247, -7.9440029634937535, 17.515422718139597, -15.421472058767515, -6.158638558596109, -8.61848330111745, -1.5975075852118574, -12.856202374635727, 2.251290903256533e-29, 4.563792076694309e-28, 1.8010327226052265e-28, -1.932733792092412, 3.1783711981978118, -5.120722708510876, 5.405511583142565e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0124449606539594e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 14.077687104982155, 0.0, 0.0, 0.0, 2.040311612322501, 0.2266070493094131, -23.495108232978403], 'intercept': -49.36666666666522, 'uncertainty': 26.423459276250384}, 'bad': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.1307306666666666, -0.0008176666666666633, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.14399999999999982, 1.0, 0.0, 0.75, 0.6700000000000003, 1.1960000000000008, 1.673999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 0.03605628243300866, 0.03220251104943354, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029393876913398138, 1.0, 1.0, 1.0, 0.11392980294900883, 0.22553048574416715, 0.263066531508666], 'beta': [0.0, -1.6380319541901553e-31, -1.6380319541901553e-31, 0.0, 0.0, 0.0, 0.0, 0.0019060493769445077, 0.047954341141996704, -0.016161973449527054, 0.005034213146834776, 0.042023367241824545, 0.013797260529452791, -0.010900956679260583, -0.021333185190674307, 1.5430943674211152e-31, -6.552127816760634e-31, 1.2344754939368922e-30, -0.028929496643010214, 0.00332455829032517, -0.027102501170014413, -5.051413478275263e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.895739949942722e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06281961618462509, 0.0, 0.0, 0.0, 0.07313338107117215, 0.06570123446546806, -0.0604022571118556], 'intercept': 0.041666666666658296, 'uncertainty': 0.18921242698027274}, 'train_mean_delta': -49.36666666666667, 'train_min_delta': -149.0, 'train_positive_rate': 0.1, 'train_bad_rate': 0.041666666666666664}, 'MILK|336|360|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.1307306666666666, -0.0008176666666666633, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.14399999999999982, 1.0, 0.0, 0.75, 0.6700000000000003, 1.1960000000000008, 1.673999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 0.03605628243300866, 0.03220251104943354, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029393876913398138, 1.0, 1.0, 1.0, 0.11392980294900883, 0.22553048574416715, 0.263066531508666], 'beta': [0.0, 1.737669247076075e-28, 1.737669247076075e-28, 0.0, 0.0, 0.0, 0.0, 3.8099931309064097, -15.816627387332229, 25.32261182235459, -18.79580752853666, -19.16580659567511, -14.949450562618885, 1.187044507118406, -19.45735960356332, 1.9617447503355453e-29, 6.950676988304294e-28, 1.5693958002684363e-28, -5.838830689586164, 0.3505225104939264, -5.111264970198615, 7.5699702304461915e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3668865338706317e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 17.82297425274169, 0.0, 0.0, 0.0, -0.13130675994611427, -2.455569659072598, -29.786931910940524], 'intercept': -81.98333333333048, 'uncertainty': 37.12031467765266}, 'bad': {'mean': [0.0, 0.466666666666667, 0.466666666666667, 0.0, 0.1875, 0.28125, 0.33333333333333337, 0.5416111111111109, 0.9990299999999993, 0.004388888888888889, 0.009277777777777777, 0.024166666666666687, -9.999999999999999e-05, -0.00021000000000000012, -0.0005500000000000001, 0.05999999999999985, 0.5200000000000014, 0.4799999999999988, 0.12991299999999997, 0.1307306666666666, -0.0008176666666666633, 0.9000000000000017, 0.5, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.14399999999999982, 1.0, 0.0, 0.75, 0.6700000000000003, 1.1960000000000008, 1.673999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1618990216458288, 0.0037046052421276987, 0.0034689985034998152, 0.018945259024520426, 0.027726341266023538, 7.958224257542218e-05, 0.0003160696125855824, 0.0005463515351859094, 1.0, 1.0, 1.0, 0.026025249937960893, 0.03605628243300866, 0.03220251104943354, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029393876913398138, 1.0, 1.0, 1.0, 0.11392980294900883, 0.22553048574416715, 0.263066531508666], 'beta': [0.0, -1.939218693084495e-31, -1.939218693084495e-31, 0.0, 0.0, 0.0, 0.0, 0.0019067862752596481, 0.062066614242729246, -0.028785326804569304, -0.0035707010864870115, 0.05277412790390252, 0.01904244145713291, -0.022710259521420414, -0.029363187350939002, 2.0000509909059595e-31, -7.756874772337993e-31, 1.6000407927247676e-30, -0.0400791447681669, -0.0015403403101696119, -0.030666267394765145, 9.597735743595976e-32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3288293558327174e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0689945497109729, 0.0, 0.0, 0.0, 0.08724847313367694, 0.07915730479911304, -0.07631395439356342], 'intercept': 0.04999999999998921, 'uncertainty': 0.18974891120871173}, 'train_mean_delta': -81.98333333333333, 'train_min_delta': -223.0, 'train_positive_rate': 0.06666666666666667, 'train_bad_rate': 0.05}, 'MILK|360|377|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.1653933333333333, -0.0139475, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.3119999999999997, 0.9399999999999992, 0.0, 0.75, 0.7293333333333326, 1.1466666666666674, 1.5639999999999976], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 0.03519437529239896, 0.021645715097527576, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12812493902437563, 0.09165151389911674, 1.0, 1.0, 0.0957403897120867, 0.11562391140628683, 0.17036431551237483], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0575294282932697e-28, 9.341061797333502, -13.791019741759932, 11.140342815781791, -8.637588214235343, -21.44594317635358, -19.684612061295557, -1.0201791330196284, 21.462230707211432, -5.64952204832762e-29, 0.8040300425508885, -0.8040300425508102, 1.0330252612233601, 4.629968601505053, -6.324107006166412, -2.2368802837071203e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.714198704871785e-29, 0.0, 0.0, 0.0, 0.0, 0.0, -11.099207126340639, -17.144818030369365, 0.0, 0.0, -7.602878776634023, -0.29163446665962744, 3.8767669130136486], 'intercept': -8.866666666664008, 'uncertainty': 21.824053719849186}, 'bad': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.1653933333333333, -0.0139475, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.3119999999999997, 0.9399999999999992, 0.0, 0.75, 0.7293333333333326, 1.1466666666666674, 1.5639999999999976], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 0.03519437529239896, 0.021645715097527576, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12812493902437563, 0.09165151389911674, 1.0, 1.0, 0.0957403897120867, 0.11562391140628683, 0.17036431551237483], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8719414070179335e-31, -0.061870723727094255, 0.03741202391106292, -0.014026234133276689, -0.009668081877499516, 0.014771038404767914, -0.01722035944778098, 0.0518109114108121, -0.17299445940593786, 3.327186099536033e-31, -0.008326456398732772, 0.008326456398732527, -0.036862050349936074, -0.037696589439229544, 0.01833286705748562, -8.493918475277902e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.01849145759154e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10146683370512696, 0.12023869077659549, 0.0, 0.0, 0.10310077426049483, 0.05830441689073216, 0.021200808092688216], 'intercept': 0.06666666666665282, 'uncertainty': 0.17976168367955417}, 'train_mean_delta': -8.866666666666667, 'train_min_delta': -105.0, 'train_positive_rate': 0.5666666666666667, 'train_bad_rate': 0.06666666666666667}, 'MILK|360|377|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.1653933333333333, -0.0139475, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.3119999999999997, 0.9399999999999992, 0.0, 0.75, 0.7293333333333326, 1.1466666666666674, 1.5639999999999976], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 0.03519437529239896, 0.021645715097527576, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12812493902437563, 0.09165151389911674, 1.0, 1.0, 0.0957403897120867, 0.11562391140628683, 0.17036431551237483], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.6556785161464807e-28, 22.845495479270607, -36.44924494825299, 27.364348797763057, -21.96958670249504, -54.437756476014506, -50.77516098341663, -5.4173505305975445, 56.80374650394313, -1.503476952839023e-28, 2.213565484553526, -2.2135654845533614, 3.2341644867039965, 12.353362151936281, -16.316579070304268, -5.4425660986043265e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.3668132497467944e-28, 0.0, 0.0, 0.0, 0.0, 0.0, -30.733466974010316, -47.03832076913794, 0.0, 0.0, -22.47523133096625, -3.164148544098999, 8.33557743795539], 'intercept': -27.049999999992846, 'uncertainty': 56.609447126275626}, 'bad': {'mean': [0.0, 0.5, 0.5, 0.0, 0.28125, 0.375, 0.23611111111111077, 0.48900000000000016, 0.9998099999999994, 0.0038888888888888875, 0.005333333333333331, -0.05261111111111105, -9.999999999999999e-05, -0.00018, 0.00078, 0.0899999999999999, 0.46025000000000066, 0.5397499999999994, 0.15144583333333328, 0.1653933333333333, -0.0139475, 0.9000000000000017, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.4200000000000011, 1.0, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.3119999999999997, 0.9399999999999992, 0.0, 0.75, 0.7293333333333326, 1.1466666666666674, 1.5639999999999976], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1900714290685167, 0.004053257948860367, 0.0031720227608044893, 0.01784916223783605, 0.04120900506886125, 7.958224257542218e-05, 0.0003501428280002319, 0.00056, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.025225940768163407, 0.03519437529239896, 0.021645715097527576, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12812493902437563, 0.09165151389911674, 1.0, 1.0, 0.0957403897120867, 0.11562391140628683, 0.17036431551237483], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2107001977142197e-31, -0.04204447048479061, 0.02055857624925202, 0.005256026396054548, -0.013876659789322772, 0.03407041224575728, 0.025638387250784864, 0.06097685998533998, -0.17272115001584323, 3.8475857581064125e-31, -0.007800239582809176, 0.007800239582808879, -0.026588589152177385, -0.038922260829903306, 0.03229842380648626, -9.336922225226573e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.323997076689593e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12118250864829097, 0.13750866917154939, 0.0, 0.0, 0.11758102117873721, 0.060437360788259974, 0.015958478747431556], 'intercept': 0.07499999999998863, 'uncertainty': 0.17801008928680032}, 'train_mean_delta': -27.05, 'train_min_delta': -295.0, 'train_positive_rate': 0.5666666666666667, 'train_bad_rate': 0.075}, 'MILK|377|381|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.1752301666666666, -0.018688333333333338, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.3849999999999992, 0.014999999999999996, 0.14400000000000004, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5613333333333337, 0.9786666666666675, 1.3960000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 0.03653953833368573, 0.019785775291579783, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.045000000000000054, 0.045000000000000026, 0.18821264569629742, 0.09165151389911674, 0.05937171043518962, 1.0, 0.1320538610651813, 0.10442647599574019, 0.1257934815481312], 'beta': [0.0, -1.2024556218623632e-28, 0.0, -6.012278109311816e-29, 0.0, 0.0, -1.1836164067727084e-29, -2.060118856357111, -4.960548383331827, -0.3803568268000709, -2.004047750385201, 0.8056161145434385, -0.6662150812748517, -1.5981839132301272, -0.11640891144018611, 2.2276937196877308e-29, 0.3751720432984622, -0.37517204329846277, -0.6405270665501415, 0.683676495356197, -2.1198701308113415, 0.0, 0.0, -2.2434820701883236e-29, 0.0, 0.0, 0.0, 0.0, 9.959529123841382e-30, 0.0, 0.0, 0.0, 4.971550348929419, -4.971550348929351, -5.4602645786838355, 3.94468907492233, -0.5802103658809578, 0.0, -4.523908740638089, -1.600249251603084, 2.092181828099461], 'intercept': -9.516666666667266, 'uncertainty': 9.920570921672384}, 'bad': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.1752301666666666, -0.018688333333333338, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.3849999999999992, 0.014999999999999996, 0.14400000000000004, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5613333333333337, 0.9786666666666675, 1.3960000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 0.03653953833368573, 0.019785775291579783, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.045000000000000054, 0.045000000000000026, 0.18821264569629742, 0.09165151389911674, 0.05937171043518962, 1.0, 0.1320538610651813, 0.10442647599574019, 0.1257934815481312], 'beta': [0.0, 4.041569450898707e-31, 0.0, 2.0207847254493534e-31, 0.0, 0.0, 5.9021477827788e-32, -0.029287543642542047, -0.01539739954381777, -0.008749266029972553, -0.017345808524869135, -0.047905000939347825, -0.01243885357048526, -0.01714972078740427, -0.07232713046112886, 1.86428732037244e-31, -0.00033923568759854604, 0.00033923568759919895, -0.011294551953015467, -0.004555218457278973, -0.00670430727651345, 0.0, 0.0, 1.0960282787821846e-31, 0.0, 0.0, 0.0, 0.0, 7.503241325026734e-32, 0.0, 0.0, 0.0, -0.07159861077826814, 0.07159861077826622, 0.010482673445233097, 0.009837227571411592, -0.0371261658134162, 0.0, 0.00867156025770901, 0.0030380640623656925, -0.004059068557778768], 'intercept': 0.058333333333331655, 'uncertainty': 0.18691319133945872}, 'train_mean_delta': -9.516666666666667, 'train_min_delta': -55.0, 'train_positive_rate': 0.38333333333333336, 'train_bad_rate': 0.058333333333333334}, 'MILK|377|381|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.1752301666666666, -0.018688333333333338, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.3849999999999992, 0.014999999999999996, 0.14400000000000004, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5613333333333337, 0.9786666666666675, 1.3960000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 0.03653953833368573, 0.019785775291579783, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.045000000000000054, 0.045000000000000026, 0.18821264569629742, 0.09165151389911674, 0.05937171043518962, 1.0, 0.1320538610651813, 0.10442647599574019, 0.1257934815481312], 'beta': [0.0, -2.4283066660439373e-28, 0.0, -1.2141533330219686e-28, 0.0, 0.0, -2.3826887206375787e-29, -3.189920857658272, -8.464980949011611, 0.20125832794798212, -4.404912767286749, 3.476870240820059, -0.4011574412592983, -2.1926756570908745, -2.3145965955587537, 4.4464834740650134e-29, 0.7206101998365709, -0.7206101998365605, -0.9290857712520302, 1.1105820376362527, -3.2944699761156246, 0.0, 0.0, -4.6691408842765954e-29, 0.0, 0.0, 0.0, 0.0, 2.1953666045280267e-29, 0.0, 0.0, 0.0, 10.291655674766417, -10.291655674766524, -11.220841045984661, 8.085257419325332, -0.8387376191930038, 0.0, -9.228732269178511, -3.1167888646801805, 4.513263857333509], 'intercept': -19.150000000001004, 'uncertainty': 22.280758860637327}, 'bad': {'mean': [0.0, 0.52361111111111, 0.5, 0.7083333333333328, 0.375, 0.09375, 0.05555555555555543, 0.44044444444444436, 1.0005750000000007, 0.00011111111111110998, -0.0007222222222222223, -0.047055555555555566, -5.0000000000000036e-05, -7.500000000000001e-05, 0.000705, 0.1199999999999997, 0.46025000000000066, 0.5397499999999994, 0.1565418333333332, 0.1752301666666666, -0.018688333333333338, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.25, 0.1799999999999998, 1.0, 0.39999999999999913, 1.0, 0.3849999999999992, 0.014999999999999996, 0.14400000000000004, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5613333333333337, 0.9786666666666675, 1.3960000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2235914484108663, 0.0044384165720070395, 0.018063331658840887, 0.024442613567797773, 0.043119522375329525, 0.0003024896692450836, 0.0004157823950096975, 0.0006346718312114799, 1.0, 0.0015612494995995937, 0.0015612494995996024, 0.02648139544986673, 0.03653953833368573, 0.019785775291579783, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.045000000000000054, 0.045000000000000026, 0.18821264569629742, 0.09165151389911674, 0.05937171043518962, 1.0, 0.1320538610651813, 0.10442647599574019, 0.1257934815481312], 'beta': [0.0, 4.780373302253307e-31, 0.0, 2.3901866511266535e-31, 0.0, 0.0, 6.816115546794013e-32, -0.024930777805624253, -0.018297572241916027, -0.009239478826166183, -0.016077416538690006, -0.06788839254470934, -0.012944465980467577, -0.02052349472531236, -0.07210048989289539, 1.9339431345032539e-31, -0.0010995258329012, 0.0010995258329018478, -0.02155587999731365, -0.008097368786424284, -0.013896633376918667, 0.0, 0.0, 8.900460143529175e-32, 0.0, 0.0, 0.0, 0.0, 7.569876451996842e-32, 0.0, 0.0, 0.0, -0.07980379416138995, 0.07980379416138893, 0.013442304278195412, 0.00924512945343608, -0.038472354584699525, 0.0, 0.011236932756939929, 0.004191935594298985, -0.0048363573864788475], 'intercept': 0.06666666666666415, 'uncertainty': 0.17536853540583452}, 'train_mean_delta': -19.15, 'train_min_delta': -151.0, 'train_positive_rate': 0.38333333333333336, 'train_bad_rate': 0.06666666666666667}, 'MILK|381|406|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.1809701666666667, -0.005397000000000001, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.39999999999999913, 1.0, 0.3899999999999991, 0.01, 0.08400000000000006, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5013333333333341, 0.9186666666666653, 1.3359999999999992], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 0.036077662641014994, 0.018258999087208117, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029999999999999992, 0.030000000000000044, 0.10799999999999998, 0.09165151389911674, 0.05937171043518962, 1.0, 0.06259570450296273, 0.08732060976017567, 0.1509436981129058], 'beta': [0.0, 3.1261406901679665e-29, 0.0, 0.0, 0.0, 0.0, 2.3698157301007975e-30, 0.8212049050626204, -14.355466919260724, 9.34620165937896, -1.336849685006403, -13.466677506977886, -7.34371331977018, 0.09097150978729876, 5.1459613445984616, 5.728662508098683e-30, -0.4667180808935185, 0.46671808089362055, -2.1296868353437586, -2.712245386308917, 1.8210877821919607, 0.0, 0.0, 2.788457535775177e-29, 0.0, 0.0, 0.0, 0.0, 2.2914650032394663e-29, 0.0, 0.0, 0.0, -3.0707064546200002, 3.0707064546199963, -0.8749571003922788, 1.864315442317384, -1.8670924278134458, 0.0, -2.702913475263763, -2.792996753193785, -2.11059878538989], 'intercept': -35.48333333333502, 'uncertainty': 29.101919205835554}, 'bad': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.1809701666666667, -0.005397000000000001, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.39999999999999913, 1.0, 0.3899999999999991, 0.01, 0.08400000000000006, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5013333333333341, 0.9186666666666653, 1.3359999999999992], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 0.036077662641014994, 0.018258999087208117, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029999999999999992, 0.030000000000000044, 0.10799999999999998, 0.09165151389911674, 0.05937171043518962, 1.0, 0.06259570450296273, 0.08732060976017567, 0.1509436981129058], 'beta': [0.0, 4.471753795080407e-31, 0.0, 0.0, 0.0, 0.0, -5.814840337094453e-31, -0.01659361327196327, -0.019169947996801082, 0.015158166494665724, -0.0018601389142890866, 0.008909127393330683, 0.0045337844841980375, 0.00010805243086819083, -0.014747662574076408, 8.275211341437701e-32, 0.0011057620656452808, -0.0011057620656448016, -0.007207388883830839, -0.0018048019333974307, -0.00840737165346304, 0.0, 0.0, 4.074363604455935e-31, 0.0, 0.0, 0.0, 0.0, 3.3100845365750655e-31, 0.0, 0.0, 0.0, -0.07599822027456664, 0.07599822027455447, 0.014271424908243775, 0.010861995881382584, -0.03669464129540427, 0.0, 0.010245360472571666, -0.002962448126117655, -0.007676247002454707], 'intercept': 0.05833333333333383, 'uncertainty': 0.1911338035159438}, 'train_mean_delta': -35.483333333333334, 'train_min_delta': -91.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.058333333333333334}, 'MILK|381|406|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.1809701666666667, -0.005397000000000001, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.39999999999999913, 1.0, 0.3899999999999991, 0.01, 0.08400000000000006, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5013333333333341, 0.9186666666666653, 1.3359999999999992], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 0.036077662641014994, 0.018258999087208117, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029999999999999992, 0.030000000000000044, 0.10799999999999998, 0.09165151389911674, 0.05937171043518962, 1.0, 0.06259570450296273, 0.08732060976017567, 0.1509436981129058], 'beta': [0.0, 9.146131336079983e-30, 0.0, 0.0, 0.0, 0.0, 4.8551682100409506e-29, 2.05162853914773, -26.06399747497245, 22.83860946128496, -2.240175385214696, -21.67633849600502, -17.568452820628302, 0.2030870271089967, 9.949651959925369, 5.463776251587191e-30, -1.0143724646436842, 1.0143724646438739, -7.250058214887657, -6.9464224807448804, 1.6809882264232572, 0.0, 0.0, -2.6550649800094203e-31, 0.0, 0.0, 0.0, 0.0, 2.1855105006348698e-29, 0.0, 0.0, 0.0, -3.198920130717288, 3.1989201307183204, -2.020568142111824, 3.031250929661537, 0.008762822067932463, 0.0, -4.461845167144905, -3.897857846251076, -2.659496959358513], 'intercept': -76.05000000000321, 'uncertainty': 35.793590184455304}, 'bad': {'mean': [0.0, 0.5291666666666672, 0.5, 0.875, 0.09375, 0.65625, 0.3472222222222216, 0.36605555555555563, 1.0018700000000005, -0.07377777777777776, -0.07427777777777785, -0.12294444444444447, 0.0012700000000000012, 0.0012450000000000015, 0.002060000000000002, 0.029999999999999926, 0.38025000000000053, 0.6197499999999997, 0.1755731666666665, 0.1809701666666667, -0.005397000000000001, 0.3000000000000001, 0.625, 0.45000000000000084, 0.75, 0.39999999999999913, 1.0, 0.3000000000000001, 0.1199999999999997, 1.0, 0.39999999999999913, 1.0, 0.3899999999999991, 0.01, 0.08400000000000006, 0.9399999999999992, 0.5850000000000007, 0.75, 0.5013333333333341, 0.9186666666666653, 1.3359999999999992], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24981807578246673, 0.004483425030041211, 0.03906531409616594, 0.04267487623565412, 0.07101927663895902, 0.0004208325082500162, 0.0004769433928675392, 0.000785748051222527, 1.0, 0.0015612494995996026, 0.001561249499599603, 0.030333208067049037, 0.036077662641014994, 0.018258999087208117, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.029999999999999992, 0.030000000000000044, 0.10799999999999998, 0.09165151389911674, 0.05937171043518962, 1.0, 0.06259570450296273, 0.08732060976017567, 0.1509436981129058], 'beta': [0.0, 5.483219644040132e-31, 0.0, 0.0, 0.0, 0.0, -6.534773544103181e-31, -0.015190378573879386, -0.024895419571256985, 0.005689918420388145, -0.0037660053603726785, -0.0027483930839852628, 0.009554786342879071, -0.0007931014592627215, -0.015061278310403119, 9.489944506548377e-32, 0.0014689651999152177, -0.00146896519991478, -0.0006897159683998151, 0.0021650269177901286, -0.005423649360645945, 0.0, 0.0, 5.056152541488497e-31, 0.0, 0.0, 0.0, 0.0, 3.795977802619334e-31, 0.0, 0.0, 0.0, -0.08098159088616486, 0.08098159088615199, 0.014074527927782875, 0.012686535809587915, -0.04424171682984837, 0.0, 0.008278601455673041, -0.0055386607662803785, -0.009841307585023612], 'intercept': 0.06666666666666682, 'uncertainty': 0.184711815291006}, 'train_mean_delta': -76.05, 'train_min_delta': -178.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.06666666666666667}, 'MILK|406|408|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.20447491666666656, 0.023630749999999996, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.955, 0.53, 0.75, 0.4133333333333333, 0.8266666666666665, 1.2399999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 0.046982263905219826, 0.025335630219202498, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1350000000000004, 0.09000000000000008, 1.0, 0.0689605362185906, 0.1379210724371812, 0.20688160865577182], 'beta': [0.0, 0.0, -1.6005417501439907e-29, 1.0386728958077299e-28, 0.0, 0.0, -1.789852035092776e-30, -27.380946229643545, 3.876154973840882, -13.262577932137836, -21.57475253895031, 3.7011973993999705, -9.68645151595737, -16.839725926194916, -4.210507252343031, 1.6005417501440094e-29, 1.8342661269879366, -1.8342661269880147, 4.144914370105985, 2.937757097788244, 1.6529442347770682, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.887677591444871e-29, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5701324170786266, -7.435869687763709, 0.0, 1.3005914655053161, 1.3005914655053157, 1.3005914655052542], 'intercept': -33.11666666666746, 'uncertainty': 31.091802768805096}, 'bad': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.20447491666666656, 0.023630749999999996, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.955, 0.53, 0.75, 0.4133333333333333, 0.8266666666666665, 1.2399999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 0.046982263905219826, 0.025335630219202498, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1350000000000004, 0.09000000000000008, 1.0, 0.0689605362185906, 0.1379210724371812, 0.20688160865577182], 'beta': [0.0, 0.0, -1.0101372915367474e-31, -5.783752792699194e-31, 0.0, 0.0, 6.57221365097693e-32, 0.0022292813598851144, -0.015943200289244647, 0.01785977796426373, 0.00715456576593195, -0.04939337426777418, 0.0005189006585417503, 0.0027451494118353233, -0.07630793752528488, 1.0101372915367494e-31, 0.002334411471882722, -0.0023344114718826434, 0.015530819764426822, 0.0036446626927446714, 0.019847402920327686, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5144544857669407e-31, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03262875325359901, 0.003992111960284477, 0.0, -0.029029689665242196, -0.02902968966524219, -0.02902968966524028], 'intercept': 0.04166666666666619, 'uncertainty': 0.1771164333581049}, 'train_mean_delta': -33.11666666666667, 'train_min_delta': -122.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.041666666666666664}, 'MILK|406|408|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.20447491666666656, 0.023630749999999996, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.955, 0.53, 0.75, 0.4133333333333333, 0.8266666666666665, 1.2399999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 0.046982263905219826, 0.025335630219202498, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1350000000000004, 0.09000000000000008, 1.0, 0.0689605362185906, 0.1379210724371812, 0.20688160865577182], 'beta': [0.0, 0.0, -5.955412196503438e-29, 3.4637247053316087e-28, 0.0, 0.0, -3.2266788851559405e-30, -86.16952475391471, 6.288338084483963, -42.43723358368526, -62.96009754840414, 8.158592144409724, -20.706830084009646, -43.919433173678286, -26.445366835552882, 5.955412196503483e-29, 4.600052665949097, -4.600052665949132, 10.586113316358533, 13.343643058197763, -6.609177122848469, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1239322519021228e-28, 0.0, 0.0, 0.0, 0.0, 0.0, -3.205610908496977, -33.01246683157258, 0.0, 7.5041523775695165, 7.504152377569514, 7.504152377569476], 'intercept': -109.98333333333534, 'uncertainty': 95.19065746140414}, 'bad': {'mean': [0.0, 0.5638888888888893, 0.5333333333333328, 0.9166666666666681, 0.65625, 0.09375, 0.027777777777777714, 0.312611111111111, 1.0026716666666666, 0.006166666666666665, 0.01144444444444445, -0.03422222222222223, -0.00014000000000000004, -0.0002550000000000005, 0.0004766666666666664, 0.21000000000000055, 0.37024999999999947, 0.6297500000000001, 0.22810566666666668, 0.20447491666666656, 0.023630749999999996, 0.19999999999999957, 0.625, 0.6500000000000002, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.5499999999999988, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.0, 0.955, 0.53, 0.75, 0.4133333333333333, 0.8266666666666665, 1.2399999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2729639045915716, 0.004811101456238714, 0.00900257164905097, 0.01172788173015669, 0.04082422422962397, 0.00018814887722226782, 0.0002616772821625905, 0.0006950699405255729, 1.0, 0.001561249499599601, 0.0015612494995996032, 0.043402793664565974, 0.046982263905219826, 0.025335630219202498, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1350000000000004, 0.09000000000000008, 1.0, 0.0689605362185906, 0.1379210724371812, 0.20688160865577182], 'beta': [0.0, 0.0, -1.0101372915367474e-31, -5.783752792699194e-31, 0.0, 0.0, 6.57221365097693e-32, 0.0022292813598851144, -0.015943200289244647, 0.01785977796426373, 0.00715456576593195, -0.04939337426777418, 0.0005189006585417503, 0.0027451494118353233, -0.07630793752528488, 1.0101372915367494e-31, 0.002334411471882722, -0.0023344114718826434, 0.015530819764426822, 0.0036446626927446714, 0.019847402920327686, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5144544857669407e-31, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03262875325359901, 0.003992111960284477, 0.0, -0.029029689665242196, -0.02902968966524219, -0.02902968966524028], 'intercept': 0.04166666666666619, 'uncertainty': 0.1771164333581049}, 'train_mean_delta': -109.98333333333333, 'train_min_delta': -415.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.041666666666666664}, 'MILK|408|432|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.2067549166666667, 0.037582749999999984, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2740000000000001, 0.81, 0.0, 0.75, 0.6440000000000002, 1.0139999999999982, 1.3840000000000037], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 0.05040891374541864, 0.026884527519327905, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14312232530251875, 0.23430749027719944, 1.0, 1.0, 0.062481997407253145, 0.1600124995117567, 0.2925474320516245], 'beta': [0.0, 5.931574165928113e-29, 5.931574165928113e-29, 0.0, 0.0, 0.0, 0.0, -12.0324443010833, 1.5493569122857502, -7.5873362219148195, 17.114039834298648, 1.8468306462684887, 23.3258760315743, -1.727927691763433, -10.7796830968664, -6.90642851319997e-30, -4.089385206959888, 4.089385206959997, -3.7786310353529835, -5.576139652870994, 3.2260433225602014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.27795569874197e-29, 0.0, 0.0, 0.0, 0.0, 0.0, -6.242424588144312, 0.5974379665069229, 0.0, 0.0, -1.445124010597046, 4.454913556061762, 5.181990251906985], 'intercept': -6.700000000000012, 'uncertainty': 36.66447291887198}, 'bad': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.2067549166666667, 0.037582749999999984, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2740000000000001, 0.81, 0.0, 0.75, 0.6440000000000002, 1.0139999999999982, 1.3840000000000037], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 0.05040891374541864, 0.026884527519327905, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14312232530251875, 0.23430749027719944, 1.0, 1.0, 0.062481997407253145, 0.1600124995117567, 0.2925474320516245], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -6.7, 'train_min_delta': -90.0, 'train_positive_rate': 0.18333333333333332, 'train_bad_rate': 0.0}, 'MILK|408|432|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.2067549166666667, 0.037582749999999984, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2740000000000001, 0.81, 0.0, 0.75, 0.6440000000000002, 1.0139999999999982, 1.3840000000000037], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 0.05040891374541864, 0.026884527519327905, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14312232530251875, 0.23430749027719944, 1.0, 1.0, 0.062481997407253145, 0.1600124995117567, 0.2925474320516245], 'beta': [0.0, 3.1098370940673845e-28, 3.1098370940673845e-28, 0.0, 0.0, 0.0, 0.0, -6.122500877562848, -5.468910114071121, -5.024942962261264, 7.560477920318026, -4.1971742456391965, 11.274617343148323, -3.452365041471858, -9.145767120645184, -1.2710008095991635e-29, -8.795387183210858, 8.795387183211222, -6.525897579772065, -12.769324475325416, 11.457286531863582, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.935811410603515e-29, 0.0, 0.0, 0.0, 0.0, 0.0, -11.680397611004395, 10.796024922822898, 0.0, 0.0, -10.310014854955712, 2.395719239700262, 4.822740568554965], 'intercept': -12.133333333329574, 'uncertainty': 51.071392440748305}, 'bad': {'mean': [0.0, 0.5666666666666679, 0.5666666666666679, 0.0, 0.09375, 0.28125, 0.33333333333333337, 0.20722222222222234, 1.0044616666666648, -0.10450000000000002, -0.09394444444444444, -0.13961111111111102, 0.0017650000000000012, 0.0015349999999999988, 0.002266666666666668, 0.029999999999999926, 0.6134166666666662, 0.38658333333333367, 0.2443376666666666, 0.2067549166666667, 0.037582749999999984, 1.0, 0.625, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.21999999999999972, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2740000000000001, 0.81, 0.0, 0.75, 0.6440000000000002, 1.0139999999999982, 1.3840000000000037], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26762755122154275, 0.004300817428763495, 0.060890992461635596, 0.055795564892994795, 0.07187461218360211, 0.0008588412736549949, 0.0007540612265150179, 0.0008846217019469718, 1.0, 0.005243382072246461, 0.005243382072246468, 0.05143568940488264, 0.05040891374541864, 0.026884527519327905, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.14312232530251875, 0.23430749027719944, 1.0, 1.0, 0.062481997407253145, 0.1600124995117567, 0.2925474320516245], 'beta': [0.0, -8.12912649097954e-32, -8.12912649097954e-32, 0.0, 0.0, 0.0, 0.0, 0.0045994038745759205, -0.0010736390068196656, -0.006678210320723005, -0.030478627553707595, -0.00351935981625799, -0.011782418968848572, 0.003214467034048493, -0.014917260736954718, -1.0652666606576141e-32, 0.008942324219757675, -0.00894232421975796, 0.00953336090264542, 0.008597160656496712, 0.0021194890013713423, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2900838566263953e-33, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01080754373290026, -0.00915149671858563, 0.0, 0.0, -0.01563036047803844, -0.0025399984195306267, 0.0005597490635442634], 'intercept': 0.008333333333332348, 'uncertainty': 0.10041590008560844}, 'train_mean_delta': -12.133333333333333, 'train_min_delta': -174.0, 'train_positive_rate': 0.21666666666666667, 'train_bad_rate': 0.008333333333333333}, 'MILK|432|455|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.24073041666666678, 0.01675966666666667, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2080000000000003, 0.8700000000000009, 0.0, 0.75, 0.5840000000000007, 0.9600000000000011, 1.336000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 0.05476082751985881, 0.023900734819015276, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.07493997598078075, 0.16155494421403488, 1.0, 1.0, 0.13499629624548967, 0.2152208168370335, 0.29983995731056273], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.192353452752076e-28, -1.8995205849470678, -1.9397487509931233, 13.05980952226862, -16.091658903615517, 12.426421956321219, 6.382230087433733, 12.925508104979237, -13.807928922442004, -3.990441815940087e-29, 2.2582629731186112, -2.258262973118646, 1.3298935416603561, -0.7402198574196377, 4.615278485113334, 3.725433585097812e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9664853660035575, -5.040293451636525, 0.0, 0.0, 1.5749665947824134, 2.312312124077429, 2.610395086321441], 'intercept': -15.183333333332422, 'uncertainty': 22.25930754951009}, 'bad': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.24073041666666678, 0.01675966666666667, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2080000000000003, 0.8700000000000009, 0.0, 0.75, 0.5840000000000007, 0.9600000000000011, 1.336000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 0.05476082751985881, 0.023900734819015276, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.07493997598078075, 0.16155494421403488, 1.0, 1.0, 0.13499629624548967, 0.2152208168370335, 0.29983995731056273], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -15.183333333333334, 'train_min_delta': -99.0, 'train_positive_rate': 0.16666666666666666, 'train_bad_rate': 0.0}, 'MILK|432|455|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.24073041666666678, 0.01675966666666667, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2080000000000003, 0.8700000000000009, 0.0, 0.75, 0.5840000000000007, 0.9600000000000011, 1.336000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 0.05476082751985881, 0.023900734819015276, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.07493997598078075, 0.16155494421403488, 1.0, 1.0, 0.13499629624548967, 0.2152208168370335, 0.29983995731056273], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.242048205816247e-28, -5.736693989590415, -6.142170257832147, 34.43686339612167, -41.04669158380871, 27.030461638422864, 16.641603108550036, 32.053576320136806, -33.194488578377914, -1.0302560257270284e-28, 5.9299249215669105, -5.929924921566818, 2.99541732620704, -3.3684465616224926, 14.293071493138182, 9.623850203717032e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.273011438509639, -13.673820338946781, 0.0, 0.0, 3.7189086915549034, 5.8050016528049895, 6.659137472498635], 'intercept': -40.38333333333068, 'uncertainty': 58.12117881511}, 'bad': {'mean': [0.0, 0.6000000000000002, 0.6000000000000002, 0.0, 0.28125, 0.28125, 0.31944444444444364, 0.2009444444444444, 1.004509999999999, -0.008055555555555557, -0.00894444444444444, -0.006277777777777777, 0.00010333333333333327, 6.16666666666665e-05, 4.833333333333335e-05, 0.0899999999999999, 0.7829166666666675, 0.21708333333333332, 0.2574900833333333, 0.24073041666666678, 0.01675966666666667, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.2080000000000003, 0.8700000000000009, 0.0, 0.75, 0.5840000000000007, 0.9600000000000011, 1.336000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.26038155170659777, 0.0042981274992722265, 0.02080145709521212, 0.0359603262183256, 0.02740702577311875, 0.0003114303917232372, 0.000545310818606131, 0.0004591628856468646, 1.0, 0.00454529671443155, 0.004545296714431555, 0.05246547416310773, 0.05476082751985881, 0.023900734819015276, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.07493997598078075, 0.16155494421403488, 1.0, 1.0, 0.13499629624548967, 0.2152208168370335, 0.29983995731056273], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5573043691007853e-31, 0.010484586572791242, 0.009553123561128347, 0.001181925213391028, 0.01703577849641596, 0.024525846287341134, -0.0030582187114742737, 0.002670424483359562, 0.020774231725579805, 3.196630461376026e-32, 0.012596144959890533, -0.012596144959890913, 0.005779755726776645, 0.010525016768421555, -0.011427305696458822, -1.0756880633452169e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.018908296475525926, -0.018665267879692603, 0.0, 0.0, 0.0018845418906550152, -0.004219735555066222, -0.006906204429612917], 'intercept': 0.008333333333332948, 'uncertainty': 0.10309665991769563}, 'train_mean_delta': -40.38333333333333, 'train_min_delta': -241.0, 'train_positive_rate': 0.18333333333333332, 'train_bad_rate': 0.008333333333333333}, 'MILK|455|456|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.28889674999999987, 0.02167349999999998, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.003999999999999999, 0.8700000000000009, 0.5650000000000005, 0.75, 0.3800000000000007, 0.7560000000000014, 1.1320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 0.06337634481364085, 0.026385397939327475, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.011999999999999953, 0.16155494421403488, 0.045000000000000054, 1.0, 0.08246211251235319, 0.170129362545094, 0.2580232547659224], 'beta': [0.0, 2.3936174932167517e-30, 0.0, 2.2305933703247922e-29, 0.0, 0.0, 9.61467707450994e-31, -0.012687111486949097, 0.5205759606439125, 0.04986931071947021, -1.512900534169606, 1.3452793124790403, -1.1484993256531904, 0.21100040262278716, 1.1462385747724, 1.3941208564529957e-30, -0.3101592451781685, 0.3101592451781676, -0.8058673354029997, -0.3331600020654306, -0.7961600946357538, 0.0, 0.0, 2.3936174932167556e-30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.300283974059723, 0.22226165611087065, -0.07463061979824986, 0.0, -0.2755498077128068, -0.10486979165218255, -0.0502297452740703], 'intercept': -1.1833333333333995, 'uncertainty': 5.34122043422796}, 'bad': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.28889674999999987, 0.02167349999999998, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.003999999999999999, 0.8700000000000009, 0.5650000000000005, 0.75, 0.3800000000000007, 0.7560000000000014, 1.1320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 0.06337634481364085, 0.026385397939327475, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.011999999999999953, 0.16155494421403488, 0.045000000000000054, 1.0, 0.08246211251235319, 0.170129362545094, 0.2580232547659224], 'beta': [0.0, -1.009244852958323e-31, 0.0, -4.663585927819741e-31, 0.0, 0.0, -3.279526165401613e-32, -0.015097335332454068, -0.01958358940553958, -0.035286122302955934, -0.06259261144264451, -0.023201774537786088, -0.02590114941994835, -0.08201291741500011, -0.033201683435243795, -2.914741204887333e-32, -0.00314681120514474, 0.003146811205143753, 0.015603860873817075, 0.012748648222261318, 0.0002890776916994416, 0.0, 0.0, -1.009244852958326e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04324035669550452, -0.004684957685267682, -0.005700249360684315, 0.0, 0.006850060071885201, 0.003590541686890056, 0.0025456725324857076], 'intercept': 0.016666666666669137, 'uncertainty': 0.11458499587037359}, 'train_mean_delta': -1.1833333333333333, 'train_min_delta': -34.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.016666666666666666}, 'MILK|455|456|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.28889674999999987, 0.02167349999999998, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.003999999999999999, 0.8700000000000009, 0.5650000000000005, 0.75, 0.3800000000000007, 0.7560000000000014, 1.1320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 0.06337634481364085, 0.026385397939327475, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.011999999999999953, 0.16155494421403488, 0.045000000000000054, 1.0, 0.08246211251235319, 0.170129362545094, 0.2580232547659224], 'beta': [0.0, -1.5018682925214977e-28, 0.0, 1.593545810103511e-28, 0.0, 0.0, 3.073977500104218e-30, -8.7835064180884, -2.0410292589225962, -13.027897786169383, -18.368824054225556, -13.551361222875734, -24.15164060575753, -12.107389820965311, -9.231846460162656, 9.959661313146926e-30, -1.0691764199437228, 1.069176419943561, -10.139993799042907, -5.320014926548252, -7.308550196516186, 0.0, 0.0, -1.5018682925214968e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -13.746835214426142, 3.3473246441953672, 2.8208272514262798, 0.0, -3.4810625307999015, -2.4049318204734744, -2.0588968446576352], 'intercept': -11.0666666666664, 'uncertainty': 20.04295688034123}, 'bad': {'mean': [0.0, 0.6319444444444432, 0.6000000000000002, 0.9583333333333318, 0.28125, 0.09375, 0.013888888888888857, 0.1777222222222223, 1.0048383333333328, -0.019166666666666676, -0.0037222222222222257, -0.026611111111111085, 0.00029166666666666696, 1.666666666666663e-05, 0.00038333333333333275, 0.0899999999999999, 0.3934166666666667, 0.6065833333333334, 0.31057025, 0.28889674999999987, 0.02167349999999998, 0.19999999999999957, 0.75, 0.5499999999999988, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.04000000000000003, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.003999999999999999, 0.8700000000000009, 0.5650000000000005, 0.75, 0.3800000000000007, 0.7560000000000014, 1.1320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2504355403622721, 0.004238360204397729, 0.02462778467052563, 0.022119973098458867, 0.047205193007905176, 0.00039845186520944925, 0.0004054489966554223, 0.0007158134455910203, 1.0, 0.005243382072246455, 0.005243382072246458, 0.05226848420196278, 0.06337634481364085, 0.026385397939327475, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.011999999999999953, 0.16155494421403488, 0.045000000000000054, 1.0, 0.08246211251235319, 0.170129362545094, 0.2580232547659224], 'beta': [0.0, 6.645073860847212e-31, 0.0, -6.666703370397627e-31, 0.0, 0.0, -5.31427967447994e-32, -0.01416637566070182, 0.0003878787670449654, -0.06664185405132642, 0.02572918723884277, -0.016423064854054245, -0.006272298954886314, -0.0719584718807031, -0.027848945172217095, -4.166689606498503e-32, 0.0018275339598547414, -0.0018275339598555307, 0.024665033969002215, 0.012391175185671323, 0.01909755343826943, 0.0, 0.0, 6.645073860847212e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11057504944644436, -0.006463317356939238, -0.023283766680530296, 0.0, 0.00893577311791304, 0.000863018945786068, -0.0017177250260423907], 'intercept': 0.041666666666666394, 'uncertainty': 0.17143507624926646}, 'train_mean_delta': -11.066666666666666, 'train_min_delta': -136.0, 'train_positive_rate': 0.1, 'train_bad_rate': 0.041666666666666664}, 'MILK|456|480|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.29013008333333307, 0.025207666666666653, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.37000000000000016, 0.5549999999999997, 0.0, 0.75, 0.592000000000001, 0.8140000000000017, 1.0359999999999971], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 0.06456831070651498, 0.028503362016124023, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.040249223594996254, 0.04153311931459039, 1.0, 1.0, 0.0391918358845308, 0.0447660585711988, 0.05499090833947012], 'beta': [0.0, 6.863555753086766e-30, 6.863555753086766e-30, 0.0, 0.0, 0.0, 0.0, -0.03264946234553206, 1.479848984066106, -3.146864869463106, 6.357774786896097, 6.322426006226553, 3.655483552131366, -4.777740864204706, -3.3561755001763345, -1.0302195999079775e-30, -0.4033423652636114, 0.4033423652636479, 0.1479880871595862, 0.918158035192889, -1.7828404262281834, -9.070318836548825e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38074116574544103, 0.05819873572494999, 0.0, 0.0, 0.4156836738609014, 0.3855214618083853, 0.3314210822122325], 'intercept': -9.316666666667118, 'uncertainty': 14.739647668298915}, 'bad': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.29013008333333307, 0.025207666666666653, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.37000000000000016, 0.5549999999999997, 0.0, 0.75, 0.592000000000001, 0.8140000000000017, 1.0359999999999971], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 0.06456831070651498, 0.028503362016124023, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.040249223594996254, 0.04153311931459039, 1.0, 1.0, 0.0391918358845308, 0.0447660585711988, 0.05499090833947012], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -9.316666666666666, 'train_min_delta': -111.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.0}, 'MILK|456|480|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.29013008333333307, 0.025207666666666653, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.37000000000000016, 0.5549999999999997, 0.0, 0.75, 0.592000000000001, 0.8140000000000017, 1.0359999999999971], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 0.06456831070651498, 0.028503362016124023, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.040249223594996254, 0.04153311931459039, 1.0, 1.0, 0.0391918358845308, 0.0447660585711988, 0.05499090833947012], 'beta': [0.0, -1.9603815062088512e-29, -1.9603815062088512e-29, 0.0, 0.0, 0.0, 0.0, -1.2369021891304688, 3.0997752712551816, -0.2921465272639253, 13.18247595681941, 10.276062828172508, 6.02309481754621, -6.326218781379252, -3.075074876403246, 3.532919867722071e-31, -0.3953851301167126, 0.39538513011676313, 0.31625998520002563, 1.2730737796282479, -2.249060097828548, -5.997321586791395e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3360246329351513, -2.0555436704725865, 0.0, 0.0, 0.5007343373351567, -0.3244547030007332, -0.8851247581980847], 'intercept': -17.350000000000936, 'uncertainty': 18.386702402522044}, 'bad': {'mean': [0.0, 0.6333333333333341, 0.6333333333333341, 0.0, 0.09375, 0.4375, 0.33333333333333337, 0.1450000000000002, 1.0053949999999987, -0.05188888888888886, -0.03527777777777782, -0.05594444444444444, 0.000848333333333334, 0.0005500000000000001, 0.0008850000000000003, 0.029999999999999926, 0.6873333333333328, 0.3126666666666667, 0.3153377499999998, 0.29013008333333307, 0.025207666666666653, 0.9000000000000017, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.28000000000000047, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.37000000000000016, 0.5549999999999997, 0.0, 0.75, 0.592000000000001, 0.8140000000000017, 1.0359999999999971], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24321191795846883, 0.003971205736297256, 0.04291406860169072, 0.035928383841081685, 0.06643875393692568, 0.0006286206239767264, 0.0005220153254455273, 0.0009505831543496515, 1.0, 0.004955356249106123, 0.004955356249106174, 0.057213843159712376, 0.06456831070651498, 0.028503362016124023, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.040249223594996254, 0.04153311931459039, 1.0, 1.0, 0.0391918358845308, 0.0447660585711988, 0.05499090833947012], 'beta': [0.0, 2.938961845534246e-31, 2.938961845534246e-31, 0.0, 0.0, 0.0, 0.0, -0.015910365267036285, -0.05649365599836602, 0.015779951837051116, 0.026793994707607453, -0.007161735444501596, 0.03879933553867282, -0.016909524044408195, 0.022288897231300155, -1.670474209129182e-32, -0.006130844063173786, 0.006130844063174524, -0.011227631010207299, 0.0017688670227308776, -0.0265438398027332, 7.800449345607304e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.025205826187346275, 0.024802736055655118, 0.0, 0.0, -0.015372102942945406, -0.004253377300709181, 0.004030612835742923], 'intercept': 0.025000000000018497, 'uncertainty': 0.165620516398663}, 'train_mean_delta': -17.35, 'train_min_delta': -122.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.025}, 'MILK|480|502|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.33676049999999996, 0.03803083333333332, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.19800000000000004, 0.6300000000000014, 0.0, 0.75, 0.4500000000000007, 0.7019999999999982, 0.9539999999999994], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 0.0739597808592616, 0.031009911791321745, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.03841874542459708, 0.045825756949558406, 1.0, 1.0, 0.030000000000000002, 0.03155946767611899, 0.041999999999999996], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.086702487223886e-28, 185.89545574174667, 76.81352120261108, -40.598188384623334, 24.379794099242115, 209.19447604777653, -67.6405595653089, 127.37511605084308, 117.8433030429044, 0.0, 9.427025916286562, -9.427025916286775, 99.76312805926794, 59.37800701619507, 61.75711615816558, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.093603051852741e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 45.11733594960747, -39.36714273565163, 0.0, 0.0, 33.72465993876687, 9.193062291782434, -10.27341651616692], 'intercept': 91.51666666663807, 'uncertainty': 301.8260872740526}, 'bad': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.33676049999999996, 0.03803083333333332, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.19800000000000004, 0.6300000000000014, 0.0, 0.75, 0.4500000000000007, 0.7019999999999982, 0.9539999999999994], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 0.0739597808592616, 0.031009911791321745, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.03841874542459708, 0.045825756949558406, 1.0, 1.0, 0.030000000000000002, 0.03155946767611899, 0.041999999999999996], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.04204822111008e-31, 0.005530617647086164, 0.004439281025631178, -0.008138274589709622, 0.027948075086622448, 0.010106441981535527, -0.0018076660175202588, -0.0049564084995128805, 0.023971784402575583, 0.0, -0.00797761766868704, 0.00797761766868694, 0.0017547579581129568, 0.0037235230751640714, -0.005303512240660504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2906313165867556e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0035164057545172276, -0.0018672829111802591, 0.0, 0.0, 0.0033622678784950995, 0.0021115747547213806, 0.0007717217638049026], 'intercept': 0.008333333333331965, 'uncertainty': 0.10625798821376128}, 'train_mean_delta': 91.51666666666667, 'train_min_delta': -131.0, 'train_positive_rate': 0.2833333333333333, 'train_bad_rate': 0.008333333333333333}, 'MILK|480|502|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.33676049999999996, 0.03803083333333332, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.19800000000000004, 0.6300000000000014, 0.0, 0.75, 0.4500000000000007, 0.7019999999999982, 0.9539999999999994], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 0.0739597808592616, 0.031009911791321745, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.03841874542459708, 0.045825756949558406, 1.0, 1.0, 0.030000000000000002, 0.03155946767611899, 0.041999999999999996], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0851368845695484e-28, 202.77690315421077, 67.11486287455595, -47.150444425766125, 4.458979951996378, 189.73785828306157, -73.12465177719392, 122.81718322301718, 132.67034087403573, 0.0, 14.058643001954923, -14.0586430019546, 95.59809822291973, 58.47148593431902, 55.42842527722758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.97264064889509e-28, 0.0, 0.0, 0.0, 0.0, 0.0, 46.85600815739978, -40.568005924569434, 0.0, 0.0, 35.21750724136318, 9.9146597997511, -10.255296340502765], 'intercept': 110.06666666664134, 'uncertainty': 299.3080017626095}, 'bad': {'mean': [0.0, 0.6666666666666667, 0.6666666666666667, 0.0, 0.4375, 0.28125, 0.30555555555555636, 0.14916666666666686, 1.0052666666666652, -0.022166666666666668, -0.016333333333333335, 0.004166666666666667, 0.0003500000000000002, 0.00023166666666666675, -0.00012833333333333322, 0.14000000000000024, 0.7468333333333333, 0.2531666666666669, 0.37479133333333325, 0.33676049999999996, 0.03803083333333332, 1.0, 0.75, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.6000000000000002, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.19800000000000004, 0.6300000000000014, 0.0, 0.75, 0.4500000000000007, 0.7019999999999982, 0.9539999999999994], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2465049206931294, 0.00418583590483697, 0.022067447116107985, 0.021914818257500076, 0.027111737242314374, 0.0003427827300200523, 0.00037125537781364937, 0.0004895888297563805, 1.0, 0.0046517619123176215, 0.00465176191231762, 0.06321641305775652, 0.0739597808592616, 0.031009911791321745, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.03841874542459708, 0.045825756949558406, 1.0, 1.0, 0.030000000000000002, 0.03155946767611899, 0.041999999999999996], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7773853340157366e-31, 0.005528624304443495, 0.012537312568660822, -0.008847327876849857, 0.03991624083814579, 0.06247257652165675, 0.006119510806162007, 0.014984147128057542, 0.004495011915515676, 0.0, -0.002337463744960824, 0.0023374637449605842, -0.009241232436213786, -0.005761175951424316, -0.0050984426223256996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.287386746084601e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.020872640112778288, -0.013918647245179478, 0.0, 0.0, 0.018225587618077913, 0.009240796240468818, 0.0008690855229275871], 'intercept': 0.02499999999999513, 'uncertainty': 0.16391368806081827}, 'train_mean_delta': 110.06666666666666, 'train_min_delta': -176.0, 'train_positive_rate': 0.3333333333333333, 'train_bad_rate': 0.025}, 'MILK|502|504|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.3723580833333333, 0.08961041666666665, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.018000000000000013, 0.6300000000000014, 0.6700000000000014, 0.75, 0.26999999999999985, 0.521999999999999, 0.774000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 0.0731826475937185, 0.03885863309046937, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.02749545416973506, 0.045825756949558406, 0.045825756949558406, 1.0, 0.045825756949558406, 0.06415605972938174, 0.08248636250920516], 'beta': [0.0, 4.029785258084554e-30, 0.0, 5.899091823717068e-30, 0.0, 0.0, -1.8812924641620387e-30, -3.697496347989802, 1.3829840053513296, -5.501168189641359, -4.458581923025653, 0.5444912931071977, -2.023771655227283, -2.0237716552273284, -0.6716509163503065, -1.7609854027007153e-30, -0.42515179803782815, 0.4251517980378587, 1.8218363485878382, 1.1664903259276653, 1.0146416513418217, 0.0, 0.0, 2.817576644321111e-29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.232760640191089, -1.2327606401908542, -3.3869417820390026, 0.0, -1.2327606401908129, -1.2327606401909499, -1.232760640190741], 'intercept': -11.61666666666685, 'uncertainty': 13.595976404317414}, 'bad': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.3723580833333333, 0.08961041666666665, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.018000000000000013, 0.6300000000000014, 0.6700000000000014, 0.75, 0.26999999999999985, 0.521999999999999, 0.774000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 0.0731826475937185, 0.03885863309046937, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.02749545416973506, 0.045825756949558406, 0.045825756949558406, 1.0, 0.045825756949558406, 0.06415605972938174, 0.08248636250920516], 'beta': [0.0, -3.7267206745993306e-32, 0.0, -1.0420443248734776e-31, 0.0, 0.0, -1.8347442924926132e-32, -0.012133689370540532, -0.02794432097479, -0.026968677770655484, -0.025116477146093847, 0.027252476909277762, -0.025664659960078075, -0.025664659960078842, 0.01443681151594669, -1.6368310238925584e-32, -0.0022130803196394495, 0.002213080319639807, 0.0009418243258261799, 0.01540840348664105, -0.027358489228787686, 0.0, 0.0, 2.6189296382281075e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0023587294882785707, -0.0023587294882792954, 0.01348292825264769, 0.0, -0.0023587294882790894, -0.002358729488278889, -0.00235872948827914], 'intercept': 0.016666666666666656, 'uncertainty': 0.13021443900844926}, 'train_mean_delta': -11.616666666666667, 'train_min_delta': -52.0, 'train_positive_rate': 0.0, 'train_bad_rate': 0.016666666666666666}, 'MILK|502|504|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.3723580833333333, 0.08961041666666665, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.018000000000000013, 0.6300000000000014, 0.6700000000000014, 0.75, 0.26999999999999985, 0.521999999999999, 0.774000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 0.0731826475937185, 0.03885863309046937, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.02749545416973506, 0.045825756949558406, 0.045825756949558406, 1.0, 0.045825756949558406, 0.06415605972938174, 0.08248636250920516], 'beta': [0.0, 6.283905465188191e-29, 0.0, 2.2068646093546303e-28, 0.0, 0.0, -1.0097482470410438e-29, -10.007830264643424, 2.436455913943408, -20.6480299299709, -26.971539814779046, 2.42292023420831, -8.349460918210418, -8.349460918210841, -0.9617153301643506, 4.522163739663501e-32, -0.47519419882237673, 0.4751941988227739, -1.8328022354180276, -0.1584852178265714, -2.932352898070871, 0.0, 0.0, -7.235461983483277e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6336598381651657, 0.6336598381658488, -20.437345294667402, 0.0, 0.6336598381663463, 0.633659838165844, 0.633659838166526], 'intercept': -43.45000000000069, 'uncertainty': 23.598828097872353}, 'bad': {'mean': [0.0, 0.6972222222222227, 0.6666666666666667, 0.9166666666666681, 0.28125, 0.09375, 0.027777777777777714, 0.1621111111111112, 1.0050866666666667, 0.019055555555555555, 0.028944444444444488, 0.01294444444444445, -0.00030999999999999973, -0.0004650000000000009, -0.00017999999999999998, 0.0899999999999999, 0.2836666666666668, 0.7163333333333327, 0.46196849999999995, 0.3723580833333333, 0.08961041666666665, 0.19999999999999957, 0.75, 0.7000000000000015, 0.75, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.6000000000000002, 0.39999999999999913, 1.0, 0.39999999999999913, 0.0, 0.018000000000000013, 0.6300000000000014, 0.6700000000000014, 0.75, 0.26999999999999985, 0.521999999999999, 0.774000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23378762653525456, 0.003986622073329197, 0.012737981505698078, 0.021365355018387357, 0.03980550709464367, 0.0002233830790368868, 0.0003350746185553301, 0.0005949229642006884, 1.0, 0.006182412330330455, 0.006182412330330471, 0.06849925800875512, 0.0731826475937185, 0.03885863309046937, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.02749545416973506, 0.045825756949558406, 0.045825756949558406, 1.0, 0.045825756949558406, 0.06415605972938174, 0.08248636250920516], 'beta': [0.0, 6.305292863243678e-32, 0.0, 3.151028058478568e-31, 0.0, 0.0, -4.3615036537501365e-32, -0.049776805393412484, 0.0005065138454161987, -0.005636931231828627, -0.03520837464418755, 0.0016168006979886149, -0.06099964444959668, -0.06099964444959738, 0.01810960091861926, -5.706778003792953e-32, 0.0009400150218759173, -0.0009400150218759149, -0.016723013443525777, 0.021008978040283944, -0.06904531722801033, 0.0, 0.0, 9.130844806068747e-31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.006767441635634016, -0.006767441635635533, 0.021403704437735965, 0.0, -0.006767441635635403, -0.0067674416356350575, -0.006767441635635392], 'intercept': 0.0416666666666673, 'uncertainty': 0.1997871946995136}, 'train_mean_delta': -43.45, 'train_min_delta': -204.0, 'train_positive_rate': 0.08333333333333333, 'train_bad_rate': 0.041666666666666664}, 'MILK|504|518|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.3782227499999999, 0.09375441666666665, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.30600000000000066, 0.5499999999999997, 0.0, 0.75, 0.5259999999999994, 0.7459999999999977, 0.9659999999999983], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 0.0761606845640901, 0.04222868611887409, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0764460594144656, 0.03872983346207418, 1.0, 1.0, 0.06264183905346328, 0.049839743177508444, 0.0390384425918863], 'beta': [0.0, 2.390349933894973e-29, 2.390349933894973e-29, 0.0, 0.0, 0.0, -5.9758748347374326e-30, 2.6236807372144066, -0.4448278204486427, -8.078714349553419, 2.2689969171458246, -1.497579867509321, -1.2863911556668075, -1.1151257654241602, -2.3744131537144515, -1.5299503203013217e-30, -0.4277773665665039, 0.42777736656653637, 0.46236587773508586, 0.4025993668779569, 0.06590804928344643, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2239602562410592e-29, -2.2004870662403316e-29, 0.0, 0.0, 0.0, 0.0, 0.8776577672037289, -0.009940303623730669, 0.0, 0.0, 1.0686066103284229, 1.3400046733031807, 1.7068174299139718], 'intercept': 8.750000000000016, 'uncertainty': 6.403324623993064}, 'bad': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.3782227499999999, 0.09375441666666665, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.30600000000000066, 0.5499999999999997, 0.0, 0.75, 0.5259999999999994, 0.7459999999999977, 0.9659999999999983], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 0.0761606845640901, 0.04222868611887409, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0764460594144656, 0.03872983346207418, 1.0, 1.0, 0.06264183905346328, 0.049839743177508444, 0.0390384425918863], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 8.75, 'train_min_delta': 0.0, 'train_positive_rate': 0.5166666666666667, 'train_bad_rate': 0.0}, 'MILK|504|518|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.3782227499999999, 0.09375441666666665, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.30600000000000066, 0.5499999999999997, 0.0, 0.75, 0.5259999999999994, 0.7459999999999977, 0.9659999999999983], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 0.0761606845640901, 0.04222868611887409, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0764460594144656, 0.03872983346207418, 1.0, 1.0, 0.06264183905346328, 0.049839743177508444, 0.0390384425918863], 'beta': [0.0, 5.073285822409134e-29, 5.073285822409134e-29, 0.0, 0.0, 0.0, -1.2683214556022835e-29, 5.8027903684447875, -1.1723005392812, -15.87795184294111, 3.5050597349278165, -4.601042719210317, -3.6761251344044044, -2.7061990359474937, -4.862767360036072, -2.1746053291876855e-30, -0.472370473287357, 0.4723704732874089, 0.4956284873039769, 1.02083330112126, -0.9921180750603042, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7396842633501515e-29, -3.5617703709932447e-29, 0.0, 0.0, 0.0, 0.0, 1.6451726280603918, -0.33576765242707757, 0.0, 0.0, 1.924676481236032, 2.3146905844299983, 2.8218824021964126], 'intercept': 16.35000000000025, 'uncertainty': 10.580431838533853}, 'bad': {'mean': [0.0, 0.7000000000000015, 0.7000000000000015, 0.0, 0.09375, 0.1875, 0.19444444444444406, 0.11494444444444434, 1.005831666666665, -0.038277777777777744, -0.018222222222222233, -0.034222222222222244, 0.0005900000000000004, 0.0002800000000000004, 0.0005650000000000003, 0.029999999999999926, 0.7636666666666679, 0.23633333333333278, 0.4719771666666669, 0.3782227499999999, 0.09375441666666665, 1.0, 0.875, 0.0, 0.75, 0.39999999999999913, 1.0, 0.0, 0.2399999999999994, 0.5499999999999988, 0.39999999999999913, 1.0, 0.0, 0.39999999999999913, 0.30600000000000066, 0.5499999999999997, 0.0, 0.75, 0.5259999999999994, 0.7459999999999977, 0.9659999999999983], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2277535149679296, 0.003786400738901753, 0.02920864703693747, 0.018393503952600244, 0.055720422038053205, 0.00037224543874528104, 0.00030484968973796455, 0.0008176847395746934, 1.0, 0.01861600267392428, 0.018616002673924267, 0.07233547137220364, 0.0761606845640901, 0.04222868611887409, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0764460594144656, 0.03872983346207418, 1.0, 1.0, 0.06264183905346328, 0.049839743177508444, 0.0390384425918863], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 16.35, 'train_min_delta': 0.0, 'train_positive_rate': 0.5333333333333333, 'train_bad_rate': 0.0}, 'MILK|518|524|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.39250750000000006, 0.09976933333333333, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.39999999999999913, 1.0, 0.3499999999999997, 0.04999999999999995, 0.13799999999999993, 0.5499999999999997, 0.6150000000000012, 0.75, 0.35800000000000043, 0.578, 0.7979999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 0.08023690371071987, 0.05442008051159874, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.06708203932499375, 0.06708203932499367, 0.14951922953252542, 0.03872983346207418, 0.022912878474779214, 1.0, 0.13518875692896956, 0.1211445417672624, 0.10749883720301344], 'beta': [0.0, 0.0, 3.074276847107715e-28, 0.0, 0.0, 0.0, 0.0, 0.7448504306226569, -0.06416905634594064, -0.30363731366699925, -1.2540721934226715, -2.959881653907184, -1.129007581490504, -0.6395081576251801, -1.5557577776306013, -2.2450926670553393e-29, -0.8353659656607799, 0.8353659656607659, 0.2880306752038094, -0.2552088280126898, 0.7784024760436808, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.4901853341106814e-29, -1.7791333193916712e-28, 0.0, 0.0, -0.7031387503130159, 0.7031387503130491, -2.7811251494130227, 6.562656619770929, -2.692157011432386, 0.0, -2.3238874114200954, -1.7540634362213734, -1.030959728628894], 'intercept': -10.950000000000141, 'uncertainty': 11.617757906890809}, 'bad': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.39250750000000006, 0.09976933333333333, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.39999999999999913, 1.0, 0.3499999999999997, 0.04999999999999995, 0.13799999999999993, 0.5499999999999997, 0.6150000000000012, 0.75, 0.35800000000000043, 0.578, 0.7979999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 0.08023690371071987, 0.05442008051159874, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.06708203932499375, 0.06708203932499367, 0.14951922953252542, 0.03872983346207418, 0.022912878474779214, 1.0, 0.13518875692896956, 0.1211445417672624, 0.10749883720301344], 'beta': [0.0, 0.0, 5.511444305155057e-31, 0.0, 0.0, 0.0, 0.0, -0.022991204649680113, 0.01300196629029716, 0.0135645639961131, 0.01832372175124371, 0.003509188188147241, 0.0020734069953993353, -0.017811821632696997, -0.016636002366299387, -7.078723706552226e-32, 0.0002094947699663718, -0.00020949476996608222, -0.0025973380625218083, 0.010712298077772773, -0.019420372859876595, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4157447413104433e-31, -2.8399526023186653e-32, 0.0, 0.0, -0.026805573515330784, 0.026805573515328616, 0.0024296608863391914, 0.018927472976376356, -0.004337823104879798, 0.0, 0.004856203941493076, 0.00783962125641356, 0.011562455069936828], 'intercept': 0.01666666666665948, 'uncertainty': 0.13751015526299407}, 'train_mean_delta': -10.95, 'train_min_delta': -41.0, 'train_positive_rate': 0.25, 'train_bad_rate': 0.016666666666666666}, 'MILK|518|524|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.39250750000000006, 0.09976933333333333, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.39999999999999913, 1.0, 0.3499999999999997, 0.04999999999999995, 0.13799999999999993, 0.5499999999999997, 0.6150000000000012, 0.75, 0.35800000000000043, 0.578, 0.7979999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 0.08023690371071987, 0.05442008051159874, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.06708203932499375, 0.06708203932499367, 0.14951922953252542, 0.03872983346207418, 0.022912878474779214, 1.0, 0.13518875692896956, 0.1211445417672624, 0.10749883720301344], 'beta': [0.0, 0.0, 4.50889755360866e-28, 0.0, 0.0, 0.0, 0.0, 2.275618024653636, -0.4536753058818198, -0.2824866825904339, -0.9516749080575243, -5.038802196104727, -0.6513774520637086, -0.3751128797307397, -2.860767646128782, -3.26673329990152e-29, -1.1631474937494537, 1.1631474937494828, 0.37066143487140985, -0.21605563675352935, 0.8360367920527513, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.533466599803046e-29, -2.606570821859557e-28, 0.0, 0.0, -0.3924278671582379, 0.3924278671582987, -4.259689043422395, 10.13513266131262, -3.9564136478796255, 0.0, -3.5497968520957763, -2.665244500821298, -1.542965747628816], 'intercept': -16.88333333333336, 'uncertainty': 18.99615722662091}, 'bad': {'mean': [0.0, 0.7194444444444446, 0.7000000000000015, 0.5833333333333338, 0.1875, 0.09375, 0.08333333333333334, 0.151222222222222, 1.005204999999998, 0.022166666666666685, 0.033944444444444465, 0.008166666666666666, -0.00036999999999999967, -0.0005700000000000012, -0.0001916666666666667, 0.05999999999999985, 0.35683333333333384, 0.6431666666666671, 0.49227683333333333, 0.39250750000000006, 0.09976933333333333, 0.09999999999999978, 0.875, 0.6000000000000002, 0.75, 0.39999999999999913, 1.0, 0.19999999999999957, 0.1199999999999997, 0.5499999999999988, 0.39999999999999913, 1.0, 0.3499999999999997, 0.04999999999999995, 0.13799999999999993, 0.5499999999999997, 0.6150000000000012, 0.75, 0.35800000000000043, 0.578, 0.7979999999999978], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22650072902071114, 0.003964064622749403, 0.01510242806085608, 0.021477737540917682, 0.022684837487887206, 0.00025053276565484745, 0.00036207733980463314, 0.00040550037676376503, 1.0, 0.010486764144492929, 0.010486764144492932, 0.07597648841344863, 0.08023690371071987, 0.05442008051159874, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.06708203932499375, 0.06708203932499367, 0.14951922953252542, 0.03872983346207418, 0.022912878474779214, 1.0, 0.13518875692896956, 0.1211445417672624, 0.10749883720301344], 'beta': [0.0, 0.0, 1.2450570977743614e-30, 0.0, 0.0, 0.0, 0.0, -0.0077543679114104765, -0.02522281666577837, -0.015000333929402596, -0.02517930476113972, 0.09821496549082108, -0.038312452545492086, -0.04995395998854945, 0.10457965259220546, -1.573258221749308e-31, -0.0042149788151110154, 0.004214978815110337, 0.0031538183825404937, 0.023040819671304707, -0.029568276424000086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.146516443498613e-31, -4.398804207165213e-31, 0.0, 0.0, -0.04261453478032299, 0.042614534780319, 0.0028653591388221157, 0.033777458014347304, -0.00779992964203984, 0.0, 0.007039819300491525, 0.012175394187701897, 0.018588672502553114], 'intercept': 0.03333333333334581, 'uncertainty': 0.15667914437273836}, 'train_mean_delta': -16.883333333333333, 'train_min_delta': -62.0, 'train_positive_rate': 0.25, 'train_bad_rate': 0.03333333333333333}, 'MILK|524|527|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.41173900000000013, 0.10816066666666667, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3899999999999991, 0.014999999999999996, 0.09600000000000004, 0.5499999999999997, 0.6150000000000012, 0.75, 0.316, 0.5360000000000009, 0.7560000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 0.08611828617663031, 0.056627729819899685, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.029999999999999992, 0.045000000000000026, 0.1346996659238618, 0.03872983346207418, 0.022912878474779214, 1.0, 0.12158947322856539, 0.10910545357588679, 0.09748846085563147], 'beta': [0.0, -8.106980791149623e-30, 2.4767340758151408e-29, -9.820829394687135e-30, 0.0, 0.0, 0.0, -0.847691653658162, -0.16633611016360583, 1.1642318830552971, -0.849969951822044, -0.2541194201575461, 0.5785990959820893, 0.2041041859037189, -0.7468814858780723, -7.341346130968787e-31, 0.0655806763497858, -0.06558067634977824, 0.15653996177347998, -0.17824527130810058, 0.49274016844511453, 0.0, 0.0, 0.0, 0.0, -2.241000613843154e-30, 0.0, 0.0, -1.4682692261937582e-30, -1.8720257209186143e-30, 0.24145244840218577, 0.0, -0.24145244840218097, 0.24145244840220506, 0.0687092487076538, 0.4181160851009744, 0.035707493647936336, 0.0, 0.12939063692411223, 0.2035642141973568, 0.29426449251068454], 'intercept': -0.9833333333332533, 'uncertainty': 3.447344183213805}, 'bad': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.41173900000000013, 0.10816066666666667, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3899999999999991, 0.014999999999999996, 0.09600000000000004, 0.5499999999999997, 0.6150000000000012, 0.75, 0.316, 0.5360000000000009, 0.7560000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 0.08611828617663031, 0.056627729819899685, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.029999999999999992, 0.045000000000000026, 0.1346996659238618, 0.03872983346207418, 0.022912878474779214, 1.0, 0.12158947322856539, 0.10910545357588679, 0.09748846085563147], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -0.9833333333333333, 'train_min_delta': -9.0, 'train_positive_rate': 0.03333333333333333, 'train_bad_rate': 0.0}, 'MILK|524|527|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.41173900000000013, 0.10816066666666667, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3899999999999991, 0.014999999999999996, 0.09600000000000004, 0.5499999999999997, 0.6150000000000012, 0.75, 0.316, 0.5360000000000009, 0.7560000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 0.08611828617663031, 0.056627729819899685, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.029999999999999992, 0.045000000000000026, 0.1346996659238618, 0.03872983346207418, 0.022912878474779214, 1.0, 0.12158947322856539, 0.10910545357588679, 0.09748846085563147], 'beta': [0.0, -1.1565541146338623e-29, 2.2258777626805728e-29, 2.249142080619146e-29, 0.0, 0.0, 0.0, -1.1965365265714074, -0.7616895256145823, 3.047737270219094, -3.4750271361923253, 0.08460788171125407, 0.6457652081930749, 1.5768194149149959, -2.0560518878046037, 1.852847333387018e-30, 0.357214383819224, -0.3572143838191806, -0.12273323540178084, 0.5260097613192737, -0.9737412301317626, 0.0, 0.0, 0.0, 0.0, -5.1333890668826104e-30, 0.0, 0.0, 3.705694666774037e-30, 4.686971457547836e-29, 0.06291307452187665, 0.0, -0.06291307452184466, 0.06291307452187013, -0.0016889604051110742, -0.23491269259097472, -0.009892091509537158, 0.0, -0.031801718393201305, -0.06879588257100198, -0.1143238662208705], 'intercept': -3.4499999999996125, 'uncertainty': 3.841155057602421}, 'bad': {'mean': [0.0, 0.7277777777777756, 0.7000000000000015, 0.8333333333333323, 0.09375, 0.09375, 0.04166666666666667, 0.11883333333333335, 1.0057149999999977, -0.0323888888888889, -0.010222222222222226, -0.03438888888888888, 0.0005099999999999995, 0.00014000000000000007, 0.00047333333333333326, 0.029999999999999926, 0.4168333333333324, 0.5831666666666676, 0.5198996666666663, 0.41173900000000013, 0.10816066666666667, 0.3000000000000001, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.3000000000000001, 0.05999999999999985, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3899999999999991, 0.014999999999999996, 0.09600000000000004, 0.5499999999999997, 0.6150000000000012, 0.75, 0.316, 0.5360000000000009, 0.7560000000000003], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2308453266037993, 0.00394612996069147, 0.02205289656515852, 0.020311159737359745, 0.03364020779137541, 0.00031606961258558207, 0.0003152776554086888, 0.0005022836737232149, 1.0, 0.010486764144492934, 0.010486764144492913, 0.0801877022567814, 0.08611828617663031, 0.056627729819899685, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.029999999999999992, 0.045000000000000026, 0.1346996659238618, 0.03872983346207418, 0.022912878474779214, 1.0, 0.12158947322856539, 0.10910545357588679, 0.09748846085563147], 'beta': [0.0, 2.2834483108453296e-31, 3.078854541871054e-30, -1.2820892861277532e-30, 0.0, 0.0, 0.0, 0.01798440857855525, -0.028243291659152507, -0.02358767758480697, 0.0018119307598353047, 0.0826886208291473, 0.07395103458753688, 0.022010601171977635, 0.06481434945910955, -1.3355987566778309e-31, -0.001261836419917814, 0.0012618364199180057, 0.0056489743541179, 0.012454114502842074, -0.010940730367233463, 0.0, 0.0, 0.0, 0.0, -1.5182085920724542e-31, 0.0, 0.0, -2.671197513355668e-31, -7.791008357807559e-31, 0.02886270175992201, 0.0, -0.028862701759923334, 0.028862701759921904, -0.01369638178068419, 0.0517498310357031, -0.017198673404745398, 0.0, -0.008579633478480132, -0.00221334655735842, 0.005746492970586676], 'intercept': 0.033333333333348134, 'uncertainty': 0.14442684357245136}, 'train_mean_delta': -3.45, 'train_min_delta': -18.0, 'train_positive_rate': 0.016666666666666666, 'train_bad_rate': 0.03333333333333333}, 'MILK|527|552|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.4277393333333333, 0.10584433333333336, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3949999999999992, 0.01, 0.07200000000000005, 0.5499999999999997, 0.6150000000000012, 0.75, 0.2920000000000002, 0.5120000000000002, 0.7320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 0.09148734942359822, 0.05421888372964618, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999994, 0.030000000000000044, 0.10998181667894023, 0.03872983346207418, 0.022912878474779214, 1.0, 0.09724196624914572, 0.0854166260162505, 0.07493997598078077], 'beta': [0.0, -2.790026343993029e-29, 1.0059115500590118e-28, -1.0059115500590118e-28, 0.0, 0.0, -6.090240640500498e-30, -3.3775316358317555, -0.23412763722627583, 2.822462586821752, 1.4973924411537074, -1.9815568076109462, 0.44838554600343855, 2.979509213564227, -2.224336866536554, 4.289265313175345e-31, -0.055268430182299495, 0.0552684301822495, -0.23828252350438037, -2.210886045803841, 3.3708032163541772, 0.0, 0.0, 0.0, 0.0, -2.4960884691451668e-29, 0.0, 5.029557750295046e-29, 0.0, -1.2180481281001565e-29, 1.3979203086809828, 0.0, -1.3979203086809562, 1.3979203086810064, 1.715209468226076, -0.5224527781478727, 1.7152094682260906, 0.0, 1.8566885947374412, 2.0189774997204113, 2.193227849915681], 'intercept': 0.9500000000001481, 'uncertainty': 11.268790989938019}, 'bad': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.4277393333333333, 0.10584433333333336, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3949999999999992, 0.01, 0.07200000000000005, 0.5499999999999997, 0.6150000000000012, 0.75, 0.2920000000000002, 0.5120000000000002, 0.7320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 0.09148734942359822, 0.05421888372964618, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999994, 0.030000000000000044, 0.10998181667894023, 0.03872983346207418, 0.022912878474779214, 1.0, 0.09724196624914572, 0.0854166260162505, 0.07493997598078077], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 0.95, 'train_min_delta': -32.0, 'train_positive_rate': 0.23333333333333334, 'train_bad_rate': 0.0}, 'MILK|527|552|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.4277393333333333, 0.10584433333333336, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3949999999999992, 0.01, 0.07200000000000005, 0.5499999999999997, 0.6150000000000012, 0.75, 0.2920000000000002, 0.5120000000000002, 0.7320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 0.09148734942359822, 0.05421888372964618, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999994, 0.030000000000000044, 0.10998181667894023, 0.03872983346207418, 0.022912878474779214, 1.0, 0.09724196624914572, 0.0854166260162505, 0.07493997598078077], 'beta': [0.0, -6.48987259419292e-29, 2.2256588425123436e-28, -2.2256588425123436e-28, 0.0, 0.0, -3.161714695677778e-29, -4.136011213560598, -1.1764198249359952, 7.118205303758388, 1.3901808340882273, -4.828343456291572, -1.8447086831840707, 5.215650115304169, -2.758132698117983, 2.1495797991481488e-30, -0.18610717801703933, 0.18610717801698756, -0.7034141456541565, -2.983144456795955, 3.9715908597094, 0.0, 0.0, 0.0, 0.0, -2.2132168875588628e-29, 0.0, 1.1128294212561693e-28, 0.0, -6.323429391355672e-29, 1.7809540915984103, 0.0, -1.7809540915983442, 1.7809540915985107, 3.128506299778992, -0.7551050844529505, 3.1285062997788073, 0.0, 3.418079472217599, 3.7543361985098076, 4.123095708220624], 'intercept': -0.03333333333291761, 'uncertainty': 19.22801373782191}, 'bad': {'mean': [0.0, 0.7319444444444463, 0.7000000000000015, 0.9583333333333318, 0.09375, 0.28125, 0.3472222222222216, 0.11466666666666678, 1.0057849999999993, -0.015888888888888897, -0.026277777777777785, -0.0002777777777777776, 0.00025500000000000024, 0.0004099999999999999, -4.666666666666667e-05, 0.029999999999999926, 0.35683333333333384, 0.6431666666666671, 0.5335836666666669, 0.4277393333333333, 0.10584433333333336, 0.19999999999999957, 0.875, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.35000000000000075, 0.0, 0.5499999999999988, 0.40499999999999914, 1.0, 0.3949999999999992, 0.01, 0.07200000000000005, 0.5499999999999997, 0.6150000000000012, 0.75, 0.2920000000000002, 0.5120000000000002, 0.7320000000000001], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22968754944569908, 0.003910917411554507, 0.01426426078710247, 0.018166836560537556, 0.01968541165682764, 0.0002052843556305902, 0.00026185237571323784, 0.0003383620677453206, 1.0, 0.010486764144492929, 0.010486764144492932, 0.08186465485720247, 0.09148734942359822, 0.05421888372964618, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999994, 0.030000000000000044, 0.10998181667894023, 0.03872983346207418, 0.022912878474779214, 1.0, 0.09724196624914572, 0.0854166260162505, 0.07493997598078077], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -0.03333333333333333, 'train_min_delta': -52.0, 'train_positive_rate': 0.23333333333333334, 'train_bad_rate': 0.0}, 'MILK|552|571|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.44558099999999995, 0.1143823333333333, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.40499999999999914, 1.0, 0.0, 0.40499999999999914, 0.32999999999999996, 0.5649999999999997, 0.0, 0.75, 0.5560000000000007, 0.7819999999999989, 1.007999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 0.08434069786091804, 0.05639723425153245, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.014999999999999972, 0.10129165809680477, 0.05937171043518962, 1.0, 1.0, 0.08284926070883196, 0.06779380502671319, 0.05878775382679635], 'beta': [0.0, 1.655839509728906e-29, 1.655839509728906e-29, 0.0, 0.0, 0.0, 1.0599261304694218e-30, 5.1189765946108645, -1.2020014221090987, -2.7018765138833842, -3.306987833457935, 8.053082017012152, 2.7709450146179107, 4.085450150883742, -0.6387718611436384, 2.0697993871611408e-30, 0.13109687297749606, -0.13109687297736908, -0.42664933990680726, 1.8766676576787085, -3.45800505762912, -2.74454276933021e-29, 0.0, 0.0, 0.0, -1.3722713846651116e-29, 0.0, 0.0, -2.6941326779830974e-30, -3.311679019457828e-29, -0.5938690595748637, 0.0, 0.0, -0.5938690595748635, -0.11341868920841591, -1.1102081950355176, 0.0, 0.0, -0.4569063206669959, -0.9472891313080518, -1.5409045688339162], 'intercept': 3.566666666666816, 'uncertainty': 10.404246638141272}, 'bad': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.44558099999999995, 0.1143823333333333, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.40499999999999914, 1.0, 0.0, 0.40499999999999914, 0.32999999999999996, 0.5649999999999997, 0.0, 0.75, 0.5560000000000007, 0.7819999999999989, 1.007999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 0.08434069786091804, 0.05639723425153245, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.014999999999999972, 0.10129165809680477, 0.05937171043518962, 1.0, 1.0, 0.08284926070883196, 0.06779380502671319, 0.05878775382679635], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': 3.566666666666667, 'train_min_delta': -29.0, 'train_positive_rate': 0.3333333333333333, 'train_bad_rate': 0.0}, 'MILK|552|571|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.44558099999999995, 0.1143823333333333, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.40499999999999914, 1.0, 0.0, 0.40499999999999914, 0.32999999999999996, 0.5649999999999997, 0.0, 0.75, 0.5560000000000007, 0.7819999999999989, 1.007999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 0.08434069786091804, 0.05639723425153245, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.014999999999999972, 0.10129165809680477, 0.05937171043518962, 1.0, 1.0, 0.08284926070883196, 0.06779380502671319, 0.05878775382679635], 'beta': [0.0, 1.9977761001285585e-28, 1.9977761001285585e-28, 0.0, 0.0, 0.0, -5.670124601772031e-29, 22.529860400077272, -5.293972796565352, -8.683362737126135, -5.437495705291986, 20.179716625415313, 12.30309071520952, 6.590907474560108, 3.332532389521914, 2.4972201251606996e-29, -1.5071560338925278, 1.5071560338928447, -0.6477096411473315, 3.7077925420941154, -6.533967304129617, -2.211497332351531e-28, 0.0, 0.0, 0.0, -1.1057486661757683e-28, 0.0, 0.0, -3.3656053527280377e-29, -3.9955522002571247e-28, -2.5855494603049904, 0.0, 0.0, -2.58554946030499, -5.025275925321219, 2.6841799302330314, 0.0, 0.0, -5.37449321370682, -5.627756811414034, -5.405569798048739], 'intercept': -4.93333333333277, 'uncertainty': 33.82230866711017}, 'bad': {'mean': [0.0, 0.7666666666666659, 0.7666666666666659, 0.0, 0.28125, 0.1875, 0.26388888888888923, 0.1654444444444446, 1.0049983333333332, 0.00861111111111112, 0.025000000000000026, 0.05900000000000002, -0.0001516666666666666, -0.00036833333333333396, -0.000931666666666666, 0.0899999999999999, 0.7572500000000003, 0.2427500000000001, 0.5599633333333335, 0.44558099999999995, 0.1143823333333333, 0.9000000000000017, 0.875, 0.0, 0.75, 0.45000000000000084, 1.0, 0.0, 0.2399999999999994, 0.7000000000000015, 0.40499999999999914, 1.0, 0.0, 0.40499999999999914, 0.32999999999999996, 0.5649999999999997, 0.0, 0.75, 0.5560000000000007, 0.7819999999999989, 1.007999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2318228623457421, 0.004073266161475605, 0.007965008350572594, 0.03236653507239524, 0.04058507297206877, 0.00014432794447214847, 0.0005149406654061117, 0.000625963701468029, 1.0, 0.010951598056904759, 0.01095159805690476, 0.08611864109406793, 0.08434069786091804, 0.05639723425153245, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.014999999999999972, 0.10129165809680477, 0.05937171043518962, 1.0, 1.0, 0.08284926070883196, 0.06779380502671319, 0.05878775382679635], 'beta': [0.0, -1.1666566289615913e-31, -1.1666566289615913e-31, 0.0, 0.0, 0.0, -1.5624126084990485e-31, -0.013362259832615993, 0.003519214409175786, 0.01262605464549058, -0.01994041713878416, 0.042512788899695654, 2.3775062683350334e-05, 0.014926073214857699, -0.02040322165334143, -1.4583207862020228e-32, 0.003726673373367979, -0.0037266733733673525, 0.0009891950636820358, 0.006600854391064957, -0.008360915874995616, -1.1810194617713205e-30, 0.0, 0.0, 0.0, -5.9050973088566e-31, 0.0, 0.0, -3.4028416272829663e-31, 2.33331325792332e-31, 0.029509462438851577, 0.0, 0.0, 0.029509462438851587, 0.0014181503814013892, -0.012513285082384026, 0.0, 0.0, -0.0018530913918299169, -0.006648119353429144, -0.012721618261618106], 'intercept': 0.01666666666666351, 'uncertainty': 0.13703474894771114}, 'train_mean_delta': -4.933333333333334, 'train_min_delta': -74.0, 'train_positive_rate': 0.31666666666666665, 'train_bad_rate': 0.016666666666666666}, 'MILK|571|597|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.4754304166666668, 0.0992323333333334, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.08400000000000006, 0.5649999999999997, 0.6550000000000015, 0.75, 0.3099999999999999, 0.5359999999999999, 0.7619999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 0.09026263536107501, 0.061273665982123915, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.10799999999999998, 0.05937171043518962, 0.041533119314590396, 1.0, 0.09889388252060899, 0.09499473669630334, 0.096932966528421], 'beta': [0.0, 6.788546623875628e-30, -6.251492884002343e-30, -6.788546623875628e-30, 0.0, 0.0, -4.897412042414799e-29, 5.101493073311118, -5.74498463343801, -3.5506320190095595, -4.979270489886368, -3.9439575882436113, 5.784159579409467, 5.501047423518273, -0.01999499475764943, -5.454365720309635e-30, 3.2383733038687312, -3.238373303868682, -2.7194329971847613, 1.2339575934701223, -6.1285705419597605, 0.0, 0.0, 1.2502985768004857e-29, 0.0, 2.71619205790077e-29, 0.0, 0.0, -1.0908731440619275e-29, 1.2502985768004857e-29, 0.5369877545612199, 0.0, 0.0, 0.536987754561758, 3.7550688868107684, -2.8527226551277236, -3.119832771875708, 0.0, 3.415772763621783, 2.8427956160685723, 2.087032089311169], 'intercept': -5.9916666666664895, 'uncertainty': 25.063732757084452}, 'bad': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.4754304166666668, 0.0992323333333334, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.08400000000000006, 0.5649999999999997, 0.6550000000000015, 0.75, 0.3099999999999999, 0.5359999999999999, 0.7619999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 0.09026263536107501, 0.061273665982123915, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.10799999999999998, 0.05937171043518962, 0.041533119314590396, 1.0, 0.09889388252060899, 0.09499473669630334, 0.096932966528421], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'intercept': 0.0, 'uncertainty': 0.01}, 'train_mean_delta': -5.991666666666666, 'train_min_delta': -60.0, 'train_positive_rate': 0.2, 'train_bad_rate': 0.0}, 'MILK|571|597|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.4754304166666668, 0.0992323333333334, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.08400000000000006, 0.5649999999999997, 0.6550000000000015, 0.75, 0.3099999999999999, 0.5359999999999999, 0.7619999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 0.09026263536107501, 0.061273665982123915, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.10799999999999998, 0.05937171043518962, 0.041533119314590396, 1.0, 0.09889388252060899, 0.09499473669630334, 0.096932966528421], 'beta': [0.0, -2.3860991005404568e-29, -1.84380464876832e-29, 2.3860991005404568e-29, 0.0, 0.0, -3.93709713414259e-29, 9.221693415453618, -9.175053376487673, -5.240949445612817, -8.23251449298052, -4.53888454558127, 5.7285351011995935, 9.143052363397274, 1.4557314538587778, -2.0045811638265127e-30, 4.006000857561935, -4.006000857561899, -2.5119489925911895, 3.302105934772496, -8.846271605631024, 0.0, 0.0, 3.6876092975366794e-29, 0.0, 4.599988118454189e-29, 0.0, 0.0, -4.00916232765301e-30, 3.6876092975366794e-29, -1.1104837497430464, 0.0, 0.0, -1.110483749742291, 4.306265739225068, -4.558366108745045, -3.827984538707189, 0.0, 3.608125130445408, 2.6166323982575963, 1.4475065982419142], 'intercept': -14.241666666666147, 'uncertainty': 42.84636465786555}, 'bad': {'mean': [0.0, 0.7930555555555545, 0.7666666666666659, 0.7916666666666676, 0.1875, 0.34375, 0.3611111111111117, 0.1496666666666666, 1.0051949999999996, 0.0016666666666666672, 0.02011111111111113, -0.006222222222222221, -2.1666666666666684e-05, -0.00032166666666666704, 2.666666666666667e-05, 0.05999999999999985, 0.1972500000000005, 0.8027500000000012, 0.5746627500000001, 0.4754304166666668, 0.0992323333333334, 0.19999999999999957, 0.875, 0.7000000000000015, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.1199999999999997, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.08400000000000006, 0.5649999999999997, 0.6550000000000015, 0.75, 0.3099999999999999, 0.5359999999999999, 0.7619999999999999], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2359046417517044, 0.004255287886853258, 0.013102162671355696, 0.02672886265379491, 0.0281297698287292, 0.0002017355254342234, 0.00041396121664823734, 0.0005137660513848257, 1.0, 0.010951598056904757, 0.010951598056904755, 0.09713043869768201, 0.09026263536107501, 0.061273665982123915, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.10799999999999998, 0.05937171043518962, 0.041533119314590396, 1.0, 0.09889388252060899, 0.09499473669630334, 0.096932966528421], 'beta': [0.0, 1.4692908334074284e-31, 8.898146044662959e-32, -1.4692908334074284e-31, 0.0, 0.0, -9.518079104485394e-32, 0.0027522898028837675, 0.0002743128902571722, -0.00471059897368995, 0.009218771546281573, -0.007585690183313629, 0.0035732712029702225, -0.0018723291215664612, 0.0013572303600331958, -8.023838147985326e-33, 0.007960201310480225, -0.007960201310480376, -0.0065543482969379965, -0.006003172947858408, -0.0015465781773957524, 0.0, 0.0, -1.779629208932605e-31, 0.0, -6.446542756203555e-32, 0.0, 0.0, -1.6047676295970726e-32, -1.779629208932605e-31, 0.007613966731322143, 0.0, 0.0, 0.007613966731321885, 0.0008285929463206698, 0.005160296661997017, 0.0009758247770213122, 0.0, 0.00214409919469726, 0.0035221798720589084, 0.004716030279332251], 'intercept': 0.008333333333332386, 'uncertainty': 0.10689802269484933}, 'train_mean_delta': -14.241666666666667, 'train_min_delta': -137.0, 'train_positive_rate': 0.15, 'train_bad_rate': 0.008333333333333333}, 'MILK|597|600|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.4952645833333334, 0.10727691666666668, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.044000000000000025, 0.6550000000000007, 0.6550000000000014, 0.75, 0.30599999999999994, 0.568000000000001, 0.8299999999999982], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 0.09954961650600697, 0.06654067098143601, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.054990908339470096, 0.04153311931459039, 0.05678908345800272, 1.0, 0.06988562083862467, 0.08541662601625048, 0.10129165809680477], 'beta': [0.0, -9.23236412650547e-29, 0.0, 0.0, 0.0, 0.0, 0.0, -12.666563857944828, -2.5178415625166934, -12.678400790457282, -11.850876292504648, -6.111684018572139, -4.951082973897306, -5.265216948360302, -5.207540727296231, -5.749413133842702e-29, 0.11491869233900942, -0.11491869233841505, 0.518034754700245, 5.132100609398076, -6.831198749939799, 0.0, 0.0, -1.6521448253868397e-28, 0.0, -1.1541002353604454e-29, 0.0, 0.0, 0.0, 1.8464728253011368e-28, 1.0368488188438114, 0.0, 0.0, 1.0368488188443012, -13.391167851177476, 1.9543656873681075, -24.92704138307138, 0.0, -10.072517266132799, -7.860949238531986, -6.308391154170703], 'intercept': -41.0750000000005, 'uncertainty': 36.71374546805299}, 'bad': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.4952645833333334, 0.10727691666666668, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.044000000000000025, 0.6550000000000007, 0.6550000000000014, 0.75, 0.30599999999999994, 0.568000000000001, 0.8299999999999982], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 0.09954961650600697, 0.06654067098143601, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.054990908339470096, 0.04153311931459039, 0.05678908345800272, 1.0, 0.06988562083862467, 0.08541662601625048, 0.10129165809680477], 'beta': [0.0, 4.1212126733233065e-31, 0.0, 0.0, 0.0, 0.0, 0.0, -0.031126495854534045, -0.007498554301944849, 0.008004794347865638, 0.004778753390444653, -0.006162318493156427, -0.0281321734136969, -0.004484411574764999, -0.07499232569634624, 1.5952630820845952e-32, -0.013231372765664558, 0.013231372765664055, 0.01769596591282812, 0.02502617547992649, -0.00851470892784405, 0.0, 0.0, 7.153482624910098e-32, 0.0, -6.763105977199135e-31, 0.0, 0.0, 0.0, -8.242425346646503e-31, 0.05460414933039282, 0.0, 0.0, 0.05460414933038966, -0.0019218418635490095, -0.032277546970446824, 0.02007306510459045, 0.0, -0.009185275967513966, -0.013793024289463508, -0.01692526821438855], 'intercept': 0.03333333333333245, 'uncertainty': 0.152816255758099}, 'train_mean_delta': -41.075, 'train_min_delta': -198.0, 'train_positive_rate': 0.008333333333333333, 'train_bad_rate': 0.03333333333333333}, 'MILK|597|600|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.4952645833333334, 0.10727691666666668, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.044000000000000025, 0.6550000000000007, 0.6550000000000014, 0.75, 0.30599999999999994, 0.568000000000001, 0.8299999999999982], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 0.09954961650600697, 0.06654067098143601, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.054990908339470096, 0.04153311931459039, 0.05678908345800272, 1.0, 0.06988562083862467, 0.08541662601625048, 0.10129165809680477], 'beta': [0.0, -1.5638118818576073e-28, 0.0, 0.0, 0.0, 0.0, 0.0, -3.3576966587475203, -10.280590615481547, -14.281628164125705, -29.109879103231695, -30.96483264006019, -2.4353440580258345, 7.62341595089539, 3.7592891371254975, -9.040278524188497e-29, 2.4249466883684243, -2.424946688367866, -3.826435765981096, 0.015347453650432438, -6.277743340573638, 0.0, 0.0, -3.111867580080048e-28, 0.0, 2.0769574197272804e-28, 0.0, 0.0, 0.0, 3.1276237637152447e-28, -18.69547247093819, 0.0, 0.0, -18.6954724709367, -20.377659645995585, 17.536030206309906, -50.30735557374072, 0.0, -11.865897302902187, -6.2976637097932215, -2.4345022735692936], 'intercept': -78.26666666666635, 'uncertainty': 54.13623146313814}, 'bad': {'mean': [0.0, 0.8291666666666659, 0.7999999999999983, 0.875, 0.34375, 0.09375, 0.04166666666666667, 0.18200000000000008, 1.004608333333333, 0.011666666666666669, 0.026000000000000023, 0.054944444444444455, -0.00021333333333333341, -0.0004600000000000001, -0.0009316666666666667, 0.10999999999999986, 0.48724999999999924, 0.5127500000000008, 0.6025415, 0.4952645833333334, 0.10727691666666668, 0.39999999999999913, 1.0, 0.5499999999999988, 0.75, 0.45000000000000084, 1.0, 0.39999999999999913, 0.08000000000000006, 0.7000000000000015, 0.40499999999999914, 1.0, 0.39999999999999913, 0.005, 0.044000000000000025, 0.6550000000000007, 0.6550000000000014, 0.75, 0.30599999999999994, 0.568000000000001, 0.8299999999999982], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.23767367918593302, 0.004619245669539084, 0.026973924308349943, 0.0299895043368571, 0.04237177461412983, 0.00044701478971307236, 0.0005112729212465686, 0.0007251417485950978, 1.0, 0.010951598056904748, 0.01095159805690476, 0.10876895535683273, 0.09954961650600697, 0.06654067098143601, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 1.0, 0.015000000000000022, 0.054990908339470096, 0.04153311931459039, 0.05678908345800272, 1.0, 0.06988562083862467, 0.08541662601625048, 0.10129165809680477], 'beta': [0.0, 5.137758318556127e-31, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03356680299393587, -0.006652964077374391, 0.001782553969909034, 0.012267260490923323, 0.010084624419630702, -0.02147319866524166, -0.010946212761967116, -0.07868843979996072, 2.7486562573528996e-32, -0.0063216499926851445, 0.00632164999268444, 0.01637677183209865, 0.02352933326701273, -0.008431711475155191, 0.0, 0.0, 1.8489130384523355e-31, 0.0, -8.617925199649046e-31, 0.0, 0.0, 0.0, -1.027551663711212e-30, 0.07193350588095823, 0.0, 0.0, 0.07193350588095396, 0.0007642485802361882, -0.04210729570139063, 0.03115742423311187, 0.0, -0.009408404805421836, -0.015887435605593313, -0.020303647098629408], 'intercept': 0.04166666666666389, 'uncertainty': 0.15684495875227644}, 'train_mean_delta': -78.26666666666667, 'train_min_delta': -424.0, 'train_positive_rate': 0.025, 'train_bad_rate': 0.041666666666666664}, 'MILK|600|620|DELAY_25': {'support': 114, 'rows': 114, 'margin': {'mean': [0.0, 0.8333333333333325, 0.8333333333333325, 0.0, 0.09375, 0.34375, 0.27777777777777835, 0.12114035087719308, 1.0055578947368407, -0.06055555555555559, -0.041023391812865496, -0.005409356725146196, 0.0009421052631578953, 0.0005947368421052636, 6.140350877193017e-06, 0.029999999999999933, 0.9057894736842121, 0.09421052631578936, 0.6139526315789475, 0.5060407894736844, 0.10791184210526318, 0.9000000000000015, 1.0, 0.0, 0.75, 0.45000000000000073, 1.0, 0.0, 0.32000000000000023, 0.7000000000000015, 0.405263157894736, 1.0, 0.0, 0.405263157894736, 0.30210526315789477, 0.5657894736842103, 0.0, 0.75, 0.5284210526315801, 0.7547368421052645, 0.9810526315789451], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2410049739230031, 0.004486566591324987, 0.04152764331375287, 0.027784071309206994, 0.022913475485954925, 0.0005675269120818952, 0.0003738941836569323, 0.00047021553246176154, 1.0, 0.009070362073481098, 0.009070362073481093, 0.11762724713480882, 0.10528750094213407, 0.06602741371881928, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.015344610249592876, 1.0, 1.0, 0.015344610249592876, 0.048076577880712616, 0.06081168425416489, 1.0, 1.0, 0.040165861112292485, 0.04581064241471357, 0.06137844099837153], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3322157326240437e-29, 1.8504000222360442, -2.418109010891165, 2.6457232574973433, -8.168120317366487, -4.420158930215624, -5.308631129214112, 1.308451427848106, -1.421661017911531, -5.7903000277655085e-31, 0.22849024221709763, -0.22849024221707415, -0.4507982665878226, 0.6067390845632342, -1.7706009438339319, 1.8288639147087595e-29, 0.0, 0.0, 0.0, 9.14431957354379e-30, 0.0, 0.0, 0.0, -1.8167587297176384e-29, 0.09245792033096681, 0.0, 0.0, 0.09245792033096689, -0.13322787415755233, 0.16553923016388297, 0.0, 0.0, -0.05921577288253911, 0.0359793087898724, 0.09245792033092552], 'intercept': -1.8070175438588256, 'uncertainty': 12.695520885957862}, 'bad': {'mean': [0.0, 0.8333333333333325, 0.8333333333333325, 0.0, 0.09375, 0.34375, 0.27777777777777835, 0.12114035087719308, 1.0055578947368407, -0.06055555555555559, -0.041023391812865496, -0.005409356725146196, 0.0009421052631578953, 0.0005947368421052636, 6.140350877193017e-06, 0.029999999999999933, 0.9057894736842121, 0.09421052631578936, 0.6139526315789475, 0.5060407894736844, 0.10791184210526318, 0.9000000000000015, 1.0, 0.0, 0.75, 0.45000000000000073, 1.0, 0.0, 0.32000000000000023, 0.7000000000000015, 0.405263157894736, 1.0, 0.0, 0.405263157894736, 0.30210526315789477, 0.5657894736842103, 0.0, 0.75, 0.5284210526315801, 0.7547368421052645, 0.9810526315789451], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2410049739230031, 0.004486566591324987, 0.04152764331375287, 0.027784071309206994, 0.022913475485954925, 0.0005675269120818952, 0.0003738941836569323, 0.00047021553246176154, 1.0, 0.009070362073481098, 0.009070362073481093, 0.11762724713480882, 0.10528750094213407, 0.06602741371881928, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.015344610249592876, 1.0, 1.0, 0.015344610249592876, 0.048076577880712616, 0.06081168425416489, 1.0, 1.0, 0.040165861112292485, 0.04581064241471357, 0.06137844099837153], 'beta': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.080296972413081e-31, -0.019159907717147182, 0.01006648074445578, -0.05350229778142312, -0.010485538361979624, 0.03800057145829738, 0.028250155424202412, -0.03488843870879757, -0.020606348065436594, 1.7729648171261774e-32, -0.013741923498215345, 0.013741923498215732, 0.013490949320656127, 0.016382241059808424, -0.0020891624164211062, -8.22869319424283e-31, 0.0, 0.0, 0.0, -4.114346597121425e-31, 0.0, 0.0, 0.0, -9.72213953830454e-31, 0.01213292115884496, 0.0, 0.0, 0.012132921158844977, -0.02023254987054765, 0.023534560917120875, 0.0, 0.0, -0.009964712157841879, 0.0037595908152943604, 0.012132921158845046], 'intercept': 0.017543859649114812, 'uncertainty': 0.12611061683953442}, 'train_mean_delta': -1.8070175438596492, 'train_min_delta': -53.0, 'train_positive_rate': 0.12280701754385964, 'train_bad_rate': 0.017543859649122806}, 'MILK|600|620|DELAY_50': {'support': 81, 'rows': 81, 'margin': {'mean': [0.0, 0.833333333333334, 0.833333333333334, 0.0, 0.09375, 0.34375, 0.27777777777777823, 0.10588477366255139, 1.0059370370370357, -0.0653086419753087, -0.04753086419753085, -0.013086419753086428, 0.0009950617283950617, 0.0006987654320987658, 0.00016666666666666663, 0.029999999999999985, 0.8999999999999997, 0.09999999999999984, 0.6265604938271605, 0.5231422222222224, 0.10341827160493827, 0.8999999999999997, 1.0, 0.0, 0.75, 0.44999999999999984, 1.0, 0.0, 0.3200000000000002, 0.700000000000001, 0.40555555555555495, 1.0, 0.0, 0.40555555555555495, 0.30222222222222217, 0.5666666666666669, 0.0, 0.75, 0.5288888888888894, 0.7555555555555548, 0.9822222222222216], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2114839278608835, 0.0037151157516302357, 0.04088116465480159, 0.024943160503555113, 0.018553875711481028, 0.0005708778673231972, 0.0002795471766808699, 0.0002434322477800738, 1.0, 1.0, 1.0, 0.12025273573440227, 0.11323788655135385, 0.0627733157271704, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.015713484026367748, 1.0, 1.0, 0.015713484026367748, 0.052020888492087206, 0.06382847385042252, 1.0, 1.0, 0.0426296135405575, 0.04724509250104298, 0.0628539361054708], 'beta': [0.0, 2.1534073924678007e-29, 2.1534073924678007e-29, 0.0, 0.0, 0.0, 0.0, 3.71323092817632, -3.1674772189478277, 4.2733125032741, -17.372598992052527, -5.592381861324217, -10.839972539925743, 2.8297564766399854, -8.661091101530928, 0.0, -1.0767036962339008e-29, -5.383518481169504e-30, 0.05467020538764751, 1.3218856271275514, -2.279842816325288, -1.0767036962339008e-29, 0.0, 0.0, 0.0, -5.383518481169498e-30, 0.0, 0.0, 5.383518481169498e-30, 1.4070179171486995e-29, -0.029434951404732186, 0.0, 0.0, -0.029434951404732117, -1.7237586745715914, 1.146580174757301, 0.0, 0.0, -1.4168008503976137, -0.6587750385912944, -0.02943495140474238], 'intercept': 0.3950617283962029, 'uncertainty': 16.622177798700687}, 'bad': {'mean': [0.0, 0.833333333333334, 0.833333333333334, 0.0, 0.09375, 0.34375, 0.27777777777777823, 0.10588477366255139, 1.0059370370370357, -0.0653086419753087, -0.04753086419753085, -0.013086419753086428, 0.0009950617283950617, 0.0006987654320987658, 0.00016666666666666663, 0.029999999999999985, 0.8999999999999997, 0.09999999999999984, 0.6265604938271605, 0.5231422222222224, 0.10341827160493827, 0.8999999999999997, 1.0, 0.0, 0.75, 0.44999999999999984, 1.0, 0.0, 0.3200000000000002, 0.700000000000001, 0.40555555555555495, 1.0, 0.0, 0.40555555555555495, 0.30222222222222217, 0.5666666666666669, 0.0, 0.75, 0.5288888888888894, 0.7555555555555548, 0.9822222222222216], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2114839278608835, 0.0037151157516302357, 0.04088116465480159, 0.024943160503555113, 0.018553875711481028, 0.0005708778673231972, 0.0002795471766808699, 0.0002434322477800738, 1.0, 1.0, 1.0, 0.12025273573440227, 0.11323788655135385, 0.0627733157271704, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.015713484026367748, 1.0, 1.0, 0.015713484026367748, 0.052020888492087206, 0.06382847385042252, 1.0, 1.0, 0.0426296135405575, 0.04724509250104298, 0.0628539361054708], 'beta': [0.0, -1.4683503912976301e-31, -1.4683503912976301e-31, 0.0, 0.0, 0.0, 0.0, -0.017421205648479647, 0.017321628426943136, -0.061723458055563805, -0.025618906950332723, 0.05733325783719061, 0.03287567515122829, -0.023138453108996693, -0.019178214465384723, 0.0, 7.341751956488076e-32, 3.670875978244038e-32, 0.019766750974858402, 0.02177629317618383, -0.0014161357235405983, 7.341751956488076e-32, 0.0, 0.0, 0.0, 3.6708759782440206e-32, 0.0, 0.0, -3.6708759782440206e-32, 1.4946733899863706e-31, 0.013402788743364143, 0.0, 0.0, 0.013402788743364138, -0.02396207674176103, 0.027272918906953002, 0.0, 0.0, -0.012906826030400075, 0.0030924378187322074, 0.013402788743363212], 'intercept': 0.02469135802468386, 'uncertainty': 0.13079941344372936}, 'train_mean_delta': 0.3950617283950617, 'train_min_delta': -41.0, 'train_positive_rate': 0.1728395061728395, 'train_bad_rate': 0.024691358024691357}, 'MILK|620|624|DELAY_25': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.5295818333333332, 0.12901949999999998, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.40499999999999914, 1.0, 0.40499999999999914, 0.0, 0.018000000000000013, 0.5649999999999997, 0.6300000000000014, 0.75, 0.2440000000000005, 0.4700000000000003, 0.695999999999998], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 0.10755274514537304, 0.06550514728948151, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999972, 1.0, 0.02749545416973506, 0.05937171043518962, 0.045825756949558406, 1.0, 0.019595917942265447, 0.03376388603226828, 0.05499090833947002], 'beta': [0.0, -9.630414302644951e-29, -3.5805299267230692e-28, -3.5805299267230692e-28, 0.0, 0.0, 6.019008939153094e-30, -12.964848459939418, -7.867443491148799, -3.5009141698891892, -11.439559549178671, 9.47959592141429, -2.521817454258704, 3.177004631177427, -1.3979817788416682, -4.4756624084038685e-29, 1.8526929247380226, -1.8526929247382635, -1.8858710482199645, -2.3429168127963367, 0.6229584502069271, 0.0, 0.0, 0.0, 0.0, -1.15897860561546e-28, 0.0, -1.15897860561546e-28, 0.0, 5.902931109136176e-28, 8.373685053509316, 0.0, 8.373685053509314, 0.0, 2.773430393811376, -4.8676377908856825, 2.773430393811747, 0.0, -2.0077275481281736, -4.589023215663497, -4.919781425551036], 'intercept': -12.683333333334092, 'uncertainty': 21.622138611633844}, 'bad': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.5295818333333332, 0.12901949999999998, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.40499999999999914, 1.0, 0.40499999999999914, 0.0, 0.018000000000000013, 0.5649999999999997, 0.6300000000000014, 0.75, 0.2440000000000005, 0.4700000000000003, 0.695999999999998], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 0.10755274514537304, 0.06550514728948151, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999972, 1.0, 0.02749545416973506, 0.05937171043518962, 0.045825756949558406, 1.0, 0.019595917942265447, 0.03376388603226828, 0.05499090833947002], 'beta': [0.0, -1.0539072839466186e-30, 5.449579754649276e-31, 5.449579754649276e-31, 0.0, 0.0, 6.586920524666366e-32, -0.016476151485661276, -0.00500438010155873, 0.013138008367212175, 0.048317255900517386, -0.02307542459192495, 0.01159324993250906, 0.0002795570549470984, 0.018196274722648797, 6.811974693311535e-32, -0.0019400170927148061, 0.0019400170927146227, 0.008811224015305713, 0.014972889139259975, -0.009521282564473404, 0.0, 0.0, 0.0, 0.0, -1.4315476323029603e-31, 0.0, -1.4315476323029603e-31, 0.0, -5.873823496428267e-31, 0.010582512676661283, 0.0, 0.010582512676661302, 0.0, -0.01185346076432077, 0.00861299235954506, -0.011853460764321122, 0.0, -0.006193588492007463, 0.0024635251820097855, 0.005232236155245567], 'intercept': 0.016666666666664446, 'uncertainty': 0.1350368612019947}, 'train_mean_delta': -12.683333333333334, 'train_min_delta': -108.0, 'train_positive_rate': 0.05, 'train_bad_rate': 0.016666666666666666}, 'MILK|620|624|DELAY_50': {'support': 120, 'rows': 120, 'margin': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.5295818333333332, 0.12901949999999998, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.40499999999999914, 1.0, 0.40499999999999914, 0.0, 0.018000000000000013, 0.5649999999999997, 0.6300000000000014, 0.75, 0.2440000000000005, 0.4700000000000003, 0.695999999999998], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 0.10755274514537304, 0.06550514728948151, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999972, 1.0, 0.02749545416973506, 0.05937171043518962, 0.045825756949558406, 1.0, 0.019595917942265447, 0.03376388603226828, 0.05499090833947002], 'beta': [0.0, -3.1734666009702886e-28, -6.349567373924051e-28, -6.349567373924051e-28, 0.0, 0.0, 1.9834166256064304e-29, -23.30371097572261, -17.4000868699226, -8.47608813489591, -31.175603402775266, 13.678057750700665, -4.054916848603886, 4.640728248486699, -3.6465835548502685, -7.936959217405128e-29, 3.551555715224048, -3.5515557152248407, -6.6520313068695165, -6.032187436092257, -1.4673234801100132, 0.0, 0.0, 0.0, 0.0, -2.2337577612355757e-28, 0.0, -2.2337577612355757e-28, 0.0, 1.0391935616446096e-27, 16.389340781838836, 0.0, 16.389340781838847, 0.0, 3.412504910144463, -8.499030667796173, 3.4125049101454676, 0.0, -5.511985867292692, -9.177060293757688, -9.30506566333274], 'intercept': -36.85833333333499, 'uncertainty': 49.94242756390284}, 'bad': {'mean': [0.0, 0.8611111111111132, 0.8333333333333323, 0.8333333333333323, 0.34375, 0.15625, 0.05555555555555543, 0.1608055555555556, 1.0049408333333334, -0.002111111111111112, 0.020888888888888898, -0.008472222222222225, 1.4166666666666674e-05, -0.00034083333333333345, 0.00011583333333333333, 0.10999999999999986, 0.23725000000000024, 0.7627499999999994, 0.6586013333333337, 0.5295818333333332, 0.12901949999999998, 0.3000000000000001, 1.0, 0.6000000000000002, 0.75, 0.45000000000000084, 1.0, 0.45000000000000084, 0.0, 0.7000000000000015, 0.40499999999999914, 1.0, 0.40499999999999914, 0.0, 0.018000000000000013, 0.5649999999999997, 0.6300000000000014, 0.75, 0.2440000000000005, 0.4700000000000003, 0.695999999999998], 'scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2391663746933306, 0.004548030633020066, 0.019085253825407895, 0.03247144994804506, 0.045797250106135765, 0.00029588731901782403, 0.0005323682674824095, 0.0006955807924956592, 1.0, 0.010951598056904764, 0.01095159805690476, 0.11198016429955596, 0.10755274514537304, 0.06550514728948151, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.014999999999999972, 1.0, 0.014999999999999972, 1.0, 0.02749545416973506, 0.05937171043518962, 0.045825756949558406, 1.0, 0.019595917942265447, 0.03376388603226828, 0.05499090833947002], 'beta': [0.0, -2.5752127494604052e-30, 1.2465284278719634e-30, 1.2465284278719634e-30, 0.0, 0.0, 1.6095079684127533e-31, -0.02086206576323683, -0.037150097309371334, -0.009474185080351931, -0.005862838013657261, 0.0020961609920262234, -0.009258115878191872, -0.06201166734359232, 0.06547341839707671, 1.558160534839941e-31, -0.00581884916902775, 0.0058188491690273104, 0.013763830017763682, 0.017266363477054462, -0.004820519560992282, 0.0, 0.0, 0.0, 0.0, -5.46100169494846e-31, 0.0, -5.46100169494846e-31, 0.0, -1.6237847674977618e-30, 0.01924427767601479, 0.0, 0.019244277676014814, 0.0, -0.01603559594909374, 0.017339546251619542, -0.016035595949094494, 0.0, -0.001485747487664145, 0.011333909296890637, 0.014447265074207235], 'intercept': 0.033333333333328906, 'uncertainty': 0.1608709976303288}, 'train_mean_delta': -36.858333333333334, 'train_min_delta': -229.0, 'train_positive_rate': 0.09166666666666666, 'train_bad_rate': 0.03333333333333333}}, 'variant': 'rl010c_bidirectional_opp'}
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
