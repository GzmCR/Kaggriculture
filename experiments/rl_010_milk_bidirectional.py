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
RL010_MIN_GAP = 1
RL010_MAX_GAP = 72
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


def rl010_route_opportunities(actions):
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
            if RL010_MIN_GAP <= gap <= RL010_MAX_GAP:
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


def rl010_shed_total(obs):
    private = rl010_get(obs, "private", {}) or {}
    shed = rl010_get(private, "shed", {}) or {}
    return sum(max(0, rl010_int(value)) for value in shed.values()) if isinstance(shed, dict) else 0


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
    shed_total = rl010_shed_total(obs)

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
        min(2.0, rl010_private_inventory(obs) / 100.0),
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

    def _apply_pending(self, action, step):
        pending = self.pending
        if not pending or int(pending.get("due_step", -1)) != int(step):
            return action, True, 0
        trial = copy.deepcopy(action)
        ok = rl010_adjust_sell(trial, int(pending["delta"]))
        if not ok:
            self.repayment_failures += 1
            self.pending = None
            self.fallbacks += 1
            return action, False, 0
        self.pending = None
        self.repayment_successes += 1
        return trial, True, abs(int(pending["delta"]))

    def _legal_units(self, action, obs, opportunity, action_name):
        current = rl010_sell_quantity(action)
        if current <= 0:
            return 0
        if action_name.startswith("ADVANCE"):
            ratio = RL010_RATIOS[action_name]
            desired = rl010_round_half_up(opportunity["future_quantity"] * ratio)
            visible = rl010_private_inventory(obs)
            return max(0, min(desired, opportunity["future_quantity"], max(0, visible - current)))
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
        if len(farms) == 2 and seat in (0, 1):
            mine = rl010_float(rl010_get(farms[seat], "money", 0))
            other = rl010_float(rl010_get(farms[1 - seat], "money", 0))
            if action_name.startswith("DELAY") and mine + 1000 < other:
                return False, "cash_lag"
        if action_name.startswith("ADVANCE"):
            if rl010_sell_quantity(action) + units > rl010_private_inventory(obs):
                return False, "inventory_short"
        if int(opportunity["future_step"]) <= step:
            return False, "future_not_after_current"
        return True, "safe"

    def act(self, obs, base_action):
        step = rl010_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl010_align_hands(base_action, obs)
        self.last_route_action = copy.deepcopy(action)
        action, repayment_ok, repayment_units = self._apply_pending(action, step)
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
