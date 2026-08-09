"""Observation-aware, conservative sale-timing policy for RL-004.

The v22 farmer and hands route stays fixed. RL-004 only moves one unit from
an existing premium SELL event to the next same-product event when that exact
event has enough paired counterfactual support and a positive lower confidence
bound.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np


RL004_PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
RL004_SUPPORTED_ITEMS = ("MILK", "STRAWBERRY")
RL004_MIN_SUPPORT = 12
RL004_MIN_EXPECTED_DELTA = 5.0
RL004_LCB_Z = 1.5
RL004_MAX_DELAYED_ORDERS = 8
RL004_MIN_GAP = 4
RL004_MAX_GAP = 72
RL004_CUTOFF = 648
RL004_FEATURE_DIM = 29


def rl004_as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def rl004_as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rl004_get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def rl004_step(obs):
    raw_step = rl004_get(obs, "step", None)
    if raw_step is not None:
        return max(0, rl004_as_int(raw_step))
    day = rl004_as_int(rl004_get(obs, "day", 0))
    hour = rl004_as_int(rl004_get(obs, "hour", 0))
    return max(0, day * 24 + hour)


def rl004_item_key(item, current_step, future_step):
    return f"{str(item).upper()}|{int(current_step)}|{int(future_step)}"


def rl004_route_opportunities(actions):
    events = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if (
                not isinstance(order, (list, tuple))
                or len(order) < 3
                or str(order[0]).upper() != "SELL"
            ):
                continue
            item = str(order[1]).upper()
            if item not in RL004_PREMIUM:
                continue
            quantity = max(0, rl004_as_int(order[2]))
            if quantity:
                events[item][int(step)] = events[item].get(int(step), 0) + quantity
    opportunities = []
    for item, rows in events.items():
        ordered = sorted(rows.items())
        for index, (current_step, current_quantity) in enumerate(ordered[:-1]):
            future_step, future_quantity = ordered[index + 1]
            gap = future_step - current_step
            if RL004_MIN_GAP <= gap <= RL004_MAX_GAP:
                opportunities.append({
                    "item": item,
                    "current_step": int(current_step),
                    "future_step": int(future_step),
                    "current_quantity": int(current_quantity),
                    "future_quantity": int(future_quantity),
                    "gap": int(gap),
                })
    return sorted(opportunities, key=lambda row: (row["current_step"], row["item"]))


def rl004_opportunity_index(opportunities):
    index = defaultdict(list)
    for row in opportunities or []:
        if int(row["current_step"]) >= RL004_CUTOFF:
            continue
        if int(row["future_step"]) >= 672:
            continue
        index[int(row["current_step"])].append(row)
    return index


def rl004_public_supply(farm, item):
    total = 0
    for row in rl004_get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            animal = str(tile.get("animal", "")).upper()
            crop = str(tile.get("crop", "")).upper()
            if item == "MILK" and animal == "COW":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
            elif item == "WOOL" and animal == "SHEEP":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
            elif item == "MELON" and crop == "MELON":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
            elif item == "STRAWBERRY" and crop == "STRAWBERRY":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
    return total


def rl004_private_inventory(obs, item):
    private = rl004_get(obs, "private", {}) or {}
    shed = rl004_get(private, "shed", {}) or {}
    carried = 0
    for inventory in rl004_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            carried += max(0, rl004_as_int(inventory.get(item, 0)))
    return max(0, rl004_as_int(shed.get(item, 0))) + carried


class RL004FeatureState:
    """Observable price and market-inventory history."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_step = -1
        self.prices = {item: [] for item in RL004_PREMIUM}
        self.inventories = {item: [] for item in RL004_PREMIUM}

    def observe(self, obs):
        step = rl004_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        market = rl004_get(obs, "market", {}) or {}
        prices = rl004_get(market, "prices", {}) or {}
        inventory = rl004_get(market, "inventory", {}) or {}
        for item in RL004_PREMIUM:
            self.prices[item].append((step, rl004_as_float(prices.get(item, 0))))
            self.inventories[item].append((step, rl004_as_float(inventory.get(item, 0))))
            self.prices[item] = self.prices[item][-96:]
            self.inventories[item] = self.inventories[item][-96:]
        self.last_step = step

    @staticmethod
    def _lagged(rows, step, lag):
        target = int(step) - int(lag)
        values = [value for seen_step, value in rows if seen_step <= target]
        return values[-1] if values else None

    def momentum(self, item, step):
        current_price = self.prices[item][-1][1] if self.prices[item] else 0.0
        current_inventory = self.inventories[item][-1][1] if self.inventories[item] else 0.0
        price12 = self._lagged(self.prices[item], step, 12)
        price24 = self._lagged(self.prices[item], step, 24)
        inv12 = self._lagged(self.inventories[item], step, 12)
        inv24 = self._lagged(self.inventories[item], step, 24)
        return (
            current_price,
            current_inventory,
            0.0 if price12 is None else current_price - price12,
            0.0 if price24 is None else current_price - price24,
            0.0 if inv12 is None else current_inventory - inv12,
            0.0 if inv24 is None else current_inventory - inv24,
        )


def rl004_features(obs, opportunity, history, base_action=None):
    item = str(opportunity["item"]).upper()
    step = rl004_as_int(opportunity["current_step"])
    future_step = rl004_as_int(opportunity["future_step"])
    current_quantity = max(0, rl004_as_int(opportunity["current_quantity"]))
    future_quantity = max(0, rl004_as_int(opportunity["future_quantity"]))
    gap = max(0, future_step - step)
    values = [
        1.0,
        step / 720.0,
        future_step / 720.0,
        (step // 24) / 30.0,
        (step % 24) / 24.0,
        min(1.0, current_quantity / 32.0),
        min(1.0, future_quantity / 32.0),
        max(-1.0, min(1.0, (future_quantity - current_quantity) / 32.0)),
        min(1.0, gap / 72.0),
    ]
    values.extend(1.0 if item == name else 0.0 for name in RL004_PREMIUM)

    price, market_inventory, price12, price24, inv12, inv24 = history.momentum(item, step)
    values.extend([
        min(2.0, price / 300.0),
        min(2.0, market_inventory / 10000.0),
        max(-2.0, min(2.0, price12 / 300.0)),
        max(-2.0, min(2.0, price24 / 300.0)),
        max(-2.0, min(2.0, inv12 / 10000.0)),
        max(-2.0, min(2.0, inv24 / 10000.0)),
    ])

    farms = list(rl004_get(obs, "farms", []) or [])
    seat = rl004_as_int(rl004_get(obs, "player", 0))
    mine = farms[seat] if 0 <= seat < len(farms) else {}
    other = farms[1 - seat] if len(farms) > 1 and seat in (0, 1) else {}
    mine_money = rl004_as_float(rl004_get(mine, "money", 0))
    other_money = rl004_as_float(rl004_get(other, "money", 0))
    market_orders = len((base_action or {}).get("market", []) or [])
    values.extend([
        min(2.0, rl004_private_inventory(obs, item) / 100.0),
        min(2.0, rl004_public_supply(mine, item) / 100.0),
        min(2.0, rl004_public_supply(other, item) / 100.0),
        max(-2.0, min(2.0, (mine_money - other_money) / 100000.0)),
        min(1.0, market_orders / 10.0),
        min(1.0, len(rl004_get(rl004_get(obs, "town", {}) or {}, "unlocked_shops", []) or []) / 8.0),
        min(1.0, len(rl004_get(mine, "hands", []) or []) / 20.0),
        min(1.0, len(rl004_get(other, "hands", []) or []) / 20.0),
        min(1.0, len(rl004_get(mine, "unlocked_quadrants", []) or []) / 4.0),
        min(1.0, len(rl004_get(other, "unlocked_quadrants", []) or []) / 4.0),
    ])
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (RL004_FEATURE_DIM,):
        raise AssertionError(f"RL004 feature size {array.size} != {RL004_FEATURE_DIM}")
    return array


def rl004_normalize_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(value or ["PASS"]) for value in action.get("hands", []) or []],
        "market": [list(value) for value in action.get("market", []) or [] if isinstance(value, list)],
    }


def rl004_align_hands(action, obs):
    action = rl004_normalize_action(action)
    farms = list(rl004_get(obs, "farms", []) or [])
    seat = rl004_as_int(rl004_get(obs, "player", 0))
    expected = len(rl004_get(farms[seat], "hands", []) or []) if 0 <= seat < len(farms) else 0
    action["hands"].extend([["PASS"] for _ in range(max(0, expected - len(action["hands"])) )])
    action["hands"] = action["hands"][:expected]
    return action


def rl004_adjust_sell(action, item, delta):
    for index, order in enumerate(action.get("market", []) or []):
        if (
            len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == str(item).upper()
        ):
            current = max(0, rl004_as_int(order[2]))
            updated = current + int(delta)
            if updated < 0:
                return 0
            if updated == 0:
                action["market"].pop(index)
            else:
                action["market"][index] = [order[0], order[1], updated]
            return abs(int(delta))
    return 0


class RL004Policy:
    def __init__(self, payload=None):
        payload = payload or {}
        self.feature_dim = int(payload.get("feature_dim", RL004_FEATURE_DIM))
        self.min_support = int(payload.get("min_support", RL004_MIN_SUPPORT))
        self.min_expected_delta = float(payload.get("min_expected_delta", RL004_MIN_EXPECTED_DELTA))
        self.lcb_z = float(payload.get("lcb_z", RL004_LCB_Z))
        self.models = dict(payload.get("models", {}))
        if self.feature_dim != RL004_FEATURE_DIM:
            raise ValueError("RL004 feature dimension mismatch")

    def predict(self, key, features):
        model = self.models.get(key)
        if not model or int(model.get("support", 0)) < self.min_support:
            return None
        mean = np.asarray(model.get("mean", [0.0] * self.feature_dim), dtype=np.float64)
        scale = np.asarray(model.get("scale", [1.0] * self.feature_dim), dtype=np.float64)
        beta = np.asarray(model.get("beta", [0.0] * self.feature_dim), dtype=np.float64)
        x = (np.asarray(features, dtype=np.float64) - mean) / np.maximum(scale, 1e-9)
        prediction = float(model.get("intercept", 0.0) + x @ beta)
        uncertainty = max(1.0, float(model.get("uncertainty", 1.0)))
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "lcb": prediction - self.lcb_z * uncertainty,
            "support": int(model.get("support", 0)),
        }

    def select(self, key, item, features):
        if str(item).upper() not in RL004_SUPPORTED_ITEMS:
            return {"selected": False, "reason": "item_not_enabled"}
        result = self.predict(key, features)
        if result is None:
            return {"selected": False, "reason": "unsupported_event"}
        model = self.models.get(key, {})
        # Require a positive training-set margin before trusting a contextual
        # prediction. This keeps sparse event models from extrapolating into a
        # profitable-looking but unsupported market state.
        if float(model.get("train_mean_delta", -math.inf)) < self.min_expected_delta:
            result["selected"] = False
            result["reason"] = "training_mean_below_gate"
            return result
        if float(model.get("train_min_delta", -math.inf)) < 0.0:
            result["selected"] = False
            result["reason"] = "training_min_negative"
            return result
        if result["prediction"] < self.min_expected_delta:
            result["selected"] = False
            result["reason"] = "below_expected_delta"
            return result
        if result["lcb"] <= 0.0:
            result["selected"] = False
            result["reason"] = "non_positive_lcb"
            return result
        result["selected"] = True
        result["reason"] = "positive_lcb"
        return result


class RL004Runtime:
    def __init__(self, payload=None, opportunities=None):
        self.policy = RL004Policy(payload)
        self.index = rl004_opportunity_index(opportunities or [])
        self.history = RL004FeatureState()
        self.pending = defaultdict(int)
        self.last_step = -1
        self.delayed_orders = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.decisions = []
        self.errors = 0

    def reset(self):
        self.history.reset()
        self.pending.clear()
        self.last_step = -1
        self.delayed_orders = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.decisions = []

    def _apply_pending(self, action, step):
        changed = 0
        for key in [key for key in self.pending if key[0] == int(step)]:
            _, item = key
            quantity = self.pending.pop(key, 0)
            if quantity:
                changed += rl004_adjust_sell(action, item, quantity)
        return changed

    def act(self, obs, base_action):
        step = rl004_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl004_align_hands(base_action, obs)
        changed = self._apply_pending(action, step)
        candidates = self.index.get(step, [])
        if candidates and self.delayed_orders < RL004_MAX_DELAYED_ORDERS:
            scored = []
            for opportunity in candidates:
                key = rl004_item_key(
                    opportunity["item"], opportunity["current_step"], opportunity["future_step"]
                )
                features = rl004_features(obs, opportunity, self.history, action)
                result = self.policy.select(key, opportunity["item"], features)
                scored.append((float(result.get("lcb", -math.inf)), opportunity, result))
            _, opportunity, result = max(scored, key=lambda row: row[0])
            moved = 0
            if result.get("selected"):
                moved = rl004_adjust_sell(action, opportunity["item"], -1)
                if moved:
                    self.pending[(int(opportunity["future_step"]), opportunity["item"])] += moved
                    self.delayed_orders += moved
                    changed += moved
            self.decisions.append({
                "step": int(step),
                "item": opportunity["item"],
                "future_step": int(opportunity["future_step"]),
                "selected": bool(result.get("selected", False) and moved),
                "moved": int(moved),
                "reason": result.get("reason", "unsupported_event"),
                "prediction": float(result.get("prediction", 0.0)),
                "lcb": float(result.get("lcb", 0.0)),
                "support": int(result.get("support", 0)),
            })
        if changed:
            self.changed_calls += 1
            self.changed_units += int(changed)
        self.last_step = step
        return action


def rl004_fit_models(samples, ridge=8.0, min_support=RL004_MIN_SUPPORT):
    grouped = defaultdict(list)
    for row in samples:
        key = rl004_item_key(row["item"], row["current_step"], row["future_step"])
        grouped[key].append(row)
    models = {}
    report = {"groups": {}, "skipped_groups": {}}
    for key, rows in sorted(grouped.items()):
        unique_episodes = {(row.get("seed"), row.get("seat"), row.get("opponent", "v22")) for row in rows}
        support = len(unique_episodes)
        if support < int(min_support):
            report["skipped_groups"][key] = {"rows": len(rows), "support": support}
            continue
        matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
        target = np.asarray([rl004_as_float(row["cash_delta"]) for row in rows], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != RL004_FEATURE_DIM:
            raise ValueError(f"invalid feature matrix for {key}: {matrix.shape}")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        mean[0] = 0.0
        scale[0] = 1.0
        scale = np.where(scale < 1e-9, 1.0, scale)
        normalized = (matrix - mean) / scale
        design = np.column_stack((np.ones(len(rows), dtype=np.float64), normalized))
        penalty = np.eye(RL004_FEATURE_DIM + 1, dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        intercept = float(coefficients[0])
        beta = coefficients[1:]
        residual = target - design @ coefficients
        residual_std = float(np.std(residual))
        uncertainty = max(1.0, residual_std * (1.0 + 1.0 / math.sqrt(max(1, support))))
        prediction = intercept + normalized @ beta
        train_mean_delta = float(target.mean())
        train_min_delta = float(target.min())
        train_positive_rate = float(np.mean(target > 0))
        models[key] = {
            "support": support,
            "rows": len(rows),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "beta": beta.tolist(),
            "intercept": intercept,
            "uncertainty": uncertainty,
            "train_mean_delta": train_mean_delta,
            "train_min_delta": train_min_delta,
            "train_positive_rate": train_positive_rate,
        }
        report["groups"][key] = {
            "rows": len(rows),
            "support": support,
            "mean_delta": train_mean_delta,
            "median_delta": float(np.median(target)),
            "min_delta": train_min_delta,
            "max_delta": float(target.max()),
            "positive_rate": train_positive_rate,
            "residual_std": residual_std,
            "mean_prediction": float(prediction.mean()),
        }
    target_values = [rl004_as_float(row["cash_delta"]) for row in samples]
    report["samples"] = len(samples)
    report["models"] = len(models)
    report["mean_cash_delta"] = float(np.mean(target_values)) if target_values else 0.0
    report["min_support"] = int(min_support)
    return {
        "version": "rl004",
        "feature_dim": RL004_FEATURE_DIM,
        "min_support": int(min_support),
        "min_expected_delta": RL004_MIN_EXPECTED_DELTA,
        "lcb_z": RL004_LCB_Z,
        "models": models,
    }, report


def rl004_load_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
