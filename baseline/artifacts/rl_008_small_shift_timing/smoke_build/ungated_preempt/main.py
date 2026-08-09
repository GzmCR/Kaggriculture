"""RL-006 bidirectional premium-sale timing overlay.

The V022c/V22 farmer and hands route remains fixed.  A shared feature contract
and action-aware ridge heads choose between keeping, preempting, or delaying a
single existing premium SELL wave.  The module is intentionally self-contained
so the builder can prepend it to a single-file Kaggle submission.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np


RL006_PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
RL006_ACTION_NAMES = {
    0: "CONTROL",
    1: "PREEMPT_1",
    2: "PREEMPT_25",
    3: "PREEMPT_50",
    4: "DELAY_1",
    5: "DELAY_25",
    6: "DELAY_50",
}
RL006_NONCONTROL = tuple(range(1, 7))
RL006_MIN_SUPPORT = 12
RL006_MIN_EXPECTED_DELTA = 5.0
RL006_LCB_Z = 1.5
RL006_MIN_GAP = 4
RL006_MAX_GAP = 72
RL006_CUTOFF = 648
RL006_TERMINAL_CUTOFF = 672
RL006_FEATURE_DIM = 78

RL006_BASE_PRICE = {"WOOL": 200.0, "STRAWBERRY": 120.0, "MILK": 160.0, "MELON": 250.0}
RL006_INITIAL_INVENTORY = {item: 10000.0 for item in RL006_PREMIUM}
RL006_PRODUCER = {"MILK": ("animal", "COW"), "WOOL": ("animal", "SHEEP")}


def rl006_get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def rl006_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def rl006_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rl006_step(obs):
    raw = rl006_get(obs, "step", None)
    if raw is not None:
        return max(0, rl006_int(raw))
    return max(0, rl006_int(rl006_get(obs, "day", 0)) * 24 + rl006_int(rl006_get(obs, "hour", 0)))


def rl006_normalize_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(value or ["PASS"]) for value in action.get("hands", []) or []],
        "market": [list(value) for value in action.get("market", []) or [] if isinstance(value, list)],
    }


def rl006_align_hands(action, obs):
    action = rl006_normalize_action(action)
    farms = list(rl006_get(obs, "farms", []) or [])
    seat = rl006_int(rl006_get(obs, "player", 0))
    expected = 0
    if 0 <= seat < len(farms):
        expected = len(rl006_get(farms[seat], "hands", []) or [])
    action["hands"].extend([["PASS"] for _ in range(max(0, expected - len(action["hands"])) )])
    action["hands"] = action["hands"][:expected]
    return action


def rl006_event_key(opportunity):
    return "{}|{}|{}".format(
        str(opportunity["item"]).upper(),
        rl006_int(opportunity["current_step"]),
        rl006_int(opportunity["future_step"]),
    )


def rl006_route_opportunities(actions):
    events = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            if str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item not in RL006_PREMIUM:
                continue
            quantity = max(0, rl006_int(order[2]))
            if quantity:
                events[item][int(step)] = events[item].get(int(step), 0) + quantity
    opportunities = []
    for item, rows in events.items():
        ordered = sorted(rows.items())
        for index, (current_step, current_quantity) in enumerate(ordered[:-1]):
            future_step, future_quantity = ordered[index + 1]
            gap = future_step - current_step
            if RL006_MIN_GAP <= gap <= RL006_MAX_GAP:
                opportunities.append({
                    "item": item,
                    "current_step": int(current_step),
                    "future_step": int(future_step),
                    "current_quantity": int(current_quantity),
                    "future_quantity": int(future_quantity),
                    "gap": int(gap),
                    "event_index": int(index),
                })
    return sorted(opportunities, key=lambda row: (row["current_step"], row["item"]))


def rl006_opportunity_index(opportunities):
    index = defaultdict(list)
    for row in opportunities or []:
        if rl006_int(row["current_step"]) >= RL006_CUTOFF:
            continue
        if rl006_int(row["future_step"]) >= RL006_TERMINAL_CUTOFF:
            continue
        index[rl006_int(row["current_step"])].append(row)
    return index


def rl006_adjust_sell(action, item, delta):
    item = str(item).upper()
    for index, order in enumerate(action.get("market", []) or []):
        if len(order) < 3 or str(order[0]).upper() != "SELL" or str(order[1]).upper() != item:
            continue
        current = max(0, rl006_int(order[2]))
        updated = current + int(delta)
        if updated < 0:
            return 0
        if updated == 0:
            action["market"].pop(index)
        else:
            action["market"][index] = [order[0], order[1], updated]
        return abs(updated - current)
    return 0


def rl006_private_inventory(obs, item):
    private = rl006_get(obs, "private", {}) or {}
    shed = rl006_get(private, "shed", {}) or {}
    total = max(0, rl006_int(rl006_get(shed, item, 0)))
    for inventory in rl006_get(private, "inventories", []) or []:
        total += max(0, rl006_int(rl006_get(inventory, item, 0)))
    return total


def rl006_inventory_stats(obs, item):
    private = rl006_get(obs, "private", {}) or {}
    shed = rl006_get(private, "shed", {}) or {}
    shed_item = max(0, rl006_int(rl006_get(shed, item, 0)))
    carried_item = 0
    shed_total = 0
    for name, quantity in (shed or {}).items():
        if str(name).upper() not in {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"}:
            continue
        shed_total += max(0, rl006_int(quantity))
    for inventory in rl006_get(private, "inventories", []) or []:
        carried_item += max(0, rl006_int(rl006_get(inventory, item, 0)))
    total_item = shed_item + carried_item
    return shed_item, carried_item, total_item, shed_total, max(0, 100 - shed_total)


def rl006_public_production(farm, item):
    count = 0
    yield_units = 0
    rows = rl006_get(farm, "tiles", []) or []
    for row in rows:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            animal = str(tile.get("animal", "")).upper()
            crop = str(tile.get("crop", "")).upper()
            matches = (
                item == "MILK" and animal == "COW"
                or item == "WOOL" and animal == "SHEEP"
                or item == "MELON" and crop == "MELON"
                or item == "STRAWBERRY" and crop == "STRAWBERRY"
            )
            if matches:
                count += 1
                yield_units += max(0, rl006_int(tile.get("yield_units", 0)))
    return count, yield_units


def rl006_public_signature(farm):
    crops = defaultdict(int)
    animals = defaultdict(int)
    for row in rl006_get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop", "")).upper()
            animal = str(tile.get("animal", "")).upper()
            if crop:
                crops[crop] += 1
            if animal:
                animals[animal] += 1
    return {
        "hands": len(rl006_get(farm, "hands", []) or []),
        "quadrants": len(rl006_get(farm, "unlocked_quadrants", []) or []),
        "COW": animals["COW"],
        "SHEEP": animals["SHEEP"],
        "GOOSE": animals["GOOSE"],
        "MELON": crops["MELON"],
        "STRAWBERRY": crops["STRAWBERRY"],
        "TOMATO": crops["TOMATO"],
        "CARROT": crops["CARROT"],
        "WHEAT": crops["WHEAT"],
    }


def rl006_clone_distance(first, second):
    a = rl006_public_signature(first)
    b = rl006_public_signature(second)
    weights = {"hands": 2.0, "quadrants": 3.0, "COW": 2.0, "SHEEP": 2.0, "GOOSE": 1.0}
    distance = 0.0
    for key, value in a.items():
        distance += weights.get(key, 1.0) * abs(float(value) - float(b.get(key, 0)))
    return min(1.0, distance / 48.0)


class RL006History:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_step = -1
        self.prices = {item: [] for item in RL006_PREMIUM}
        self.inventories = {item: [] for item in RL006_PREMIUM}

    def observe(self, obs):
        step = rl006_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        market = rl006_get(obs, "market", {}) or {}
        prices = rl006_get(market, "prices", {}) or {}
        inventories = rl006_get(market, "inventory", {}) or {}
        for item in RL006_PREMIUM:
            self.prices[item].append((step, rl006_float(rl006_get(prices, item, 0))))
            self.inventories[item].append((step, rl006_float(rl006_get(inventories, item, 0))))
            self.prices[item] = self.prices[item][-128:]
            self.inventories[item] = self.inventories[item][-128:]
        self.last_step = step

    @staticmethod
    def _value(rows, step, lag):
        target = int(step) - int(lag)
        values = [value for seen, value in rows if seen <= target]
        return values[-1] if values else None

    def state(self, item, step):
        prices = self.prices[item]
        inventories = self.inventories[item]
        current_price = prices[-1][1] if prices else 0.0
        current_inventory = inventories[-1][1] if inventories else 0.0
        price_changes = []
        inventory_changes = []
        for lag in (6, 12, 24, 48):
            old_price = self._value(prices, step, lag)
            old_inventory = self._value(inventories, step, lag)
            price_changes.append(0.0 if old_price is None else current_price - old_price)
            inventory_changes.append(0.0 if old_inventory is None else current_inventory - old_inventory)
        recent_prices = [value for seen, value in prices if seen >= int(step) - 24]
        long_prices = [value for seen, value in prices if seen >= int(step) - 96]
        return {
            "price": current_price,
            "inventory": current_inventory,
            "price_changes": price_changes,
            "inventory_changes": inventory_changes,
            "volatility_short": float(np.std(recent_prices)) if len(recent_prices) > 1 else 0.0,
            "volatility_long": float(np.std(long_prices)) if len(long_prices) > 1 else 0.0,
        }


def rl006_town_drain(obs, item, horizon, step):
    del item
    shops = len(rl006_get(rl006_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    intervals = max(0, (int(step) + int(horizon)) // 12 - int(step) // 12)
    center_mult = 1 if int(step) < 240 else 2 if int(step) < 480 else 4
    center = intervals * center_mult
    shop_turns = max(0, int(horizon) // 4)
    return float(center + shops * shop_turns * 0.25)


def rl006_cash_needed(action, obs):
    prices = rl006_get(rl006_get(obs, "market", {}) or {}, "prices", {}) or {}
    seed_prices = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
    animal_prices = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
    total = 0.0
    for order in action.get("market", []) or []:
        if not isinstance(order, list) or len(order) < 3:
            continue
        op = str(order[0]).upper()
        quantity = max(0, rl006_int(order[2]))
        if op == "BUY_SEED":
            total += seed_prices.get(str(order[1]).upper(), 0) * quantity
        elif op == "BUY_ANIMAL":
            total += animal_prices.get(str(order[1]).upper(), 0) * quantity
        elif op == "BUY_PRODUCT":
            total += rl006_float(rl006_get(prices, str(order[1]).upper(), 0)) * quantity
    return total


def rl006_queue_stats(action, item, current_quantity, future_quantity):
    market = action.get("market", []) or []
    target_indices = []
    item_count = 0
    item_total = 0
    premium_count = 0
    current_order_quantity = 0
    for index, order in enumerate(market):
        if not isinstance(order, list) or len(order) < 3:
            continue
        if str(order[0]).upper() != "SELL":
            continue
        order_item = str(order[1]).upper()
        quantity = max(0, rl006_int(order[2]))
        if order_item in RL006_PREMIUM:
            premium_count += 1
        if order_item == str(item).upper():
            target_indices.append(index)
            item_count += 1
            item_total += quantity
            current_order_quantity += quantity
    target_index = min(target_indices) if target_indices else 10
    return (
        min(1.0, len(market) / 10.0),
        min(1.0, max(0, 10 - len(market)) / 10.0),
        min(1.0, target_index / 10.0),
        min(1.0, item_count / 3.0),
        min(2.0, item_total / 100.0),
        min(1.0, premium_count / 6.0),
        min(2.0, current_order_quantity / 64.0),
        min(2.0, max(0, rl006_int(future_quantity)) / 64.0),
    )


def rl006_features(obs, opportunity, history, base_action=None, pending=None, blocked=False):
    item = str(opportunity["item"]).upper()
    step = rl006_int(opportunity["current_step"])
    future_step = rl006_int(opportunity["future_step"])
    current_quantity = max(0, rl006_int(opportunity["current_quantity"]))
    future_quantity = max(0, rl006_int(opportunity["future_quantity"]))
    gap = max(0, future_step - step)
    days_left = max(0, 720 - step) / 720.0
    ratio = future_quantity / max(1.0, float(current_quantity))
    values = [
        1.0,
        step / 720.0,
        future_step / 720.0,
        (step // 24) / 30.0,
        (step % 24) / 24.0,
        min(1.0, gap / 72.0),
        days_left,
        min(2.0, current_quantity / 64.0),
        min(2.0, future_quantity / 64.0),
        max(-2.0, min(2.0, (future_quantity - current_quantity) / 64.0)),
        max(0.0, min(4.0, ratio)) / 4.0,
    ]
    values.extend(1.0 if item == name else 0.0 for name in RL006_PREMIUM)

    market_state = history.state(item, step)
    base_price = RL006_BASE_PRICE.get(item, 100.0)
    current_price = market_state["price"]
    inventory = market_state["inventory"]
    price_changes = market_state["price_changes"]
    inventory_changes = market_state["inventory_changes"]
    values.extend([
        min(3.0, current_price / base_price),
        float(current_price <= 1.0),
        min(2.0, inventory / RL006_INITIAL_INVENTORY[item]),
        max(-2.0, min(2.0, price_changes[0] / base_price)),
        max(-2.0, min(2.0, price_changes[1] / base_price)),
        max(-2.0, min(2.0, price_changes[2] / base_price)),
        max(-2.0, min(2.0, price_changes[3] / base_price)),
        max(-2.0, min(2.0, inventory_changes[0] / 10000.0)),
        max(-2.0, min(2.0, inventory_changes[1] / 10000.0)),
        max(-2.0, min(2.0, inventory_changes[2] / 10000.0)),
        max(-2.0, min(2.0, inventory_changes[3] / 10000.0)),
        min(2.0, market_state["volatility_short"] / base_price),
        min(2.0, market_state["volatility_long"] / base_price),
        max(-2.0, min(2.0, inventory_changes[1] / max(1.0, gap))),
    ])

    momentum = 0.5 * price_changes[1] + 0.25 * price_changes[2]
    forecast_price = max(1.0, min(base_price * 3.0, current_price + momentum * gap / 24.0))
    slope = abs(price_changes[1]) / max(1.0, abs(inventory_changes[1]))
    if slope <= 0.0:
        slope = max(0.05, current_price * 0.002)

    def marginal(quantity, price):
        quantity = max(1, int(quantity))
        impacted = max(1.0, price - slope * quantity)
        return impacted * quantity

    quantities = [1, max(1, int(math.ceil(current_quantity * 0.25))), max(1, int(math.ceil(current_quantity * 0.50)))]
    future_quantities = [1, max(1, int(math.ceil(future_quantity * 0.25))), max(1, int(math.ceil(future_quantity * 0.50)))]
    now_revenue = [marginal(quantity, current_price) for quantity in quantities]
    future_revenue = [marginal(quantity, forecast_price) for quantity in future_quantities]
    values.extend([
        min(3.0, forecast_price / base_price),
        max(-2.0, min(2.0, (current_price - forecast_price) / base_price)),
        min(3.0, now_revenue[0] / 10000.0),
        min(3.0, now_revenue[1] / 10000.0),
        min(3.0, now_revenue[2] / 10000.0),
        min(3.0, future_revenue[0] / 10000.0),
        min(3.0, future_revenue[1] / 10000.0),
        min(3.0, future_revenue[2] / 10000.0),
        min(2.0, (current_price - max(1.0, current_price - slope)) / base_price),
        min(2.0, (current_price - max(1.0, current_price - slope * quantities[1])) / base_price),
        min(2.0, (current_price - max(1.0, current_price - slope * quantities[2])) / base_price),
    ])

    own_farm = {}
    other_farm = {}
    farms = list(rl006_get(obs, "farms", []) or [])
    seat = rl006_int(rl006_get(obs, "player", 0))
    if 0 <= seat < len(farms):
        own_farm = farms[seat]
    if len(farms) > 1 and seat in (0, 1):
        other_farm = farms[1 - seat]
    own_count, own_yield = rl006_public_production(own_farm, item)
    other_count, other_yield = rl006_public_production(other_farm, item)
    own_signature = rl006_public_signature(own_farm)
    other_signature = rl006_public_signature(other_farm)
    own_money = rl006_float(rl006_get(own_farm, "money", 0))
    other_money = rl006_float(rl006_get(other_farm, "money", 0))
    own_item, carried_item, total_item, shed_total, capacity = rl006_inventory_stats(obs, item)
    drain12 = rl006_town_drain(obs, item, 12, step)
    drain24 = rl006_town_drain(obs, item, 24, step)
    drain72 = rl006_town_drain(obs, item, 72, step)
    projected_supply = float(max(1, own_yield + other_yield + current_quantity + future_quantity))
    supply_gap = (drain72 - projected_supply) / projected_supply
    values.extend([
        min(2.0, drain12 / 32.0),
        min(2.0, drain24 / 64.0),
        min(2.0, drain72 / 192.0),
        max(-2.0, min(2.0, supply_gap)),
    ])
    cash_needed = rl006_cash_needed(base_action or {}, obs)
    values.extend([
        min(2.0, own_item / 100.0),
        min(2.0, carried_item / 100.0),
        min(2.0, total_item / 100.0),
        min(2.0, shed_total / 100.0),
        min(1.0, capacity / 100.0),
        min(3.0, own_money / 100000.0),
        max(-2.0, min(2.0, (own_money - other_money) / 100000.0)),
        min(2.0, cash_needed / 50000.0),
    ])
    distance = rl006_clone_distance(own_farm, other_farm)
    premium_own = sum(own_signature.get(name, 0) for name in ("COW", "SHEEP", "MELON", "STRAWBERRY"))
    premium_other = sum(other_signature.get(name, 0) for name in ("COW", "SHEEP", "MELON", "STRAWBERRY"))
    values.extend([
        min(2.0, own_yield / 100.0),
        min(2.0, other_yield / 100.0),
        min(2.0, other_yield / max(1.0, own_yield)),
        min(1.0, own_count / 20.0),
        min(1.0, other_count / 20.0),
        min(2.0, own_yield / max(1.0, own_count) / 10.0),
        min(2.0, other_yield / max(1.0, other_count) / 10.0),
        max(-1.0, min(1.0, (len(rl006_get(own_farm, "hands", []) or []) - len(rl006_get(other_farm, "hands", []) or [])) / 20.0)),
        max(-1.0, min(1.0, (len(rl006_get(own_farm, "unlocked_quadrants", []) or []) - len(rl006_get(other_farm, "unlocked_quadrants", []) or [])) / 4.0)),
        distance,
        float(distance <= 0.15),
        min(2.0, premium_own / 20.0),
        min(2.0, premium_other / 20.0),
    ])
    queue = rl006_queue_stats(base_action or {}, item, current_quantity, future_quantity)
    values.extend(queue)
    pending = pending or {}
    pending_item = sum(max(0, rl006_int(value.get("quantity", 0))) for value in pending.values() if value.get("item") == item)
    pending_total = sum(max(0, rl006_int(value.get("quantity", 0))) for value in pending.values())
    due_distances = [max(0, rl006_int(value.get("due_step", step)) - step) for value in pending.values() if value.get("item") == item]
    values.extend([
        min(2.0, pending_item / 100.0),
        min(1.0, pending_total / 100.0),
        min(1.0, (min(due_distances) if due_distances else 72) / 72.0),
        float(bool(blocked)),
        float(any(value.get("item") == item and value.get("current_step") == step for value in pending.values())),
    ])
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (RL006_FEATURE_DIM,):
        raise AssertionError(f"RL006 feature size {array.size} != {RL006_FEATURE_DIM}")
    return array


def rl006_action_quantity(action_id, opportunity):
    current = max(0, rl006_int(opportunity["current_quantity"]))
    future = max(0, rl006_int(opportunity["future_quantity"]))
    if action_id in (1, 4):
        return 1
    if action_id in (2, 5):
        return max(1, int(math.ceil((future if action_id == 2 else current) * 0.25)))
    if action_id in (3, 6):
        return max(1, int(math.ceil((future if action_id == 3 else current) * 0.50)))
    return 0


def rl006_action_direction(action_id):
    if action_id in (1, 2, 3):
        return "PREEMPT"
    if action_id in (4, 5, 6):
        return "DELAY"
    return "CONTROL"


class RL006Policy:
    def __init__(self, payload=None):
        payload = payload or {}
        self.feature_dim = int(payload.get("feature_dim", RL006_FEATURE_DIM))
        self.min_support = int(payload.get("min_support", RL006_MIN_SUPPORT))
        self.min_expected_delta = float(payload.get("min_expected_delta", RL006_MIN_EXPECTED_DELTA))
        self.lcb_z = float(payload.get("lcb_z", RL006_LCB_Z))
        self.feature_mean = np.asarray(payload.get("feature_mean", [0.0] * self.feature_dim), dtype=np.float64)
        self.feature_scale = np.asarray(payload.get("feature_scale", [1.0] * self.feature_dim), dtype=np.float64)
        self.models = dict(payload.get("models", {}))
        self.allowed_actions = tuple(int(value) for value in payload.get("allowed_actions", RL006_NONCONTROL))
        self.supported_events = set(str(value) for value in payload.get("supported_events", []))
        if self.feature_dim != RL006_FEATURE_DIM:
            raise ValueError("RL006 feature dimension mismatch")

    def predict(self, action_id, features):
        model = self.models.get(str(int(action_id)))
        if not model or int(model.get("support", 0)) < self.min_support:
            return None
        x = (np.asarray(features, dtype=np.float64) - self.feature_mean) / np.maximum(self.feature_scale, 1e-9)
        beta = np.asarray(model.get("beta", [0.0] * self.feature_dim), dtype=np.float64)
        prediction = float(model.get("intercept", 0.0) + x @ beta)
        uncertainty = max(1.0, float(model.get("uncertainty", 1.0)))
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "lcb": prediction - self.lcb_z * uncertainty,
            "support": int(model.get("support", 0)),
        }


class RL006Runtime:
    def __init__(self, payload=None, opportunities=None):
        self.policy = RL006Policy(payload)
        self.index = rl006_opportunity_index(opportunities or [])
        self.history = RL006History()
        self.pending = {}
        self.last_step = -1
        self.changed_calls = 0
        self.changed_units = 0
        self.preempt_units = 0
        self.delay_units = 0
        self.errors = 0
        self.last_error = ""
        self.fallbacks = 0
        self.decisions = []

    def reset(self):
        self.history.reset()
        self.pending.clear()
        self.last_step = -1
        self.changed_calls = 0
        self.changed_units = 0
        self.preempt_units = 0
        self.delay_units = 0
        self.errors = 0
        self.last_error = ""
        self.fallbacks = 0
        self.decisions = []

    def _pending_view(self):
        return {key: value for key, value in self.pending.items()}

    def _apply_pending(self, action, step):
        changed = 0
        for key in [key for key in self.pending if key[0] == int(step)]:
            _, item, direction = key
            debt = self.pending.pop(key)
            quantity = max(0, rl006_int(debt.get("quantity", 0)))
            delta = -quantity if direction == "PREEMPT" else quantity
            moved = rl006_adjust_sell(action, item, delta)
            if moved != quantity:
                self.errors += 1
                continue
            changed += moved
        return changed

    @staticmethod
    def _has_item_order(action, item):
        return any(
            len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == str(item).upper()
            for order in action.get("market", []) or []
        )

    def _feasible(self, obs, base_action, opportunity, action_id, pending):
        item = opportunity["item"]
        direction = rl006_action_direction(action_id)
        quantity = rl006_action_quantity(action_id, opportunity)
        if direction == "CONTROL" or quantity <= 0:
            return False
        if len(base_action.get("market", []) or []) > 10:
            return False
        if any(value.get("item") == item for value in pending.values()):
            return False
        if not self._has_item_order(base_action, item):
            return False
        current_quantity = max(0, rl006_int(opportunity["current_quantity"]))
        future_quantity = max(0, rl006_int(opportunity["future_quantity"]))
        if direction == "PREEMPT":
            if quantity > future_quantity:
                return False
            if rl006_private_inventory(obs, item) < current_quantity + quantity:
                return False
        else:
            if quantity > current_quantity:
                return False
        return True

    def act(self, obs, base_action):
        step = rl006_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl006_align_hands(base_action, obs)
        changed = self._apply_pending(action, step)
        pending_view = self._pending_view()
        selected = None
        if step < RL006_CUTOFF:
            candidates = self.index.get(step, [])
            scored = []
            for opportunity in candidates:
                if self.policy.supported_events and rl006_event_key(opportunity) not in self.policy.supported_events:
                    continue
                for action_id in self.policy.allowed_actions:
                    if action_id not in RL006_NONCONTROL:
                        continue
                    if not self._feasible(obs, action, opportunity, action_id, pending_view):
                        continue
                    features = rl006_features(obs, opportunity, self.history, action, pending_view)
                    result = self.policy.predict(action_id, features)
                    if result is None:
                        continue
                    if result["prediction"] < self.policy.min_expected_delta or result["lcb"] <= 0.0:
                        continue
                    scored.append((float(result["lcb"]), opportunity, int(action_id), result))
            if scored:
                _, opportunity, action_id, result = max(scored, key=lambda row: row[0])
                direction = rl006_action_direction(action_id)
                quantity = rl006_action_quantity(action_id, opportunity)
                delta = quantity if direction == "PREEMPT" else -quantity
                moved = rl006_adjust_sell(action, opportunity["item"], delta)
                if moved == quantity:
                    key = (int(opportunity["future_step"]), str(opportunity["item"]).upper(), direction)
                    self.pending[key] = {
                        "item": str(opportunity["item"]).upper(),
                        "quantity": int(quantity),
                        "due_step": int(opportunity["future_step"]),
                        "current_step": int(step),
                        "direction": direction,
                    }
                    selected = {
                        "step": int(step),
                        "item": opportunity["item"],
                        "future_step": int(opportunity["future_step"]),
                        "action_id": int(action_id),
                        "action": RL006_ACTION_NAMES.get(action_id, "UNKNOWN"),
                        "direction": direction,
                        "moved": int(moved),
                        "prediction": float(result["prediction"]),
                        "uncertainty": float(result["uncertainty"]),
                        "lcb": float(result["lcb"]),
                        "support": int(result["support"]),
                    }
                    self.decisions.append(selected)
                    if direction == "PREEMPT":
                        self.preempt_units += int(moved)
                    else:
                        self.delay_units += int(moved)
                    changed += moved
                else:
                    self.fallbacks += 1
            elif candidates:
                self.fallbacks += 1
        if changed:
            self.changed_calls += 1
            self.changed_units += int(changed)
        self.last_step = step
        return action


def rl006_fit_models(samples, ridge=12.0, min_support=RL006_MIN_SUPPORT):
    if not samples:
        raise ValueError("RL006 requires at least one counterfactual sample")
    matrix = np.asarray([row["features"] for row in samples], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != RL006_FEATURE_DIM:
        raise ValueError(f"invalid RL006 feature matrix: {matrix.shape}")
    feature_mean = matrix.mean(axis=0)
    feature_scale = matrix.std(axis=0)
    feature_mean[0] = 0.0
    feature_scale[0] = 1.0
    feature_scale = np.where(feature_scale < 1e-9, 1.0, feature_scale)
    normalized = (matrix - feature_mean) / feature_scale
    grouped = defaultdict(list)
    for row in samples:
        grouped[str(rl006_int(row["action_id"]))].append(row)
    models = {}
    report = {"actions": {}, "skipped_actions": {}}
    rng = np.random.default_rng(6006)
    for action_key, rows in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        if int(action_key) == 0:
            continue
        support_keys = {
            (row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", "")))
            for row in rows
        }
        support = len(support_keys)
        if support < int(min_support):
            report["skipped_actions"][action_key] = {"rows": len(rows), "support": support}
            continue
        indices = [samples.index(row) for row in rows]
        x = normalized[indices]
        target = np.asarray([rl006_float(row.get("cash_delta", 0.0)) for row in rows], dtype=np.float64)
        design = np.column_stack((np.ones(len(rows), dtype=np.float64), x))
        penalty = np.eye(RL006_FEATURE_DIM + 1, dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        residual = target - design @ coefficients
        group_values = defaultdict(list)
        for row, value in zip(rows, residual):
            key = (row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", "")))
            group_values[key].append(float(value))
        group_means = np.asarray([np.mean(values) for values in group_values.values()], dtype=np.float64)
        if len(group_means) > 1:
            bootstrap = []
            for _ in range(128):
                bootstrap.append(float(np.mean(rng.choice(group_means, size=len(group_means), replace=True))))
            bootstrap_std = float(np.std(bootstrap))
        else:
            bootstrap_std = 0.0
        residual_std = float(np.std(residual))
        uncertainty = max(1.0, residual_std * (1.0 + 1.0 / math.sqrt(max(1, support))), bootstrap_std)
        models[action_key] = {
            "support": support,
            "rows": len(rows),
            "intercept": float(coefficients[0]),
            "beta": coefficients[1:].tolist(),
            "uncertainty": uncertainty,
            "mean_delta": float(target.mean()),
            "min_delta": float(target.min()),
            "positive_rate": float(np.mean(target > 0)),
            "residual_std": residual_std,
            "bootstrap_std": bootstrap_std,
        }
        report["actions"][action_key] = {
            **models[action_key],
            "name": RL006_ACTION_NAMES.get(int(action_key), action_key),
        }
    report["samples"] = len(samples)
    report["models"] = len(models)
    report["feature_dim"] = RL006_FEATURE_DIM
    report["mean_cash_delta"] = float(np.mean([rl006_float(row.get("cash_delta", 0.0)) for row in samples]))
    supported_events = sorted({
        rl006_event_key({
            "item": row["item"],
            "current_step": row["current_step"],
            "future_step": row["future_step"],
        })
        for row in samples
    })
    payload = {
        "version": "rl006_bidirectional_timing",
        "feature_dim": RL006_FEATURE_DIM,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "min_support": int(min_support),
        "min_expected_delta": RL006_MIN_EXPECTED_DELTA,
        "lcb_z": RL006_LCB_Z,
        "allowed_actions": list(RL006_NONCONTROL),
        "supported_events": supported_events,
        "models": models,
    }
    return payload, report


def rl006_load_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


"""RL-008 small-quantity bidirectional timing overlay.

V022c owns production and the base market route.  This module only moves one
small premium SELL quantity at a time and keeps an exact repayment debt.
"""


import json
import math
from collections import defaultdict

import numpy as np

try:
    from rl_006_bidirectional_timing import (
        RL006History,
        rl006_adjust_sell,
        rl006_align_hands,
        rl006_clone_distance,
        rl006_features,
        rl006_get,
        rl006_int,
        rl006_normalize_action,
        rl006_private_inventory,
        rl006_public_signature,
        rl006_step,
    )
except ImportError:
    pass


RL008_ACTION_NAMES = {
    0: "CONTROL",
    1: "PREEMPT_H1_1",
    2: "PREEMPT_H1_25",
    3: "PREEMPT_H2_1",
    4: "PREEMPT_H2_25",
    5: "PREEMPT_H3_1",
    6: "PREEMPT_H3_25",
    7: "DELAY_1",
    8: "DELAY_25",
    9: "DELAY_50",
}
RL008_PREEMPT_ACTIONS = tuple(range(1, 7))
RL008_DELAY_ACTIONS = (7, 8, 9)
RL008_NONCONTROL = tuple(range(1, 10))
RL008_FEATURE_DIM = 78
RL008_MIN_SUPPORT = 12
RL008_MIN_EXPECTED_DELTA = 5.0
RL008_LCB_Z = 1.5
RL008_CUTOFF = 648
RL008_PREMIUM = {"MILK", "WOOL", "STRAWBERRY", "MELON"}


def rl008_event_key(opportunity):
    return "{}|{}|{}|{}|{}".format(
        str(opportunity["kind"]).upper(),
        str(opportunity["item"]).upper(),
        rl006_int(opportunity["current_step"]),
        rl006_int(opportunity["future_step"]),
        rl006_int(opportunity.get("horizon", 0)),
    )


def rl008_preempt_opportunities(actions, target_sales=None):
    sales = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            if str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item not in RL008_PREMIUM:
                continue
            quantity = max(0, rl006_int(order[2]))
            if quantity:
                sales[item][int(step)] = quantity
    allowed = set((str(item).upper(), int(step)) for item, step in (target_sales or ()))
    rows = []
    for item, item_sales in sales.items():
        for future_step, future_quantity in sorted(item_sales.items()):
            if allowed and (item, future_step) not in allowed:
                continue
            for horizon in (1, 2, 3):
                current_step = future_step - horizon
                if 0 <= current_step < RL008_CUTOFF:
                    rows.append({
                        "kind": "PREEMPT",
                        "item": item,
                        "current_step": current_step,
                        "future_step": int(future_step),
                        "current_quantity": 0,
                        "future_quantity": int(future_quantity),
                        "gap": horizon,
                        "horizon": horizon,
                    })
    return rows


def rl008_delay_opportunities(actions, requested_events=None):
    from_rl006 = []
    events = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            if str(order[0]).upper() != "SELL" or str(order[1]).upper() not in RL008_PREMIUM:
                continue
            item = str(order[1]).upper()
            events[item][int(step)] = max(0, rl006_int(order[2]))
    allowed = set((str(item).upper(), int(step)) for item, step in (requested_events or ()))
    for item, rows in events.items():
        ordered = sorted(rows.items())
        for index, (current_step, current_quantity) in enumerate(ordered[:-1]):
            future_step, future_quantity = ordered[index + 1]
            if not (4 <= future_step - current_step <= 72):
                continue
            if allowed and (item, current_step) not in allowed:
                continue
            from_rl006.append({
                "kind": "DELAY",
                "item": item,
                "current_step": int(current_step),
                "future_step": int(future_step),
                "current_quantity": int(current_quantity),
                "future_quantity": int(future_quantity),
                "gap": int(future_step - current_step),
                "horizon": int(future_step - current_step),
            })
    return from_rl006


def rl008_action_kind(action_id):
    if int(action_id) == 0:
        return "CONTROL"
    return "DELAY" if int(action_id) in RL008_DELAY_ACTIONS else "PREEMPT"


def rl008_action_quantity(action_id, opportunity):
    action_id = int(action_id)
    current = max(0, rl006_int(opportunity.get("current_quantity", 0)))
    future = max(0, rl006_int(opportunity.get("future_quantity", 0)))
    if action_id in (1, 3, 5, 7):
        return 1
    if action_id in (2, 4, 6):
        return min(future, max(1, int(math.ceil(future * 0.25))))
    if action_id == 8:
        return min(current, max(1, int(math.ceil(current * 0.25))))
    if action_id == 9:
        return min(current, max(1, int(math.ceil(current * 0.50))))
    return 0


def rl008_action_matches(action_id, opportunity):
    kind = str(opportunity.get("kind", "")).upper()
    if kind == "DELAY":
        return int(action_id) in RL008_DELAY_ACTIONS
    horizon = rl006_int(opportunity.get("horizon", 0))
    return {
        1: (1,), 2: (1,),
        3: (2,), 4: (2,),
        5: (3,), 6: (3,),
    }.get(int(action_id), ()) == (horizon,)


def rl008_clone_units(obs):
    farms = list(rl006_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10**9
    left = rl006_public_signature(farms[0])
    right = rl006_public_signature(farms[1])
    weights = {"hands": 2, "quadrants": 3, "COW": 2, "SHEEP": 2, "GOOSE": 1}
    return sum(weights.get(key, 1) * abs(int(value) - int(right.get(key, 0))) for key, value in left.items())


def rl008_gate(obs, action, opportunity, quantity):
    market = rl006_get(obs, "market", {}) or {}
    prices = rl006_get(market, "prices", {}) or {}
    item = str(opportunity["item"]).upper()
    if rl008_clone_units(obs) > 6:
        return False
    if float(rl006_get(prices, item, 0) or 0) <= 1:
        return False
    if len(action.get("market", []) or []) >= 10:
        return False
    if rl006_private_inventory(obs, item) < quantity:
        return False
    return True


def rl008_append_sell(action, item, quantity):
    if quantity <= 0 or len(action.get("market", []) or []) >= 10:
        return 0
    action.setdefault("market", []).append(["SELL", str(item).upper(), int(quantity)])
    return int(quantity)


def rl008_shift_sell(action, item, delta):
    """Apply an exact signed quantity across existing same-item SELL orders."""
    item = str(item).upper()
    delta = int(delta)
    if delta == 0:
        return 0
    market = list(action.get("market", []) or [])
    if delta > 0:
        for order in market:
            if len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == item:
                order[2] = max(0, rl006_int(order[2])) + delta
                action["market"] = market
                return delta
        return rl008_append_sell(action, item, delta)

    need = -delta
    available = sum(
        max(0, rl006_int(order[2]))
        for order in market
        if len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == item
    )
    if available < need:
        return 0
    remaining = need
    updated = []
    for order in market:
        if remaining and len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == item:
            quantity = max(0, rl006_int(order[2]))
            take = min(quantity, remaining)
            quantity -= take
            remaining -= take
            if quantity <= 0:
                continue
            updated.append([order[0], order[1], quantity])
        else:
            updated.append(order)
    if remaining:
        return 0
    action["market"] = updated
    return need


class RL008Policy:
    def __init__(self, payload=None):
        payload = payload or {}
        self.feature_dim = int(payload.get("feature_dim", RL008_FEATURE_DIM))
        self.min_support = int(payload.get("min_support", RL008_MIN_SUPPORT))
        self.min_expected_delta = float(payload.get("min_expected_delta", RL008_MIN_EXPECTED_DELTA))
        self.lcb_z = float(payload.get("lcb_z", RL008_LCB_Z))
        self.feature_mean = np.asarray(payload.get("feature_mean", [0.0] * self.feature_dim), dtype=np.float64)
        self.feature_scale = np.asarray(payload.get("feature_scale", [1.0] * self.feature_dim), dtype=np.float64)
        self.models = dict(payload.get("models", {}))
        self.allowed_actions = tuple(int(value) for value in payload.get("allowed_actions", RL008_NONCONTROL))
        self.supported_events = set(str(value) for value in payload.get("supported_events", []))
        self.gate_mode = str(payload.get("gate_mode", "ungated"))
        if self.feature_dim != RL008_FEATURE_DIM:
            raise ValueError("RL-008 feature dimension mismatch")

    def predict(self, action_id, features):
        model = self.models.get(str(int(action_id)))
        if not model or int(model.get("support", 0)) < self.min_support:
            return None
        x = (np.asarray(features, dtype=np.float64) - self.feature_mean) / np.maximum(self.feature_scale, 1e-9)
        beta = np.asarray(model.get("beta", [0.0] * self.feature_dim), dtype=np.float64)
        prediction = float(model.get("intercept", 0.0) + x @ beta)
        uncertainty = max(1.0, float(model.get("uncertainty", 1.0)))
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "lcb": prediction - self.lcb_z * uncertainty,
            "support": int(model.get("support", 0)),
        }


class RL008Runtime:
    def __init__(self, payload=None, opportunities=None):
        self.policy = RL008Policy(payload)
        self.index = defaultdict(list)
        for opportunity in opportunities or []:
            self.index[rl006_int(opportunity["current_step"])].append(opportunity)
        self.history = RL006History()
        self.pending = {}
        self.last_step = -1
        self.changed_calls = 0
        self.changed_units = 0
        self.preempt_units = 0
        self.delay_units = 0
        self.fallbacks = 0
        self.errors = 0
        self.last_error = ""
        self.decisions = []

    def reset(self):
        self.history.reset()
        self.pending.clear()
        self.last_step = -1
        self.changed_calls = 0
        self.changed_units = 0
        self.preempt_units = 0
        self.delay_units = 0
        self.fallbacks = 0
        self.errors = 0
        self.last_error = ""
        self.decisions = []

    def _repay(self, action, step):
        changed = 0
        for key in [key for key in self.pending if key[0] == int(step)]:
            _, item = key
            debt = self.pending.pop(key)
            kind = str(debt.get("kind", "PREEMPT")).upper()
            quantity = max(0, rl006_int(debt.get("quantity", 0)))
            delta = -quantity if kind == "PREEMPT" else quantity
            moved = rl008_shift_sell(action, item, delta)
            if moved != abs(delta):
                self.errors += 1
                self.fallbacks += 1
                continue
            self.decisions.append({
                "step": int(step), "future_step": int(step), "item": item,
                "kind": kind, "action": "REPAY", "moved": int(moved),
            })
            changed += moved
        return changed

    def act(self, obs, base_action):
        step = rl006_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl006_align_hands(base_action, obs)
        changed = self._repay(action, step)
        if step < RL008_CUTOFF:
            candidates = self.index.get(step, [])
            scored = []
            for opportunity in candidates:
                event_key = rl008_event_key(opportunity)
                if self.policy.supported_events and event_key not in self.policy.supported_events:
                    continue
                item = str(opportunity["item"]).upper()
                quantity = rl008_action_quantity(7 if opportunity["kind"] == "DELAY" else 1, opportunity)
                if quantity <= 0 or any(value.get("item") == item for value in self.pending.values()):
                    continue
                for action_id in self.policy.allowed_actions:
                    action_id = int(action_id)
                    if not rl008_action_matches(action_id, opportunity):
                        continue
                    quantity = rl008_action_quantity(action_id, opportunity)
                    if quantity <= 0 or len(action.get("market", []) or []) >= 10:
                        continue
                    if opportunity["kind"] == "PREEMPT":
                        if self.policy.gate_mode == "gated" and not rl008_gate(obs, action, opportunity, quantity):
                            continue
                        if rl006_private_inventory(obs, item) < quantity:
                            continue
                    else:
                        if rl006_int(opportunity.get("current_quantity", 0)) < quantity:
                            continue
                    features = rl006_features(obs, opportunity, self.history, action, self.pending)
                    result = self.policy.predict(action_id, features)
                    if result is None or result["prediction"] < self.policy.min_expected_delta or result["lcb"] <= 0:
                        continue
                    scored.append((float(result["lcb"]), opportunity, action_id, quantity, result))
            if scored:
                _, opportunity, action_id, quantity, result = max(scored, key=lambda row: row[0])
                item = str(opportunity["item"]).upper()
                kind = str(opportunity["kind"]).upper()
                if kind == "PREEMPT":
                    moved = rl008_append_sell(action, item, quantity)
                else:
                    moved = rl008_shift_sell(action, item, -quantity)
                if abs(moved) == quantity:
                    key = (int(opportunity["future_step"]), item)
                    self.pending[key] = {
                        "item": item, "quantity": quantity, "kind": kind,
                        "due_step": int(opportunity["future_step"]),
                        "current_step": int(step),
                    }
                    self.decisions.append({
                        "step": int(step), "future_step": int(opportunity["future_step"]),
                        "item": item, "kind": kind, "action": RL008_ACTION_NAMES[action_id],
                        "moved": int(quantity), "prediction": float(result["prediction"]),
                        "uncertainty": float(result["uncertainty"]), "lcb": float(result["lcb"]),
                    })
                    changed += quantity
                    if kind == "PREEMPT":
                        self.preempt_units += quantity
                    else:
                        self.delay_units += quantity
                else:
                    self.fallbacks += 1
            elif candidates:
                self.fallbacks += 1
        if changed:
            self.changed_calls += 1
            self.changed_units += int(changed)
        self.last_step = step
        return action


def rl008_fit_models(samples, ridge=12.0, min_support=RL008_MIN_SUPPORT):
    samples = [row for row in samples if int(row.get("shift_applied", 1)) and int(row.get("future_repaid", 1))]
    if not samples:
        raise ValueError("RL-008 has no valid applied samples")
    matrix = np.asarray([row["features"] for row in samples], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != RL008_FEATURE_DIM:
        raise ValueError(f"invalid RL-008 feature matrix: {matrix.shape}")
    feature_mean = matrix.mean(axis=0)
    feature_scale = np.where(matrix.std(axis=0) < 1e-9, 1.0, matrix.std(axis=0))
    feature_mean[0] = 0.0
    feature_scale[0] = 1.0
    normalized = (matrix - feature_mean) / feature_scale
    grouped = defaultdict(list)
    for index, row in enumerate(samples):
        grouped[str(rl006_int(row["action_id"]))].append((index, row))
    models = {}
    report = {"actions": {}, "skipped_actions": {}}
    rng = np.random.default_rng(8008)
    for key, indexed in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        action_id = int(key)
        if action_id not in RL008_NONCONTROL:
            continue
        rows = [row for _, row in indexed]
        support_keys = {(row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", ""))) for row in rows}
        support = len(support_keys)
        if support < int(min_support):
            report["skipped_actions"][key] = {"rows": len(rows), "support": support}
            continue
        indices = [index for index, _ in indexed]
        x = normalized[indices]
        target = np.asarray([float(row.get("cash_delta", 0.0)) for row in rows], dtype=np.float64)
        design = np.column_stack((np.ones(len(rows)), x))
        penalty = np.eye(RL008_FEATURE_DIM + 1) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        residual = target - design @ coefficients
        groups = defaultdict(list)
        for row, value in zip(rows, residual):
            groups[(row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", "")))].append(float(value))
        group_means = np.asarray([np.mean(values) for values in groups.values()])
        boot = [float(np.mean(rng.choice(group_means, size=len(group_means), replace=True))) for _ in range(128)] if len(group_means) > 1 else []
        residual_std = float(np.std(residual))
        bootstrap_std = float(np.std(boot)) if boot else 0.0
        uncertainty = max(1.0, residual_std * (1.0 + 1.0 / math.sqrt(max(1, support))), bootstrap_std)
        models[key] = {
            "support": support, "rows": len(rows), "intercept": float(coefficients[0]),
            "beta": coefficients[1:].tolist(), "uncertainty": uncertainty,
            "mean_delta": float(target.mean()), "min_delta": float(target.min()),
            "positive_rate": float(np.mean(target > 0)), "residual_std": residual_std,
            "bootstrap_std": bootstrap_std,
        }
        report["actions"][key] = {**models[key], "name": RL008_ACTION_NAMES[action_id]}
    report.update({
        "samples": len(samples), "models": len(models), "feature_dim": RL008_FEATURE_DIM,
        "mean_cash_delta": float(np.mean([float(row.get("cash_delta", 0.0)) for row in samples])),
    })
    payload = {
        "version": "rl008_small_shift_timing", "feature_dim": RL008_FEATURE_DIM,
        "feature_mean": feature_mean.tolist(), "feature_scale": feature_scale.tolist(),
        "min_support": int(min_support), "min_expected_delta": RL008_MIN_EXPECTED_DELTA,
        "lcb_z": RL008_LCB_Z, "allowed_actions": list(RL008_NONCONTROL),
        "supported_events": sorted({rl008_event_key(row) for row in samples}), "models": models,
        "gate_mode": "ungated",
    }
    return payload, report


def rl008_load_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


"""v22 price-impact route agent for Kaggriculture.

The production route is a real fit-only trajectory.  Runtime adaptation is
limited to identity-free weed recovery and in-place ranking of existing SELL
slots by the official 1.32.4 price-impact curve.  Ordinary turns never create,
delete, or resize a market order.
"""
import base64
import copy
import json
import math
import zlib


_ACTIONS = json.loads(zlib.decompress(base64.b85decode(
    (
    'c-rk<U2hyoa{MoR<^$)06zMmvG-nCN6$MJV!FfR}7VsGcjPt|VZ^r$1Yei0XPiJIgWLEVowYs+fIn!O0Rb8DK85#N0|DFBEFTei%'
    'Z@-@X%TH$?Za;oJdptk;&tLxg-~Z>!AHID2`!B!#*Wdp4%jciY-oAU-efh8U;fK$E{rUF&yB}}w&d$$1zTNFUoSm=DKVIMOCx8C9'
    '+r9bn$Nk&g?WeQzSF=C=xVyW5e|ElFKR*8B{AkqgUjO;?hso83@&9zT-+lb_bv*CyA3nYP`ssO+liy8;_w<9~iT^f;4-'
    'fZmKEM7n4$lnZhtKcs-'
    'u(RK>YqM;wZUW*<IUMHh6|71n~q~X>$}_6yXQ&kH#7evcX+nl<kIsg!dtjsBDW&88&>ekgx`<$KQ`gBEuI##(SC>fJngryd*b%f?'
    '&0{3fBJ1tPDl0p?UXsk>yDE=+~D=?v+<t4)X8|`q=q{U-'
    '?3XioPk{vU@N;JW<TSrbaVsJdUiv^W;|RkX})16G?<TWwP7ddTWx+_(Q4yP=wdAVpw1^8S#ADK60J7*lWsP1t4`JebMP%<{xx~H3'
    'dRBk@ogkLkYp<6Lnjm25025gjeBORZsR`waQe$WpCyii2mPFl>uwESNL|nPP0t5t(>3Nt>+dy=f_=?34ldOnVluni_J!#&j@S2hc'
    'e~f0e*V+$;nTajcmH<$@~T|%<NfFMW$HhyHxKt8mVKH&?(TjI-'
    '6lgGBe+F2M0f(N8n5?ao;YUs^3KWZ+g>*TF>P`;sTf0HbvY^!M~?HAo?d2j*7fVn&$pxNp%pM57BuPja5$D)J%$0wI1u3fTA!}r-'
    'qxt26K0Ltb=pn#kByKx9CHvsY=q3MNkCUx`(D!qVas<qZ*Y<<G;uc~>U8h969A_>e0ciya<~2tX3byZN-VsX4alwkm?kKM_Am9$e'
    'Xsv5U2W#yZZrPvR`qYWqr1h~G{v)0l48#nQ&30dK!ID#Z!bhjDOWXl%QkZzWsz#${v0LkZ7UQ2F}HH|zsfDG5$%jXP7)rpRVSX@u'
    '`tEVn~c5I>o+7e4I%g*u$y?lmZ)&k@ui)3ktGIX(39U`3@^?oAhG$~+XNiC|1?Uktk#Ph!Z)25T<WrNuF&(7owt7%5Bkz0p7rsdr'
    'vX}TADZ&G7ILBorc09(O{h){QvqIe3?VLOJv+&fCZsq>32>ZI#~M&_!5x(17Gb)Z#Q^f*e{XLe{-Qd<8v!YOcK-'
    'R)byCeRJa`Wj&(1e@m3L@|KaNGAv{}<ke~$xZ%orqhLB5o^%wR4l$`inFW~BN0XY#G%pQfL{rK3R;C>zlTaVAS(W)TYJ;eOLaw=;'
    'oHUuFdeqtHvx$<Lm(;?Wae4LPn^Y`f3{HxmOp;+fJh{4cj9h6#Wx&*v1L(lA?}oHO11&T_^xPb0F2SIng|uZ>^MfN@rXoSFxv;4l'
    '?{#MG<c=_09tp(RQUGmIMC*f}-MG#OI(i3j?n24>S*;lPqMuNcs+hX{x3Vqy&kV{*bC0@ZRfS3~CS<tHtn{bhS-'
    '4}X2tH+0|hkJwKZr*}(n*;mA(L3tkou?bb|!^Vva)RBxvkdw}gpj@EXm7S5JVc8Wt-L&J?-'
    'Xt<7cG?U?($A1hE(xqWj2To!A<iX27e5Ls4p8A5HL<s{{)-'
    ')JGF;1`4)%gdgs#VR&E~MSCM&mZ9XPqRVi!Buh%{U^%iKM=3}f8;`@63d^Ekx4n*6=ue%;-'
    '^f0Hlvj{~=PJwNt$kh>tdi8r&v&&N*>xBDM<4-bDmJHL+K$g~XJ&;Bfzwwbe##`D*gI7o^CcwsEQ-gq=-'
    '?x9$EyfXOTVIV^WWcHn<t?gBv`@mNC+F^DdE<Vr9p)AB?kGHM?D0e~c>b|sD;J^S&fTYP_iG_LO$k0avVVikAgHEFqBN<OV<M?C?'
    'Hc|^W+ibdF9fuF|1Zt3lX>iKrr3DpA=m-?R^(|rfCYID%4IRYUg-'
    'i^NVIrfOg{=w3EUfD#VRsahG43j(Povi>1mlUp0tcSFRHx<}M-'
    '(3=9j%9UerVBo%+P79gpg>mcSXKe&wXdd)<i|CnWIToLY*P0wbJj#sFiZ)bs`SiRf(y_h0vza$~F;BO^<W<f)|Cz8u0nmMf}DXrL'
    '}CCKOT7UlMn@5KZ==C=T5^MfIr$cvYSUeKtOjIhE2a!;m2i`*2t&L-'
    'w?aGBLl|PWsqcHL2zT&)s7}`U&L`ykpUJ<BE>Mzx_~^l(1LUCGo&X)hL$r&7ulCHk8m0C;SRXdK856~iF>o-'
    '?lxvQb+IWxv&{pK7*7uI%8@|}tB$mj9da+hs!u#6pN*z5D}`YiAC9o_oJR+pn69V*&XkstEMN<=Y%FCF7s#y|Ay7C)!J^v#wWXoB'
    'CLdztRz@HvDC~0ud7YBzXJuG0g!Jz!)mwJW;4q506jC+k!IC#-V+o<>xs({cuDdhg&7ZCf!>1qL-TfIbdEhzSZ$Kohrfk=ok)rUF'
    'q$0@x9gF4Dh0ibaO(|&QqTbV>^_7#4T;+q=wHnN>nu6J<4>baq6;x=6Vds}iy&Gp*#W=pog{;No;jBg}0Q&olgk&0QKID+#(_&^P'
    'XEI)?wp75K(qkn$-'
    '~x5oE*n`$KT+dHB?Z#94aH;ofTA%uRIcEY^wvCzB#T<{vA~Y7OBkjy#XNJ~<Ixh9nFZF95iAEA?m|@6;^S$m_uA|)qGxcIdl(zxb'
    'AnH543b5JuZ7(dZnDr>ON|ivw}_+es-'
    'Z8jdJ#M(EmP;UY|>!qH3;JOFE@r6v2VMa;72brwMfI9Y7c*(R%w>;&&G&KG|jReN|d)jf4tNKJ`8ZLGm)R=UgGp6NR2GS4rPw!Fb'
    'bN^Y#5HB1j4eDeW|mOh=q-k?O!+zxx7-'
    '=*&i%oS4!qwR=I+l&uiDpN=Nuvk0lDEAW#nWx|l)iNmWG#1fC>XTVIT$f#V2#C^|5DTBQhkn|}IbeBo_a3%@Vz&jbaM#b6X<V~9^'
    '*3aMNWgH9N503c3XSMH(sK(WjtJfyk~QWX+GpDeEgWW^U7CqKZ)kpLfoVNvrjUg>4qnvMw?Vvy9Gec;1QJ~jal10h7?VTH%d(quz'
    'B)bl7T6eWlv{FH06#CtY=sKLIH7j&+YBUx`+0r!+eQ{I&J=23z#_vc|MEawL%aQIWAawe>Z?FP|ExLK0E+p|{k*?OA`Csw^FB}KI'
    'o$*99I0Gkl0m)9P`|Dt2CN>`qyc-LXmu5Cd2o!SUN>OzDV@*3+rW<hAgD?;=UTjm9@iL;1$p|E*c%;VQQvPe~$5So8wMuO;r<t77'
    'b(uHBcTY4h;_KMiPAfF!BvjzMoFfYFtu8;UUr8Cm{J0zsZS_lW3V8=F+o8BZ#GAtykeAa2yG$0`v`_X2mq7SrK?pnM$-'
    'RiSbCt9_TfNotPdP3FeoipHvk8_bHdfr%D&Wevel&c4bQfUJ?0lZZJea6gnmC8$PxKJf3Ae^ck41GgWvUhmE`|wJq5ML9PJTI2zv'
    'S<Ux={f<ZuN`z*9Y!6J==9)O*NHS4t=L!Y^j|$z3H;H6;q`>i5F{w6TC0y}FvWZZj=QZs3=;i9Cvp+{##mZrL)dzWaF+^NS(G_5&'
    'MAUy6$1No&H$~^%)k>g(OT6xLX0rx0XLmi9#>)&UkW)E(rDTdd&?g)nlWZ@qii0%$X<Fb;<l%9VOwtElmJ31UZ`$(expkjBMXp>0'
    'QjHVtDOg)>d2b<kkFP`M2F!}7IV%(^P~Eci!N#n=^~cPB6w>VMkO`8)wWW(FCkjJsZ1918i#>u&E+DXDk5QDk|15`lPCIsv21ZHk'
    '*GjjT31#kf+Jl;ht4Y#dXG%!p@nplSvk9p0!~W{)^u4GLCASyvE>_})GHlQ()81{@+s0-'
    'MhCr8^4CoHq^6m+)=dX}a;Ab}!w1k1s*}~2CW22S&8bZACoHjyxkrb|WMrCuS?_Mlv@+iw;eW+a`r&=xGYkNpr9Wn?rZ`d=RKngZ'
    'BH4Va#yP?UA!<ncQXYwrsE{XnsF)=2`8R#H1Qk}`FRP?>0I}AL6vPy1NXgK3On9oa=1E6{Yz?W+(!3rU;LVPYZ7%3bM|YcibhnC|'
    '(e#ij$Od&;dP$Si8YM>eB2lpL&1M~2@2{)?2urA#?_PgDNcJ~x0c!ost5LB&EcUj0!-J-'
    '4w%1DXiT_n?VoKB5<WDX`p;?BC4kXEa$n<w1x23w@J6Q=q($QzkqQ>d70(7*s089)Fz_gxAa$9~WN2qdGH~=UegFkS@W`&-'
    'a=ki!61F$_lU|R0|<w0m+fYwyO?X%FMGmoSxgBwm4Mx#>^NJD5{jBx=o0g8eq^UEQ*9yXJe&x%zj3E@E)43Z@Gbc5xCpT-_fTP@3'
    'V<E6B0c~DJ~T*w-'
    'ARBKzub^L8&Y|}t%?(#3zfzmCz?^9S$YjkJPM4!Y{oH@xDbzQ^ly$bn0X(Uz+KyBKwz@64n@ZDRcAET06Km9OIdM;N>O`?%hE0#_'
    'jcbF-<(xPcW1an>EUD4E71pp`3bB#Aef-uWuGlC$!A*p!vm{Go_<+bsOB!(fCr@Fc#%W*bQ{Hkh$Xg8J0TJ1giwIbCy^rc8FEVl-'
    '4B*sep>in(iZJl!dIR70Tp(qZF0JX?ufbdxWPC~f>vEZU-'
    '7+^}3tx^AoIgaI$9b(!e&%uM^ZO)A#pjuTaPUH|;Qz$1IW^1NJ>SP;`*%nX8sJRISVfR^adROL!WRS%rlRK&hNuje-'
    '<ba4{LziVIgP`qsq7JpjCrBr;SpgqQitoDToU*2*<x3s2Q=Xnu@RU<6q|ls!LNd+P@Ekfzr}&AuzG<FxE{wJ`9oRyWK$k9k#EvBJ'
    'h_GT8TBejpVqrL+3@=emNUe<<m(3>Hg)2xiznaQCu(4&7g*|3A2|x&JV@NZUnu8@3qO#jXPxQ&kDAH_}P-'
    'K@jK`xjF4{wjDaK*%ErGelZDTrI3F|f?HZ4f+bAX_%Kwkj6ft=2RbdHV9YwV=i={bt)LB))y@lC@lig>OqMiPKp4sA=F~=+ljG=}'
    'd3>oy~GdQHx4gZz(XKtuSrQ?Gce1h_*>2KsDP(mKs1ecBl#=v%?xXry&{7@XdJ?IQ)6}tSwbcra(np%GFWzIrM=L-'
    'XxHemrp(7s29W@$Gc(rGjT&?#!G6!Y-'
    'J7b>xs&j>AcFvoC}!vC#tl!ZxH(5B&z!`<&QBYk5?Z4y|vPh^r@~PXjSx!fQ7jh2nhzU13ZB)7Ty!6rwKD<e?~Y77F1?SLj|I-'
    '3;<E^DQ1(}*t#5WK~N6zjHYX7ewAyQr7TM<TL(rNb{O6y&qX>g)@(SL`n5)4T#C>(zLHtSHgd~|2p>TrRuYN2r`GpT=^uzxmo5q1'
    'c1oA8Ut+qPe}XXdGAVH<u8i}{y|bg{?VD{VJVCh7oK_2@!kJq|6R3UC-Y#v<?KefKu_s7#2i-'
    'rk_^6WC*4|6Br%x0YcS>WE9z2fRaw3HZJaVO84UZt~+_P(F!AvVEcQk22%ZO60AY-'
    '~~y~+yKO1{?#rouhcDf=1XKeW2CvZfW?bu?a_A*<{=sqZov3wq6~)((B{tJv`fB1wK|h2zh)DAF^uT8%zlXCkzk4`Os8F=w`s<|6'
    'FHn0><Y$xOXRlrr*`>QMPDX2xOj4M~~tX{#M8>2lQx`h^ojpb|V|!IRq3AvwZCns^=y(+li+3l`V2_O;yaQM;yC7ATz!o?hQmTzv'
    ')h?g^Oc<r35d`RQ_d>s;bD&SjR9*KIYgaSC`XrHNa6M3-'
    'tnlaV|{zoM!4F4f(+{c+<q*J52n=M*&SM@*q0%Bftay}3j<2vjM_E0l_jEL3%Db__~VDpDQ3b9|y4H?QjeX>B<&^U@=I8watPNAO'
    '^>!~yeI-F#)IX@DSwoUyzl!;AhH19<U80sDu^n4xWHvmy?SL7-'
    'VAN_lDD5xFa`Smq(nJp<WKZBljZD|9#<n;+DAdeW}#nl$N><5)7uZ9dT#7m=ELg5NQ#78tsE>uqY&kXuw3EX(^+w4c=|p*DA!AaT'
    ';A=D47EEoIN)MoTmu6aA#rl-zj6E<hA-Q<+Fv(JH@lTC?@wp(>T}2en<fJ(m!+U=5sza`Nz@wsEF_#&$8)<oLjwerP^*(%gw-'
    'D`W}%O5MRf`4e%8{9+Ie%x8<r1KHxV2DE@=x9n3+;&1xOY#DK;`3u~uSxyFdXD-'
    'w+ov}8Kxz(UB1k)u>2h(JeSiw5b3jIuPmL|wXY6OMz_7VQ%B2k^1eX}5%>ffbUNoC%8lW3ikRmD+pep&h!jR5UCO^JZc8GNe*IW_'
    '6$NoY3=nc<>nyoD1q5LXL1bZU8A$FNvWa9%OaBH!Sbr4_m|aqw!J==5zNc0Ce~j|aMPei!kuF8JE^OnFsrVoqO3k+uT<D<jA$&ca'
    'r}Ktqa~TY<p&Cr!<cybL>{xA$%K1K_!6?XU%jaz=pSe0}`3ebUt&?*%JL`vFwiWV@vZtM)at>=ReaD8c$GsfUDf(M25#3NN}mq|j'
    '@+0Up;<Z@XA?Bg+h8IMCBKDP2^C0LhTOZKFePS6!J&t1H}-y;-'
    'Kht1LQ&w&wc8V3*Hf;l(&tD#M48qs$d0>~B>tNBK=d#2rKuS5N}9X{Lcy$x%e9QfPqPr>?9r6T6EoxyWI*0w5@^G_SYOt0Q!bCB@'
    '6BP%r`N#zHZ#K42Xy3$vRnmObYJtNSk9inoak=Wz=nAQF1d$CuFA(2`ncLvSB?MV#-'
    'mQv@d|4wcnz?b@P%?O=+iv62P0bQb<tPU|0bod6^_E*$FpBn~1)Srjd^q{>N(qN~xe+OZQIkz9!%5m2EOU@R)piemYLNV*jZ8ajI'
    'YSF}RMzl-17wYI0|tsK<L*d5?bvD~jyFC=6_MigI!NWw07<x#=qM2RK<5@Ibdx41moJbSGS9blPN!?QDmnmcCBLhz%YffK-'
    '}8fQw7j1C7sH&KtuUeDo|I9?$TmHaw|BW(?-UZEb)WQpDZXH7!KwXB#^STu`Ac@H!J-'
    'DF@aqme{bgYe&MX=zbeR!05w)l%~n+s8h=FsGOHiX0Cy^hFN_+vQ0ZZ0p~h!B(!)TZb8jWEEV;t~#`wQe%Y)f9lp!$HE<4N`R}wV'
    '>O`#TI>J~R8ba&Zkx_McEAn@c^}$9TY7~cN5*ag)tpK%_1dxrf{GBe;wFs{NhJzq#4>1Exq){ifP?>^*c0ipC7HEEn%9SrOllL0('
    'I}ftOdJTlLHPsv_C}AjBOxqV-UA#o4RWcgMXc2uv9{$WFe*(h%hb@a)+nBg9zh}&l#)~?T|yhgmN*$j9D7*gq#wtW*3XDASu>kv0'
    'e~aTpJVtugjIV|HGnqJ1i)sO#71+Vvy-A~Bq1xLd$|p$e%#&Nzeh~SbVU?A;8``ht?&r-'
    'oT4~a5LXKQNn2h9U%&Qi^R~VCPwxN1v2Rv(-PY-45^A&~r~tq&YWm;xu2U{QDha}w{QxwP*nXk}a<epbj^&o{aEF;Fjtjw};-'
    'zY$Zf!Ah%Sq8L$LZRfgwwTonefjl!E*7LORPInFPhe6Cv-^Am-hejVrg8o9*&qL&*MsLqdvSYsVYwKRnsi4R2jl!7vu`e(Q$TJ{S'
    '#a_KW*Q2+xb*2k0GLU&h=}}YDHO_$EGgS$s$t9UZP=cjlP!QG|@iQOv2ZLKvdb}242(UV_V~EDDwtmNviimIzjc>rLTkv3#}Qotn'
    '!`aK*j5Lo!F<+I;tQ`yt+PI!WR(E=K3+w=!tV~qQX@A$z{}q%))BV?uKr`@hK8Ncr_ZGcHT*p5GgMsfUu=T2|*Eh;tXe{O#$g6ju'
    'ufra;XhD+)1hg9D5e*L)Jw#&D+~yy%$q=qz;FlywD#Mg}K5{U_YG_&6nXZG`%@PF|X6IW-{Uke}oel-'
    'm0^?3B?MlUaD1klP{eaMc$}TMK1N|NfbBjeA1dpy5N}|e<eYXTbFP-xHqIXmP<~PB^5*!`cSUy`O=#UtTaD+_#NWdMF;fo@+2r>X'
    'PGk*@NAbq)5u|Ebg&8e1Da<P5b^=8Y~^1QQ_L3s1)Zeo6^c#Ufrz4qJcPD!<~j_9l4cAg$SS#)*kOSKV(}VmUs4*GVH#*mF~ksz{'
    'ind8XC+i}=*oy=bPKYFQs(4&Qz&=$mD^DTZV(W4)x?G3D&KhB!DIC1-'
    'M@OCBo~(fbJJrTQp~ICN!v9}C?rPd<_xJ*?F+57QLNj)@(MhxKYRWXd`JyYsPJT2t7~su0W~JrB4Y5>rHwf=|6z&3nc)M;=UElyq'
    'a9#~&4rf*=E5;f(8z+R_>2$c8n1*<b5u~YXZVz;Ma76=r`1je%&ZDC)sp?XDD(@eGyw_)76b^O;nS$SCEe<#g{j7r+9D!+-'
    '+p_sS9v%d!x#BTfHXc2-(oETp1;UqT9$3<cf-'
    'V!+M$)zT2<R(e6!`0x4Lx=B;Cd!=&*JFbJ<GUTJuhYCso(7#ZYP5*7`+gQjsv9b(HEt%W9j-'
    'rlPmJ&y^a`?m)_3?!&@@I1p9I(mE4Tku<K_+pMcadi^n$53NiT6u_5P)=ITflvz7u2M1MWX{B#jHB0N2zR$H*r_G!hA5fj|s*PsG'
    'fYvCsleGASjk<2)CMB9wBbuY6z>!o)s}bt7ng#Y?rn&**)n%x-'
    '((_#<(dDGIoMICc0zNzTVyg#fzs7rzg%gp*bL^N3{HbP)ov_zpn{3Dqn~CD!Kr4pMj9DOYsID$DWI<<Q43y+&ke0L{zpS$cJ;nvE'
    'u=NU<0wl6lW~-'
    'w}(T*!HXy~8XG`)DHAS$aRWtutFQm*n<VbmPXKtT;Id^bsg`^qAfO5l`diO|lI(qsDwQck3_=6~YhK@2vBbq3KkJN}(rR+4*^gxC'
    'xtslCr`F6RL1%LACy8bR!;5bou5UtRw-'
    'CwvX>P}9+;10Z?z`&EQ)7`rt;Hw9ExIvW})RC6(DLvzeBIZVPA)5U_zvBCf_XmOo9D@-ZbuMvY-JljAI6S9XjnwYd9g25#kBV~N('
    'RxFv5CrqHq+{!+CgXV_=7B`8^<T|N@Y_%l+PKrH)q$LFfT!bYxU<P{f9Zp?cyiCJp(sg>NGB_Bs-'
    'QOW6h_z%QTU_5Js1kE0lE`9#!h=zn5S3UONIZ%wp}aIW$`aKHI$_S%AGpw9ZkOf!YY(50o2;1t11)i4MyP~tC-Tb7j@qk~(Op#S('
    '?dm^9z1ars?I?lxhzY)*JEYoAR(F^Itm%7g}0{GtE$3Eq`U?mRY=_L;+~7mZ4=S>C<%;l)nax|*d~0&v8h3Il+Pxso`T;NSZr72)'
    '|TOZlVgRXf?Oxop>4m|PoG`nLtA2Yfh}g|Aq&p$eWg7160Z##V@6wp25q{6Qrk6#n;dq~g3P$fmxEB9UbCz>`R$t&DqDqot8Yq&E'
    'Ikt+L|967OrpEh=dldija5W4sv8&lNzieY+Exq_mEN6HnoDoJw<RxtlE4^meMlF`(obT@iOp@J^W0dqWgI_-CE0R8Xei#NHig^@<'
    'S)K1QomJ^<dsHu()w_b(vzU?N7jV#xtu-'
    '>T}8zRkqyZZhEn!r^{v$RbI?}Ujy9B(q&3DEY8ipLR96p1bxJ^sa=wKSrX&=srZ2_{5!vg9@vyYb#lBVQT!<S=3se)E97uN%VXIu'
    'kO6__`>I0fc0Gf=)*l7W{jAdg<x+pE_+sA#VQcpR{ywE<I{28W2CbaIPhwxk2d55ybG;LW%Q7WuMpshy-JRJGq+l9_Hq*HBTidb0'
    'cDbNAo+e}ww0Z?+LU8LIR#5TOFwE>VjBmh&$jUo@>>9vDit-'
    '7LeHwKh@g*yv?x{pSD?)v4l4(Lx5Tc1QGtSl$rkrxKJR2du&Dx7u3<O&$cP?_GL(p0J0Y}6DZj#paVU&}W3))-'
    '^YgQ%X>HeALU3zFAtlTc(W6=XGasDz;!hslCEvxK;INh%wjES89o3#ou`O32wuNv9<=RGz0|2QVxela%HH+->VYFdr5j2g-'
    'p`Wpcz|sfm2kU^x;+3CaRceejTd%}dz<G&t%mTLXBQ#rFiGYFq(IrRCJfu#M|3r4gNDTltSGIkKs3HN&@w*=ewOfIrc_&c?2X*<5'
    'QP>R`H0ky@x|GwWPYCAr01JiP%dVV@`C)3E*M`XY`40=FU|y1lRR_F#k`vUT=JWqaF!(^7OY{UX~{!nQUp-'
    '_}(ZvNep50cTGDBJ7qYV;UX`$k-'
    'JsLRq8~6%X3^rKI5MLU(q`u60y#fMG82(F@*A<B}g<l#5MX6v$5@VVjjwH0h#pTdb(xQSR&y!AZ;GI?c6?oe5>hF)oTxDJ4EMy>k'
    'Mwd!<QOeT;=j*4k>+KbQ}mm9%H%vQC9MvBEm4dX*96m+BO)G_^xjq1t3(ooE9blOtLYDgq__0p~ckXL)?)Z(yyhrlJN{<kMKw@yv'
    'l<phq>%vRI>$wXpj)4AEp@LlXTcwXWnWQ;{7R)xwoF1TqdP+6vI+=n$cdi0r1=OS;ezqDZ#_Z|qS+dXQNo3{7nbrWNUSFf^J(0E4'
    'Q9rJ0=!FhRLENyPrR&OV!SV?RO}R1-C|0GopmQJ<sIb4Z+4h!1cpDMCe5B|}r^*ZUag;6-zVDS<pyO?Zi6oKxnLJNU7-'
    '072m*m>$QcXh%MSHylefSvJg7)0T|@F@vW>XsAxR0)!|2o3f`}u=EySSlG@t;BmTaf?w>WO|;-'
    '*TC#@zB7zhaaFjrYwze64TcpNWkV<Go;P{56SvM>C?kllLrp-sVA#;{RMM%X?t5Tc$@NK8cB_{&ajCt|Rh=bZAQOGrTGTd05h7t('
    'e>aBX2rccNorthpCSP92%%Wpl_QQoEuiQCL8?IgZbkg5_{$(&x!U&M-'
    'E3qmT8M53(<quegI1<6C&WeSr~H?Fh*pi0Mu9e!HO8+I;Bcp1Al%O&efQ=Y&+L`I`Fwr~CVHV66mkH`N9*sTax'
    )
)).decode("utf-8"))
_PRICE_FLOOR = 1
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
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


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


def _impact_slots(obs, action):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_impact_score(obs, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def _rl008_v22_agent(obs):
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
        action = _weed_repair_action(obs, _copy_action(_ACTIONS[step]), _ACTIONS, step)
        return _align_hands(_impact_slots(obs, action), obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs):
    return agent(obs)

# RL-008: small-quantity bidirectional premium timing selector.
RL008_PAYLOAD = {"version":"rl008_small_shift_timing","feature_dim":78,"feature_mean":[0.0,0.3875000000000001,0.3902777777777778,0.36666666666666664,0.6250000000000001,0.02777777777777778,0.6124999999999998,0.0,0.09375,0.09375,1.0,0.0,0.0,0.0,1.0,0.86,0.0,1.0059000000000002,0.008000000000000002,0.008000000000000002,-0.25199999999999995,-0.24800000000000008,-0.00020000000000000006,-0.00020000000000000006,0.0083,0.007900000000000003,0.10009011016002627,0.09167752902488878,-1.222222222222222,0.8550833333333331,0.004916666666666666,0.0214,0.0214,0.0214,0.021277083333333332,0.04235416666666667,0.06323125,0.004000000000000001,0.004000000000000001,0.004000000000000001,0.1328125,0.1328125,0.1328125,-0.20260569403714565,0.0,0.12000000000000004,0.12000000000000004,0.4699999999999998,0.5300000000000001,0.10878999999999996,0.10043333333333333,0.005319999999999999,0.06999999999999999,0.18999999999999997,2.0,0.35000000000000003,0.5,0.09999999999999999,0.18999999999999997,0.0,0.0,0.6319444444444445,0.0,2.0,1.8166666666666673,0.19999999999999998,0.7999999999999999,1.0,0.0,0.0,0.0,0.0,0.09375,0.0,0.0,1.0,0.0,0.0],"feature_scale":[1.0,0.001134023029066282,1.0,1.0,0.03402069087198856,0.011340230290662862,0.001134023029066282,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.004605936186466661,0.0017516489906241413,0.5665577237325317,0.0020072207614473324,0.0020072207614473107,1.0,1.0,1.0,5.01805190361833e-05,0.00010036103807236661,0.0001505415571085499,1.0,1.0,1.0,1.0,1.0,1.0,0.020355866390691212,1.0,1.0,1.0,1.0,1.0,1.0,0.00017913371790058796,1.0,0.008164965809277261,1.0,1.0,0.04082482904638632,1.0,1.0,1.0,1.0,1.0,0.009820927516479845,1.0,1.0,0.023570226039551605,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"min_support":12,"min_expected_delta":5.0,"lcb_z":1.5,"allowed_actions":[1,2,3,4,5,6],"supported_events":["PREEMPT|MELON|278|281|3","PREEMPT|MELON|279|281|2","PREEMPT|MELON|280|281|1"],"models":{},"gate_mode":"ungated"}
_RL008_OPPORTUNITIES = rl008_preempt_opportunities(_ACTIONS) + rl008_delay_opportunities(_ACTIONS)
_RL008_RUNTIME = RL008Runtime(payload=RL008_PAYLOAD, opportunities=_RL008_OPPORTUNITIES)

def agent(obs, config=None):
    """Public entry point; V022 owns every farmer/hand action."""
    try:
        base = _rl008_v22_agent(obs)
        return _RL008_RUNTIME.act(obs, base)
    except Exception as exc:
        _RL008_RUNTIME.errors += 1
        _RL008_RUNTIME.last_error = f'{type(exc).__name__}: {exc}'
        return rl006_align_hands(_rl008_v22_agent(obs), obs)
