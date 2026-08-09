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
                sales[item][int(step)] = sales[item].get(int(step), 0) + quantity
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
            events[item][int(step)] = events[item].get(int(step), 0) + max(0, rl006_int(order[2]))
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
RL008_PAYLOAD = {"version":"rl008_small_shift_timing","feature_dim":78,"feature_mean":[0.0,0.5710470085470248,0.5871794871794987,0.5487179487179574,0.6698717948717909,0.1613247863247901,0.4289529914529846,0.06009615384615385,0.11177884615384616,0.051682692307692304,0.682910839160842,0.38461538461538464,0.23076923076923078,0.23076923076923078,0.15384615384615385,1.232887413512403,0.002442002442002442,0.995426984126983,0.001153897028897004,0.013284035409035413,-0.08322283272283214,-0.09221036833536826,-6.715506715506732e-05,-0.0003462148962148917,0.0019141025641025656,0.001843894993894986,0.046258360365874655,0.0602829544616013,-0.869185654995497,1.226016850770747,0.006870562741656507,0.01993642134731392,0.034444183275861635,0.05309368529527018,0.019820063428611867,0.04387154725926313,0.07287099848966494,0.005883582961604933,0.009014350061568494,0.013379913697667709,0.1935096153846154,0.1935096153846154,0.1935096153846154,0.9415685794380784,0.05076923076922998,0.04923076923077054,0.09999999999999937,0.5262637362637466,0.47373626373625477,0.3565916941391963,0.17815823565323527,0.002600927960927993,0.060750915750916235,0.037832722832722994,0.46686944008372583,0.49990842490841964,0.44905372405372307,0.07893831391218001,0.04004231369749992,0.019780219780220074,0.0,0.48529075091575397,0.14285714285714285,1.9769230769230697,1.9212454212454257,0.2076923076923107,0.7923076923077049,0.62161172161172,0.12820512820512625,0.038461538461538464,0.11111111111111113,0.06009615384615385,0.11177884615384616,0.0,0.0,1.0,0.0,0.0],"feature_scale":[1.0,0.1341781383017006,0.12647063882659582,0.13688561861578563,0.24575900936883557,0.18691187663233422,0.13417813830170125,0.10630056663055246,0.06423264046099597,0.10797257058401741,0.3655291417080495,0.48650425541051784,0.42132504423474837,0.42132504423474787,0.36080121229410944,0.4587026917257247,0.049356246474743126,0.008616523273460842,0.029787259852484663,0.03821082031913173,0.13782945803813887,0.1468835020649227,0.00044813610911326364,0.0005745527178723322,0.0032976130256802513,0.0033346750001304443,0.05348964356902288,0.05517330808940232,1.0154118868828412,0.4643132513530345,0.029936118281774164,0.005025731421679686,0.03323193047483981,0.0662841715421645,0.005164538966960711,0.025284105115061656,0.043018530734030835,0.005221204127222345,0.012589485477849518,0.021676253661396746,0.0542265815179347,0.0542265815179347,0.0542265815179347,0.802543195071434,0.07829545015384153,0.041779924966154715,0.05363695626080116,0.1634240124787681,0.1634240124787687,0.18830134933468537,0.1683836644818835,0.0033897174281455195,0.049161320089419175,0.06695088757289587,0.7455875450378966,0.27716016521457076,0.3179231226064582,0.0756837813499737,0.071990567426873,0.06748183348502962,1.0,0.296682855568623,0.3499271061118843,0.07994080650318057,0.19326362792388507,0.2152930208172599,0.21529302081725943,0.47917240484216783,0.16216808513683797,0.06803236264355285,0.1378163717657887,0.10630056663055246,0.06423264046099597,1.0,1.0,1.0,1.0,1.0],"min_support":12,"min_expected_delta":5.0,"lcb_z":1.5,"allowed_actions":[1,2,3,4,5,6,7,8,9],"supported_events":["DELAY|MILK|215|260|45","DELAY|MILK|310|336|26","DELAY|MILK|452|473|21","DELAY|STRAWBERRY|480|503|23","DELAY|WOOL|450|470|20","PREEMPT|MELON|278|281|3","PREEMPT|MELON|279|281|2","PREEMPT|MELON|280|281|1","PREEMPT|MILK|470|473|3","PREEMPT|MILK|471|473|2","PREEMPT|MILK|472|473|1","PREEMPT|STRAWBERRY|500|503|3","PREEMPT|STRAWBERRY|501|503|2","PREEMPT|STRAWBERRY|502|503|1","PREEMPT|WOOL|467|470|3","PREEMPT|WOOL|468|470|2","PREEMPT|WOOL|469|470|1"],"models":{"1":{"support":84,"rows":336,"intercept":-73.18699632899575,"beta":[0.0,6.032977146463295,6.400644840886304,6.089933496314694,-2.9456505506668784,-1.1109806329885797e-11,-6.032977146461928,-2.616706297818246e-13,10.77340139753471,6.409072366872132,0.9773561379980786,-2.9372992839369276,11.867687409289939,-0.938951266214993,-8.801360776131176,-3.727690686221073,-4.2774502940317345e-13,-21.508508448662774,-13.635204933355052,25.632960161727645,24.353508575184275,-22.780296685065416,14.494691360467874,12.437980503238872,1.8719473383619007,16.595312742905236,-11.416565994881896,42.17973379813047,2.816950571299451,-3.563394969642137,-1.8496134655895102,-2.7929643616784037,-0.42238559576819323,-0.2117653193136141,-2.538784209592869,28.786311149254146,31.286591391602318,8.206541720208179,3.4034774157496,1.9767267060580203,3.661581379811589,3.6615813798115813,3.6615813798115875,-14.681142483419963,8.940769199235122,-11.93046200328002,3.758001123091706,-47.54375853669244,47.543758536691946,-39.6567807304455,-4.871594475367832,1.5682964356846396,-2.2505536098282968,-2.9530037242646583,-3.912021557023586,-4.124828337842872,-0.023135978224036107,1.776549561971714,1.9130325290043286,-5.203398482797655,0.0,-3.7031935996474403,-5.180320632024529,-2.524151750619042e-12,1.967835431393001,1.0312531080538654,-1.031253108048743,-6.7583260054750584e-12,9.520840903371004e-12,8.123424972235744e-12,6.982536193427561e-12,-3.873085321655042e-12,10.773401397534087,0.0,0.0,0.0,0.0,0.0],"uncertainty":64.00617995213145,"mean_delta":-73.51190476190476,"min_delta":-401.0,"positive_rate":0.07142857142857142,"residual_std":57.70955164852803,"bootstrap_std":3.6355676733071363},"2":{"support":84,"rows":336,"intercept":-124.99414922800254,"beta":[0.0,12.070371258607151,12.80597583057442,12.192882207286482,-6.036435411557722,-2.5031937159350212e-11,-12.070371258604704,-3.453489023196371e-12,13.304924519500604,7.91506980337702,-0.9711902313654384,2.9187685638097527,17.490963479574948,-5.338735727583671,-18.126411439786484,-11.392629307574037,-1.047607394823187e-12,-13.107867722248207,-10.90593177644869,28.402584273592463,15.331099059306759,-29.039176469689952,24.91965991267389,21.185085368768583,-4.005595593802631,36.61251324105744,-26.010603053749353,31.810213071680824,-0.7445262956383087,-11.158863910676082,-1.4905521493537723,4.072192091632269,0.6158457681371589,0.3087576305748053,4.274688028346262,29.161554979969953,33.39954353531832,9.68655780384522,4.017280584894719,2.3332212463463393,5.88755541470949,5.887555414709481,5.887555414709477,8.276296116166142,13.177181210609422,-21.052014647456282,2.8368824947278317,-59.47278256203122,59.47278256202938,-61.10694399106029,-18.07502073748352,6.7241046903148405,3.7063113503686362,-5.601398287222121,48.72900282654768,-8.106420631637171,-31.47896491498353,7.095337719493888,-39.27442890307064,5.935615129714075,0.0,-9.235306504885578,11.21632179149708,-2.121372731465172e-12,36.50384273810363,1.430001211666734,-1.4300012116574183,-1.0745112045491028e-11,1.35407296747539e-11,1.0905657225802258e-11,1.0578630691313096e-11,-8.783895648446735e-12,13.304924519499808,0.0,0.0,0.0,0.0,0.0],"uncertainty":97.91057387425549,"mean_delta":-125.92559523809524,"min_delta":-807.0,"positive_rate":0.07738095238095238,"residual_std":88.27859003863587,"bootstrap_std":4.9715679139042965},"3":{"support":84,"rows":336,"intercept":-43.95351070519904,"beta":[0.0,7.7722155832597375,8.245877676564175,7.784578045072792,-2.7753241657164742,-4.9836930043293945e-14,-7.772215583261255,-8.735690915215224e-13,2.45949913362891,1.4631505271240948,-2.1952969427577194,6.597640192111164,5.85654361924117,-3.547117460273438,-11.593084959781175,-2.7152561751928057,6.637819856585248,-15.043935777746343,2.645258258655823,6.946922723150093,26.42766046866866,-25.427261857903005,22.140852883035617,26.534595302157367,1.6364181612672128,14.594477972957552,-13.728523164726901,38.06986318940287,-12.562475947954749,-2.489235644886351,-2.9967218789409675,1.1289334316722002,0.170730864546866,0.08559684896935531,1.3997021027841696,31.203692130441958,34.48532011385331,24.724545584070587,10.253945617902362,5.955452517938387,3.7010902741802045,3.7010902741802068,3.7010902741802063,-12.299172687951271,4.412148972163197,-10.26589313393811,-1.5559617984125569,-47.052307099467164,47.05230709946759,-40.45614942214091,-5.651022819459882,-6.365918256462623,5.590440842562916,-0.5342723876809762,-7.641301541180013,-3.3709914494675375,-0.4204561240274434,5.1436084499271955,1.8054713821965316,-4.688167453418577,0.0,-3.8756354511302993,-2.3355970956992556,-1.1096558538310712e-12,4.035025023484665,1.3883305786528684,-1.3883305786517233,-3.1196560394477734e-12,3.619820531848669e-12,4.673149461435127e-12,1.8041810802368531e-12,-3.3241164011687247e-12,2.459499133629322,0.0,0.0,0.0,0.0,0.0],"uncertainty":63.07651694625851,"mean_delta":-74.55952380952381,"min_delta":-401.0,"positive_rate":0.07142857142857142,"residual_std":56.87134453644489,"bootstrap_std":3.3962717923065417},"4":{"support":84,"rows":336,"intercept":-94.30493082746034,"beta":[0.0,11.037512275909656,11.710171328334004,11.192329502762327,-6.234900507418173,9.13622383787348e-13,-11.037512275910919,4.400393406921504e-13,2.3927756705441756,1.4234568883480208,-4.622751138793608,13.892994663909144,8.934146760201996,-10.080203058211698,-17.3950047563132,-10.722117640382619,9.601252454510536,-6.00937131872703,-2.246730177688205,0.5672107257869152,21.793641258343964,-31.454819207458325,11.572981476698613,38.081371439650994,-6.070615247582836,31.461271586355828,-32.10124634920905,24.132881694623116,-26.392817643124197,-10.447221736841382,-2.2541576623795616,8.105491497806899,1.2258097175407023,0.614566379287353,8.390588439332578,30.7600062472984,34.96593400888818,37.9224871647623,15.727493140808784,9.134468049328671,3.2428908437070945,3.2428908437070953,3.2428908437071016,7.7701635074362665,6.730725323324352,-16.451919325317512,-2.99000534499004,-57.93174139646798,57.93174139646911,-62.19401175020366,-16.595224564493897,-8.36633645179903,12.823381630633612,-5.96092998949777,26.11055342645457,-9.419954831213472,-32.18731068985534,11.612949514909637,-25.707803054532526,-14.973764626647268,0.0,-15.936696296522825,14.927549722687212,-2.1085134209450304e-12,29.25026808200401,3.9453596621703753,-3.9453596621692752,-1.0959031440081358e-11,1.1506422088779886e-11,9.5210715688941e-12,6.830265597375543e-12,-6.148092023020272e-12,2.392775670544666,0.0,0.0,0.0,0.0,0.0],"uncertainty":97.97775276319581,"mean_delta":-129.81845238095238,"min_delta":-807.0,"positive_rate":0.07142857142857142,"residual_std":88.33916018301711,"bootstrap_std":4.997545999206377},"5":{"support":84,"rows":336,"intercept":-35.475480883276056,"beta":[0.0,8.405317739400237,8.91756297407281,8.515883577863542,-4.625519805667614,4.048322749963573e-12,-8.4053177394026,-9.43254509372003e-14,1.4933425175278832,0.8883861196864316,-3.574137025659028,10.741540077460161,6.472550256460505,-7.555992559739698,-13.218701651994799,-2.846517326025094,3.5269843857033365,-12.149714458161629,31.71958710072553,2.549841749116847,28.8643477514659,-28.387111003471432,26.838154432434305,23.785915680137236,-8.262266345329701,15.351038594607306,-5.4066316631754106,30.296643217226723,-5.732255791145611,-2.5265030275537255,-4.429977292515729,3.882562073868094,0.5871676406527565,0.2943796951402852,4.1999564301992125,20.426520082825174,27.05023405378272,11.787167298711474,4.888460823643235,2.8391994073168276,2.5148645600473256,2.514864560047327,2.514864560047326,-13.629636022245553,4.876230387404743,-12.289463053167875,-2.4547662689845153,-45.33119698339379,45.33119698339428,-34.369344850328794,-3.3222506493526995,4.553037280104381,10.725214292817022,0.15187153376968304,-4.226674844094773,-6.154887851844088,-3.472969493169646,8.767498939302925,0.6105076749454207,-3.192825517397079,0.0,-8.571357147137329,-3.972945484534591,-1.5694753552208876e-12,6.696916524700754,5.384690461531407,-5.3846904615310285,-2.815279216271081e-12,4.165754126284076e-12,4.718767487478608e-12,6.3197679762749,-2.1180575166161527e-12,1.4933425175272304,0.0,0.0,0.0,0.0,0.0],"uncertainty":63.81568264863223,"mean_delta":-74.55952380952381,"min_delta":-401.0,"positive_rate":0.07142857142857142,"residual_std":57.53779457782931,"bootstrap_std":3.0238972605024634},"6":{"support":84,"rows":336,"intercept":-86.04684004121557,"beta":[0.0,12.18569557684134,12.928328358121638,12.444455272406843,-8.35123140317136,9.76790936799461e-12,-12.185695576836183,7.331712346918153e-13,2.41460706962801,1.4364443387753272,-5.901716362266096,17.736735434601197,10.625161589798907,-14.172521120950842,-19.773785706775993,-10.507231126936574,5.268804188447199,-5.58406624398911,32.69616444437039,-0.6838582609781717,30.591699981524034,-34.08898657746904,12.781933361297398,38.68879143917698,-17.721970118728052,32.540003362489706,-16.326186437182358,15.454011357174023,-14.179552697966265,-10.092913466790607,-4.456881545766084,9.671370417152271,1.46262072355288,0.7332928671398775,10.165008348768303,18.08615891138827,26.520300053120756,18.293121010841812,7.58665785743389,4.406301956673676,2.368497654610576,2.3684976546105867,2.3684976546105867,1.5129942391818672,8.004686524227228,-19.533015301328945,-3.530352803325875,-55.46779751563995,55.467797515641976,-58.273358778512595,-15.417731594734073,7.372638374223274,18.425759769276738,-9.61868709935252,38.95017082523098,-11.967689990985736,-29.749333110674137,15.385207581961117,-31.650669163685027,-13.716906302938039,0.0,-15.228878406994747,14.748221663220015,-2.393311224820909e-12,32.20731606474325,9.555104649626145,-9.555104649625315,-1.1522208115919274e-11,1.2678808156600751e-11,9.474622505150589e-12,10.435380010215868,-5.38689500358379e-12,2.414607069627222,0.0,0.0,0.0,0.0,0.0],"uncertainty":98.98231802616455,"mean_delta":-129.81845238095238,"min_delta":-807.0,"positive_rate":0.07142857142857142,"residual_std":89.24490101883903,"bootstrap_std":5.288241223800191},"7":{"support":84,"rows":420,"intercept":5.37030046854686,"beta":[5.497510706273628e-15,0.16071351844187748,0.2599387849773583,0.19189920890045414,-0.5742212132650673,0.6051185024263018,-0.1607135184429302,0.17340090434182748,-1.5961098862051024,-1.120237910317696,-1.4900935472768186,-0.06878976539852696,-0.3157292264421067,0.39516080558259065,-2.309898210541907e-13,1.1736719377161609,-4.4423629745495105e-14,1.3780360635980629,-3.3293415164848645,3.0894865576411505,-1.0560795121371167,-3.567109076828658,-1.7974314334838777,0.3567289874402841,-2.585707480831737,-2.6434519728164894,0.5930664158783063,2.636303925493545,-0.34331395748253957,1.1114571895334497,0.7449922335128768,1.4928446293914057,-1.0985526117400117,-0.8033814195605679,1.332549524742444,-0.46632821620238635,-0.19726067354314966,0.899473407551708,0.18177658755632983,-2.0246151213593335,-0.18708345699793197,-0.18708345699793225,-0.18708345699793213,-0.9234259168407112,-0.033550297461719905,0.34523060683134404,0.21993927376305053,1.29272695643371,-1.2927269564337158,-2.306660062106504,-0.10133332427922587,0.06876313627520535,-0.07402659824195519,-1.5340730119599693,0.5816234915858045,0.22287282322613902,-2.8187855851088424,-0.3608943397787707,-6.177792716502166,-1.0918450738635332,0.0,0.014027925356228512,-0.7956159151559637,-1.9298909946183145,3.113977140430496,0.1235366001286032,-0.12353660012843816,0.2604624592635763,1.0313769267285476e-12,0.17340090434202374,-1.0877666630935157,0.1734009043419795,-1.5961098862052168,0.0,0.0,0.0,0.0,0.0],"uncertainty":7.877596503421366,"mean_delta":8.023809523809524,"min_delta":-92.0,"positive_rate":0.8214285714285714,"residual_std":7.1026354427096265,"bootstrap_std":0.409199066534915},"8":{"support":84,"rows":420,"intercept":-3.3698174738067137,"beta":[-1.58161901210283e-13,0.6986598461396737,0.8796883222951708,0.8093192336713747,-2.079999411756984,0.9367979682777637,-0.6986598461416617,1.9190557604307243,-1.775320722907399,-2.945472639087085,-3.0559075747225855,-1.4845323368979253,0.42328286986058994,1.2909074187972132,1.510182960984326e-13,3.9239255692418573,-1.982554767709941e-13,-0.8904636915247541,-3.757957140612128,-2.7472260566988496,-1.8040742009508848,-16.59782484684338,-4.24318257440499,-2.8677942381119546,-7.841884963494784,-13.669567662616789,-2.5846227118103666,4.041393807057037,-2.4667860219876987,3.7775787097817983,1.5344463610664172,4.799352592658941,-7.083222772958642,-5.512309417041482,4.227774167878364,-5.525807382321879,-4.155091561622265,4.512843028725878,-1.9820715009578211,-11.618106924248375,0.20074694744503072,0.20074694744503035,0.20074694744503047,-5.2738478015378245,1.6125763126116488,0.10293242811584731,2.4341033967678474,4.415886537934858,-4.415886537935075,-15.0461977193787,3.270055179314081,0.22463473478106963,1.503576398813715,-6.67454265675602,0.7591914188943202,0.8007746975744886,-3.7876949712013137,0.7074923496733249,-39.10554857424969,0.12904991932336074,0.0,-8.80626002792534,-6.923632643743434,-3.248334515884708,6.690507826692736,1.2607064777718868,-1.260706477770402,0.5418411087467787,1.3857941314584334e-12,1.9190557604316998,-1.321841403480523,1.9190557604313638,-1.7753207229082508,0.0,0.0,0.0,0.0,0.0],"uncertainty":40.550574251617185,"mean_delta":11.49047619047619,"min_delta":-622.0,"positive_rate":0.8333333333333334,"residual_std":36.561398616529125,"bootstrap_std":1.6904924823351888},"9":{"support":84,"rows":420,"intercept":-1.5905869354077269,"beta":[-1.320479443594762e-13,1.740395201932501,2.058748529891458,1.9328004193695614,-3.7902487938628973,1.436412395920108,-1.7403952019355804,3.36899796004821,-3.437648193797929,-5.361876719826808,-5.666871341764169,-2.6325638294579052,0.7579200535730857,2.2819028176203635,4.114089563832704e-13,6.237178050595263,-3.483783738201701e-13,-4.02489313277499,-5.097292178380624,-5.575362255495658,-1.3094451013913089,-35.12837870375144,-8.104816422420562,-5.903645365335051,-15.752331803794956,-26.01992559470336,-2.485644380872574,5.489033359885244,-5.207518578391571,6.108999521837061,0.819108538114609,8.200357362754875,-14.333940959555447,-11.771295106971133,7.55030648340198,-10.324027386299093,-7.785803814612342,4.785485225573894,-3.73344543036164,-19.05973938162297,0.7199386066711715,0.7199386066711704,0.7199386066711719,-9.351193401075507,2.8619849872888503,0.12255857457426177,4.273189735984706,9.36257592748482,-9.362575927485253,-28.877416114650504,6.417160541856224,0.39708086479949944,2.6694635188115283,-11.970926156963914,1.1392294197108184,1.4424041382666484,-6.016263382123024,1.264107774379334,-73.4328226411641,-0.8037254156646191,0.0,-15.323779272911949,-11.962771517967964,-5.446916928237898,10.211657016230564,2.1316361097204592,-2.1316361097180914,1.0534498874287521,3.0251854605124415e-12,3.3689979600503563,-2.5888262743434547,3.368997960049092,-3.437648193799314,0.0,0.0,0.0,0.0,0.0],"uncertainty":77.5211871532358,"mean_delta":20.061904761904763,"min_delta":-1196.0,"positive_rate":0.8476190476190476,"residual_std":69.89501571911715,"bootstrap_std":3.7292059348304654}},"gate_mode":"ungated"}
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
