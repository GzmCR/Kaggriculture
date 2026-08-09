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
