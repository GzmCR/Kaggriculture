"""RL-001 high-level market selector.

The field/labor policy is supplied by V022c.  This module only selects a
market overlay at 48-turn boundaries and keeps all farmer/hand actions intact.
Training uses NumPy, while the generated submission can use the same small
linear model with embedded weights.
"""

from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np


BLOCK_STEPS = 48
RL_STOP_STEP = 672
ACTION_COUNT = 4
FEATURE_DIM = 96
PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
SELLABLE = PREMIUM + ("WHEAT", "TOMATO", "CARROT", "EGG")
PRODUCTS = ("WHEAT", "FERTILIZER", "MILK", "WOOL", "STRAWBERRY", "MELON")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("COW", "SHEEP", "GOOSE")


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _copy_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in action.get("hands", []) or []],
        "market": [list(item) for item in action.get("market", []) or [] if isinstance(item, list) and item],
    }


def _align_hands(action, obs):
    action = _copy_action(action)
    seat = _as_int(_get(obs, "player", 0), 0)
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[seat] if 0 <= seat < len(farms) else {}
    expected = len(_get(farm, "hands", []) or [])
    hands = list(action.get("hands", []) or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(item or ["PASS"]) for item in hands[:expected]]
    return action


def _farm(obs):
    seat = _as_int(_get(obs, "player", 0), 0)
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if 0 <= seat < len(farms) else {}


def _opponent_farm(obs):
    seat = _as_int(_get(obs, "player", 0), 0)
    farms = list(_get(obs, "farms", []) or [])
    other = 1 - seat
    return farms[other] if 0 <= other < len(farms) else {}


def _tile_counts(farm):
    crops = Counter()
    animals = Counter()
    structures = Counter()
    weeds = 0
    yield_total = Counter()
    maintenance_risk = 0.0
    for row in _get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "")).upper()
            crop = str(tile.get("crop", "")).upper()
            animal = str(tile.get("animal", "")).upper()
            if crop:
                crops[crop] += 1
                yield_total[crop] += max(0, _as_int(tile.get("yield_units", 0)))
                maintenance_risk += min(2, max(0, _as_int(tile.get("consecutive_unwatered", 0))))
            if animal:
                animals[animal] += 1
                yield_total[animal] += max(0, _as_int(tile.get("yield_units", 0)))
                maintenance_risk += min(2, max(0, _as_int(tile.get("consecutive_unfed", 0))))
            if kind in ("COOP", "PASTURE"):
                structures[kind] += 1
            if kind == "WEED":
                weeds += 1
    return crops, animals, structures, weeds, yield_total, maintenance_risk


def _signature(farm):
    crops, animals, structures, weeds, _, _ = _tile_counts(farm)
    hands = len(_get(farm, "hands", []) or [])
    unlocked = len(_get(farm, "unlocked_quadrants", []) or [])
    return (
        hands,
        unlocked,
        *(crops.get(item, 0) for item in CROPS),
        *(animals.get(item, 0) for item in ANIMALS),
        structures.get("COOP", 0),
        structures.get("PASTURE", 0),
        weeds,
    )


# Coarse public route prototypes keep the state feature observable and do not
# depend on a username, episode id, or private replay data.
PUBLIC_PROTOTYPES = (
    ("high_output", (14, 3, 92, 0, 0, 42, 24, 0, 8, 6, 0, 14, 1)),
    ("stable", (12, 3, 66, 0, 0, 44, 21, 0, 4, 5, 0, 12, 0)),
    ("livestock", (10, 3, 48, 0, 0, 34, 18, 0, 10, 8, 0, 12, 0)),
    ("wheat", (8, 2, 80, 0, 0, 12, 8, 0, 4, 2, 0, 8, 0)),
)


def route_distance(obs):
    actual = _signature(_opponent_farm(obs))
    distance = []
    for _, target in PUBLIC_PROTOTYPES:
        value = abs(actual[0] - target[0]) * 1.5
        value += abs(actual[1] - target[1]) * 3.0
        value += sum(abs(left - right) for left, right in zip(actual[2:], target[2:]))
        distance.append(float(value))
    return min(distance) if distance else 999.0


class FeatureEncoder:
    """Create a stable, bounded feature vector for macro decisions."""

    def __init__(self):
        self.last_prices = {}
        self.last_step = -1

    def reset(self):
        self.last_prices.clear()
        self.last_step = -1

    def encode(self, obs):
        step = _as_int(_get(obs, "step", 0), 0)
        day = _as_int(_get(obs, "day", step // 24), step // 24)
        hour = _as_int(_get(obs, "hour", step % 24), step % 24)
        farm = _farm(obs)
        opponent = _opponent_farm(obs)
        crops, animals, structures, weeds, yields, risk = _tile_counts(farm)
        other_crops, other_animals, other_structures, other_weeds, other_yields, other_risk = _tile_counts(opponent)
        private = _get(obs, "private", {}) or {}
        shed = _get(private, "shed", {}) or {}
        inventories = _get(private, "inventories", []) or []
        carried = Counter()
        for inventory in inventories:
            for item, quantity in (inventory or {}).items():
                carried[str(item).upper()] += max(0, _as_int(quantity))
        market = _get(obs, "market", {}) or {}
        prices = _get(market, "prices", {}) or {}
        market_inventory = _get(market, "inventory", {}) or {}
        shops = _get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or []
        money = _as_float(_get(farm, "money", 0))
        other_money = _as_float(_get(opponent, "money", 0))
        values = [
            1.0,
            step / 720.0,
            day / 30.0,
            hour / 24.0,
            min(3.0, money / 200000.0),
            max(-3.0, min(3.0, (money - other_money) / 100000.0)),
            len(_get(farm, "hands", []) or []) / 16.0,
            _as_int(_get(farm, "hires_today", 0)) / 16.0,
            len(_get(farm, "unlocked_quadrants", []) or []) / 4.0,
            len(shops) / 8.0,
            len(_get(opponent, "hands", []) or []) / 16.0,
            len(_get(opponent, "unlocked_quadrants", []) or []) / 4.0,
            len(_get(opponent, "unlocked_quadrants", []) or []) / 4.0,
            weeds / 20.0,
            other_weeds / 20.0,
            min(4.0, risk / 20.0),
            min(4.0, other_risk / 20.0),
            route_distance(obs) / 100.0,
        ]
        values.extend(min(1.0, crops.get(item, 0) / 160.0) for item in CROPS)
        values.extend(min(1.0, yields.get(item, 0) / 400.0) for item in CROPS)
        values.extend(min(1.0, animals.get(item, 0) / 16.0) for item in ANIMALS)
        values.extend(min(1.0, yields.get(item, 0) / 64.0) for item in ANIMALS)
        values.extend(min(1.0, other_crops.get(item, 0) / 160.0) for item in CROPS)
        values.extend(min(1.0, other_animals.get(item, 0) / 16.0) for item in ANIMALS)
        values.extend(min(1.0, structures.get(item, 0) / 16.0) for item in ("COOP", "PASTURE"))
        values.extend(min(1.0, other_structures.get(item, 0) / 16.0) for item in ("COOP", "PASTURE"))
        values.extend(min(1.0, _as_float(shed.get(item, 0)) / 100.0) for item in PRODUCTS)
        values.extend(min(1.0, carried.get(item, 0) / 30.0) for item in PRODUCTS)
        for item in PRODUCTS:
            price = _as_float(prices.get(item, 0))
            inventory = _as_float(market_inventory.get(item, 0))
            previous = self.last_prices.get(item, price)
            values.append(min(4.0, price / 250.0))
            values.append(min(1.0, inventory / 10000.0))
            values.append(max(-1.0, min(1.0, (price - previous) / 100.0)))
            self.last_prices[item] = price
        self.last_step = step
        values = [max(-5.0, min(5.0, float(item))) for item in values]
        if len(values) < FEATURE_DIM:
            values.extend([0.0] * (FEATURE_DIM - len(values)))
        return np.asarray(values[:FEATURE_DIM], dtype=np.float64)


class DoubleLinearQ:
    def __init__(self, weights_a=None, weights_b=None, seed=0, alpha=0.05, gamma=1.0):
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q_a = np.asarray(weights_a if weights_a is not None else np.zeros((ACTION_COUNT, FEATURE_DIM)), dtype=np.float64)
        self.q_b = np.asarray(weights_b if weights_b is not None else np.zeros((ACTION_COUNT, FEATURE_DIM)), dtype=np.float64)
        if self.q_a.shape != (ACTION_COUNT, FEATURE_DIM) or self.q_b.shape != (ACTION_COUNT, FEATURE_DIM):
            raise ValueError("RL weights must have shape (4, 96)")

    def values(self, features):
        vector = np.asarray(features, dtype=np.float64)
        return 0.5 * (self.q_a @ vector + self.q_b @ vector)

    def select(self, features, epsilon=0.0):
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(0, ACTION_COUNT))
        return int(np.argmax(self.values(features)))

    def update(self, features, action, reward, next_features=None, done=False):
        features = np.asarray(features, dtype=np.float64)
        action = int(action)
        if self.rng.random() < 0.5:
            current = float(self.q_a[action] @ features)
            if done or next_features is None:
                target = float(reward)
            else:
                next_action = int(np.argmax(self.q_a @ next_features))
                target = float(reward) + self.gamma * float(self.q_b[next_action] @ next_features)
            self.q_a[action] += self.alpha * (target - current) * features
        else:
            current = float(self.q_b[action] @ features)
            if done or next_features is None:
                target = float(reward)
            else:
                next_action = int(np.argmax(self.q_b @ next_features))
                target = float(reward) + self.gamma * float(self.q_a[next_action] @ next_features)
            self.q_b[action] += self.alpha * (target - current) * features

    def payload(self):
        return {"q_a": self.q_a.tolist(), "q_b": self.q_b.tolist()}


def _shed_count(obs, item):
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    return max(0, _as_int(shed.get(item, 0)))


def _market_order_item(order):
    if isinstance(order, list) and len(order) >= 3 and str(order[0]).upper() == "SELL":
        return str(order[1]).upper()
    return None


def _replace_sell_slots(market, priority):
    """Reorder existing premium SELL orders without changing quantities."""
    slots = []
    orders = list(market)
    for index, order in enumerate(orders):
        item = _market_order_item(order)
        if item in PREMIUM:
            slots.append(index)
    if len(slots) < 2:
        return orders
    ranked = sorted(
        (orders[index] for index in slots),
        key=lambda order: (priority.get(_market_order_item(order), 99),),
    )
    for index, order in zip(slots, ranked):
        orders[index] = list(order)
    return orders


def _profile_priority(obs, mode):
    farm = _farm(obs)
    opponent = _opponent_farm(obs)
    crops, animals, _, _, _, _ = _tile_counts(opponent)
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    money = _as_float(_get(farm, "money", 0))
    other_money = _as_float(_get(opponent, "money", 0))
    if mode == 2:
        # Conditional memory: when the public opponent resembles a livestock
        # route, move its likely collision products earlier among existing slots.
        if animals.get("COW", 0) + animals.get("SHEEP", 0) >= 10:
            order = ("MILK", "WOOL", "STRAWBERRY", "MELON")
        elif crops.get("STRAWBERRY", 0) + crops.get("MELON", 0) >= 50:
            order = ("STRAWBERRY", "MELON", "MILK", "WOOL")
        else:
            order = ("WOOL", "MILK", "MELON", "STRAWBERRY")
    else:
        # Frontier-style delayed profile: wait for visible economic context,
        # then prioritize the best quoted product while preserving the queue.
        if _as_int(_get(obs, "step", 0)) < 288 or len(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or []) < 4:
            return {}
        if money < other_money - 1000:
            order = tuple(sorted(PREMIUM, key=lambda item: -_as_float(prices.get(item, 0))))
        else:
            order = ("MILK", "WOOL", "STRAWBERRY", "MELON")
    return {item: index for index, item in enumerate(order)}


def apply_overlay(base_action, obs, mode, base_actions=None, state=None):
    """Apply one safe market mode to a V022c action."""
    action = _align_hands(base_action, obs)
    mode = int(mode)
    step = _as_int(_get(obs, "step", 0), 0)
    if step >= RL_STOP_STEP or mode == 0:
        return action
    if state is None:
        state = {}
    market = list(action.get("market", []) or [])

    if mode == 1 and base_actions is not None and step + 1 < len(base_actions):
        pending = state.setdefault("pending", {})
        # Repay a previous one-turn shift by reducing the planned next SELL.
        if pending:
            for item, quantity in list(pending.items()):
                if quantity <= 0:
                    pending.pop(item, None)
                    continue
                for index, order in enumerate(market):
                    if _market_order_item(order) == item:
                        current = max(0, _as_int(order[2]))
                        reduce = min(current, quantity)
                        if reduce == current:
                            market.pop(index)
                        else:
                            market[index] = [order[0], order[1], current - reduce]
                        pending[item] -= reduce
                        if pending[item] <= 0:
                            pending.pop(item, None)
                        break
        next_market = (base_actions[step + 1] or {}).get("market", []) or []
        existing = {_market_order_item(order) for order in market}
        if len(market) < 10:
            for order in next_market:
                item = _market_order_item(order)
                if item not in PREMIUM or item in existing:
                    continue
                next_quantity = max(0, _as_int(order[2]))
                quantity = min(8, next_quantity // 2, _shed_count(obs, item))
                if quantity <= 0:
                    continue
                market.append(["SELL", item, quantity])
                pending[item] = pending.get(item, 0) + quantity
                break
    elif mode in (2, 3):
        priority = _profile_priority(obs, mode)
        if priority:
            market = _replace_sell_slots(market, priority)

    action["market"] = market[:10]
    return action


class SelectorRuntime:
    """Stateful runtime used by local training and the generated submission."""

    def __init__(self, weights=None, training=False, seed=0):
        self.encoder = FeatureEncoder()
        payload = weights or {}
        self.q = DoubleLinearQ(payload.get("q_a"), payload.get("q_b"), seed=seed)
        self.training = bool(training)
        self.rng = np.random.default_rng(seed)
        self.mode = 0
        self.last_step = -1
        self.block_start_cash = None
        self.last_features = None
        self.last_action = None
        self.pending_state = {}
        self.transitions = []
        self.epsilon = 0.10 if self.training else 0.0
        self.fixed_mode = None
        self.forced_modes = {}
        self.boundary_features = {}

    def reset(self):
        self.encoder.reset()
        self.mode = 0
        self.last_step = -1
        self.block_start_cash = None
        self.last_features = None
        self.last_action = None
        self.pending_state = {}
        self.transitions = []
        self.boundary_features = {}

    def _cash(self, obs):
        return _as_float(_get(_farm(obs), "money", 0))

    def _boundary(self, step):
        return step == 0 or (step % BLOCK_STEPS == 0 and step <= RL_STOP_STEP)

    def choose(self, obs):
        step = _as_int(_get(obs, "step", 0), 0)
        if step == 0 or step < self.last_step:
            self.reset()
        if self._boundary(step):
            features = self.encoder.encode(obs)
            if self.training and self.last_features is not None and self.last_action is not None and self.block_start_cash is not None:
                reward = self._cash(obs) - self.block_start_cash
                self.q.update(self.last_features, self.last_action, reward, features, done=False)
                self.transitions.append({"step": step, "action": self.last_action, "reward": reward})
            if step >= RL_STOP_STEP:
                self.mode = 0
                self.last_features = None
                self.last_action = None
                self.block_start_cash = None
                self.last_step = step
                return self.mode
            self.boundary_features[step] = features.copy()
            if step in self.forced_modes:
                self.mode = int(self.forced_modes[step])
            elif self.fixed_mode is not None:
                self.mode = int(self.fixed_mode)
            else:
                self.mode = self.q.select(features, self.epsilon)
            self.last_features = features
            self.last_action = self.mode
            self.block_start_cash = self._cash(obs)
        self.last_step = step
        return self.mode

    def finish(self, obs):
        if self.last_features is None or self.last_action is None or self.block_start_cash is None:
            return
        if self.training:
            reward = self._cash(obs) - self.block_start_cash
            self.q.update(self.last_features, self.last_action, reward, None, done=True)
            self.transitions.append({"step": _as_int(_get(obs, "step", 0), 0), "action": self.last_action, "reward": reward})
        self.last_features = None
        self.last_action = None
        self.block_start_cash = None

    def act(self, obs, base_action, base_actions=None):
        mode = self.choose(obs)
        return apply_overlay(base_action, obs, mode, base_actions=base_actions, state=self.pending_state)


def make_agent(base_agent, base_actions=None, weights=None, training=False, seed=0):
    runtime = SelectorRuntime(weights=weights, training=training, seed=seed)

    def agent(obs, config=None):
        try:
            step = _as_int(_get(obs, "step", 0), 0)
            if step == 0:
                runtime.reset()
            base = base_agent(obs, config) if config is not None else base_agent(obs)
            return runtime.act(obs, base, base_actions=base_actions)
        except Exception:
            return _align_hands({"farmer": ["PASS"], "hands": [], "market": []}, obs)

    agent.runtime = runtime
    return agent
