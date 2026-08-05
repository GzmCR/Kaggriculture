"""V017 product-level market rollout controller.

This module is intentionally market-only.  It uses the current V012 expert
schedule as a scaffold, evaluates a small set of quantities for each premium
SELL order, and leaves farmer/hands plus every non-premium order untouched.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict


PREMIUM_PRODUCTS = ("MELON", "STRAWBERRY", "MILK", "WOOL")
MAX_MARKET_ORDERS = 10
HORIZON = 8
MARKET_I0 = 10000
MIN_REWRITE_CASH = 3000.0

MARKET = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "log", 0.20, "sqrt", 0.70),
    "TOMATO": (60, 200, "linear", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "linear", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}

SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

CROP_RULES = {
    "STRAWBERRY": {"first": 10, "max_yield": 4},
    "MELON": {"first": 10, "max_yield": 6},
}
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL"}
ANIMAL_MAX_HELD = {"COW": 6, "SHEEP": 6}

VARIANTS = {
    "curve_only": {
        "scenario_multipliers": (0.0,),
        "scenario_weights": (1.0,),
        "terminal_weight": 0.25,
        "min_gain": 20.0,
        "quota": 1.0,
    },
    "opponent_aware": {
        "scenario_multipliers": (1.0,),
        "scenario_weights": (1.0,),
        "terminal_weight": 0.25,
        "min_gain": 20.0,
        "quota": 1.0,
    },
    "robust_quota": {
        "scenario_multipliers": (0.5, 1.0, 1.5),
        "scenario_weights": (0.25, 0.50, 0.25),
        "terminal_weight": 0.35,
        "min_gain": 20.0,
        "quota": 0.50,
    },
}


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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
    return value


def _market_parameters(obs, item):
    base, throughput, below_fn, below_move, above_fn, above_move = MARKET[item]
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    custom = market.get("params", {}) if isinstance(market, dict) else {}
    custom = custom.get(item, {}) if isinstance(custom, dict) else {}
    if not isinstance(custom, dict):
        custom = {}
    return (
        _num(custom.get("base", base), base),
        _num(custom.get("T", throughput), throughput),
        str(custom.get("below_func", below_fn)),
        _num(custom.get("below_target", below_move), below_move),
        str(custom.get("above_func", above_fn)),
        _num(custom.get("above_target", above_move), above_move),
        _num(custom.get("I0", MARKET_I0), MARKET_I0),
    )


def price_at(item, inventory, obs=None):
    """Match kaggriculture.market_price, including rounding and floor."""
    if item not in MARKET:
        return 1
    base, throughput, below_fn, below_move, above_fn, above_move, equilibrium = _market_parameters(obs or {}, item)
    inventory = _num(inventory, equilibrium)
    if inventory < equilibrium:
        amplitude = below_move * base / max(1e-9, _shape(below_fn, throughput))
        value = base + amplitude * _shape(below_fn, equilibrium - inventory)
    else:
        amplitude = above_move * base / max(1e-9, _shape(above_fn, throughput))
        value = base - amplitude * _shape(above_fn, inventory - equilibrium)
    return max(1, int(round(value)))


def _copy_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or [])],
    }


def _farm_for(obs, player):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(player)
    if isinstance(farms, list) and 0 <= player < len(farms) and isinstance(farms[player], dict):
        return farms[player]
    return {}


def _inventory_total(obs, item):
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    if not isinstance(private, dict):
        return 0
    total = _int((private.get("shed", {}) or {}).get(item, 0))
    for inventory in private.get("inventories", []) or []:
        if isinstance(inventory, dict):
            total += _int(inventory.get(item, 0))
    return max(0, total)


def _town_demand(obs, item, step):
    if item == "FERTILIZER":
        return 0
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    town = obs.get("town", {}) if isinstance(obs, dict) else {}
    config = obs.get("configuration", {}) if isinstance(obs, dict) else {}
    shop_interval = max(1, _int(config.get("townShopSellInterval", 4))) if isinstance(config, dict) else 4
    center_interval = max(1, _int(config.get("townCenterSellInterval", 12))) if isinstance(config, dict) else 12
    day = _int(obs.get("day", step // 24)) + max(0, (step - _int(obs.get("step", step))) // 24)
    demand = 0
    if step % shop_interval == 0:
        for name in (town.get("unlocked_shops", []) if isinstance(town, dict) else []) or []:
            products = SHOPS.get(name, ())
            multiplier = 2 if len(products) == 1 else 1
            if item in products:
                demand += multiplier
    if step % center_interval == 0:
        center_multiplier = 4 if day >= 20 else 2 if day >= 10 else 1
        demand += center_multiplier
    return demand


def _opponent_supply_total(obs, item, horizon):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    opponent = 1 - player
    if not isinstance(farms, list) or not (0 <= opponent < len(farms)):
        return 0.0
    farm = farms[opponent] if isinstance(farms[opponent], dict) else {}
    day = _int(obs.get("day", 0)) if isinstance(obs, dict) else 0
    days_ahead = max(1, int(math.ceil((_int(obs.get("hour", 0)) + horizon) / 24.0))) if isinstance(obs, dict) else 1
    total = 0.0
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop", "")).upper()
            if tile.get("kind") == "PLANT" and crop == item and crop in CROP_RULES:
                planted_day = _int(tile.get("planted_day", day), day)
                age = day - planted_day
                held = max(0, _int(tile.get("yield_units", 0)))
                rule = CROP_RULES[crop]
                if age >= rule["first"]:
                    total += held
                elif age + days_ahead >= rule["first"]:
                    total += max(1, min(rule["max_yield"], held + 1))
            animal = str(tile.get("animal", "")).upper()
            if ANIMAL_PRODUCT.get(animal) == item:
                held = max(0, _int(tile.get("yield_units", 0)))
                total += held + min(days_ahead, ANIMAL_MAX_HELD.get(animal, 1))
    return float(total)


def _town_and_market_inventory(obs, item):
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    inventory = market.get("inventory", {}) if isinstance(market, dict) else {}
    return max(0, _int(inventory.get(item, MARKET_I0)))


def _sell_lockstep(inventory, own_quantity, opponent_quantity, item, obs):
    """Simulate one turn of two-player unit-wise SELL processing."""
    revenue = 0.0
    own_left = max(0, int(own_quantity))
    opponent_left = max(0, int(round(opponent_quantity)))
    while own_left or opponent_left:
        price = price_at(item, inventory, obs)
        sellers = 0
        if own_left:
            revenue += price
            own_left -= 1
            sellers += 1
        if opponent_left:
            opponent_left -= 1
            sellers += 1
        if price > 1:
            inventory += sellers
    return inventory, revenue


class MarketRolloutController:
    """Adjust only current premium SELL quantities using short rollout."""

    def __init__(self, variant="opponent_aware", runtime=None, selected_state=None, pipeline_fn=None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown V017 variant: {variant}")
        self.variant = variant
        self.config = dict(VARIANTS[variant])
        self.runtime = runtime or {}
        self.selected_state = selected_state if isinstance(selected_state, dict) else {}
        self.pipeline_fn = pipeline_fn
        self._states = {}

    def _state(self, player):
        player = _int(player)
        if player not in self._states:
            self._states[player] = {
                "last_step": -1,
                "stats": Counter(),
                "items": defaultdict(lambda: {"base": 0, "chosen": 0, "gain": 0.0}),
                "pending": defaultdict(int),
                "history": [],
            }
        return self._states[player]

    def reset(self, player=None):
        if player is None:
            self._states.clear()
        else:
            self._states.pop(_int(player), None)

    def _expert_name(self, obs):
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        name = self.selected_state.get(player)
        experts = self.runtime.get("experts", {}) if isinstance(self.runtime, dict) else {}
        if name in experts:
            return name
        board = self.runtime.get("board_by_seat", {}) if isinstance(self.runtime, dict) else {}
        fallback = board.get(str(player)) if isinstance(board, dict) else None
        if fallback in experts:
            return fallback
        names = sorted(experts)
        return names[0] if names else None

    def _expert_orders(self, expert_name, step):
        experts = self.runtime.get("experts", {}) if isinstance(self.runtime, dict) else {}
        expert = experts.get(expert_name, {}) if isinstance(experts, dict) else {}
        actions = expert.get("actions", []) if isinstance(expert, dict) else []
        result = []
        for offset in range(HORIZON):
            index = min(max(0, step + offset), max(0, len(actions) - 1))
            action = actions[index] if actions else {}
            quantities = defaultdict(int)
            for order in (action.get("market", []) if isinstance(action, dict) else []) or []:
                if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS:
                    quantities[order[1]] += max(0, _int(order[2]))
            result.append(dict(quantities))
        return result

    def _scenarios(self, obs, item):
        if self.variant == "curve_only":
            return [(0.0, 1.0)]
        total = _opponent_supply_total(obs, item, HORIZON)
        per_turn = total / float(HORIZON)
        return [(per_turn * multiplier, weight) for multiplier, weight in zip(self.config["scenario_multipliers"], self.config["scenario_weights"])]

    def _simulate(self, obs, item, base_schedule, candidate, opponent_per_turn):
        step = _int(obs.get("step", 0)) if isinstance(obs, dict) else 0
        inventory = _town_and_market_inventory(obs, item)
        revenue = 0.0
        sold = 0
        for offset in range(HORIZON):
            own_quantity = candidate if offset == 0 else base_schedule[offset]
            opponent_quantity = opponent_per_turn
            inventory, turn_revenue = _sell_lockstep(inventory, own_quantity, opponent_quantity, item, obs)
            revenue += turn_revenue
            sold += max(0, int(own_quantity))
            inventory = max(0, inventory - _town_demand(obs, item, step + offset))
        available = _inventory_total(obs, item)
        day = _int(obs.get("day", step // 24)) if isinstance(obs, dict) else step // 24
        terminal_factor = max(0.0, min(1.0, (day - 23.0) / 7.0))
        unsold = max(0, available - sold)
        base_price = MARKET[item][0]
        # Carry the units that remain after the simulated window at their
        # forecast end-of-window price.  Without this mark-to-market term a
        # positive sale is always preferred, even when it depresses the
        # expert's own sales in the following turns.
        residual_value = unsold * price_at(item, inventory, obs) * 0.75
        inventory_penalty = unsold * base_price * self.config["terminal_weight"] * terminal_factor
        return {
            "revenue": revenue,
            "sold": sold,
            "unsold": unsold,
            "inventory": inventory,
            "residual_value": residual_value,
            "score": revenue + residual_value - inventory_penalty,
        }

    def _score(self, obs, item, base_schedule, candidate):
        scenario_results = []
        for opponent_per_turn, weight in self._scenarios(obs, item):
            result = self._simulate(obs, item, base_schedule, candidate, opponent_per_turn)
            scenario_results.append((result, weight))
        scores = [result["score"] for result, _ in scenario_results]
        weights = [weight for _, weight in scenario_results]
        total_weight = sum(weights) or 1.0
        mean_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
        worst_score = min(scores) if scores else 0.0
        risk_penalty = 0.35 * (mean_score - worst_score) if self.variant == "robust_quota" else 0.0
        return {
            "score": mean_score - risk_penalty,
            "mean_score": mean_score,
            "worst_score": worst_score,
            "scenario_scores": scores,
        }

    def _candidate_quantities(self, obs, item, base_quantity):
        base_quantity = max(0, int(base_quantity))
        if base_quantity <= 0:
            return [0]
        values = {0, base_quantity}
        for fraction in (0.25, 0.50, 0.75):
            values.add(int(math.ceil(base_quantity * fraction)))
        if self.variant == "robust_quota" and _int(obs.get("day", 0)) < 27:
            available = max(base_quantity, _inventory_total(obs, item))
            quota = max(1, int(math.ceil(available * self.config["quota"])))
            values = {min(value, quota) for value in values}
        return sorted(values)

    def _item_slots(self, orders):
        slots = defaultdict(list)
        for index, order in enumerate(orders):
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS:
                slots[order[1]].append((index, max(0, _int(order[2]))))
        return slots

    def _cash_safe_to_rewrite(self, obs, orders):
        """Do not trade away cash needed by the fixed route's next buys."""
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        farm = _farm_for(obs, player)
        if _num(farm.get("money", 0)) < MIN_REWRITE_CASH:
            return False
        critical = {"HIRE", "BUY_LAND", "BUY_ANIMAL", "BUY_PRODUCT"}
        if any(isinstance(order, list) and order and order[0] in critical for order in orders):
            return False
        return True

    def apply(self, obs, base_action):
        action = _copy_action(base_action)
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        step = _int(obs.get("step", 0)) if isinstance(obs, dict) else 0
        state = self._state(player)
        if step == 0 or step <= state["last_step"]:
            self.reset(player)
            state = self._state(player)
        state["last_step"] = step

        orders = action["market"]
        if not self._cash_safe_to_rewrite(obs, orders):
            state["stats"]["cash_guard_skips"] += 1
            return action
        slots = self._item_slots(orders)
        if not slots:
            state["stats"]["no_premium_slot"] += 1
            return action
        expert_name = self._expert_name(obs)
        schedules = self._expert_orders(expert_name, step)
        chosen_by_item = {}
        for item, item_slots in slots.items():
            base_quantity = sum(quantity for _, quantity in item_slots)
            base_schedule = [int(schedule.get(item, 0)) for schedule in schedules]
            pending_before = max(0, int(state["pending"][item]))
            release = min(pending_before, max(1, int(math.ceil(base_quantity * 0.25)))) if base_quantity > 0 else 0
            candidates = self._candidate_quantities(obs, item, base_quantity)
            scored = {quantity: self._score(obs, item, base_schedule, quantity + release) for quantity in candidates}
            base_score = self._score(obs, item, base_schedule, base_quantity + release)["score"]
            best_quantity = max(candidates, key=lambda quantity: (scored[quantity]["score"], quantity))
            best_score = scored[best_quantity]["score"]
            if best_score - base_score < self.config["min_gain"]:
                # Robust-quota is deliberately a hard pacing guard before
                # day 27.  The other variants remain no-op by default when
                # rollout confidence is too small.
                if self.variant == "robust_quota" and _int(obs.get("day", 0)) < 27:
                    best_quantity = max(candidates)
                    best_score = scored[best_quantity]["score"]
                else:
                    best_quantity = base_quantity
                    best_score = base_score
            actual_quantity = best_quantity + release
            deferred = max(0, base_quantity - best_quantity)
            state["pending"][item] = max(0, pending_before - release) + deferred
            chosen_by_item[item] = best_quantity
            item_state = state["items"][item]
            item_state.update({"base": base_quantity, "chosen": actual_quantity, "gain": best_score - base_score})
            state["stats"][f"base_{item}"] += base_quantity
            state["stats"][f"chosen_{item}"] += actual_quantity
            state["stats"][f"released_{item}"] += release
            state["stats"]["released_units"] += release
            state["stats"][f"pending_{item}"] = state["pending"][item]
            state["stats"][f"candidate_evaluations_{item}"] += len(candidates)
            state["stats"][f"scenario_evaluations_{item}"] += len(self._scenarios(obs, item))
            if best_quantity != base_quantity:
                state["stats"]["adjusted_items"] += 1
                state["stats"][f"adjusted_{item}"] += 1
                state["stats"]["adjusted_units"] += abs(best_quantity - base_quantity)
            chosen_by_item[item] = actual_quantity

        allocations = {}
        for item, item_slots in slots.items():
            remaining = max(0, int(chosen_by_item[item]))
            for position, (index, original_quantity) in enumerate(item_slots):
                if position == len(item_slots) - 1:
                    quantity = remaining
                else:
                    quantity = min(original_quantity, remaining)
                allocations[index] = max(0, int(quantity))
                remaining = max(0, remaining - allocations[index])

        rewritten = []
        for index, order in enumerate(orders):
            if not (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS):
                rewritten.append(list(order))
                continue
            quantity = allocations.get(index, 0)
            if quantity > 0:
                rewritten.append(["SELL", order[1], quantity])
        action["market"] = rewritten[:MAX_MARKET_ORDERS]
        state["stats"]["market_actions"] += 1
        if action["market"] != orders:
            state["stats"]["changed_market_actions"] += 1
        state["history"].append({
            "step": step,
            "expert": expert_name,
            "chosen": dict(chosen_by_item),
        })
        return action

    def diagnostics(self, player=0):
        state = self._state(player)
        result = dict(state["stats"])
        result.update({
            "variant": self.variant,
            "last_step": state["last_step"],
            "history": copy.deepcopy(state["history"]),
            "items": copy.deepcopy(dict(state["items"])),
            "pending": dict(state["pending"]),
        })
        return result
