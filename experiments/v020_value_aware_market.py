"""V020 value-aware market controller.

The controller is intentionally market-only.  It receives a V012 action,
simulates a short premium-product rollout against public supply scenarios,
and rewrites only existing premium SELL quantities.  Farmer/hands actions and
non-premium market orders remain untouched.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict


PREMIUM_PRODUCTS = ("MELON", "STRAWBERRY", "MILK", "WOOL")
MAX_MARKET_ORDERS = 10
HORIZON = 72
BALANCED_START = 648
FORCE_FLUSH_STEP = 708

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

ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
ANIMAL_DAILY = {"GOOSE": 1.0, "COW": 0.5, "SHEEP": 1.0 / 3.0}
LAND_COST = {"NE": 1000.0, "SW": 2000.0, "SE": 4000.0}
ANIMAL_COST = {"GOOSE": 250.0, "COW": 400.0, "SHEEP": 500.0}
PRODUCT_BUY_COST = {"WHEAT": 10.0, "FERTILIZER": 100.0}

VARIANTS = {
    "balanced": {
        "scenario_multipliers": (0.5, 1.0, 1.5),
        "scenario_weights": (0.25, 0.50, 0.25),
        "risk_weight": 0.35,
        "collision_weight": 0.35,
        "residual_weight": 0.75,
        "carried_weight": 0.25,
        "min_gain": 25.0,
        "terminal_penalty": 0.30,
        "active_products": ("MILK",),
    },
    "sensitive": {
        "scenario_multipliers": (0.5, 1.0, 1.5),
        "scenario_weights": (0.20, 0.55, 0.25),
        "risk_weight": 0.50,
        "collision_weight": 0.45,
        "residual_weight": 0.85,
        "carried_weight": 0.25,
        "min_gain": 10.0,
        "terminal_penalty": 0.35,
        "active_products": ("MILK", "WOOL", "STRAWBERRY", "MELON"),
    },
    "conservative": {
        "scenario_multipliers": (0.5, 1.0, 1.5),
        "scenario_weights": (0.30, 0.50, 0.20),
        "risk_weight": 0.25,
        "collision_weight": 0.25,
        "residual_weight": 0.65,
        "carried_weight": 0.20,
        "min_gain": 40.0,
        "terminal_penalty": 0.25,
        "active_products": ("MILK", "WOOL"),
    },
}


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name in {"log", "log10"}:
        return math.log1p(value) if name == "log" else math.log10(1.0 + value)
    return value


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
    if isinstance(farms, list) and 0 <= player < len(farms):
        farm = farms[player]
        return farm if isinstance(farm, dict) else {}
    return {}


def _market_for(obs):
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    return market if isinstance(market, dict) else {}


def _market_parameters(obs, item):
    base, throughput, below_fn, below_move, above_fn, above_move = MARKET[item]
    custom = _market_for(obs).get("params", {})
    custom = custom.get(item, {}) if isinstance(custom, dict) else {}
    custom = custom if isinstance(custom, dict) else {}
    return (
        _num(custom.get("base", base), base),
        _num(custom.get("T", throughput), throughput),
        str(custom.get("below_func", below_fn)),
        _num(custom.get("below_target", below_move), below_move),
        str(custom.get("above_func", above_fn)),
        _num(custom.get("above_target", above_move), above_move),
        _num(custom.get("I0", 10000), 10000),
    )


def price_at(item, inventory, obs=None):
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


def _market_inventory(obs, item):
    return max(0, _int((_market_for(obs).get("inventory", {}) or {}).get(item, 10000), 10000))


def _shed_inventory(obs, item):
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    shed = private.get("shed", {}) if isinstance(private, dict) else {}
    return max(0, _int((shed or {}).get(item, 0)))


def _carried_inventory(obs, item):
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    inventories = private.get("inventories", []) if isinstance(private, dict) else []
    total = 0
    for inventory in inventories if isinstance(inventories, list) else []:
        if isinstance(inventory, dict):
            total += max(0, _int(inventory.get(item, 0)))
    return total


def liquidation_value(item, quantity, inventory, obs):
    total = 0.0
    for offset in range(max(0, _int(quantity))):
        total += price_at(item, inventory + offset, obs)
    return total


def _town_demand(obs, item, step):
    if item == "FERTILIZER":
        return 0
    market = _market_for(obs)
    town = obs.get("town", {}) if isinstance(obs, dict) else {}
    config = obs.get("configuration", {}) if isinstance(obs, dict) else {}
    config = config if isinstance(config, dict) else {}
    shop_interval = max(1, _int(config.get("townShopSellInterval", 4), 4))
    center_interval = max(1, _int(config.get("townCenterSellInterval", 12), 12))
    day = _int(obs.get("day", step // 24), step // 24) + max(0, (step - _int(obs.get("step", step), step)) // 24)
    demand = 0
    if step % shop_interval == 0:
        for name in (town.get("unlocked_shops", []) if isinstance(town, dict) else []) or []:
            products = SHOPS.get(name, ())
            multiplier = 2 if len(products) == 1 else 1
            if item in products:
                demand += multiplier
    if step % center_interval == 0:
        demand += 4 if day >= 20 else 2 if day >= 10 else 1
    return demand


def _sell_lockstep(inventory, own_quantity, opponent_quantity, item, obs):
    revenue = 0.0
    own_left = max(0, _int(own_quantity))
    opponent_left = max(0, int(round(opponent_quantity)))
    while own_left or opponent_left:
        price = price_at(item, inventory, obs)
        sellers = 0
        if own_left:
            own_left -= 1
            revenue += price
            sellers += 1
        if opponent_left:
            opponent_left -= 1
            sellers += 1
        if price > 1:
            inventory += sellers
    return inventory, revenue


def _opponent_supply_profile(obs, item, horizon):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    opponent = 1 - player
    if not isinstance(farms, list) or not (0 <= opponent < len(farms)):
        return 0.0, 0.0
    farm = farms[opponent] if isinstance(farms[opponent], dict) else {}
    day = _int(obs.get("day", 0)) if isinstance(obs, dict) else 0
    hour = _int(obs.get("hour", 0)) if isinstance(obs, dict) else 0
    days_ahead = max(1, int(math.ceil((hour + horizon) / 24.0)))
    held = 0.0
    future = 0.0
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop", "")).upper()
            if tile.get("kind") == "PLANT" and crop == item:
                units = max(0, _int(tile.get("yield_units", 0)))
                if crop == "STRAWBERRY":
                    first = 10
                    if _int(tile.get("planted_day", day), day) + first <= day:
                        held += units
                    elif _int(tile.get("planted_day", day), day) + first <= day + days_ahead:
                        future += max(1, units)
                elif crop == "MELON":
                    first = 10
                    if _int(tile.get("planted_day", day), day) + first <= day:
                        held += units
                    elif _int(tile.get("planted_day", day), day) + first <= day + days_ahead:
                        future += max(1, units)
            animal = str(tile.get("animal", "")).upper()
            if ANIMAL_PRODUCT.get(animal) == item:
                held += max(0, _int(tile.get("yield_units", 0)))
                future += ANIMAL_DAILY.get(animal, 0.0) * days_ahead
    return held, future


def _expected_future_orders(runtime, obs, step, horizon):
    if not isinstance(runtime, dict):
        return [defaultdict(int) for _ in range(horizon)]
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    board = runtime.get("board_by_seat", {})
    expert_name = board.get(str(player)) if isinstance(board, dict) else None
    experts = runtime.get("experts", {})
    expert = experts.get(expert_name, {}) if isinstance(experts, dict) and expert_name else {}
    actions = expert.get("actions", []) if isinstance(expert, dict) else []
    result = []
    for offset in range(horizon):
        index = min(max(0, step + offset), max(0, len(actions) - 1))
        action = actions[index] if actions else {}
        quantities = defaultdict(int)
        for order in (action.get("market", []) if isinstance(action, dict) else []) or []:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS:
                quantities[order[1]] += max(0, _int(order[2]))
        result.append(quantities)
    return result


def _future_costs(runtime, obs, step, horizon):
    schedules = _expected_future_orders(runtime, obs, step, horizon)
    costs = 0.0
    hires = _int(_farm_for(obs, _int(obs.get("player", 0))).get("hires_today", 0)) if isinstance(obs, dict) else 0
    # Premium sell quantities have no cost.  The route's non-premium
    # purchases are approximated conservatively for the cash reserve.
    return costs, hires


class ValueAwareMarketController:
    """Use portfolio value and public supply scenarios to pace premium sales."""

    def __init__(self, variant="balanced", runtime=None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown V020 variant: {variant}")
        self.variant = variant
        self.config = dict(VARIANTS[variant])
        self.runtime = runtime if isinstance(runtime, dict) else {}
        self._states = {}

    def _new_state(self):
        return {
            "last_step": -1,
            "stats": Counter(),
            "items": defaultdict(dict),
            "pending": defaultdict(int),
            "defer_used": defaultdict(bool),
            "prices": {item: [] for item in PREMIUM_PRODUCTS},
            "supplies": {item: [] for item in PREMIUM_PRODUCTS},
            "history": [],
        }

    def _state(self, player):
        player = _int(player)
        if player not in self._states:
            self._states[player] = self._new_state()
        return self._states[player]

    def reset(self, player=None):
        if player is None:
            self._states.clear()
        else:
            self._states.pop(_int(player), None)

    def _item_slots(self, orders):
        slots = defaultdict(list)
        for index, order in enumerate(orders):
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS:
                slots[order[1]].append((index, max(0, _int(order[2]))))
        return slots

    def _candidate_quantities(self, base_quantity):
        base_quantity = max(0, _int(base_quantity))
        if base_quantity <= 0:
            return [0]
        values = {0, 1, base_quantity}
        for fraction in (0.25, 0.50, 0.75):
            values.add(int(math.ceil(base_quantity * fraction)))
        return sorted(value for value in values if 0 <= value <= base_quantity)

    def _market_signal(self, obs, item, state, step):
        market = _market_for(obs)
        prices = market.get("prices", {}) if isinstance(market, dict) else {}
        supplies = market.get("inventory", {}) if isinstance(market, dict) else {}
        price = _num(prices.get(item, MARKET[item][0]), MARKET[item][0])
        supply = _int(supplies.get(item, 10000), 10000)
        history = state["prices"][item]
        supply_history = state["supplies"][item]
        previous = history[-1] if history else None
        previous_supply = supply_history[-1] if supply_history else None
        drop = price / previous - 1.0 if previous else 0.0
        supply_jump = supply - previous_supply if previous_supply is not None else 0
        held, future = _opponent_supply_profile(obs, item, HORIZON)
        day = _int(obs.get("day", step // 24), step // 24)
        in_wave_window = 10 <= day <= 27
        # A public mature inventory alone is not enough to rewrite orders.  A
        # recent price move or a corroborating supply jump is required.  This
        # prevents the forecast from treating every animal as an immediate
        # sale, which was the main V020 replay failure mode.
        shock = in_wave_window and (
            (previous is not None and drop <= -0.06 and held >= 3)
            or (previous is not None and drop <= -0.10)
            or (supply_jump >= 6 and held >= 6 and price <= MARKET[item][0] * 1.02)
        )
        history.append(price)
        supply_history.append(supply)
        del history[:-8]
        del supply_history[:-8]
        return {
            "price": price,
            "previous": previous,
            "drop": drop,
            "supply_jump": supply_jump,
            "held": held,
            "future": future,
            "shock": shock,
        }

    def _scenario_scores(self, obs, item, step, current_quantity, future_schedule, multiplier):
        market_inventory = _market_inventory(obs, item)
        shed_available = _shed_inventory(obs, item)
        carried_available = _carried_inventory(obs, item)
        held, future = _opponent_supply_profile(obs, item, HORIZON)
        opponent_now = held * multiplier
        opponent_per_turn = future * multiplier / float(HORIZON)
        own_sold = 0
        revenue = 0.0
        inventory = market_inventory
        for offset in range(HORIZON):
            own_quantity = current_quantity if offset == 0 else max(0, _int(future_schedule[offset]))
            if offset == 0:
                own_quantity = min(own_quantity, shed_available)
            opponent_quantity = opponent_now if offset == 0 else opponent_per_turn
            inventory, turn_revenue = _sell_lockstep(inventory, own_quantity, opponent_quantity, item, obs)
            own_sold += own_quantity
            revenue += turn_revenue
            inventory = max(0, inventory - _town_demand(obs, item, step + offset))

        remaining_shed = max(0, shed_available - own_sold)
        residual_value = liquidation_value(item, remaining_shed, inventory, obs)
        carried_value = carried_available * price_at(item, inventory, obs)
        day = _int(obs.get("day", step // 24), step // 24)
        late_factor = max(0.0, min(1.0, (day - 23.0) / 7.0))
        terminal_penalty = remaining_shed * MARKET[item][0] * self.config["terminal_penalty"] * late_factor
        collision = max(0.0, opponent_now) * price_at(item, market_inventory, obs)
        score = (
            revenue
            + self.config["residual_weight"] * residual_value
            + self.config["carried_weight"] * carried_value
            - terminal_penalty
            - self.config["collision_weight"] * collision
        )
        return {
            "score": score,
            "revenue": revenue,
            "residual_value": residual_value,
            "carried_value": carried_value,
            "terminal_penalty": terminal_penalty,
            "collision": collision,
            "remaining_shed": remaining_shed,
            "opponent_now": opponent_now,
            "opponent_future": future * multiplier,
        }

    def _score(self, obs, item, step, current_quantity, future_schedule):
        results = []
        for multiplier, weight in zip(self.config["scenario_multipliers"], self.config["scenario_weights"]):
            results.append((self._scenario_scores(obs, item, step, current_quantity, future_schedule, multiplier), weight))
        total_weight = sum(weight for _, weight in results) or 1.0
        mean_score = sum(result["score"] * weight for result, weight in results) / total_weight
        worst_score = min((result["score"] for result, _ in results), default=mean_score)
        score = mean_score - self.config["risk_weight"] * max(0.0, mean_score - worst_score)
        merged = dict(results[1][0] if len(results) > 1 else results[0][0])
        merged.update({"score": score, "mean_score": mean_score, "worst_score": worst_score})
        return merged

    def _cash_reserve(self, obs):
        farm = _farm_for(obs, _int(obs.get("player", 0))) if isinstance(obs, dict) else {}
        animals = 0
        for row in farm.get("tiles", []) or []:
            for tile in row if isinstance(row, list) else []:
                if isinstance(tile, dict) and str(tile.get("animal", "")).upper() in ANIMAL_PRODUCT:
                    animals += 1
        wheat = _shed_inventory(obs, "WHEAT") + _carried_inventory(obs, "WHEAT")
        return max(500.0, (animals * 3.0 + 2.0 - wheat) * 10.0)

    def _apply_terminal_flush(self, obs, action, state, step):
        if step < FORCE_FLUSH_STEP:
            return action
        # V012 already has a terminal liquidation schedule.  The controller
        # only records the phase; inventing extra floor-price orders here can
        # change the shared market more than the deferred unit is worth.
        state["stats"]["terminal_phase"] += 1
        return action

    def apply(self, obs, base_action):
        action = _copy_action(base_action)
        if not isinstance(obs, dict):
            return action
        player = _int(obs.get("player", 0))
        step = _int(obs.get("step", 0))
        state = self._state(player)
        if step == 0 or step <= state["last_step"]:
            self.reset(player)
            state = self._state(player)
        state["last_step"] = step
        original_orders = [list(order) for order in action["market"]]
        slots = self._item_slots(original_orders)
        active_products = set(self.config.get("active_products", PREMIUM_PRODUCTS))
        slots = defaultdict(list, {
            item: entries for item, entries in slots.items() if item in active_products
        })
        if not slots:
            state["stats"]["no_premium_slot"] += 1
            return self._apply_terminal_flush(obs, action, state, step)

        signals = {item: self._market_signal(obs, item, state, step) for item in PREMIUM_PRODUCTS}
        day = _int(obs.get("day", step // 24), step // 24)
        pending_exists = any(state["pending"].get(item, 0) > 0 for item in PREMIUM_PRODUCTS)
        if day < 27 and not pending_exists and not any(signal["shock"] for signal in signals.values()):
            state["stats"]["normal_market_passthrough"] += 1
            return action

        future = _expected_future_orders(self.runtime, obs, step, HORIZON)
        chosen = {}
        for item, item_slots in slots.items():
            base_quantity = sum(quantity for _, quantity in item_slots)
            future_schedule = [max(0, _int(row.get(item, 0))) for row in future]
            pending_before = max(0, _int(state["pending"].get(item, 0)))
            signal = signals[item]
            release = 0
            if pending_before and not signal["shock"] and base_quantity > 0:
                release = min(pending_before, max(1, int(math.ceil(base_quantity * 0.25))))
            effective_base = base_quantity + release
            baseline = self._score(obs, item, step, effective_base, future_schedule)
            best_quantity = base_quantity
            best = baseline
            candidates = self._candidate_quantities(base_quantity) if signal["shock"] or day >= 27 else [base_quantity]
            if signal["shock"] and pending_before <= 0 and not state["defer_used"].get(item, False):
                # Keep the live intervention bounded: a route may defer at
                # most one unit per product until the next recovery.  This
                # preserves V012's production cadence under noisy forecasts.
                candidates = sorted({base_quantity, max(0, base_quantity - 1)})
            for candidate in candidates:
                scored = self._score(obs, item, step, candidate + release, future_schedule)
                if scored["score"] > best["score"] + 1e-9 or (
                    abs(scored["score"] - best["score"]) <= 1e-9 and candidate > best_quantity
                ):
                    best_quantity = candidate
                    best = scored
            min_gain = self.config["min_gain"]
            if day >= 27:
                min_gain *= 0.5
            if best["score"] - baseline["score"] < min_gain:
                best_quantity = base_quantity
                best = baseline
            actual_quantity = best_quantity + release
            chosen[item] = actual_quantity
            deferred = min(1, max(0, base_quantity - best_quantity))
            state["pending"][item] = min(1, max(0, pending_before - release) + deferred)
            if deferred:
                state["defer_used"][item] = True
            state["stats"][f"base_{item}"] += base_quantity
            state["stats"][f"chosen_{item}"] += actual_quantity
            state["stats"][f"pending_{item}"] = state["pending"][item]
            state["stats"][f"candidate_evaluations_{item}"] += len(candidates)
            state["stats"][f"released_{item}"] += release
            if actual_quantity != base_quantity:
                state["stats"]["adjusted_items"] += 1
                state["stats"]["adjusted_units"] += abs(actual_quantity - base_quantity)
            state["items"][item] = {
                "step": step,
                "base": base_quantity,
                "chosen": actual_quantity,
                "released": release,
                "deferred": deferred,
                "gain": round(best["score"] - baseline["score"], 4),
                "opponent_now": round(best.get("opponent_now", 0.0), 4),
                "opponent_future": round(best.get("opponent_future", 0.0), 4),
                "residual_value": round(best.get("residual_value", 0.0), 4),
                "carried_value": round(best.get("carried_value", 0.0), 4),
            }

        allocations = {}
        for item, item_slots in slots.items():
            remaining = max(0, _int(chosen[item]))
            for position, (index, original_quantity) in enumerate(item_slots):
                quantity = remaining if position == len(item_slots) - 1 else min(original_quantity, remaining)
                allocations[index] = max(0, _int(quantity))
                remaining = max(0, remaining - allocations[index])

        rewritten = []
        for index, order in enumerate(original_orders):
            if not (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS):
                rewritten.append(list(order))
                continue
            if index not in allocations:
                # Premium products outside this variant's active set are
                # still part of the V012 market plan and must pass through.
                rewritten.append(list(order))
                continue
            quantity = allocations.get(index, 0)
            if quantity > 0:
                rewritten.append(["SELL", order[1], quantity])
        action["market"] = rewritten[:MAX_MARKET_ORDERS]
        action = self._apply_terminal_flush(obs, action, state, step)
        state["stats"]["market_actions"] += 1
        if action["market"] != original_orders:
            state["stats"]["changed_market_actions"] += 1
        state["history"].append({"step": step, "chosen": dict(chosen), "pending": dict(state["pending"])})
        if len(state["history"]) > 96:
            del state["history"][:-96]
        return action

    def diagnostics(self, player=0):
        state = self._state(player)
        result = dict(state["stats"])
        result.update({
            "variant": self.variant,
            "last_step": state["last_step"],
            "items": copy.deepcopy(dict(state["items"])),
            "pending": dict(state["pending"]),
            "history": copy.deepcopy(state["history"]),
        })
        return result
