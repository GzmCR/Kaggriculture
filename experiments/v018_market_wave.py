"""V018 parameterized premium-market waves and daily MPC.

The module depends on the exact price/lockstep helpers from v017 when used as
a normal Python module.  The builder concatenates v017 before this file for a
self-contained Kaggle artifact.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict

from v017_market_rollout import (
    MARKET,
    PREMIUM_PRODUCTS,
    MAX_MARKET_ORDERS,
    _copy_action,
    _farm_for,
    _inventory_total,
    _opponent_supply_total,
    _sell_lockstep,
    _town_demand,
    _int,
    _num,
    price_at,
)


MPC_HORIZON = 72
MIN_REWRITE_CASH = 3000.0

# These are deliberately broad, interpretable priors rather than a fitted
# replay trace.  The daily MPC can reduce the ratio in response to price and
# public supply, but never creates a premium order absent from V012.
WAVE_PARAMS = {
    "MELON": {
        "start_day": 10,
        "end_day": 25,
        "pre_window_ratio": 0.50,
        "inventory_fraction": 0.30,
        "min_price_ratio": 0.82,
    },
    "STRAWBERRY": {
        "start_day": 13,
        "end_day": 26,
        "pre_window_ratio": 0.50,
        "inventory_fraction": 0.25,
        "min_price_ratio": 0.84,
    },
    "MILK": {
        "start_day": 11,
        "end_day": 27,
        "pre_window_ratio": 0.75,
        "inventory_fraction": 0.20,
        "min_price_ratio": 0.82,
    },
    "WOOL": {
        "start_day": 13,
        "end_day": 27,
        "pre_window_ratio": 0.50,
        "inventory_fraction": 0.18,
        "min_price_ratio": 0.86,
    },
}

VARIANTS = {
    "fixed_wave": {
        "use_mpc": False,
        "scenario_multipliers": (0.0,),
        "scenario_weights": (1.0,),
        "risk_weight": 0.0,
        "min_gain": 0.0,
    },
    "daily_mpc": {
        "use_mpc": True,
        "scenario_multipliers": (1.0,),
        "scenario_weights": (1.0,),
        "risk_weight": 0.0,
        "min_gain": 25.0,
    },
    "robust_mpc": {
        "use_mpc": True,
        "scenario_multipliers": (0.5, 1.0, 1.5),
        "scenario_weights": (0.25, 0.50, 0.25),
        "risk_weight": 0.35,
        "min_gain": 25.0,
    },
}


def _wave_upper_ratio(obs, item):
    day = _int(obs.get("day", 0)) if isinstance(obs, dict) else 0
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    prices = market.get("prices", {}) if isinstance(market, dict) else {}
    price = _num(prices.get(item, MARKET[item][0]), MARKET[item][0])
    params = WAVE_PARAMS[item]
    if day < params["start_day"]:
        upper = params["pre_window_ratio"]
    else:
        upper = 1.0
    if day < 27 and price < MARKET[item][0] * params["min_price_ratio"]:
        upper = min(upper, 0.25)
    return max(0.0, min(1.0, upper))


class MarketWaveController:
    """Apply fixed waves or daily MPC to existing premium SELL slots."""

    def __init__(self, variant="daily_mpc", runtime=None, selected_state=None, pipeline_fn=None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown V018 variant: {variant}")
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
                "plan_day": None,
                "ratios": {item: 1.0 for item in PREMIUM_PRODUCTS},
                "stats": Counter(),
                "history": [],
            }
        return self._states[player]

    def _expert_name(self, obs):
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        experts = self.runtime.get("experts", {}) if isinstance(self.runtime, dict) else {}
        selected = self.selected_state.get(player)
        if selected in experts:
            return selected
        board = self.runtime.get("board_by_seat", {}) if isinstance(self.runtime, dict) else {}
        fallback = board.get(str(player)) if isinstance(board, dict) else None
        return fallback if fallback in experts else (sorted(experts)[0] if experts else None)

    def _future_schedule(self, expert_name, step):
        experts = self.runtime.get("experts", {}) if isinstance(self.runtime, dict) else {}
        expert = experts.get(expert_name, {}) if isinstance(experts, dict) else {}
        actions = expert.get("actions", []) if isinstance(expert, dict) else []
        schedule = []
        for offset in range(MPC_HORIZON):
            index = min(max(0, step + offset), max(0, len(actions) - 1))
            action = actions[index] if actions else {}
            quantities = defaultdict(int)
            for order in (action.get("market", []) if isinstance(action, dict) else []) or []:
                if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS:
                    quantities[order[1]] += max(0, _int(order[2]))
            schedule.append(dict(quantities))
        return schedule

    def _scenarios(self, obs, item):
        if self.variant == "fixed_wave":
            return [(0.0, 1.0)]
        total = _opponent_supply_total(obs, item, MPC_HORIZON)
        per_turn = total / float(MPC_HORIZON)
        return [
            (per_turn * multiplier, weight)
            for multiplier, weight in zip(self.config["scenario_multipliers"], self.config["scenario_weights"])
        ]

    def _simulate(self, obs, item, schedule, ratio, opponent_per_turn, current_quantity):
        step = _int(obs.get("step", 0)) if isinstance(obs, dict) else 0
        market = obs.get("market", {}) if isinstance(obs, dict) else {}
        inventory_data = market.get("inventory", {}) if isinstance(market, dict) else {}
        inventory = max(0, _int(inventory_data.get(item, 10000)))
        available = _inventory_total(obs, item)
        revenue = 0.0
        sold = 0
        for offset in range(MPC_HORIZON):
            # The candidate ratio is a decision for the current SELL slot.
            # Future slots are kept at the selected V012 expert's plan and
            # will be reconsidered at the next daily boundary.  Applying the
            # current ratio to the whole horizon would double-count the
            # controller and creates an artificial cash-flow crisis.
            if offset == 0:
                own_quantity = int(math.floor(max(0, current_quantity) * ratio))
            else:
                own_quantity = max(0, _int(schedule[offset].get(item, 0)))
            inventory, turn_revenue = _sell_lockstep(inventory, own_quantity, opponent_per_turn, item, obs)
            revenue += turn_revenue
            sold += own_quantity
            inventory = max(0, inventory - _town_demand(obs, item, step + offset))
        unsold = max(0, available - sold)
        residual_value = unsold * price_at(item, inventory, obs) * 0.75
        day = _int(obs.get("day", step // 24)) if isinstance(obs, dict) else step // 24
        terminal_factor = max(0.0, min(1.0, (day - 23.0) / 7.0))
        terminal_penalty = unsold * MARKET[item][0] * 0.30 * terminal_factor
        return revenue + residual_value - terminal_penalty

    def _score_ratio(self, obs, item, schedule, ratio, current_quantity):
        results = []
        for opponent_per_turn, weight in self._scenarios(obs, item):
            results.append((self._simulate(obs, item, schedule, ratio, opponent_per_turn, current_quantity), weight))
        total_weight = sum(weight for _, weight in results) or 1.0
        mean_score = sum(score * weight for score, weight in results) / total_weight
        worst_score = min((score for score, _ in results), default=mean_score)
        return mean_score - self.config["risk_weight"] * (mean_score - worst_score), mean_score, worst_score

    def _compute_ratios(self, obs, step, expert_name, current_quantities):
        ratios = {}
        schedule = self._future_schedule(expert_name, step)
        for item in PREMIUM_PRODUCTS:
            total = sum(int(row.get(item, 0)) for row in schedule)
            if total <= 0:
                ratios[item] = 1.0
                continue
            upper = _wave_upper_ratio(obs, item)
            candidates = sorted({0.0, min(0.25, upper), min(0.50, upper), min(0.75, upper), upper})
            candidates = [candidate for candidate in candidates if candidate >= 0.0]
            current_quantity = max(0, _int(current_quantities.get(item, 0)))
            scored = {
                ratio: self._score_ratio(obs, item, schedule, ratio, current_quantity)
                for ratio in candidates
            }
            baseline = self._score_ratio(obs, item, schedule, 1.0, current_quantity)[0]
            best = max(candidates, key=lambda ratio: (scored[ratio][0], ratio))
            if self.variant != "fixed_wave" and scored[best][0] - baseline < self.config["min_gain"]:
                best = 1.0
            ratios[item] = min(upper, best) if self.variant == "fixed_wave" else min(upper, best)
        return ratios

    def _cash_safe(self, obs, orders):
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        farm = _farm_for(obs, player)
        if _num(farm.get("money", 0)) < MIN_REWRITE_CASH:
            return False
        critical = {"HIRE", "BUY_LAND", "BUY_ANIMAL", "BUY_PRODUCT"}
        return not any(isinstance(order, list) and order and order[0] in critical for order in orders)

    def _slots(self, orders):
        result = defaultdict(list)
        for index, order in enumerate(orders):
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS:
                result[order[1]].append((index, max(0, _int(order[2]))))
        return result

    def apply(self, obs, base_action):
        action = _copy_action(base_action)
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        step = _int(obs.get("step", 0)) if isinstance(obs, dict) else 0
        state = self._state(player)
        if step == 0 or step <= state["last_step"]:
            self._states.pop(player, None)
            state = self._state(player)
        state["last_step"] = step
        orders = action["market"]
        slots = self._slots(orders)
        if not slots or not self._cash_safe(obs, orders):
            state["stats"]["guard_skips"] += 1
            return action
        expert = self._expert_name(obs)
        day = _int(obs.get("day", step // 24)) if isinstance(obs, dict) else step // 24
        hour = _int(obs.get("hour", step % 24)) if isinstance(obs, dict) else step % 24
        if self.config["use_mpc"] and (state["plan_day"] != day or hour == 0):
            current_quantities = {
                item: sum(quantity for _, quantity in item_slots)
                for item, item_slots in slots.items()
            }
            state["ratios"] = self._compute_ratios(obs, step, expert, current_quantities)
            state["plan_day"] = day
            state["stats"]["replans"] += 1
        elif not self.config["use_mpc"]:
            state["ratios"] = {item: _wave_upper_ratio(obs, item) for item in PREMIUM_PRODUCTS}
        chosen = {}
        for item, item_slots in slots.items():
            base_quantity = sum(quantity for _, quantity in item_slots)
            ratio = max(0.0, min(1.0, _num(state["ratios"].get(item, 1.0), 1.0)))
            chosen[item] = min(base_quantity, int(math.ceil(base_quantity * ratio)))
            state["stats"][f"base_{item}"] += base_quantity
            state["stats"][f"chosen_{item}"] += chosen[item]
            state["stats"][f"reduced_{item}"] += max(0, base_quantity - chosen[item])
        allocations = {}
        for item, item_slots in slots.items():
            remaining = chosen[item]
            for position, (index, original) in enumerate(item_slots):
                take = remaining if position == len(item_slots) - 1 else min(original, remaining)
                allocations[index] = max(0, int(take))
                remaining = max(0, remaining - allocations[index])
        rewritten = []
        for index, order in enumerate(orders):
            if not (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS):
                rewritten.append(list(order))
            elif allocations.get(index, 0) > 0:
                rewritten.append(["SELL", order[1], allocations[index]])
        action["market"] = rewritten[:MAX_MARKET_ORDERS]
        state["stats"]["market_actions"] += 1
        if action["market"] != orders:
            state["stats"]["changed_market_actions"] += 1
        state["history"].append({"step": step, "day": day, "expert": expert, "ratios": dict(state["ratios"]), "chosen": dict(chosen)})
        return action

    def diagnostics(self, player=0):
        state = self._state(player)
        result = dict(state["stats"])
        result.update({
            "variant": self.variant,
            "last_step": state["last_step"],
            "plan_day": state["plan_day"],
            "ratios": dict(state["ratios"]),
            "history": copy.deepcopy(state["history"]),
        })
        return result
