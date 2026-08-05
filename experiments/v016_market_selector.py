"""V016 daily market-expert value selector.

The selector is deliberately a market-only layer.  It consumes the complete
market schedules already embedded in V012 and never plans field actions.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict


PREMIUM_PRODUCTS = ("MELON", "STRAWBERRY", "MILK", "WOOL")
MAX_MARKET_ORDERS = 10
HORIZON = 24

DEFAULT_VARIANTS = {
    "value_only": {
        "revenue_weight": 1.00,
        "collision_weight": 0.00,
        "stay_bonus": 0.50,
        "liquidity_weight": 1.00,
        "unsold_weight": 0.40,
    },
    "collision_hedged": {
        "revenue_weight": 1.00,
        "collision_weight": 1.00,
        "stay_bonus": 0.50,
        "liquidity_weight": 1.00,
        "unsold_weight": 0.40,
    },
    "aggressive_value": {
        "revenue_weight": 1.25,
        "collision_weight": 0.35,
        "stay_bonus": 0.10,
        "liquidity_weight": 0.75,
        "unsold_weight": 0.20,
    },
}

GLUT_WEIGHT = {
    "MELON": 3.6,
    "WOOL": 3.2,
    "STRAWBERRY": 1.8,
    "MILK": 1.8,
}

LAND_COST = {"NE": 1000.0, "SW": 2000.0, "SE": 4000.0}
ANIMAL_COST = {"COW": 400.0, "SHEEP": 500.0, "GOOSE": 250.0}
PRODUCT_BUY_COST = {
    "WHEAT": 10.0,
    "CARROT": 20.0,
    "TOMATO": 50.0,
    "STRAWBERRY": 100.0,
    "MELON": 80.0,
    "FERTILIZER": 100.0,
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


def _copy_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or [])],
    }


def _fib(n):
    a, b = 0, 1
    for _ in range(max(0, int(n))):
        a, b = b, a + b
    return max(1, a)


def _market(obs):
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    return market if isinstance(market, dict) else {}


def _prices(obs):
    prices = _market(obs).get("prices", {})
    return prices if isinstance(prices, dict) else {}


def _own_farm(obs):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    if isinstance(farms, list) and 0 <= player < len(farms):
        return farms[player] if isinstance(farms[player], dict) else {}
    return {}


def _opponent_farm(obs):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    other = 1 - player
    if isinstance(farms, list) and 0 <= other < len(farms):
        return farms[other] if isinstance(farms[other], dict) else {}
    return {}


def fallback_pipeline(farm):
    """Small observation-only supply estimate used outside the V012 module."""
    exposure = {name: 0.0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")}
    animals = 0.0
    unfed = 0.0
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            ready = max(0.0, _num(tile.get("yield_units")))
            crop = str(tile.get("crop", "")).upper()
            if tile.get("kind") == "PLANT" and crop in exposure:
                exposure[crop] += 1.0 + 2.0 * ready
            animal = str(tile.get("animal", "")).upper()
            product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(animal)
            if product:
                animals += 1.0
                unfed += float(not tile.get("fed_today", False))
                cadence = {"EGG": 1.0, "MILK": 0.5, "WOOL": 1.0 / 3.0}[product]
                exposure[product] += cadence + 2.0 * ready
    exposure["WHEAT"] += animals + 0.5 * unfed
    exposure["ANIMALS"] = animals
    exposure["UNFED"] = unfed
    return exposure


class MarketValueSelector:
    """Choose one complete V012 market expert at each day boundary."""

    def __init__(self, variant="collision_hedged", runtime=None, pipeline_fn=None):
        if variant not in DEFAULT_VARIANTS:
            raise ValueError(f"unknown V016 variant: {variant}")
        self.variant = variant
        self.config = dict(DEFAULT_VARIANTS[variant])
        self.runtime = runtime or {}
        self.pipeline_fn = pipeline_fn or fallback_pipeline
        self._states = {}

    def _new_state(self):
        return {
            "last_step": -1,
            "selected_day": None,
            "selected_expert": None,
            "selection_count": 0,
            "scores": {},
            "history": [],
            "stats": Counter(),
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
            self._states[_int(player)] = self._new_state()

    def _experts(self):
        experts = self.runtime.get("experts", {}) if isinstance(self.runtime, dict) else {}
        return experts if isinstance(experts, dict) else {}

    def _default_expert(self, obs):
        seat = str(_int(obs.get("player", 0)))
        board = self.runtime.get("board_by_seat", {}) if isinstance(self.runtime, dict) else {}
        if isinstance(board, dict) and board.get(seat) in self._experts():
            return board[seat]
        names = sorted(self._experts())
        return names[0] if names else None

    def _orders_at(self, expert, step):
        actions = expert.get("actions", []) if isinstance(expert, dict) else []
        if not actions:
            return []
        action = actions[min(max(0, int(step)), len(actions) - 1)] or {}
        orders = action.get("market", []) if isinstance(action, dict) else []
        return [list(order) for order in (orders or []) if isinstance(order, list) and order]

    def _current_prices(self, obs):
        prices = _prices(obs)
        return {item: max(1.0, _num(prices.get(item, 1), 1)) for item in prices}

    def _unit_inventory(self, obs, item):
        private = obs.get("private", {}) if isinstance(obs, dict) else {}
        total = 0.0
        if isinstance(private, dict):
            shed = private.get("shed", {})
            if isinstance(shed, dict):
                total += max(0.0, _num(shed.get(item)))
            inventories = private.get("inventories", [])
            for inventory in inventories if isinstance(inventories, list) else []:
                if isinstance(inventory, dict):
                    total += max(0.0, _num(inventory.get(item)))
        return total

    def _buy_cost(self, order, obs, hires_so_far, missing_land):
        if len(order) < 2:
            return 0.0, hires_so_far, missing_land
        command = str(order[0])
        item = str(order[1]).upper()
        quantity = max(0, _int(order[2], 0)) if len(order) >= 3 else 1
        if command == "BUY_PRODUCT":
            unit = PRODUCT_BUY_COST.get(item, max(1.0, _num(_prices(obs).get(item, 10), 10)))
            return unit * quantity, hires_so_far, missing_land
        if command == "BUY_ANIMAL":
            return ANIMAL_COST.get(item, 500.0) * quantity, hires_so_far, missing_land
        if command == "HIRE":
            farm = _own_farm(obs)
            current = _int(farm.get("hires_today", 0)) + hires_so_far
            return float(_fib(current + 1)), hires_so_far + 1, missing_land
        if command == "BUY_LAND":
            # A complete expert records the order without the quadrant name;
            # charge the cheapest currently locked quadrant first.
            if missing_land:
                quadrant = missing_land.pop(0)
                return LAND_COST[quadrant], hires_so_far, missing_land
            return 0.0, hires_so_far, missing_land
        return 0.0, hires_so_far, missing_land

    def _feed_need(self, obs, horizon):
        own = _own_farm(obs)
        pipeline = self.pipeline_fn(own) or {}
        animals = max(0.0, _num(pipeline.get("ANIMALS")))
        # The horizon is one full day.  Keep a small two-unit reserve because
        # the fixed route can feed at a day boundary before buying again.
        return animals * max(1.0, math.ceil(horizon / 24.0)) + 2.0

    def _forecast(self, obs, expert_name, step):
        expert = self._experts().get(expert_name, {})
        prices = self._current_prices(obs)
        revenue = 0.0
        costs = 0.0
        premium_sells = defaultdict(float)
        wheat_buys = 0.0
        order_count = 0
        hires = 0
        farm = _own_farm(obs)
        unlocked = set(farm.get("unlocked_quadrants", []) or [])
        missing_land = [name for name in ("NE", "SW", "SE") if name not in unlocked]
        for offset in range(HORIZON):
            for order in self._orders_at(expert, step + offset)[:MAX_MARKET_ORDERS]:
                order_count += 1
                command = str(order[0])
                item = str(order[1]).upper() if len(order) >= 2 else ""
                quantity = max(0, _int(order[2], 0)) if len(order) >= 3 else 1
                if command == "SELL":
                    price = prices.get(item, max(1.0, _num(_prices(obs).get(item, 1), 1)))
                    revenue += price * quantity
                    if item in PREMIUM_PRODUCTS:
                        premium_sells[item] += quantity
                elif command in {"BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND"}:
                    cost, hires, missing_land = self._buy_cost(order, obs, hires, missing_land)
                    costs += cost
                    if command == "BUY_PRODUCT" and item == "WHEAT":
                        wheat_buys += quantity
        available_wheat = self._unit_inventory(obs, "WHEAT")
        feed_need = self._feed_need(obs, HORIZON)
        feed_shortfall = max(0.0, feed_need - available_wheat - wheat_buys)
        collision_raw = 0.0
        opponent = _opponent_farm(obs)
        pipeline = self.pipeline_fn(opponent) or {}
        for item, quantity in premium_sells.items():
            collision_raw += (
                quantity
                * max(0.0, _num(pipeline.get(item)))
                * prices.get(item, 1.0)
                * GLUT_WEIGHT[item]
            )
        # An expert that sells too little of current inventory leaves value
        # stranded late in the season.  Early in the game this term is nearly
        # zero, so it cannot force premature liquidation.
        day = _int(obs.get("day", step // 24))
        late_factor = max(0.0, min(1.0, (day - 18.0) / 12.0))
        unsold_value = 0.0
        for item in PREMIUM_PRODUCTS:
            unsold = max(0.0, self._unit_inventory(obs, item) - premium_sells[item])
            unsold_value += unsold * prices.get(item, 1.0) * late_factor
        cash = max(0.0, _num(farm.get("money")))
        reserve = max(500.0, feed_need * prices.get("WHEAT", 10.0))
        post_cash = cash + revenue - costs
        cash_risk = max(0.0, reserve - post_cash)
        return {
            "revenue": revenue,
            "costs": costs,
            "premium_sells": dict(premium_sells),
            "wheat_buys": wheat_buys,
            "feed_need": feed_need,
            "feed_shortfall": feed_shortfall,
            "collision_raw": collision_raw,
            "unsold_value": unsold_value,
            "cash_risk": cash_risk,
            "post_cash": post_cash,
            "order_count": order_count,
        }

    def score_expert(self, obs, expert_name, step):
        forecast = self._forecast(obs, expert_name, step)
        config = self.config
        score = (
            config["revenue_weight"] * forecast["revenue"]
            - forecast["costs"]
            - config["collision_weight"] * forecast["collision_raw"] / 100.0
            - config["liquidity_weight"] * forecast["feed_shortfall"] * max(10.0, _num(_prices(obs).get("WHEAT", 10), 10))
            - config["liquidity_weight"] * forecast["cash_risk"]
            - config["unsold_weight"] * forecast["unsold_value"]
        )
        # Hard infeasibility is penalized, but not made impossible: terminal
        # sale orders and the environment's silent no-op semantics make a
        # soft penalty more robust than dropping an expert entirely.
        if forecast["post_cash"] < -1.0:
            score -= 100000.0 + abs(forecast["post_cash"])
        forecast["score"] = score
        return forecast

    def choose(self, obs, step=None):
        step = _int(obs.get("step", 0) if step is None else step)
        player = _int(obs.get("player", 0))
        state = self._state(player)
        if step == 0 or step <= state["last_step"]:
            self.reset(player)
            state = self._state(player)
        state["last_step"] = step
        day = _int(obs.get("day", step // 24))
        hour = _int(obs.get("hour", step % 24))
        experts = self._experts()
        if not experts:
            return None
        if state["selected_expert"] is not None and hour != 0 and state["selected_day"] == day:
            return state["selected_expert"]
        if state["selected_expert"] is not None and state["selected_day"] == day and hour != 0:
            return state["selected_expert"]
        scores = {}
        for name in sorted(experts):
            scores[name] = self.score_expert(obs, name, step)
            if name == state["selected_expert"]:
                scores[name]["score"] += self.config["stay_bonus"]
        best = max(sorted(scores), key=lambda name: (scores[name]["score"], name))
        state["selected_expert"] = best
        state["selected_day"] = day
        state["selection_count"] += 1
        state["scores"] = scores
        state["history"].append({"step": step, "day": day, "expert": best, "scores": {name: data["score"] for name, data in scores.items()}})
        state["stats"][f"selected_{best}"] += 1
        return best

    def apply(self, obs, base_action):
        action = _copy_action(base_action)
        step = _int(obs.get("step", 0)) if isinstance(obs, dict) else 0
        expert_name = self.choose(obs, step)
        if not expert_name:
            return action
        expert = self._experts().get(expert_name, {})
        orders = self._orders_at(expert, step)
        action["market"] = orders[:MAX_MARKET_ORDERS]
        state = self._state(_int(obs.get("player", 0)))
        state["stats"]["market_replacements"] += 1
        return action

    def diagnostics(self, player=0):
        state = self._state(player)
        result = dict(state["stats"])
        result.update({
            "variant": self.variant,
            "last_step": state["last_step"],
            "selected_day": state["selected_day"],
            "selected_expert": state["selected_expert"],
            "selection_count": state["selection_count"],
            "score_snapshot": {name: data.get("score", 0.0) for name, data in state["scores"].items()},
            "selection_history": copy.deepcopy(state["history"]),
        })
        return result

