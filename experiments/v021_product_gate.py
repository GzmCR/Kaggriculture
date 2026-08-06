"""V021 per-product safety gates built on the V020/V012 market overlay.

The module is intentionally small: V020 supplies the market quote, town
demand, lockstep rollout, and public-supply helpers; V021 only constrains
when and how much that rollout may change an existing premium SELL order.
The build script concatenates this module after V020 so generated agents are
self-contained.
"""

from __future__ import annotations

try:
    from v020_value_aware_market import (
        BALANCED_START,
        HORIZON,
        MARKET,
        MAX_MARKET_ORDERS,
        PREMIUM_PRODUCTS,
        ValueAwareMarketController,
        _copy_action,
        _expected_future_orders,
        _int,
        _num,
        _market_for,
        _opponent_supply_profile,
    )
except ImportError:
    # The self-contained build has the V020 definitions immediately before
    # this code, so the names above already exist in that namespace.
    pass


V021_VARIANTS = {
    "safety_patch": {
        "active_products": ("MILK", "WOOL"),
        "milk_drop": 0.06,
        "milk_two_turn_drop": 0.10,
        "wool_drop": 0.06,
        "wool_two_turn_drop": 0.10,
        "min_gain": 40.0,
        "cash_guard": False,
        "extreme_other_products": False,
    },
    "product_gate": {
        "active_products": ("MILK", "WOOL"),
        "milk_drop": 0.08,
        "milk_two_turn_drop": 0.12,
        "wool_drop": 0.10,
        "wool_two_turn_drop": 0.15,
        "min_gain": 40.0,
        "cash_guard": False,
        "extreme_other_products": False,
    },
    "win_guard": {
        "active_products": ("MILK", "WOOL", "STRAWBERRY", "MELON"),
        "milk_drop": 0.08,
        "milk_two_turn_drop": 0.12,
        "wool_drop": 0.10,
        "wool_two_turn_drop": 0.15,
        "min_gain": 40.0,
        "cash_guard": True,
        "extreme_other_products": True,
    },
}


class V021ProductGateController(ValueAwareMarketController):
    """Bound V020's optimizer to one-unit, product-specific interventions."""

    def __init__(self, variant="product_gate", runtime=None):
        if variant not in V021_VARIANTS:
            raise ValueError(f"unknown V021 variant: {variant}")
        super().__init__(variant="conservative", runtime=runtime)
        self.variant = variant
        self.config.update(V021_VARIANTS[variant])
        self.config["collision_weight"] = 0.10
        self.config["terminal_penalty"] = 0.20

    def _new_state(self):
        state = super()._new_state()
        state.update({
            "pending_units": {item: 0 for item in PREMIUM_PRODUCTS},
            "defer_budget": {item: 1 for item in PREMIUM_PRODUCTS},
            "shock_cooldown": {item: False for item in PREMIUM_PRODUCTS},
            "recovery_count": {item: 0 for item in PREMIUM_PRODUCTS},
            "last_signal": {},
        })
        return state

    def _policy(self, item):
        if item == "MILK":
            return {
                "enabled": True,
                "drop": self.config["milk_drop"],
                "two_turn_drop": self.config["milk_two_turn_drop"],
                "min_pipeline": 3.0,
                "min_gain": self.config["min_gain"],
            }
        if item == "WOOL":
            return {
                "enabled": True,
                "drop": self.config["wool_drop"],
                "two_turn_drop": self.config["wool_two_turn_drop"],
                "min_pipeline": 2.0,
                "min_gain": self.config["min_gain"],
            }
        if self.config.get("extreme_other_products"):
            return {
                "enabled": True,
                "drop": 0.15,
                "two_turn_drop": 0.20,
                "min_pipeline": 8.0,
                "min_gain": self.config["min_gain"],
            }
        return {
            "enabled": False,
            "drop": 1.0,
            "two_turn_drop": 1.0,
            "min_pipeline": 10**9,
            "min_gain": self.config["min_gain"],
        }

    @staticmethod
    def _median(values):
        values = sorted(float(value) for value in values if value is not None)
        if not values:
            return None
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    def _market_signal(self, obs, item, state, step):
        market = _market_for(obs)
        prices = market.get("prices", {}) if isinstance(market, dict) else {}
        supplies = market.get("inventory", {}) if isinstance(market, dict) else {}
        base_price = _num(MARKET[item][0], 1)
        price = _num(prices.get(item, base_price), base_price)
        supply = _int(supplies.get(item, 10000), 10000)
        history = state["prices"][item]
        supply_history = state["supplies"][item]
        previous = history[-1] if history else None
        previous_two = history[-2] if len(history) >= 2 else None
        previous_supply = supply_history[-1] if supply_history else None
        drop = price / previous - 1.0 if previous else 0.0
        two_turn_drop = price / previous_two - 1.0 if previous_two else 0.0
        supply_jump = supply - previous_supply if previous_supply is not None else 0
        held, future = _opponent_supply_profile(obs, item, HORIZON)
        pipeline = held + future
        day = _int(obs.get("day", step // 24), step // 24)
        policy = self._policy(item)
        in_wave_window = 10 <= day <= 26 and step < BALANCED_START
        severe_supply = supply_jump >= 6 and price <= base_price * 0.95 and held >= policy["min_pipeline"]
        shock = bool(
            policy["enabled"]
            and in_wave_window
            and pipeline >= policy["min_pipeline"]
            and (
                drop <= -policy["drop"]
                or two_turn_drop <= -policy["two_turn_drop"]
                or severe_supply
            )
        )
        recent_median = self._median(history[-4:])
        recovered = bool(
            previous is not None
            and recent_median is not None
            and price >= recent_median * 0.95
        )
        history.append(price)
        supply_history.append(supply)
        del history[:-8]
        del supply_history[:-8]
        return {
            "price": price,
            "previous": previous,
            "drop": drop,
            "two_turn_drop": two_turn_drop,
            "supply_jump": supply_jump,
            "held": held,
            "future": future,
            "pipeline": pipeline,
            "shock": shock,
            "recovered": recovered,
        }

    def _cash_mode(self, obs):
        if not self.config.get("cash_guard"):
            return "normal", 0.0
        farms = obs.get("farms", []) if isinstance(obs, dict) else []
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        other = 1 - player
        if not isinstance(farms, list) or player >= len(farms) or other >= len(farms):
            return "normal", 0.0
        own = _num((farms[player] or {}).get("money", 0), 0)
        opponent = _num((farms[other] or {}).get("money", 0), 0)
        gap = own - opponent
        if gap < -1000.0:
            return "blocked", gap
        if gap <= 1000.0:
            return "one_unit", gap
        return "normal", gap

    def _update_recovery(self, state, item, signal):
        if signal["shock"]:
            state["recovery_count"][item] = 0
            return
        if signal["recovered"] or not signal["shock"]:
            state["recovery_count"][item] += 1
        if state["recovery_count"][item] >= 2:
            state["shock_cooldown"][item] = False
            state["defer_budget"][item] = 1
            state["pending_units"][item] = 0

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

        # Preserve V012's terminal liquidation exactly.
        if step >= BALANCED_START:
            state["stats"]["terminal_passthrough"] += 1
            return action

        signals = {
            item: self._market_signal(obs, item, state, step)
            for item in PREMIUM_PRODUCTS
        }
        for item, signal in signals.items():
            self._update_recovery(state, item, signal)
            state["last_signal"][item] = dict(signal)

        original_orders = [list(order) for order in action["market"]]
        slots = self._item_slots(original_orders)
        active_products = set(self.config.get("active_products", ()))
        cash_mode, cash_gap = self._cash_mode(obs)
        chosen = {}

        for item, item_slots in slots.items():
            base_quantity = sum(quantity for _, quantity in item_slots)
            chosen[item] = base_quantity
            if item not in active_products:
                continue
            signal = signals[item]
            state_item = state["items"].setdefault(item, {})
            if not signal["shock"] or state["shock_cooldown"][item]:
                continue
            if cash_mode == "blocked":
                state["stats"][f"cash_blocked_{item}"] += 1
                continue
            if state["defer_budget"][item] <= 0 or base_quantity <= 0:
                state["stats"][f"budget_blocked_{item}"] += 1
                continue

            future = _expected_future_orders(self.runtime, obs, step, HORIZON)
            future_schedule = [max(0, _int(row.get(item, 0))) for row in future]
            baseline = self._score(obs, item, step, base_quantity, future_schedule)
            reduced_quantity = max(0, base_quantity - 1)
            reduced = self._score(obs, item, step, reduced_quantity, future_schedule)
            gain = reduced["score"] - baseline["score"]
            required_gain = self._policy(item)["min_gain"]
            if cash_mode == "one_unit":
                required_gain *= 1.25
            if gain < required_gain:
                state["stats"][f"score_rejected_{item}"] += 1
                continue

            chosen[item] = reduced_quantity
            state["pending_units"][item] = 1
            state["defer_budget"][item] = 0
            state["shock_cooldown"][item] = True
            state["recovery_count"][item] = 0
            state["stats"][f"deferred_{item}"] += 1
            state["stats"][f"deferred_units_{item}"] += 1
            state_item.update({
                "step": step,
                "base": base_quantity,
                "chosen": reduced_quantity,
                "gain": round(gain, 4),
                "cash_mode": cash_mode,
                "cash_gap": round(cash_gap, 2),
                "pipeline": round(signal["pipeline"], 4),
            })

        allocations = {}
        for item, item_slots in slots.items():
            remaining = max(0, _int(chosen.get(item, sum(quantity for _, quantity in item_slots))))
            for position, (index, original_quantity) in enumerate(item_slots):
                quantity = remaining if position == len(item_slots) - 1 else min(original_quantity, remaining)
                allocations[index] = max(0, _int(quantity))
                remaining = max(0, remaining - allocations[index])

        rewritten = []
        for index, order in enumerate(original_orders):
            if not (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "SELL"
                and order[1] in active_products
            ):
                rewritten.append(list(order))
                continue
            quantity = allocations.get(index, max(0, _int(order[2])))
            if quantity > 0:
                rewritten.append(["SELL", order[1], quantity])
        action["market"] = rewritten[:MAX_MARKET_ORDERS]
        if action["market"] != original_orders:
            state["stats"]["changed_market_actions"] += 1
        state["stats"]["market_actions"] += 1
        return action

    def diagnostics(self, player=0):
        result = super().diagnostics(player)
        state = self._state(player)
        result.update({
            "v021_variant": self.variant,
            "pending_units": dict(state["pending_units"]),
            "defer_budget": dict(state["defer_budget"]),
            "shock_cooldown": dict(state["shock_cooldown"]),
            "recovery_count": dict(state["recovery_count"]),
        })
        return result
