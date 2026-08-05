"""V015a shared-market collision overlay.

The overlay deliberately knows nothing about field movement.  It receives a
V012 action, preserves farmer/hands and non-premium market orders, and makes
small, stateful changes only to premium-product SELL orders.
"""

from collections import defaultdict, deque
import copy
import statistics


PREMIUM_PRODUCTS = ("MELON", "STRAWBERRY", "MILK", "WOOL")
SELLABLE_PREMIUM = set(PREMIUM_PRODUCTS)
MAX_MARKET_ORDERS = 10
TERMINAL_START_STEP = 696


def _copy_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or [])],
    }


def prepare_observation(obs, overlay):
    """Fill replay observations that omit step/hour without changing live obs."""
    copied = dict(obs) if isinstance(obs, dict) else {}
    player = int(copied.get("player", 0) or 0)
    state = overlay._state(player)
    if copied.get("step") is None:
        copied["step"] = max(0, state["last_step"] + 1)
    step = int(copied.get("step", 0) or 0)
    copied.setdefault("day", step // 24)
    copied.setdefault("hour", step % 24)
    return copied


class MarketCollisionOverlay:
    """Small per-player market memory with conservative one-unit deferrals."""

    def __init__(
        self,
        single_drop_ratio=0.20,
        two_step_drop_ratio=0.30,
        recent_median_ratio=0.70,
        inventory_jump_units=12,
        hold_steps=24,
        terminal_start=TERMINAL_START_STEP,
    ):
        self.single_drop_ratio = float(single_drop_ratio)
        self.two_step_drop_ratio = float(two_step_drop_ratio)
        self.recent_median_ratio = float(recent_median_ratio)
        self.inventory_jump_units = int(inventory_jump_units)
        self.hold_steps = int(hold_steps)
        self.terminal_start = int(terminal_start)
        self._states = {}

    def _new_state(self):
        return {
            "last_step": -1,
            "prices": {item: deque(maxlen=8) for item in PREMIUM_PRODUCTS},
            "inventories": {item: deque(maxlen=8) for item in PREMIUM_PRODUCTS},
            "congested_until": {item: -1 for item in PREMIUM_PRODUCTS},
            "pending": {item: 0 for item in PREMIUM_PRODUCTS},
            "stats": defaultdict(int),
        }

    def _state(self, player):
        player = int(player)
        if player not in self._states:
            self._states[player] = self._new_state()
        return self._states[player]

    def reset(self, player=None):
        if player is None:
            self._states.clear()
        else:
            self._states[int(player)] = self._new_state()

    def _market_values(self, obs, item):
        market = obs.get("market", {}) if isinstance(obs, dict) else {}
        prices = market.get("prices", {}) if isinstance(market, dict) else {}
        inventory = market.get("inventory", {}) if isinstance(market, dict) else {}
        try:
            price = float(prices.get(item, 0) or 0)
        except (TypeError, ValueError):
            price = 0.0
        try:
            supply = int(inventory.get(item, 0) or 0)
        except (TypeError, ValueError):
            supply = 0
        return price, supply

    def _observe_market(self, obs, step, state):
        congested = {}
        for item in PREMIUM_PRODUCTS:
            price, supply = self._market_values(obs, item)
            prices = state["prices"][item]
            supplies = state["inventories"][item]
            previous = prices[-1] if prices else None
            two_back = prices[-2] if len(prices) >= 2 else None
            previous_supply = supplies[-1] if supplies else None
            drop_one = (price / previous - 1.0) if previous and price else 0.0
            drop_two = (price / two_back - 1.0) if two_back and price else 0.0
            prior_window = list(prices)[-4:]
            median_price = statistics.median(prior_window) if prior_window else 0.0
            supply_jump = (
                previous_supply is not None
                and supply - previous_supply >= self.inventory_jump_units
            )
            price_shock = bool(
                (previous and drop_one <= -self.single_drop_ratio)
                or (two_back and drop_two <= -self.two_step_drop_ratio)
                or (median_price and price <= median_price * self.recent_median_ratio)
            )
            # A supply jump alone is not enough to block a sale.  It is a
            # corroborating signal, because town consumption and our own
            # sales can also change public inventory.
            shock = price_shock and (supply_jump or drop_one <= -self.single_drop_ratio or drop_two <= -self.two_step_drop_ratio)
            if shock:
                state["congested_until"][item] = max(
                    state["congested_until"][item], step + self.hold_steps
                )
                state["stats"]["price_shocks"] += 1
                state["stats"][f"price_shocks_{item}"] += 1
            elif state["congested_until"][item] >= step:
                recovery_reference = median_price or (previous or price)
                if recovery_reference and price >= recovery_reference * 0.85:
                    state["congested_until"][item] = step - 1
                    state["stats"]["recoveries"] += 1
                    state["stats"][f"recoveries_{item}"] += 1
            congested[item] = state["congested_until"][item] >= step
            prices.append(price)
            supplies.append(supply)
        return congested

    def _shed(self, obs, player):
        farms = obs.get("farms", []) if isinstance(obs, dict) else []
        private = obs.get("private", {}) if isinstance(obs, dict) else {}
        shed = private.get("shed", {}) if isinstance(private, dict) else {}
        if not isinstance(shed, dict):
            return {}
        return shed

    def _merge_or_append_terminal(self, orders, obs, player, state, step):
        if step < self.terminal_start:
            return orders
        shed = self._shed(obs, player)
        if not shed:
            return orders
        by_item = {}
        for index, order in enumerate(orders):
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "SELL"
                and order[1] in SELLABLE_PREMIUM
            ):
                by_item.setdefault(order[1], []).append(index)
        for item in PREMIUM_PRODUCTS:
            try:
                available = max(0, int(shed.get(item, 0) or 0))
            except (TypeError, ValueError):
                available = 0
            available = max(available, int(state["pending"][item]))
            if available <= 0:
                continue
            indices = by_item.get(item, [])
            if indices:
                first = indices[0]
                orders[first] = ["SELL", item, available]
                for duplicate in reversed(indices[1:]):
                    orders.pop(duplicate)
                state["pending"][item] = 0
                state["stats"]["terminal_flush_units"] += available
                state["stats"][f"terminal_flush_{item}"] += available
            elif len(orders) < MAX_MARKET_ORDERS:
                orders.append(["SELL", item, available])
                state["pending"][item] = 0
                state["stats"]["terminal_flush_units"] += available
                state["stats"][f"terminal_flush_{item}"] += available
        return orders[:MAX_MARKET_ORDERS]

    def apply(self, obs, base_action):
        action = _copy_action(base_action)
        player = int(obs.get("player", 0) or 0) if isinstance(obs, dict) else 0
        state = self._state(player)
        step = int(obs.get("step", state["last_step"] + 1) or 0) if isinstance(obs, dict) else state["last_step"] + 1
        if step == 0 or step <= state["last_step"]:
            self.reset(player)
            state = self._state(player)
        state["last_step"] = step
        congested = self._observe_market(obs, step, state)

        original_orders = action["market"]
        rewritten = []
        premium_slots = []
        delayed_this_turn = set()
        for order in original_orders:
            if not (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "SELL"
                and order[1] in SELLABLE_PREMIUM
            ):
                rewritten.append(list(order))
                continue
            item = order[1]
            try:
                quantity = max(0, int(order[2]))
            except (TypeError, ValueError):
                rewritten.append(list(order))
                continue
            if (
                step < self.terminal_start
                and congested[item]
                and quantity > 0
                and item not in delayed_this_turn
            ):
                quantity -= 1
                delayed_this_turn.add(item)
                state["pending"][item] = min(1, int(state["pending"][item]) + 1)
                state["stats"]["delayed_units"] += 1
                state["stats"][f"delayed_{item}"] += 1
            if quantity > 0:
                premium_slots.append(["SELL", item, quantity])
            # Keep a positional marker.  The marker is replaced below after
            # premium orders have been sorted; this preserves every
            # non-premium order's position and relative order.
            rewritten.append(None)

        # Release one held unit only when the base policy has a corresponding
        # sale slot.  This avoids inventing an order in the middle of a full
        # market queue; terminal cleanup is handled separately.
        released = []
        for order in premium_slots:
            item = order[1]
            if state["pending"][item] and not congested[item] and step < self.terminal_start:
                order[2] += int(state["pending"][item])
                released.append(item)
                state["pending"][item] = 0
                state["stats"]["released_units"] += 1
                state["stats"][f"released_{item}"] += 1

        # Only reorder premium SELL entries.  Every non-premium order remains
        # byte-for-byte in its original relative position.
        if any(congested.values()):
            premium_slots.sort(
                key=lambda order: (
                    int(congested.get(order[1], False)),
                    -self._market_values(obs, order[1])[0],
                    order[1],
                )
            )
        premium_index = 0
        merged = []
        for order in rewritten:
            if order is None:
                if premium_index < len(premium_slots):
                    merged.append(premium_slots[premium_index])
                    premium_index += 1
            else:
                merged.append(order)
        merged.extend(premium_slots[premium_index:])
        merged = self._merge_or_append_terminal(merged, obs, player, state, step)
        action["market"] = merged[:MAX_MARKET_ORDERS]
        if action["market"] != original_orders:
            state["stats"]["changed_market_actions"] += 1
        return action

    def diagnostics(self, player=0):
        state = self._state(player)
        result = dict(state["stats"])
        result.update({
            "pending_MELON": int(state["pending"]["MELON"]),
            "pending_STRAWBERRY": int(state["pending"]["STRAWBERRY"]),
            "pending_MILK": int(state["pending"]["MILK"]),
            "pending_WOOL": int(state["pending"]["WOOL"]),
            "last_step": int(state["last_step"]),
        })
        return result
