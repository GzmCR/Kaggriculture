"""V032-R2 strict interval market estimator.

This module is intentionally independent from the older V032-R1 rollout.
It estimates only the target-product revenue between one SELL event and the
next same-product SELL event.  It does not preload future production: the
caller supplies the observed control shed snapshot for each step and the
candidate keeps the delayed units only after the current event.

The market implementation mirrors kaggle-environments 1.32.6:

* market orders are truncated to ten slots;
* SELL/BUY units are quoted in player lockstep before either player commits;
* town consumption happens after the market;
* a SELL at price 1 does not add market inventory.
"""

from __future__ import annotations

import copy
import math


R2_MAX_ORDERS = 10
R2_SHED_CAPACITY = 100
R2_SAFE_BUFFER = 0.20
R2_MIN_GAIN = 10.0
R2_PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
R2_TOWN_PRODUCTS = tuple(item for item in R2_PRODUCTS if item != "FERTILIZER")
R2_SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
R2_ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
R2_LAND_PRICES = (1000, 2000, 4000)
R2_MARKET_PARAMS = {
    "WHEAT": {"base": 25, "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT": {"base": 35, "I0": 10000, "T": 450, "below_func": "log", "below_target": 0.20, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO": {"base": 60, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON": {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG": {"base": 50, "I0": 10000, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK": {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL": {"base": 200, "I0": 10000, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
R2_SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


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
    raise ValueError(name)


def r2_market_price(item, inventory, params=None):
    """Official Kaggriculture price function."""
    item = str(item).upper()
    p = (params or R2_MARKET_PARAMS)[item]
    inventory = _int(inventory)
    if inventory < p["I0"]:
        func, target = p["below_func"], p["below_target"]
        distance = p["I0"] - inventory
    else:
        func, target = p["above_func"], p["above_target"]
        distance = inventory - p["I0"]
    amplitude = target * p["base"] / _shape(func, p["T"])
    return max(1, int(round(p["base"] + amplitude * _shape(func, distance)
                            if inventory < p["I0"]
                            else p["base"] - amplitude * _shape(func, distance))))


def r2_clone_orders(orders):
    return [list(order) for order in (orders or [])
            if isinstance(order, (list, tuple))]


def r2_is_sell(order, item=None):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return False
    if str(order[0]).upper() != "SELL":
        return False
    if item is not None and str(order[1]).upper() != str(item).upper():
        return False
    return str(order[1]).upper() in R2_MARKET_PARAMS


def r2_sell_quantity(orders, item):
    return sum(max(0, _int(order[2])) for order in (orders or [])
               if r2_is_sell(order, item))


def r2_reorder_existing_orders(orders, inventory, prices=None):
    """Apply V27's existing-SELL impact ranking without adding/removing slots."""
    market = r2_clone_orders(orders)
    rows = []
    for index, order in enumerate(market):
        if not r2_is_sell(order):
            rows.append(None)
            continue
        item = str(order[1]).upper()
        quantity = max(0, _int(order[2]))
        current = _int((inventory or {}).get(item, 10000), 10000)
        quote = float((prices or {}).get(item, r2_market_price(item, current)))
        after = float(r2_market_price(item, current + quantity))
        rows.append((quantity * max(0.0, quote - after), -index, list(order)))
    sells = [row for row in rows if row is not None]
    if len(sells) < 2:
        return market
    sells.sort(reverse=True)
    iterator = iter(row[2] for row in sells)
    return [next(iterator) if row is not None else order
            for row, order in zip(rows, market)]


def r2_adjust_delay(orders, item, quantity):
    """Reduce the current SELL total and add the same amount to that item.

    The caller supplies the next event separately.  Zero-valued current rows
    are retained as empty lockstep slots, matching the engine parser.
    """
    item = str(item).upper()
    quantity = max(0, _int(quantity))
    market = r2_clone_orders(orders)
    total = r2_sell_quantity(market, item)
    if quantity <= 0 or total < quantity:
        return None
    remaining = quantity
    for order in market:
        if not r2_is_sell(order, item):
            continue
        take = min(remaining, max(0, _int(order[2])))
        order[2] = max(0, _int(order[2])) - take
        remaining -= take
        if remaining <= 0:
            break
    if remaining:
        return None
    return market


def r2_adjust_future(orders, item, quantity):
    item = str(item).upper()
    quantity = max(0, _int(quantity))
    market = r2_clone_orders(orders)
    if quantity <= 0:
        return market
    for order in market:
        if r2_is_sell(order, item):
            order[2] = max(0, _int(order[2])) + quantity
            return market
    return None


def r2_apply_town_consumption(inventory, shops, step, shop_interval=4,
                              center_interval=24):
    inventory = dict(inventory or {})
    if _int(step) % max(1, _int(shop_interval, 4)) == 0:
        for shop in list(shops or []):
            products = R2_SHOPS.get(str(shop), ())
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                inventory[item] = _int(inventory.get(item, 10000)) - multiplier
    if _int(step) % max(1, _int(center_interval, 24)) == 0:
        for item in R2_TOWN_PRODUCTS:
            inventory[item] = _int(inventory.get(item, 10000)) - 1
    return inventory


def _order_state(order):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        if isinstance(order, (list, tuple)) and order and str(order[0]).upper() in {"HIRE", "BUY_LAND"}:
            return {"op": str(order[0]).upper(), "remaining": 1, "item": ""}
        return None
    op = str(order[0]).upper()
    if op not in {"SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"}:
        return None
    quantity = _int(order[2])
    if quantity <= 0:
        return None
    return {"op": op, "item": str(order[1]).upper(), "remaining": quantity}


def _quote(state, inventory):
    if state is None or state.get("remaining", 0) <= 0:
        return None
    op, item = state["op"], state.get("item", "")
    if op == "SELL" and item in R2_MARKET_PARAMS:
        return op, item, r2_market_price(item, inventory.get(item, 10000))
    if op == "BUY_PRODUCT" and item in R2_MARKET_PARAMS:
        return op, item, r2_market_price(item, inventory.get(item, 10000) - 1)
    if op == "BUY_SEED":
        return op, item, R2_SEED_COST.get(item, 0)
    if op == "BUY_ANIMAL":
        return op, item, R2_ANIMAL_COST.get(item, 0)
    if op in {"HIRE", "BUY_LAND"}:
        return op, item, float(state.get("cost", 0.0))
    return None


def _commit(state, player, quote, inventory, sheds, money, target_item,
            target_revenue, shed_capacity):
    op, item, price = quote
    shed = sheds[player]
    if op == "SELL":
        if _int(shed.get(item, 0)) <= 0:
            return False, target_revenue
        shed[item] = _int(shed.get(item, 0)) - 1
        money[player] += float(price)
        if item == target_item:
            target_revenue[player] += float(price)
        if float(price) > 1:
            inventory[item] = _int(inventory.get(item, 10000)) + 1
        return True, target_revenue
    if op == "BUY_PRODUCT":
        if money[player] < float(price) or sum(max(0, _int(v)) for v in shed.values()) >= shed_capacity:
            return False, target_revenue
        money[player] -= float(price)
        shed[item] = _int(shed.get(item, 0)) + 1
        inventory[item] = _int(inventory.get(item, 10000)) - 1
        return True, target_revenue
    if op in {"BUY_SEED", "BUY_ANIMAL"}:
        if money[player] < float(price):
            return False, target_revenue
        if op == "BUY_ANIMAL" and sum(max(0, _int(v)) for v in shed.values()) >= shed_capacity:
            return False, target_revenue
        money[player] -= float(price)
        if op == "BUY_ANIMAL":
            shed[item] = _int(shed.get(item, 0)) + 1
        return True, target_revenue
    if op in {"HIRE", "BUY_LAND"}:
        if money[player] < float(price):
            return False, target_revenue
        money[player] -= float(price)
        return True, target_revenue
    return False, target_revenue


def r2_process_market(orders_by_player, inventory, sheds, money, target_item,
                      max_orders=R2_MAX_ORDERS, shed_capacity=R2_SHED_CAPACITY,
                      order_context=None):
    """Process one turn using environment-style lockstep semantics."""
    inventory = dict(inventory or {})
    sheds = [dict((sheds or [{}, {}])[i] or {}) for i in (0, 1)]
    money = [float((money or [0, 0])[0]), float((money or [0, 0])[1])]
    target_revenue = [0.0, 0.0]
    queues = [r2_clone_orders((orders_by_player or [[], []])[i])[:max(1, _int(max_orders, 10))] for i in (0, 1)]
    states = [[_order_state(order) for order in queue] for queue in queues]
    failed = [0, 0]
    failed_orders = [[], []]
    executed = [0, 0]
    max_len = max((len(queue) for queue in states), default=0)
    for order_index in range(max_len):
        current = [states[player][order_index] if order_index < len(states[player]) else None for player in (0, 1)]
        for player in (0, 1):
            state = current[player]
            if state is None or state.get("op") not in {"HIRE", "BUY_LAND"}:
                continue
            context = (order_context or [{}, {}])[player] or {}
            costs = context.get(state["op"], []) or []
            state["cost"] = float(costs[order_index]) if order_index < len(costs) else 0.0
        while True:
            quoted = [None, None]
            for player in (0, 1):
                if current[player] is not None:
                    quoted[player] = _quote(current[player], inventory)
                    if quoted[player] is None:
                        failed[player] += 1
                        failed_orders[player].append({"index": order_index, "reason": "invalid", "op": current[player].get("op")})
                        current[player] = None
            if quoted[0] is None and quoted[1] is None:
                break
            committed_any = False
            for player in (0, 1):
                if quoted[player] is None:
                    continue
                ok, target_revenue = _commit(
                    current[player], player, quoted[player], inventory, sheds,
                    money, str(target_item).upper(), target_revenue, shed_capacity,
                )
                if ok:
                    current[player]["remaining"] -= 1
                    executed[player] += 1
                    committed_any = True
                else:
                    failed[player] += 1
                    failed_orders[player].append({"index": order_index, "reason": "commit", "op": current[player].get("op"), "item": current[player].get("item")})
                    current[player] = None
            if not committed_any:
                break
    return {
        "inventory": inventory,
        "sheds": sheds,
        "money": money,
        "target_revenue": target_revenue,
        "executed": executed,
        "failed": failed,
        "failed_orders": failed_orders,
        "truncated_orders": [len(r2_clone_orders(x)) > max(1, _int(max_orders, 10)) for x in (orders_by_player or [[], []])],
    }


def r2_storage_usage(snapshot):
    private = (snapshot or {}).get("private", {}) or {}
    used = sum(max(0, _int(v)) for v in ((private.get("shed", {}) or {}).values()))
    for inventory in private.get("inventories", []) or []:
        if isinstance(inventory, dict):
            used += sum(max(0, _int(v)) for v in inventory.values())
    return used


def r2_simulate_interval(start_inventory, start_money, sheds_by_step,
                         orders_by_step, opponent_orders_by_step, start_step,
                         end_step, target_item, shops=None, shop_interval=4,
                         center_interval=24, max_orders=10,
                         shed_capacity=R2_SHED_CAPACITY, extra_player=None,
                         extra_units=0, order_context_by_step=None):
    """Simulate a target-product interval without inventing future stock.

    ``sheds_by_step`` contains the observed control private shed at the start
    of each step.  Delayed units are added only after ``start_step`` and are
    released at ``end_step``.  This is deliberately conservative: all other
    private inventory remains on the observed control trajectory.
    """
    inventory = dict(start_inventory or {})
    money = [float(start_money[0]), float(start_money[1])]
    total_target = [0.0, 0.0]
    total_failed = [0, 0]
    total_executed = [0, 0]
    failed_orders = [[], []]
    money_trace = {}
    target_trace = {}
    for step in range(_int(start_step), _int(end_step) + 1):
        step_sheds = []
        for player in (0, 1):
            base = dict((sheds_by_step[player].get(step, {}) or {}))
            if extra_player is not None and player == int(extra_player) and step > int(start_step) and step <= int(end_step):
                item = str(target_item).upper()
                base[item] = _int(base.get(item, 0)) + max(0, _int(extra_units))
            step_sheds.append(base)
        result = r2_process_market(
            [orders_by_step.get(step, []), opponent_orders_by_step.get(step, [])],
            inventory, step_sheds, money, target_item, max_orders, shed_capacity,
            (order_context_by_step or {}).get(step, [{}, {}]),
        )
        inventory, money = result["inventory"], result["money"]
        total_target = [total_target[i] + result["target_revenue"][i] for i in (0, 1)]
        total_failed = [total_failed[i] + result["failed"][i] for i in (0, 1)]
        total_executed = [total_executed[i] + result["executed"][i] for i in (0, 1)]
        for player in (0, 1):
            failed_orders[player].extend(result["failed_orders"][player])
        target_trace[step] = list(result["target_revenue"])
        inventory = r2_apply_town_consumption(inventory, shops, step, shop_interval, center_interval)
        money_trace[step] = list(money)
    return {
        "money": money,
        "inventory": inventory,
        "target_revenue": total_target,
        "target_trace": target_trace,
        "money_trace": money_trace,
        "failed": total_failed,
        "executed": total_executed,
        "failed_orders": failed_orders,
    }


def r2_mandatory_costs(actions_by_step, obs_by_step, start_step, end_step, player=0):
    """Estimate deterministic non-SELL cash requirements in an interval."""
    costs = {}
    for step in range(_int(start_step) + 1, _int(end_step) + 1):
        obs = obs_by_step.get(step, {}) or {}
        market_prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
        farms = obs.get("farms", []) or []
        farm = farms[_int(player)] if len(farms) > _int(player) else (farms[0] if farms else {})
        orders = actions_by_step.get(step, {}) or {}
        total = 0.0
        for order in orders.get("market", []) or []:
            if not isinstance(order, (list, tuple)) or not order:
                continue
            op = str(order[0]).upper()
            if op == "HIRE":
                n = _int(farm.get("hires_today", 0))
                a, b = 0, 1
                for _ in range(n + 1):
                    a, b = b, a + b
                total += float(max(1, a))
            elif op == "BUY_LAND":
                unlocked = len(farm.get("unlocked_quadrants", []) or [])
                index = max(0, min(len(R2_LAND_PRICES) - 1, unlocked - 1))
                total += float(R2_LAND_PRICES[index])
            elif op == "BUY_SEED" and len(order) >= 3:
                total += float(R2_SEED_COST.get(str(order[1]).upper(), 0) * max(0, _int(order[2])))
            elif op == "BUY_ANIMAL" and len(order) >= 3:
                total += float(R2_ANIMAL_COST.get(str(order[1]).upper(), 0) * max(0, _int(order[2])))
            elif op == "BUY_PRODUCT" and len(order) >= 3:
                price = float(market_prices.get(str(order[1]).upper(), 0) or 0)
                total += price * max(0, _int(order[2]))
        costs[step] = total
    return costs


def r2_safety_gate(control_result, candidate_result, candidate_obs_by_step,
                   target_item, transfer, start_step, end_step,
                   capacity=R2_SHED_CAPACITY, mandatory_costs=None, player=0,
                   control_obs_by_step=None, immediate_cash_delta=0.0):
    """Return (safe, reasons) for a delay candidate."""
    reasons = []
    item = str(target_item).upper()
    transfer = max(0, _int(transfer))
    if transfer <= 0:
        reasons.append("zero_transfer")
    player = _int(player)
    other = 1 - player
    if candidate_result["failed"][player] > control_result["failed"][player]:
        reasons.append("new_own_market_failure")
    if candidate_result["failed"][other] > control_result["failed"][other]:
        reasons.append("new_opponent_market_failure")
    # The shadow simulator may report baseline failures for sells whose
    # control action depends on a same-turn DROP/harvest.  Only a *new*
    # candidate failure is unsafe; baseline failures are not attributed to
    # the timing transfer.
    max_usage = 0
    for step in range(_int(start_step) + 1, _int(end_step) + 1):
        usage = r2_storage_usage(candidate_obs_by_step.get(step, {})) + transfer
        max_usage = max(max_usage, usage)
        if usage > int(capacity):
            reasons.append("shed_capacity")
            break
    if control_obs_by_step:
        costs = mandatory_costs or {}
        for step in range(_int(start_step), _int(end_step)):
            obs = control_obs_by_step.get(step, {}) or {}
            farms = obs.get("farms", []) or []
            farm = farms[player] if len(farms) > player else {}
            control_cash = float(farm.get("money", 0) or 0)
            required = float(costs.get(step, 0.0) or 0.0)
            if control_cash + float(immediate_cash_delta) < required * (1.0 + R2_SAFE_BUFFER):
                reasons.append("cash_reserve")
                break
    elif control_result["money_trace"]:
        costs = mandatory_costs or {}
        for step, money in control_result["money_trace"].items():
            if step < int(start_step) or step >= int(end_step):
                continue
            remaining = sum(float(value) for target, value in costs.items() if int(target) > int(step) and int(target) <= int(end_step))
            # The candidate result contains the immediate cash effect of the
            # delayed sale.  Require a 20% reserve over all known costs.
            candidate_money = candidate_result["money_trace"].get(step, [0.0, 0.0])[_int(player)]
            if candidate_money < remaining * (1.0 + R2_SAFE_BUFFER):
                reasons.append("cash_reserve")
                break
    if candidate_result.get("failed", [0, 0])[player] < 0:
        reasons.append("invalid_failure_count")
    return not reasons, reasons


def r2_local_prediction(control_result, candidate_result, player):
    """Compute target-only own/opponent revenue and local margin deltas."""
    player = int(player)
    other = 1 - player
    own_delta = float(candidate_result["target_revenue"][player] - control_result["target_revenue"][player])
    opp_delta = float(candidate_result["target_revenue"][other] - control_result["target_revenue"][other])
    return {
        "own_delta": own_delta,
        "opponent_delta": opp_delta,
        "predicted_local_margin_delta": own_delta - opp_delta,
    }
