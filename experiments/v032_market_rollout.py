"""Small, dependency-free Kaggriculture market simulator used by V032-R1.

The implementation mirrors the public 1.32.6 environment market semantics:
orders are truncated to ten slots, SELL/BUY units are quoted in lockstep, the
quote is taken before either player commits that unit, and town consumption is
applied after the market.  The module is intentionally import-free so the
builder can append its source to a self-contained submission.
"""

import math


R1_MARKET_PARAMS = {
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

R1_TOWN_PRODUCTS = tuple(item for item in R1_MARKET_PARAMS if item != "FERTILIZER")
R1_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _r1_shape(name, value):
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


def r1_market_price(item, inventory, params=None):
    p = (params or R1_MARKET_PARAMS)[str(item)]
    inventory = int(inventory)
    base, equilibrium, scale = p["base"], p["I0"], p["T"]
    if inventory < equilibrium:
        func, target = p["below_func"], p["below_target"]
        price = base + target * base / _r1_shape(func, scale) * _r1_shape(func, equilibrium - inventory)
    else:
        func, target = p["above_func"], p["above_target"]
        price = base - target * base / _r1_shape(func, scale) * _r1_shape(func, inventory - equilibrium)
    return max(1, int(round(price)))


def r1_clone_orders(orders):
    return [list(order) for order in (orders or []) if isinstance(order, (list, tuple))]


def r1_sell_orders(orders, item=None):
    result = []
    for order in orders or []:
        if not isinstance(order, (list, tuple)) or len(order) < 3:
            continue
        if str(order[0]).upper() != "SELL":
            continue
        if item is not None and str(order[1]).upper() != str(item).upper():
            continue
        try:
            quantity = int(order[2])
        except (TypeError, ValueError):
            quantity = 0
        if quantity > 0:
            result.append(["SELL", str(order[1]).upper(), quantity])
    return result


def r1_clean_zero_sells(orders):
    """Remove zero/negative SELLs instead of leaving invalid order slots."""
    result = []
    for order in r1_clone_orders(orders):
        if len(order) >= 3 and str(order[0]).upper() == "SELL":
            try:
                if int(order[2]) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
        result.append(order)
    return result


def r1_reorder_sell_orders(orders, inventory):
    """Apply V27's quantity-impact ordering to an order queue.

    Non-SELL slots stay in place.  SELL slots are sorted by the official
    current-price minus post-queue-price impact, matching the route overlay.
    """
    orders = r1_clean_zero_sells(orders)
    rows = []
    for index, order in enumerate(orders):
        if len(order) >= 3 and str(order[0]).upper() == "SELL":
            item = str(order[1]).upper()
            try:
                quantity = max(0, int(order[2]))
            except (TypeError, ValueError):
                quantity = 0
            if quantity > 0 and item in R1_MARKET_PARAMS:
                current = r1_market_price(item, inventory.get(item, 10000))
                later = r1_market_price(item, inventory.get(item, 10000) + quantity)
                rows.append((quantity * max(0, current - later), -index, list(order)))
                continue
        rows.append(None)
    sell_rows = [row for row in rows if row is not None]
    if len(sell_rows) < 2:
        return orders
    sell_rows.sort(reverse=True)
    iterator = iter(row[2] for row in sell_rows)
    return [next(iterator) if row is not None else order for row, order in zip(rows, orders)]


def r1_apply_town_consumption(inventory, shops, step, shop_interval=4, center_interval=24):
    """Apply the same deterministic portion of _town_consume as the engine."""
    inventory = dict(inventory or {})
    shops = list(shops or [])
    if int(step) % max(1, int(shop_interval)) == 0:
        for shop in shops:
            products = R1_SHOP_PRODUCTS.get(str(shop), ())
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                inventory[item] = int(inventory.get(item, 10000)) - multiplier
    if int(step) % max(1, int(center_interval)) == 0:
        for item in R1_TOWN_PRODUCTS:
            inventory[item] = int(inventory.get(item, 10000)) - 1
    return inventory


def _r1_order_state(order):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return None
    op = str(order[0]).upper()
    if op not in {"SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"}:
        return None
    try:
        quantity = int(order[2])
    except (TypeError, ValueError):
        return None
    if quantity <= 0:
        return None
    return {"op": op, "item": str(order[1]).upper(), "remaining": quantity}


def _r1_commit(state, player, quote, inventory, sheds, money, shed_capacity=100):
    op, item, price = quote
    shed = sheds[player]
    if op == "SELL":
        if int(shed.get(item, 0)) <= 0:
            return False
        shed[item] = int(shed.get(item, 0)) - 1
        money[player] += float(price)
        if float(price) > 1:
            inventory[item] = int(inventory.get(item, 10000)) + 1
        return True
    if op == "BUY_PRODUCT":
        if money[player] < float(price) or sum(max(0, int(v)) for v in shed.values()) >= int(shed_capacity):
            return False
        money[player] -= float(price)
        shed[item] = int(shed.get(item, 0)) + 1
        inventory[item] = int(inventory.get(item, 10000)) - 1
        return True
    # BUY_SEED and BUY_ANIMAL are included so a route's current market queue
    # can be represented.  They do not change shared inventory.
    if op in {"BUY_SEED", "BUY_ANIMAL"}:
        if money[player] < float(price):
            return False
        money[player] -= float(price)
        return True
    return False


def _r1_quote(state, inventory):
    if state is None or state["remaining"] <= 0:
        return None
    op, item = state["op"], state["item"]
    if op == "SELL":
        if item not in R1_MARKET_PARAMS:
            return None
        return op, item, r1_market_price(item, inventory.get(item, 10000))
    if op == "BUY_PRODUCT":
        if item not in R1_MARKET_PARAMS:
            return None
        return op, item, r1_market_price(item, inventory.get(item, 10000) - 1)
    # Fixed prices are not material to V032 premium rollouts, but these values
    # match the public rules and prevent a route BUY from becoming free.
    fixed = {"WHEAT": 10, "FERTILIZER": 100, "GOOSE": 300, "COW": 400, "SHEEP": 500}
    return op, item, fixed.get(item, 0)


def r1_process_market(orders_by_player, inventory, sheds, money, max_orders=10, shed_capacity=100):
    """Process two players' queues using environment-style unit lockstep."""
    inventory = dict(inventory or {})
    sheds = [dict(sheds[0] or {}), dict(sheds[1] or {})]
    money = [float(money[0]), float(money[1])]
    queues = [r1_clone_orders((orders_by_player or [[], []])[i])[:max(1, int(max_orders))] for i in (0, 1)]
    states = [[_r1_order_state(order) for order in queue] for queue in queues]
    executed = [0, 0]
    failed = [0, 0]
    max_len = max((len(queue) for queue in states), default=0)
    for order_index in range(max_len):
        current = [states[player][order_index] if order_index < len(states[player]) else None for player in (0, 1)]
        while True:
            quotes = [None, None]
            for player in (0, 1):
                if current[player] is not None:
                    quotes[player] = _r1_quote(current[player], inventory)
                    if quotes[player] is None:
                        current[player] = None
            if quotes[0] is None and quotes[1] is None:
                break
            committed = False
            # Quotes were formed before either commit, matching _process_market.
            for player in (0, 1):
                if quotes[player] is None:
                    continue
                ok = _r1_commit(current[player], player, quotes[player], inventory, sheds, money, shed_capacity)
                if ok:
                    current[player]["remaining"] -= 1
                    executed[player] += 1
                    committed = True
                else:
                    failed[player] += 1
                    current[player] = None
            if not committed:
                break
    return {"inventory": inventory, "sheds": sheds, "money": money,
            "executed": executed, "failed": failed,
            "truncated_orders": [len(r1_clone_orders(x)) > max(1, int(max_orders)) for x in (orders_by_player or [[], []])]}


def r1_simulate_window(start_inventory, start_money, start_sheds, orders_by_step,
                       opponent_orders_by_step=None, start_step=0, end_step=0,
                       shops=None, shop_interval=4, center_interval=24,
                       max_orders=10, shed_capacity=100):
    """Roll a shared market from start_step through end_step inclusive."""
    inventory = dict(start_inventory or {})
    money = [float(start_money[0]), float(start_money[1])]
    sheds = [dict((start_sheds or [{}, {}])[0] or {}), dict((start_sheds or [{}, {}])[1] or {})]
    own_schedule = orders_by_step or {}
    opp_schedule = opponent_orders_by_step or {}
    executed = [0, 0]
    failed = [0, 0]
    truncated = 0
    for step in range(int(start_step), int(end_step) + 1):
        own_orders = r1_reorder_sell_orders(own_schedule.get(step, []), inventory)
        result = r1_process_market(
            [own_orders, opp_schedule.get(step, [])],
            inventory, sheds, money, max_orders=max_orders, shed_capacity=shed_capacity,
        )
        inventory, sheds, money = result["inventory"], result["sheds"], result["money"]
        executed = [executed[i] + result["executed"][i] for i in (0, 1)]
        failed = [failed[i] + result["failed"][i] for i in (0, 1)]
        truncated += sum(bool(x) for x in result["truncated_orders"])
        inventory = r1_apply_town_consumption(inventory, shops, step, shop_interval, center_interval)
    return {"inventory": inventory, "sheds": sheds, "money": money,
            "executed": executed, "failed": failed, "truncated_orders": truncated}
