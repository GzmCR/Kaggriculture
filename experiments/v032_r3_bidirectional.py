"""V032-R3 helpers for real-inventory constrained market timing.

R3 keeps the V27 production route fixed and evaluates two independent kinds
of single-event intervention:

* ADVANCE_H1/H2/H3: sell inventory already in the shed before the route's
  original sale at ``t`` and subtract the same quantity at ``t``;
* DELAY_1/DELAY_25/DELAY_50: the V032-R2 current-to-next-event transfer.

This module contains only pure action/event helpers.  The paired real-engine
runner lives in ``run_v032_r3_bidirectional.py``.
"""

from __future__ import annotations

import copy

from rl_010_milk_bidirectional import rl010_market_shed_state
from v032_r2_interval import (
    R2_MAX_ORDERS,
    R2_SHED_CAPACITY,
    r2_adjust_delay,
    r2_adjust_future,
    r2_local_prediction,
    r2_mandatory_costs,
    r2_process_market,
    r2_reorder_existing_orders,
    r2_safety_gate,
    r2_simulate_interval,
    r2_storage_usage,
)


R3_ITEMS = ("MILK", "STRAWBERRY", "MELON", "WOOL")
R3_MAX_ORDERS = R2_MAX_ORDERS
R3_SHED_CAPACITY = R2_SHED_CAPACITY
R3_CUTOFF = 648
R3_HORIZONS = (1, 2, 3)
R3_FRACTIONS = (0.25, 0.50)


def r3_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def r3_copy_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return copy.deepcopy(action)


def r3_market_orders(action):
    return list((action or {}).get("market", []) or [])


def r3_is_sell(order, item=None):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return False
    if str(order[0]).upper() != "SELL":
        return False
    if item is not None and str(order[1]).upper() != str(item).upper():
        return False
    return True


def r3_sell_quantity(action_or_orders, item):
    orders = (
        r3_market_orders(action_or_orders)
        if isinstance(action_or_orders, dict)
        else list(action_or_orders or [])
    )
    item = str(item).upper()
    return sum(
        max(0, r3_int(order[2]))
        for order in orders
        if r3_is_sell(order, item)
    )


def r3_round_half_up(value):
    return int(float(value) + 0.5)


def r3_quantity_candidates(source_quantity, maximum):
    """Return 1-unit, 25%, and 50% candidates capped by legal inventory."""
    source_quantity = max(0, r3_int(source_quantity))
    maximum = min(30, max(0, r3_int(maximum)))
    values = {1, r3_round_half_up(source_quantity * 0.25),
              r3_round_half_up(source_quantity * 0.50)}
    return tuple(sorted(value for value in values if 0 < value <= maximum))


def r3_adjust_sell(action, item, delta, max_orders=R3_MAX_ORDERS):
    """Adjust an existing SELL total without creating a future inventory debt.

    Positive ``delta`` adds to the first same-item SELL row, or appends a new
    SELL row when there is an order slot.  Negative ``delta`` removes units
    from same-item rows and removes zero-valued rows.
    """
    action = r3_copy_action(action)
    item = str(item).upper()
    delta = r3_int(delta)
    orders = r3_market_orders(action)
    if delta == 0:
        return action
    rows = [
        order for order in orders
        if r3_is_sell(order, item)
    ]
    if delta > 0:
        if rows:
            rows[0][2] = max(0, r3_int(rows[0][2])) + delta
            return action
        if len(orders) >= max(1, r3_int(max_orders, R3_MAX_ORDERS)):
            return None
        orders.append(["SELL", item, delta])
        action["market"] = orders
        return action

    remaining = -delta
    if r3_sell_quantity(orders, item) < remaining:
        return None
    result = []
    for order in orders:
        if remaining and r3_is_sell(order, item):
            current = max(0, r3_int(order[2]))
            take = min(current, remaining)
            order[2] = current - take
            remaining -= take
        if r3_is_sell(order, item) and r3_int(order[2]) <= 0:
            continue
        result.append(order)
    if remaining:
        return None
    action["market"] = result
    return action


def r3_non_target_signature(action, item):
    """Order multiset excluding the target SELL item.

    V27 may reorder SELL rows after an intervention.  Order position changes
    are allowed; insertion, deletion, or quantity changes of other orders are
    not.
    """
    item = str(item).upper()
    rows = []
    for order in r3_market_orders(action):
        if (isinstance(order, (list, tuple)) and len(order) >= 2
                and str(order[0]).upper() == "SELL"
                and str(order[1]).upper() == item):
            continue
        if isinstance(order, (list, tuple)):
            rows.append(tuple(copy.deepcopy(order)))
        else:
            rows.append((repr(order),))
    return tuple(sorted(rows, key=repr))


def r3_target_inventory_after_actions(obs, action, item, config=None):
    """Target shed inventory after farmer/hand actions, before market orders."""
    return rl010_market_shed_state(
        obs, action, item=str(item).upper(), config=config or {}
    )


def r3_available_extra_inventory(obs, action, item, config=None):
    """Inventory that is not already committed by the same-turn target SELL."""
    state = r3_target_inventory_after_actions(obs, action, item, config)
    return max(0, r3_int(state.get("shed_after_actions", 0))
               - r3_int(state.get("current_sell", 0)))


def _action_items(actions):
    if isinstance(actions, dict):
        return ((int(step), action) for step, action in actions.items())
    return enumerate(actions or [])


def r3_find_advance_events(actions, item, horizons=R3_HORIZONS,
                           cutoff=R3_CUTOFF, min_step=None, max_step=None):
    item = str(item).upper()
    allowed_horizons = tuple(sorted(set(r3_int(h) for h in horizons if r3_int(h) > 0)))
    result = []
    for step, action in _action_items(actions):
        step = int(step)
        if step >= int(cutoff):
            continue
        if min_step is not None and step < int(min_step):
            continue
        if max_step is not None and step > int(max_step):
            continue
        quantity = r3_sell_quantity(action, item)
        if quantity <= 0:
            continue
        for horizon in allowed_horizons:
            if step - horizon < 0:
                continue
            result.append({
                "kind": "ADVANCE",
                "item": item,
                "start_step": step - horizon,
                "end_step": step,
                "horizon": horizon,
                "current_quantity": quantity,
            })
    return sorted(result, key=lambda row: (row["end_step"], row["horizon"], row["item"]))


def r3_find_delay_events(actions, item, cutoff=R3_CUTOFF,
                         min_step=None, max_step=None):
    item = str(item).upper()
    rows = []
    for step, action in _action_items(actions):
        step = int(step)
        if step >= int(cutoff):
            continue
        if min_step is not None and step < int(min_step):
            continue
        if max_step is not None and step > int(max_step):
            continue
        quantity = r3_sell_quantity(action, item)
        if quantity > 0:
            rows.append((step, quantity))
    result = []
    for index, (step, quantity) in enumerate(rows[:-1]):
        future_step, future_quantity = rows[index + 1]
        if future_step <= step:
            continue
        result.append({
            "kind": "DELAY",
            "item": item,
            "start_step": step,
            "end_step": future_step,
            "horizon": future_step - step,
            "current_quantity": quantity,
            "future_quantity": future_quantity,
        })
    return result


def r3_event_key(event):
    return "{}|{}|{}|{}".format(
        event.get("kind", ""), event.get("item", ""),
        int(event.get("start_step", -1)), int(event.get("end_step", -1))
    )


__all__ = [
    "R3_ITEMS", "R3_MAX_ORDERS", "R3_SHED_CAPACITY", "R3_CUTOFF",
    "R3_HORIZONS", "R3_FRACTIONS", "r3_copy_action", "r3_sell_quantity",
    "r3_quantity_candidates", "r3_adjust_sell", "r3_non_target_signature",
    "r3_target_inventory_after_actions", "r3_available_extra_inventory",
    "r3_find_advance_events", "r3_find_delay_events", "r3_event_key",
    "r2_adjust_delay", "r2_adjust_future", "r2_local_prediction",
    "r2_mandatory_costs", "r2_process_market", "r2_reorder_existing_orders",
    "r2_safety_gate", "r2_simulate_interval", "r2_storage_usage",
]
