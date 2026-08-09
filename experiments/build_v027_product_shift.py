"""Build V027 product-level sell-wave shift candidates from the v22 route."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from build_v026_v22_v022c_recovery import (
    ROOT,
    SOURCE_NOTEBOOK,
    _decode_notebook_agent,
)


ARTIFACT_ROOT = ROOT / "baseline/artifacts/v027_v22_product_shift"
HISTORY_ROOT = ROOT / "baseline/history/v027_v22_product_shift"
CANDIDATES = (
    ("v027a_melon_ratio", "melon"),
    ("v027b_mirror_gated", "mirror"),
    ("v027c_product_specific", "product"),
)


README = """# V027: product-level sell-wave shift on the v22 route

V027 keeps the 44-46 v22 route, WEED recovery, and price-impact SELL ordering.
It only moves a bounded quantity from a future existing premium SELL into the
current existing SELL of the same product.  The future order receives an exact
ledger deduction, so total product sales and all farmer/hands actions remain
unchanged.

Candidates:

- `v027a_melon_ratio`: MELON only, up to 25 percent / 6 units, with an
  eight-turn cooldown and market-price/inventory gates.
- `v027b_mirror_gated`: the same MELON rule, enabled only after the public
  opponent matches the v22-like structure on both day 8 and day 12.
- `v027c_product_specific`: the gated MELON rule plus a 12.5 percent
  STRAWBERRY ablation.  At most one product is adjusted per turn.

Dynamic quantity changes stop before step 648.  No pending deduction may be
scheduled at or after step 672, where every candidate returns the v22 action.
The root `main.py` and the v22 control are not modified.
"""


RUNTIME_SUFFIX = r'''

# V027: product-level quantity-preserving sell-wave shift.
_V027_VARIANT = "__V027_VARIANT__"
_V027_STOP_STEP = 672
_V027_DYNAMIC_CUTOFF = 648
_V027_CHECK_HOUR = 6
_V027_PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")
_V027_SHIFT_CONFIG = {
    "MELON": {"ratio": 0.25, "max_units": 6, "cooldown": 8},
    "STRAWBERRY": {"ratio": 0.125, "max_units": 6, "cooldown": 8},
}
_V027_STATE = {0: {}, 1: {}}
_V027_STATS = {
    "calls": 0,
    "errors": 0,
    "action_diff_calls": 0,
    "action_diff_farmer": 0,
    "action_diff_hands": 0,
    "action_diff_market": 0,
    "planned_sell_units": {item: 0 for item in _V027_PREMIUM},
    "actual_sell_units": {item: 0 for item in _V027_PREMIUM},
    "planned_sell_orders": {item: 0 for item in _V027_PREMIUM},
    "actual_sell_orders": {item: 0 for item in _V027_PREMIUM},
    "shifted_sell_units": {item: 0 for item in _V027_PREMIUM},
    "future_reduced_units": {item: 0 for item in _V027_PREMIUM},
    "shift_events": [],
    "future_reduction_events": [],
    "price_events": [],
    "mirror_checks": [],
    "mirror_latches": 0,
    "mirror_releases": 0,
    "mirror_active_calls": 0,
    "pending_missed": 0,
    "pending_at_stop": 0,
}


def _v027_new_state():
    return {
        "last_step": -1,
        "pending": {},
        "last_shift": {item: -100000 for item in _V027_PREMIUM},
        "day8_match": None,
        "day12_match": None,
        "mirror_mode": False,
        "divergence_days": 0,
        "last_boundary_day": -1,
        "last_prices": {},
    }


def _v027_state(obs):
    seat = _seat(obs)
    state = _V027_STATE.setdefault(seat, _v027_new_state())
    step = int(_get(obs, "step", 0) or 0)
    if step == 0 or step < int(state.get("last_step", -1)):
        state = _v027_new_state()
        _V027_STATE[seat] = state
    state["last_step"] = step
    return state


def _v027_opponent(obs):
    farms = list(_get(obs, "farms", []) or [])
    seat = _seat(obs)
    return farms[1 - seat] if len(farms) > 1 else {}


def _v027_tile_counts(farm):
    counts = {"COW": 0, "SHEEP": 0}
    animal_yield = []
    for row in _get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            animal = str(tile.get("animal", "")).upper()
            if animal in counts:
                counts[animal] += 1
                try:
                    animal_yield.append(max(0.0, float(tile.get("yield_units", 0) or 0)))
                except (TypeError, ValueError):
                    animal_yield.append(0.0)
    low_output = bool(animal_yield) and (
        sum(animal_yield) < 0.5 * len(animal_yield)
        or sum(value <= 0.0 for value in animal_yield) >= max(2, len(animal_yield) // 2)
    )
    return counts, low_output


def _v027_mirror_signature(obs):
    opponent = _v027_opponent(obs)
    hands = len(_get(opponent, "hands", []) or [])
    unlocked = {str(item).upper() for item in (_get(opponent, "unlocked_quadrants", []) or [])}
    animals, low_output = _v027_tile_counts(opponent)
    match = (
        12 <= hands <= 14
        and {"NE", "SW"}.issubset(unlocked)
        and 7 <= animals["COW"] <= 9
        and 5 <= animals["SHEEP"] <= 7
        and not low_output
    )
    return {
        "hands": hands,
        "unlocked": sorted(unlocked),
        "cows": animals["COW"],
        "sheep": animals["SHEEP"],
        "low_output": bool(low_output),
        "match": bool(match),
    }


def _v027_update_mirror(obs, state, step):
    day = int(_get(obs, "day", step // 24) or 0)
    hour = int(_get(obs, "hour", step % 24) or 0)
    if hour != _V027_CHECK_HOUR or day == state.get("last_boundary_day"):
        return
    state["last_boundary_day"] = day
    signature = _v027_mirror_signature(obs)
    if day == 8:
        state["day8_match"] = signature["match"]
        _V027_STATS["mirror_checks"].append({"day": day, **signature})
    elif day == 12:
        state["day12_match"] = signature["match"]
        _V027_STATS["mirror_checks"].append({"day": day, **signature})
        if (
            _V027_VARIANT in {"mirror", "product"}
            and state.get("day8_match") is True
            and state.get("day12_match") is True
            and not state.get("mirror_mode")
        ):
            state["mirror_mode"] = True
            state["divergence_days"] = 0
            _V027_STATS["mirror_latches"] += 1
    elif day > 12 and state.get("mirror_mode"):
        if signature["match"]:
            state["divergence_days"] = 0
        else:
            state["divergence_days"] = int(state.get("divergence_days", 0)) + 1
            if state["divergence_days"] >= 2:
                state["mirror_mode"] = False
                state["divergence_days"] = 0
                _V027_STATS["mirror_releases"] += 1


def _v027_overlay_active(state, step):
    if step >= _V027_STOP_STEP:
        return False
    if _V027_VARIANT == "melon":
        return True
    return bool(state.get("mirror_mode"))


def _v027_order_quantity(market, item):
    return sum(
        max(0, int(order[2]))
        for order in market
        if _is_sell(order) and str(order[1]).upper() == item
    )


def _v027_order_count(market, item):
    return sum(
        1 for order in market
        if _is_sell(order) and str(order[1]).upper() == item
    )


def _v027_find_order(market, item):
    for index, order in enumerate(market):
        if _is_sell(order) and str(order[1]).upper() == item:
            return index
    return None


def _v027_route_quantity(step, item):
    if not _ACTIONS:
        return 0
    route_step = min(max(0, int(step)), len(_ACTIONS) - 1)
    return _v027_order_quantity((_ACTIONS[route_step] or {}).get("market", []) or [], item)


def _v027_visible_inventory(obs, item):
    private = _get(obs, "private", {}) or {}
    total = 0
    try:
        total += max(0, int((_get(private, "shed", {}) or {}).get(item, 0) or 0))
    except (TypeError, ValueError):
        pass
    for inventory in _get(private, "inventories", []) or []:
        try:
            total += max(0, int((inventory or {}).get(item, 0) or 0))
        except (TypeError, ValueError):
            continue
    return total


def _v027_current_market(market, item):
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_price = float(_get(prices, item, _market_price(item, current_inventory)) or 0)
    base, equilibrium = _MARKET_PARAMS[item][0], _MARKET_PARAMS[item][1]
    return current_inventory, current_price, base, equilibrium


def _v027_next_future(step, item, state):
    pending = state.setdefault("pending", {})
    upper = min(len(_ACTIONS), _V027_STOP_STEP)
    for future_step in range(int(step) + 1, upper):
        planned = _v027_route_quantity(future_step, item)
        already = int((pending.get(future_step, {}) or {}).get(item, 0) or 0)
        remaining = max(0, planned - already)
        if remaining > 0:
            return future_step, remaining
    return None, 0


def _v027_apply_due(market, state, step):
    due = state.setdefault("pending", {}).pop(int(step), {}) or {}
    for item, quantity in due.items():
        remaining = max(0, int(quantity or 0))
        index = _v027_find_order(market, item)
        if index is None:
            _V027_STATS["pending_missed"] += remaining
            continue
        order = list(market[index])
        current = max(0, int(order[2]))
        reduction = min(current, remaining)
        if reduction <= 0:
            _V027_STATS["pending_missed"] += remaining
            continue
        new_quantity = current - reduction
        if new_quantity:
            market[index] = [order[0], order[1], new_quantity]
        else:
            market.pop(index)
        _V027_STATS["future_reduced_units"][item] += reduction
        _V027_STATS["future_reduction_events"].append({
            "step": int(step), "product": item, "quantity": reduction,
        })
        if reduction < remaining:
            _V027_STATS["pending_missed"] += remaining - reduction


def _v027_maybe_shift(obs, market, state, step):
    if step >= _V027_DYNAMIC_CUTOFF or not _v027_overlay_active(state, step):
        return None
    if _V027_VARIANT in {"mirror", "product"}:
        _V027_STATS["mirror_active_calls"] += 1
    products = ["MELON"]
    if _V027_VARIANT == "product":
        products.append("STRAWBERRY")
    for item in products:
        config = _V027_SHIFT_CONFIG[item]
        last_shift = int(state.setdefault("last_shift", {}).get(item, -100000))
        if step - last_shift < int(config["cooldown"]):
            continue
        current_index = _v027_find_order(market, item)
        if current_index is None:
            continue
        current_quantity = max(0, int(market[current_index][2]))
        future_step, future_quantity = _v027_next_future(step, item, state)
        if future_step is None or future_step >= _V027_STOP_STEP:
            continue
        inventory, price, base, equilibrium = _v027_current_market(_get(obs, "market", {}) or {}, item)
        if inventory >= equilibrium or price * 100 < base * 105:
            continue
        available = _v027_visible_inventory(obs, item)
        quantity = min(
            int(future_quantity * float(config["ratio"])),
            int(config["max_units"]),
            max(0, available - current_quantity),
        )
        if quantity <= 0:
            continue
        market[current_index][2] = current_quantity + quantity
        by_future = state.setdefault("pending", {}).setdefault(future_step, {})
        by_future[item] = int(by_future.get(item, 0) or 0) + quantity
        state.setdefault("last_shift", {})[item] = step
        _V027_STATS["shifted_sell_units"][item] += quantity
        _V027_STATS["shift_events"].append({
            "step": int(step), "future_step": int(future_step), "product": item,
            "quantity": quantity, "price": price, "market_inventory": inventory,
            "planned_current": current_quantity,
            "planned_future": future_quantity,
            "estimated_unit_prices": [
                int(_market_price(item, inventory + offset))
                for offset in range(quantity)
            ],
        })
        return item
    return None


def _v027_sell_totals(market):
    quantities = {item: 0 for item in _V027_PREMIUM}
    orders = {item: 0 for item in _V027_PREMIUM}
    for order in market or []:
        if not _is_sell(order):
            continue
        item = str(order[1]).upper()
        if item not in quantities:
            continue
        try:
            quantities[item] += max(0, int(order[2]))
        except (TypeError, ValueError):
            continue
        orders[item] += 1
    return quantities, orders


def _v027_record_prices(obs, state, step):
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    for item in _V027_PREMIUM:
        try:
            current = float(prices.get(item, 0) or 0)
        except (TypeError, ValueError):
            continue
        previous = state.setdefault("last_prices", {}).get(item)
        if previous is not None and abs(current - previous) >= 10:
            event = {
                "step": int(step), "product": item,
                "previous": previous, "current": current,
                "kind": "cliff" if current < previous else "recovery",
            }
            _V027_STATS["price_events"].append(event)
        state.setdefault("last_prices", {})[item] = current


def _v027_record_actions(obs, planned, actual, state, step):
    planned_quantities, planned_orders = _v027_sell_totals(planned.get("market", []))
    actual_quantities, actual_orders = _v027_sell_totals(actual.get("market", []))
    for item in _V027_PREMIUM:
        _V027_STATS["planned_sell_units"][item] += planned_quantities[item]
        _V027_STATS["actual_sell_units"][item] += actual_quantities[item]
        _V027_STATS["planned_sell_orders"][item] += planned_orders[item]
        _V027_STATS["actual_sell_orders"][item] += actual_orders[item]
    if actual.get("farmer") != planned.get("farmer"):
        _V027_STATS["action_diff_farmer"] += 1
    if actual.get("hands") != planned.get("hands"):
        _V027_STATS["action_diff_hands"] += 1
    if actual.get("market") != planned.get("market"):
        _V027_STATS["action_diff_market"] += 1
    if actual != planned:
        _V027_STATS["action_diff_calls"] += 1
    _v027_record_prices(obs, state, step)


def _v027_base_action(obs, step):
    route_step = min(max(0, int(step)), len(_ACTIONS) - 1)
    action = _weed_repair_action(obs, _copy_action(_ACTIONS[route_step]), _ACTIONS, route_step)
    return _align_hands(_impact_slots(obs, action), obs)


def _v027_fallback(obs):
    farm = _farm(obs, _seat(obs))
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
        "market": [],
    }


def agent(obs):
    try:
        step = max(0, int(_get(obs, "step", 0) or 0))
        state = _v027_state(obs)
        _V027_STATS["calls"] += 1
        base = _v027_base_action(obs, step)
        _v027_update_mirror(obs, state, step)
        market = [list(order) for order in (base.get("market") or [])]
        _v027_apply_due(market, state, step)
        if step < _V027_STOP_STEP:
            _v027_maybe_shift(obs, market, state, step)
        elif state.get("pending"):
            _V027_STATS["pending_at_stop"] += sum(
                sum(max(0, int(value or 0)) for value in (items or {}).values())
                for items in state.get("pending", {}).values()
            )
        action = _align_hands({
            "farmer": list(base.get("farmer") or ["PASS"]),
            "hands": [list(item or ["PASS"]) for item in (base.get("hands") or [])],
            "market": market[:10],
        }, obs)
        _v027_record_actions(obs, base, action, state, step)
        return action
    except Exception:
        _V027_STATS["errors"] += 1
        return _v027_fallback(obs)
'''


def _write_candidate(name: str, variant: str, base_source: str, source_sha: str) -> dict:
    source = base_source.rstrip() + "\n\n" + RUNTIME_SUFFIX.replace("__V027_VARIANT__", variant) + "\n"
    history_dir = HISTORY_ROOT / name
    artifact_dir = ARTIFACT_ROOT / name
    history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_main = history_dir / "main.py"
    artifact_main = artifact_dir / "main.py"
    history_main.write_text(source, encoding="utf-8")
    artifact_main.write_text(source, encoding="utf-8")
    archive = artifact_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(artifact_main, arcname="main.py")
    manifest = {
        "candidate": name,
        "variant": variant,
        "source_notebook": str(SOURCE_NOTEBOOK.relative_to(ROOT)),
        "source_sha256": source_sha,
        "main_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "main_bytes": len(source.encode("utf-8")),
        "archive": str(archive.relative_to(ROOT)),
        "route": "v22_embedded_14hands_high_output",
        "market_layer": "v22_price_impact_plus_quantity_preserving_product_shift",
        "dynamic_cutoff_step": 648,
        "full_fallback_step": 672,
        "root_main_modified": False,
    }
    (artifact_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(README + f"\nCandidate: `{name}`\n", encoding="utf-8")
    return manifest


def build() -> dict:
    base_source, source_sha = _decode_notebook_agent(SOURCE_NOTEBOOK)
    marker = "\ndef agent(obs):"
    if base_source.count(marker) != 1:
        raise RuntimeError("Expected exactly one public v22 agent definition")
    base_source = base_source.replace(marker, "\ndef _v027_v22_original_agent(obs):", 1)
    manifests = {
        name: _write_candidate(name, variant, base_source, source_sha)
        for name, variant in CANDIDATES
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "README.md").write_text(README, encoding="utf-8")
    (ARTIFACT_ROOT / "build_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifests


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
