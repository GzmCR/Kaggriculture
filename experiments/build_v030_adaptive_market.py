"""Build V030 adaptive market variants from the clean V029 route."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_V029 = ROOT / "baseline/artifacts/v029_milk_schedule/v029a_milk_safe_schedule/main.py"
ARTIFACT_ROOT = ROOT / "baseline/artifacts/v030_adaptive_market"
HISTORY_ROOT = ROOT / "baseline/history/v030_adaptive_market"


RUNTIME = r'''

# V030: market-state gates around the V029 quantity-preserving schedule.
_V030_MODE = "__MODE__"
_V030_CUTOFF = 672
_V030_STATS = {
    "calls": 0,
    "errors": 0,
    "changed_calls": 0,
    "changed_units": 0,
    "milk_delay_accepted": 0,
    "milk_delay_blocked": 0,
    "milk_delay_pending": 0,
    "crash_advance_accepted": 0,
    "crash_advance_blocked": 0,
    "pending_failures": 0,
    "decisions": [],
}

_V030_MILK_SCHEDULE = (
    (215, 260),
    (288, 308),
    (336, 375),
    (388, 404),
    (504, 522),
    (552, 571),
)
_V030_CRASH_ITEMS = ("MELON", "STRAWBERRY")
_V030_BASE_PRICE = {"MILK": 160.0, "MELON": 250.0, "STRAWBERRY": 120.0}
_V030_MAX_ADVANCES = 4
_V030_MAX_ADVANCES_PER_ITEM = 2


def _v030_state(obs):
    seat = int(_get(obs, "player", 0) or 0)
    step = max(0, int(_get(obs, "step", 0) or 0))
    state = _V030_STATES.setdefault(seat, {
        "last_step": -1,
        "prices": {item: [] for item in ("MILK", "MELON", "STRAWBERRY")},
        "pending": {},
        "advances": 0,
        "advances_by_item": {},
    })
    if step == 0 or step < state["last_step"]:
        state.clear()
        state.update({
            "last_step": -1,
            "prices": {item: [] for item in ("MILK", "MELON", "STRAWBERRY")},
            "pending": {},
            "advances": 0,
            "advances_by_item": {},
        })
    prices = _get(obs, "market", {}).get("prices", {}) or {}
    for item in state["prices"]:
        value = _num(prices.get(item, 0))
        state["prices"][item].append((step, value))
        state["prices"][item] = state["prices"][item][-96:]
    state["last_step"] = step
    return state


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _v030_lagged_price(state, item, step, lag):
    target = int(step) - int(lag)
    values = [price for seen_step, price in state["prices"].get(item, []) if seen_step <= target]
    return values[-1] if values else None


def _v030_momentum(state, item, step):
    current_values = state["prices"].get(item, [])
    if not current_values:
        return None, None, None
    current = current_values[-1][1]
    price12 = _v030_lagged_price(state, item, step, 12)
    price24 = _v030_lagged_price(state, item, step, 24)
    return current, None if price12 is None else current - price12, None if price24 is None else current - price24


def _v030_milk_delay_allowed(obs, state, current_step, future_step):
    del obs, future_step
    current, delta12, delta24 = _v030_momentum(state, "MILK", current_step)
    if current is None:
        return True, "no_history"
    # Only suppress V029's delay when the quote is already deeply discounted
    # and still falling.  Ordinary negative momentum is common in a shared
    # market and was too noisy in the first V030 benchmark.
    if current <= 8:
        return False, "near_floor"
    if (
        delta12 is not None
        and current <= max(8.0, _V030_BASE_PRICE["MILK"] * 0.60)
        and delta12 <= -20
    ):
        return False, "low_price_fast_12_turn_drop"
    if (
        delta12 is not None
        and delta24 is not None
        and delta12 <= -30
        and delta24 <= -30
    ):
        return False, "persistent_12_24_turn_drop"
    return True, "stable_or_rising"


def _v030_crash_gate(state, item, step):
    current, delta12, delta24 = _v030_momentum(state, item, step)
    if current is None:
        return False, "no_history"
    base = _V030_BASE_PRICE[item]
    if current <= max(8.0, base * 0.20):
        return True, "near_floor"
    if delta12 is not None and delta12 <= -20:
        return True, "negative_12_turn_momentum"
    if delta24 is not None and delta24 <= -30:
        return True, "negative_24_turn_momentum"
    return False, "no_crash"


def _v030_sell_quantity(action, item):
    total = 0
    for order in action.get("market", []) or []:
        if (
            isinstance(order, (list, tuple))
            and len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == item
        ):
            total += max(0, int(_num(order[2])))
    return total


def _v030_adjust_sell(action, item, delta):
    for index, order in enumerate(action.get("market", []) or []):
        if (
            isinstance(order, (list, tuple))
            and len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == item
        ):
            current = max(0, int(_num(order[2])))
            updated = current + int(delta)
            if updated < 0:
                return 0
            if updated == 0:
                action["market"].pop(index)
            else:
                action["market"][index] = [order[0], order[1], updated]
            return abs(int(delta))
    return 0


def _v030_route_events(item):
    rows = []
    for step, action in enumerate(_ACTIONS):
        if _v030_sell_quantity(action, item) > 0:
            rows.append(step)
    return rows


_V030_NEXT_EVENTS = {}
for _item in _V030_CRASH_ITEMS:
    _rows = _v030_route_events(_item)
    for _index, _current in enumerate(_rows[:-1]):
        _future = _rows[_index + 1]
        if 8 <= _future - _current <= 72:
            _V030_NEXT_EVENTS[(_item, _current)] = _future

_V030_STATES = {}


def _v030_apply_pending(action, state, step):
    changed = 0
    for key in [key for key in state["pending"] if key[0] == int(step)]:
        _, item, direction = key
        quantity = state["pending"].pop(key, 0)
        if direction == "delay":
            delta = quantity
        else:
            delta = -quantity
        moved = _v030_adjust_sell(action, item, delta)
        changed += moved
        if moved != abs(delta):
            _V030_STATS["pending_failures"] += 1
    return changed


def _v030_adjust_market(action, obs, state, step):
    if step >= _V030_CUTOFF:
        return action, 0
    changed = _v030_apply_pending(action, state, step)

    for current_step, future_step in _V030_MILK_SCHEDULE:
        if step != current_step:
            continue
        allowed, reason = _v030_milk_delay_allowed(obs, state, current_step, future_step)
        moved = _v030_adjust_sell(action, "MILK", -1) if allowed else 0
        if allowed and moved:
            state["pending"][(future_step, "MILK", "delay")] = moved
            _V030_STATS["milk_delay_accepted"] += 1
            _V030_STATS["milk_delay_pending"] += moved
        else:
            _V030_STATS["milk_delay_blocked"] += 1
        _V030_STATS["decisions"].append({
            "step": int(step),
            "item": "MILK",
            "future_step": int(future_step),
            "mode": "delay",
            "allowed": bool(allowed and moved),
            "reason": reason,
        })
        changed += moved

    if _V030_MODE != "cross_product_guard" or state["advances"] >= _V030_MAX_ADVANCES:
        return action, changed

    for item in _V030_CRASH_ITEMS:
        future_step = _V030_NEXT_EVENTS.get((item, step))
        if future_step is None:
            continue
        if state["advances_by_item"].get(item, 0) >= _V030_MAX_ADVANCES_PER_ITEM:
            continue
        crash, reason = _v030_crash_gate(state, item, step)
        moved = _v030_adjust_sell(action, item, 1) if crash else 0
        if crash and moved:
            state["pending"][(future_step, item, "advance")] = moved
            state["advances"] += moved
            state["advances_by_item"][item] = state["advances_by_item"].get(item, 0) + moved
            _V030_STATS["crash_advance_accepted"] += moved
        elif crash:
            _V030_STATS["crash_advance_blocked"] += 1
        _V030_STATS["decisions"].append({
            "step": int(step),
            "item": item,
            "future_step": int(future_step),
            "mode": "advance",
            "allowed": bool(crash and moved),
            "reason": reason,
        })
        changed += moved
    return action, changed


def agent(obs):
    try:
        _V030_STATS["calls"] += 1
        step = max(0, int(_get(obs, "step", 0) or 0))
        state = _v030_state(obs)
        action = _v029_base_action(obs, step)
        action, changed = _v030_adjust_market(action, obs, state, step)
        if changed:
            _V030_STATS["changed_calls"] += 1
            _V030_STATS["changed_units"] += int(changed)
        return _align_hands(action, obs)
    except Exception:
        _V030_STATS["errors"] += 1
        return _v030_v029_original_agent(obs)
'''


VARIANTS = {
    "v030a_milk_momentum_gate": "milk_momentum_gate",
    "v030b_cross_product_guard": "cross_product_guard",
}


def build_variant(name, mode):
    source = SOURCE_V029.read_text(encoding="utf-8")
    marker = "\ndef agent(obs):"
    if source.count(marker) != 1:
        raise RuntimeError("Expected exactly one V029 public agent definition")
    source = source.replace(marker, "\ndef _v030_v029_original_agent(obs):", 1)
    runtime = RUNTIME.replace("__MODE__", mode)
    source = source.rstrip() + "\n\n" + runtime.lstrip() + "\n"

    history_dir = HISTORY_ROOT / name
    artifact_dir = ARTIFACT_ROOT / name
    history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data = source.encode("utf-8")
    history_main = history_dir / "main.py"
    artifact_main = artifact_dir / "main.py"
    history_main.write_bytes(data)
    artifact_main.write_bytes(data)
    archive = artifact_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(artifact_main, arcname="main.py")
    market_changes = [
        "gate V029 MILK delays only for low-price, fast 12-turn drops or persistent 12/24-turn drops",
    ]
    if mode == "cross_product_guard":
        market_changes.append(
            "cross_product_guard: advance at most four existing MELON/STRAWBERRY SELL units during a crash"
        )
    manifest = {
        "candidate": name,
        "mode": mode,
        "base": "baseline/artifacts/v029_milk_schedule/v029a_milk_safe_schedule/main.py",
        "market_changes": market_changes,
        "milk_schedule": [[215, 260], [288, 308], [336, 375], [388, 404], [504, 522], [552, 571]],
        "cutoff": 672,
        "main_sha256": hashlib.sha256(data).hexdigest(),
        "main_bytes": len(data),
        "archive": str(archive.relative_to(ROOT)),
        "root_main_modified": False,
    }
    (artifact_dir / "submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = [build_variant(name, mode) for name, mode in VARIANTS.items()]
    (ARTIFACT_ROOT / "build_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(build(), indent=2))
