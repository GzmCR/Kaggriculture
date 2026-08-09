"""Build V028 order-permutation candidates from the v22 control."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from build_v022e_adaptive_recovery import ROOT, _decode_notebook_agent


ARTIFACT_ROOT = ROOT / "baseline/artifacts/v028_order_search"
HISTORY_ROOT = ROOT / "baseline/history/v028_order_search"
SOURCE_NOTEBOOK = ROOT / "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb"

CANDIDATES = (
    ("v028a_marginal_order", "marginal", 50.0),
    ("v028b_safe_order", "marginal", 100.0),
    ("v028c_robust_order", "robust", 50.0),
)


README = """# V028: v22 premium SELL order search

V028 keeps the embedded v22 farmer/hands route and its existing price-impact
market layer. It only permutes the contents of already-existing premium SELL
slots: MELON, STRAWBERRY, MILK, and WOOL.

The runtime simulates the environment's per-order, per-unit lockstep market
against a public v22 route shadow. A permutation is used only when its
predicted cash gain exceeds the candidate safety margin. Quantities, products,
non-SELL slots, BUY/HIRE/BUY_LAND orders, and all field actions are unchanged.

Candidates:

- v028a_marginal_order: 50 coin predicted-gain margin.
- v028b_safe_order: 100 coin predicted-gain margin.
- v028c_robust_order: 50 coin margin under both the v22 shadow and a reversed
  premium-slot stress shadow.

All candidates stop changing order slots at step 672 and fall back to v22 for
the terminal clear-out. These are experimental candidates; the root main.py
and the v22 control are not modified.
"""


RUNTIME_SUFFIX = r'''

# V028: quantity-preserving premium SELL order permutation.
import itertools as _v028_itertools

_V028_VARIANT = "__V028_VARIANT__"
_V028_MARGIN = __V028_MARGIN__
_V028_PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")
_V028_CUTOFF = 672
_V028_STATS = {
    "calls": 0,
    "errors": 0,
    "evaluations": 0,
    "decision_points": 0,
    "changed_calls": 0,
    "unchanged_calls": 0,
    "action_diff_calls": 0,
    "action_diff_farmer": 0,
    "action_diff_hands": 0,
    "action_diff_market": 0,
    "predicted_base_revenue": 0.0,
    "predicted_selected_revenue": 0.0,
    "predicted_delta": 0.0,
    "permutation_histogram": {},
    "decision_events": [],
}


def _v028_fallback(obs):
    farm = _farm(obs, _seat(obs))
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
        "market": [],
    }


def _v028_base_action(obs, step):
    route_step = min(max(0, int(step)), len(_ACTIONS) - 1)
    action = _weed_repair_action(obs, _copy_action(_ACTIONS[route_step]), _ACTIONS, route_step)
    return _align_hands(_impact_slots(obs, action), obs)


def _v028_is_premium_sell(order):
    return _is_sell(order) and str(order[1]).upper() in _V028_PREMIUM


def _v028_premium_positions(market):
    return [
        index for index, order in enumerate(market or [])
        if _v028_is_premium_sell(order)
    ]


def _v028_signature(market):
    return tuple(
        (str(order[0]), str(order[1]), int(order[2]))
        for order in (market or [])
        if isinstance(order, (list, tuple)) and len(order) >= 3
    )


def _v028_permutations(market):
    positions = _v028_premium_positions(market)
    if len(positions) < 2:
        return []
    original = [list(market[index]) for index in positions]
    output = []
    seen = set()
    for permutation in _v028_itertools.permutations(original):
        signature = tuple(tuple(order) for order in permutation)
        if signature in seen:
            continue
        seen.add(signature)
        candidate = [list(order) for order in (market or [])]
        for index, order in zip(positions, permutation):
            candidate[index] = list(order)
        output.append(candidate)
    return output


def _v028_shadow_market(obs, step):
    route_step = min(max(0, int(step)), len(_ACTIONS) - 1)
    shadow_action = _impact_slots(obs, _copy_action(_ACTIONS[route_step]))
    return [list(order) for order in (shadow_action.get("market") or [])]


def _v028_reverse_premium_slots(market):
    output = [list(order) for order in (market or [])]
    positions = _v028_premium_positions(output)
    orders = [list(output[index]) for index in positions]
    for index, order in zip(positions, reversed(orders)):
        output[index] = order
    return output


def _v028_order_state(order):
    if not isinstance(order, (list, tuple)) or not order:
        return None
    operation = str(order[0]).upper()
    if operation in {"HIRE", "BUY_LAND"}:
        return {"op": operation, "remaining": 1, "item": ""}
    if operation not in {"SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"}:
        return None
    if len(order) < 3:
        return None
    try:
        quantity = int(order[2])
    except (TypeError, ValueError):
        return None
    if quantity <= 0:
        return None
    return {"op": operation, "remaining": quantity, "item": str(order[1]).upper()}


def _v028_simulate(ours, shadow, obs):
    """Approximate kaggriculture.py market lockstep for one turn.

    The real engine quotes both players from the same pre-commit inventory for
    each unit, then commits both. Atomic HIRE/BUY_LAND orders are consumed once.
    The only hidden input is the opponent's private ability to complete SELL;
    the shadow assumes its public v22 route orders are valid.
    """
    market = _get(obs, "market", {}) or {}
    inventory = {}
    for item, quantity in dict(_get(market, "inventory", {}) or {}).items():
        try:
            inventory[str(item).upper()] = max(0, int(quantity or 0))
        except (TypeError, ValueError):
            inventory[str(item).upper()] = 10000
    states = [
        [_v028_order_state(order) for order in list(ours or [])[:10]],
        [_v028_order_state(order) for order in list(shadow or [])[:10]],
    ]
    money = []
    farms = list(_get(obs, "farms", []) or [])
    for player in range(2):
        farm = farms[_seat(obs)] if player == 0 else farms[1 - _seat(obs)] if len(farms) > 1 else {}
        try:
            money.append(float(_get(farm, "money", 0) or 0))
        except (TypeError, ValueError):
            money.append(0.0)
    revenue = [0.0, 0.0]
    own_available = {}
    if _get(obs, "private", None) is not None:
        # Keep availability per product. The engine checks shed[item] for
        # every SELL unit; one product cannot borrow another product's stock.
        for item, quantity in dict(
            _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
        ).items():
            try:
                own_available[str(item).upper()] = max(0, int(quantity or 0))
            except (TypeError, ValueError):
                own_available[str(item).upper()] = 0
    max_slots = max(len(states[0]), len(states[1]))
    for slot in range(max_slots):
        current = [states[player][slot] if slot < len(states[player]) else None for player in range(2)]
        for player in range(2):
            if current[player] is not None and current[player]["op"] in {"HIRE", "BUY_LAND"}:
                current[player] = None
        while True:
            quoted = [None, None]
            for player, state in enumerate(current):
                if state is None or int(state.get("remaining", 0)) <= 0:
                    continue
                operation = state["op"]
                item = state["item"]
                if operation == "SELL" and item in _MARKET_PARAMS:
                    quoted[player] = (
                        operation, item,
                        float(_market_price(item, inventory.get(item, 10000))),
                    )
                elif operation == "BUY_PRODUCT" and item in {"WHEAT", "FERTILIZER"}:
                    quoted[player] = (
                        operation, item,
                        float(_market_price(item, inventory.get(item, 10000) - 1)),
                    )
                else:
                    # Fixed-price operations still consume this queue slot,
                    # but they do not affect dynamic product inventory.
                    quoted[player] = (operation, item, 0.0)
            if quoted[0] is None and quoted[1] is None:
                break
            committed = False
            for player, quote in enumerate(quoted):
                if quote is None:
                    continue
                operation, item, price = quote
                if operation == "SELL":
                    if player == 0 and own_available.get(item, 10**9) <= 0:
                        current[player] = None
                        continue
                    if player == 0:
                        own_available[item] = own_available.get(item, 10**9) - 1
                        revenue[0] += price
                    if price > 1:
                        inventory[item] = inventory.get(item, 10000) + 1
                elif operation == "BUY_PRODUCT":
                    if player == 0:
                        if money[0] < price:
                            current[player] = None
                            continue
                        money[0] -= price
                    if player == 1:
                        money[1] -= price
                    inventory[item] = max(0, inventory.get(item, 10000) - 1)
                current[player]["remaining"] -= 1
                committed = True
                if current[player]["remaining"] <= 0:
                    current[player] = None
            if not committed:
                break
    return float(revenue[0])


def _v028_market_candidates(obs, base_market, step):
    permutations = _v028_permutations(base_market)
    if not permutations:
        return list(base_market or []), 0.0, 0.0, 0, []
    shadow = _v028_shadow_market(obs, step)
    profiles = [("v22", shadow)]
    if _V028_VARIANT == "robust":
        profiles.append(("reverse", _v028_reverse_premium_slots(shadow)))
    base_score = min(
        _v028_simulate(base_market, profile, obs)
        for _, profile in profiles
    )
    best_market = list(base_market or [])
    best_score = base_score
    best_profiles = []
    for candidate in permutations:
        _V028_STATS["evaluations"] += 1
        profile_scores = [
            (name, _v028_simulate(candidate, profile, obs))
            for name, profile in profiles
        ]
        score = min(value for _, value in profile_scores)
        if score > best_score:
            best_market = candidate
            best_score = score
            best_profiles = profile_scores
    delta = float(best_score - base_score)
    if delta < float(_V028_MARGIN):
        return list(base_market or []), base_score, base_score, 0, []
    signature = repr(_v028_signature(best_market))
    histogram = _V028_STATS["permutation_histogram"]
    histogram[signature] = int(histogram.get(signature, 0)) + 1
    _V028_STATS["decision_events"].append({
        "step": int(step),
        "base_revenue": float(base_score),
        "selected_revenue": float(best_score),
        "predicted_delta": delta,
        "profiles": {name: float(value) for name, value in best_profiles},
        "base_market": _v028_signature(base_market),
        "selected_market": _v028_signature(best_market),
    })
    return best_market, base_score, best_score, 1, best_profiles


def _v028_record_action(base, selected):
    if selected.get("farmer") != base.get("farmer"):
        _V028_STATS["action_diff_farmer"] += 1
    if selected.get("hands") != base.get("hands"):
        _V028_STATS["action_diff_hands"] += 1
    if selected.get("market") != base.get("market"):
        _V028_STATS["action_diff_market"] += 1
    if selected != base:
        _V028_STATS["action_diff_calls"] += 1


def agent(obs):
    try:
        step = max(0, int(_get(obs, "step", 0) or 0))
        _V028_STATS["calls"] += 1
        base = _v028_base_action(obs, step)
        selected = _copy_action(base)
        base_market = [list(order) for order in (base.get("market") or [])]
        changed = 0
        if step < _V028_CUTOFF:
            _V028_STATS["decision_points"] += int(len(_v028_premium_positions(base_market)) >= 2)
            market, base_score, selected_score, changed, _ = _v028_market_candidates(
                obs, base_market, step
            )
            _V028_STATS["predicted_base_revenue"] += float(base_score)
            _V028_STATS["predicted_selected_revenue"] += float(selected_score)
            _V028_STATS["predicted_delta"] += float(selected_score - base_score)
            selected["market"] = market[:10]
        if changed:
            _V028_STATS["changed_calls"] += 1
        else:
            _V028_STATS["unchanged_calls"] += 1
        _v028_record_action(base, selected)
        return _align_hands(selected, obs)
    except Exception:
        _V028_STATS["errors"] += 1
        return _v028_fallback(obs)
'''


def _write_candidate(name: str, variant: str, margin: float, base_source: str, source_sha: str) -> dict:
    runtime = (
        RUNTIME_SUFFIX
        .replace("__V028_VARIANT__", variant)
        .replace("__V028_MARGIN__", repr(float(margin)))
    )
    source = base_source.rstrip() + "\n\n" + runtime + "\n"
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
        "margin": margin,
        "source_notebook": str(SOURCE_NOTEBOOK.relative_to(ROOT)),
        "source_sha256": source_sha,
        "main_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "main_bytes": len(source.encode("utf-8")),
        "archive": str(archive.relative_to(ROOT)),
        "route": "v22_embedded_14hands_high_output",
        "market_layer": "quantity_preserving_premium_order_permutation",
        "shadow": "public_v22_route_market_queue",
        "full_fallback_step": 672,
        "root_main_modified": False,
    }
    (artifact_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(
        README + f"\nCandidate: {name}\n", encoding="utf-8"
    )
    return manifest


def build() -> dict:
    base_source, source_sha = _decode_notebook_agent(SOURCE_NOTEBOOK)
    marker = "\ndef agent(obs):"
    if base_source.count(marker) != 1:
        raise RuntimeError("Expected exactly one public v22 agent definition")
    base_source = base_source.replace(marker, "\ndef _v028_v22_original_agent(obs):", 1)
    manifests = {
        name: _write_candidate(name, variant, margin, base_source, source_sha)
        for name, variant, margin in CANDIDATES
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "README.md").write_text(README, encoding="utf-8")
    (ARTIFACT_ROOT / "build_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifests


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
