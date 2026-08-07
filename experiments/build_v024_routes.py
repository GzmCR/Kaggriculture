"""Build V024's anonymous 14-hands route and four self-contained agents.

Only the older 70% of the 2026-08-07 Top10 episodes is used to choose the
route medoid and its optional public-state memory.  The generated files do
not contain replay ids, names, scores, seeds, or a path to the data folder.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
import zlib
from pathlib import Path

from v024_route14 import build_memory, load_top10, profile, select_medoid, split_records


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
CANDIDATES = (
    ("v024a_route14_control", "base"),
    ("v024b_route14_weed", "weed"),
    ("v024c_route14_order_memory", "memory"),
    ("v024d_route14_strict_r3", "strict"),
)


RUNTIME_SUFFIX = r'''
"""V024 runtime: anonymous 14-hands route with optional bounded overlays."""
import base64 as _v024_base64
import copy as _v024_copy
import json as _v024_json
import math as _v024_math
import zlib as _v024_zlib

_V024_PAYLOAD = _v024_json.loads(_v024_zlib.decompress(
    _v024_base64.b85decode("".join(_V024_B85_PARTS))
).decode("utf-8"))
_V024_EPISODE_STEPS = int(_V024_PAYLOAD.get("episode_steps", 720) or 720)
_V024_ACTIONS = _V024_PAYLOAD.get("actions", []) or []
_V024_MEMORY = _V024_PAYLOAD.get("memory", []) or []
_V024_VARIANT = "__V024_VARIANT__"
_V024_PREMIUM = {"MILK", "WOOL", "STRAWBERRY", "MELON"}
_V024_SELLABLE = ("MILK", "WOOL", "STRAWBERRY", "MELON", "WHEAT", "EGG", "TOMATO", "CARROT", "FERTILIZER")
_V024_STATS = {
    "weed_repairs": 0, "weed_retries": 0, "weed_catchup_actions": 0,
    "weed_abandoned": 0, "order_memory_hits": 0, "order_memory_reorders": 0,
    "strict_r3_hits": 0, "strict_r3_units": 0, "strict_r3_repayments": 0,
    "terminal_liquidations": 0, "sell_clipped": 0, "errors": 0,
}
_V024_STATE = {
    0: {"last_step": -1, "active": {}, "suppressed": {}, "cooldown": {}},
    1: {"last_step": -1, "active": {}, "suppressed": {}, "cooldown": {}},
}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _copy_action(action):
    action = _v024_copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or []) if isinstance(item, list) and item],
    }


def _align_hands(action, obs):
    action = _copy_action(action)
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[seat] if seat < len(farms) else {}
    expected = len(_get(farm, "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(item or ["PASS"]) for item in hands[:expected]]
    return action


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError, KeyError):
        return "LOCKED"


def _trace_actor(step, actor):
    trace = _V024_ACTIONS[min(max(int(step), 0), len(_V024_ACTIONS) - 1)] if _V024_ACTIONS else {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _expected_created(tile, intended):
    if not isinstance(tile, dict):
        return False
    if intended[0] == "BUILD_PASTURE":
        return str(tile.get("kind", "")).upper() == "PASTURE"
    if intended[0] == "PLANT":
        return (
            str(tile.get("kind", "")).upper() == "PLANT"
            and str(tile.get("crop", "")).upper() == str(intended[1]).upper()
        )
    return False


def _weed_overlay(obs, action, step):
    """One actor-local DIG/retry transaction; V024a intentionally skips it."""
    if _V024_VARIANT == "base":
        return action
    action = _align_hands(action, obs)
    seat = _seat(obs)
    state = _V024_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "active": {}, "suppressed": {}, "cooldown": {}})
    state["last_step"] = step
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[seat] if seat < len(farms) else {}
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [list(action.get("farmer") or ["PASS"]), *[list(item or ["PASS"]) for item in action.get("hands", []) or []]]
    active = state.setdefault("active", {})
    suppressed = state.setdefault("suppressed", {})

    for actor, txn in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions) or index >= len(positions):
            active.pop(actor, None)
            continue
        age = step - int(txn["start"])
        tile = _tile_at(farm, positions[index])
        if age == 1:
            unit_actions[index] = list(txn["intended"])
            _V024_STATS["weed_retries"] += 1
            continue
        if age >= 2 and _expected_created(tile, txn["intended"]):
            current = unit_actions[index]
            if current and current[0] in {"PASS", "WATER", "FEED", "CARE", "COLLECT_FERTILIZER"}:
                active.pop(actor, None)
                continue
        if 2 <= age <= 8:
            unit_actions[index] = _trace_actor(step - 1, actor)
            _V024_STATS["weed_catchup_actions"] += 1
        elif age > 8:
            active.pop(actor, None)
            suppressed[txn["key"]] = step + 8
            _V024_STATS["weed_abandoned"] += 1

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(position, (list, tuple)):
            continue
        if not isinstance(intended, list) or not intended or intended[0] not in {"PLANT", "BUILD_PASTURE"}:
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or str(tile.get("kind", "")).upper() != "WEED":
            continue
        key = (actor, int(position[0]), int(position[1]))
        if step < int(suppressed.get(key, -1)):
            continue
        active[actor] = {
            "start": step, "intended": list(intended), "key": key,
        }
        unit_actions[index] = ["DIG"]
        _V024_STATS["weed_repairs"] += 1
    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _signature(observation, seat):
    farms = _get(observation, "farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    counts = {}
    for row in _get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            token = str(tile.get("crop") or tile.get("animal") or tile.get("kind") or "").upper()
            if token:
                counts[token] = counts.get(token, 0) + 1
    return {
        "money": int(float(_get(farm, "money", 0) or 0) // 250),
        "farmer": list(_get(farm, "farmer", [0, 0]) or [0, 0]),
        "hands": [list(item) for item in (_get(farm, "hands", []) or [])],
        "unlocked": sorted(str(item) for item in (_get(farm, "unlocked_quadrants", []) or [])),
        "tiles": counts,
    }


def _signature_distance(left, right):
    distance = abs(int(left.get("money", 0)) - int(right.get("money", 0))) / 4.0
    distance += 2 * abs(len(left.get("hands", []) or []) - len(right.get("hands", []) or []))
    distance += sum(a != b for a, b in zip(left.get("farmer", []), right.get("farmer", [])))
    distance += sum(a != b for a, b in zip(left.get("hands", []) or [], right.get("hands", []) or []))
    distance += 3 * len(set(left.get("unlocked", []) or []) ^ set(right.get("unlocked", []) or []))
    keys = set((left.get("tiles", {}) or {})) | set((right.get("tiles", {}) or {}))
    for key in keys:
        distance += abs(int((left.get("tiles", {}) or {}).get(key, 0)) - int((right.get("tiles", {}) or {}).get(key, 0)))
    return distance


def _order_memory(obs, action, step):
    if _V024_VARIANT not in {"memory", "strict"} or step >= 672:
        return action, False
    if not _V024_MEMORY:
        return action, False
    seat = _seat(obs)
    rows = [row for row in _V024_MEMORY if int(row.get("step", -1)) == step]
    if not rows:
        return action, False
    opponent = 1 - seat
    current = _signature(obs, opponent)
    row = min(rows, key=lambda item: _signature_distance(current, item.get("signature", {})))
    if _signature_distance(current, row.get("signature", {})) > 10.0:
        return action, False
    predicted = list(row.get("sell_order", []) or [])
    if not predicted:
        return action, False
    premium_positions = []
    premium_orders = []
    for index, order in enumerate(action.get("market", []) or []):
        if len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() in _V024_PREMIUM:
            premium_positions.append(index)
            premium_orders.append(list(order))
    if not premium_orders:
        return action, False
    rank = {item: index for index, item in enumerate(predicted)}
    premium_orders.sort(key=lambda order: (rank.get(str(order[1]).upper(), 999), premium_positions[premium_orders.index(order)] if order in premium_orders else 0))
    changed = any(action["market"][position] != order for position, order in zip(premium_positions, premium_orders))
    if changed:
        for position, order in zip(premium_positions, premium_orders):
            action["market"][position] = order
        _V024_STATS["order_memory_hits"] += 1
        _V024_STATS["order_memory_reorders"] += 1
    return action, changed


_V024_MODEL_MEAN = (0.7754330004241974, 0.42216157898077045, 4.845066681290341, 0.2690391571999746, 0.649190802764908, 0.5422947717626129, 0.38474963508282034)
_V024_MODEL_SCALE = (0.9129241235454895, 0.4939040192317694, 3.256390917597153, 0.44346035797258343, 0.4421957956795104, 0.27905760741549984, 0.4865360761410145)
_V024_MODEL_COEFFICIENTS = (-0.9451280735752147, -0.4680272533738784, -1.4615973335974357, 0.599421907433265, -0.1545610769891183, 0.44222011734216526, -0.025622914463126638)
_V024_MODEL_INTERCEPT = 1.245742223898873
_V024_MODEL_THRESHOLD = 0.8065185529227787


def _farm_distance(left, right):
    distance = 0
    if list(_get(left, "farmer", []) or []) != list(_get(right, "farmer", []) or []):
        distance += 2
    lh = [tuple(item or ()) for item in (_get(left, "hands", []) or [])]
    rh = [tuple(item or ()) for item in (_get(right, "hands", []) or [])]
    distance += 3 * abs(len(lh) - len(rh)) + sum(a != b for a, b in zip(lh, rh))
    distance += 4 * len(set(_get(left, "unlocked_quadrants", []) or []) ^ set(_get(right, "unlocked_quadrants", []) or []))
    lt = _get(left, "tiles", []) or []
    rt = _get(right, "tiles", []) or []
    for y in range(max(len(lt), len(rt))):
        arow = lt[y] if y < len(lt) else []
        brow = rt[y] if y < len(rt) else []
        for x in range(max(len(arow), len(brow))):
            ta = arow[x] if x < len(arow) else "MISSING"
            tb = brow[x] if x < len(brow) else "MISSING"
            ka = (ta.get("kind"), ta.get("crop"), ta.get("animal"), ta.get("yield_units")) if isinstance(ta, dict) else ta
            kb = (tb.get("kind"), tb.get("crop"), tb.get("animal"), tb.get("yield_units")) if isinstance(tb, dict) else tb
            distance += ka != kb
    return distance


def _mirror_probability(distance, money_gap, streak, step):
    values = (
        _v024_math.log1p(max(0.0, distance)), float(distance == 0),
        _v024_math.log1p(max(0.0, money_gap)), float(money_gap <= 5.0),
        min(max(0, streak), 96) / 96.0, min(max(0, step), 718) / 718.0,
        float(step >= 480),
    )
    logit = _V024_MODEL_INTERCEPT + sum(
        coefficient * ((value - mean) / scale)
        for value, mean, scale, coefficient in zip(values, _V024_MODEL_MEAN, _V024_MODEL_SCALE, _V024_MODEL_COEFFICIENTS)
    )
    return 1.0 / (1.0 + _v024_math.exp(-min(35.0, max(-35.0, logit))))


def _shed_count(obs, item):
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    return max(0, int(shed.get(item, 0) or 0))


def _strict_r3(obs, action, step, memory_changed):
    if _V024_VARIANT != "strict" or memory_changed or not (24 <= step < 672):
        return action
    farms = _get(obs, "farms", []) or []
    if len(farms) < 2:
        return action
    distance = _farm_distance(farms[0], farms[1])
    money_gap = abs(float(_get(farms[0], "money", 0) or 0) - float(_get(farms[1], "money", 0) or 0))
    state = _V024_STATE[_seat(obs)]
    streak = int(state.get("mirror_streak", 0) or 0)
    state["mirror_streak"] = streak + 1 if distance <= 2 and money_gap <= 5 else 0
    probability = _mirror_probability(distance, money_gap, state["mirror_streak"], step)
    if probability < _V024_MODEL_THRESHOLD or distance > 2 or money_gap > 5:
        return action
    next_action = _copy_action(_V024_ACTIONS[min(step + 1, len(_V024_ACTIONS) - 1)]) if _V024_ACTIONS else {}
    existing_items = {str(order[1]).upper() for order in action.get("market", []) if len(order) >= 3 and str(order[0]).upper() == "SELL"}
    if len(action.get("market", []) or []) >= 10:
        return action
    cooldown = state.setdefault("cooldown", {})
    candidates = []
    for order in next_action.get("market", []) or []:
        if len(order) < 3 or str(order[0]).upper() != "SELL":
            continue
        item = str(order[1]).upper()
        if item not in _V024_PREMIUM or item in existing_items or step < int(cooldown.get(item, -1)):
            continue
        try:
            quantity = max(0, int(order[2]))
        except (TypeError, ValueError):
            quantity = 0
        shift = min(quantity // 2, _shed_count(obs, item))
        if shift > 0:
            candidates.append((item, shift))
    if not candidates:
        return action
    item, shift = candidates[0]
    action["market"].append(["SELL", item, shift])
    cooldown[item] = step + 8
    state["repay"] = {"step": step + 1, "item": item, "quantity": shift}
    _V024_STATS["strict_r3_hits"] += 1
    _V024_STATS["strict_r3_units"] += shift
    return action


def _apply_repayment(action, obs, step):
    state = _V024_STATE[_seat(obs)]
    repay = state.get("repay")
    if not repay or int(repay.get("step", -1)) != step:
        return action
    item = repay.get("item")
    quantity = int(repay.get("quantity", 0) or 0)
    for order in action.get("market", []) or []:
        if len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == item:
            order[2] = max(0, int(order[2] or 0) - quantity)
            _V024_STATS["strict_r3_repayments"] += quantity
            break
    state.pop("repay", None)
    return action


def _sanitize_market(obs, action, terminal=False):
    action = _copy_action(action)
    if terminal:
        shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
        prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
        remaining = {str(item).upper(): max(0, int(value or 0)) for item, value in shed.items()}
        orders = []
        for item in sorted(_V024_SELLABLE, key=lambda key: (-int(prices.get(key, 0) or 0), key)):
            if len(orders) >= 10:
                break
            quantity = remaining.get(item, 0)
            if quantity > 0:
                orders.append(["SELL", item, quantity])
        _V024_STATS["terminal_liquidations"] += 1
        action["market"] = orders
        return action
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    remaining = {str(item).upper(): max(0, int(value or 0)) for item, value in shed.items()}
    output = []
    for order in action.get("market", []) or []:
        if len(output) >= 10:
            break
        if str(order[0]).upper() != "SELL" or len(order) < 3:
            output.append(list(order))
            continue
        item = str(order[1]).upper()
        try:
            requested = max(0, int(order[2]))
        except (TypeError, ValueError):
            requested = 0
        allowed = min(requested, remaining.get(item, 0))
        if allowed < requested:
            _V024_STATS["sell_clipped"] += 1
        if allowed > 0:
            remaining[item] = remaining.get(item, 0) - allowed
            output.append(["SELL", item, allowed])
    action["market"] = output[:10]
    return action


def agent(obs):
    try:
        step = max(0, min(_V024_EPISODE_STEPS - 1, int(_get(obs, "step", 0) or 0)))
        seat = _seat(obs)
        state = _V024_STATE[seat]
        if step == 0 or step < int(state.get("last_step", -1)):
            state.clear()
            state.update({"last_step": -1, "active": {}, "suppressed": {}, "cooldown": {}, "mirror_streak": 0})
        state["last_step"] = step
        base = _copy_action(_V024_ACTIONS[min(step, len(_V024_ACTIONS) - 1)] if _V024_ACTIONS else {})
        base = _align_hands(base, obs)
        base = _apply_repayment(base, obs, step)
        base = _weed_overlay(obs, base, step)
        base = _align_hands(base, obs)
        base, memory_changed = _order_memory(obs, base, step)
        base = _strict_r3(obs, base, step, memory_changed)
        return _sanitize_market(obs, base, terminal=step >= 718)
    except Exception as exc:
        _V024_STATS["errors"] += 1
        _V024_STATS["last_error"] = repr(exc)
        seat = _seat(obs if isinstance(obs, dict) else {})
        farms = (_get(obs, "farms", []) if isinstance(obs, dict) else []) or []
        farm = farms[seat] if seat < len(farms) else {}
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])], "market": []}
'''


def _encode_payload(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.b85encode(zlib.compress(raw, 9)).decode("ascii")
    parts = [encoded[index:index + 120] for index in range(0, len(encoded), 120)]
    return "_V024_B85_PARTS = " + repr(parts) + "\n"


def _write_candidate(name: str, variant: str, payload: dict, manifest: dict) -> dict:
    source = _encode_payload(payload) + RUNTIME_SUFFIX.replace("__V024_VARIANT__", variant)
    history = ROOT / "baseline/history" / name
    artifact = ROOT / "baseline/artifacts" / name
    history.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    (history / "main.py").write_text(source, encoding="utf-8")
    (artifact / "main.py").write_text(source, encoding="utf-8")
    archive = artifact / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(artifact / "main.py", arcname="main.py")
    item = {
        "candidate": name, "variant": variant,
        "main_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "main_bytes": len(source.encode()), "archive": str(archive.relative_to(ROOT)),
        **manifest,
    }
    (artifact / "submission_manifest.json").write_text(json.dumps(item, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return item


def _find_data_root(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else []
    candidates += [ROOT / "log/2026-08-07", Path("/Users/guoziming/Desktop/比赛/kaggriculture/log/2026-08-07")]
    for path in candidates:
        if path and (path / "top10").is_dir():
            return path
    raise FileNotFoundError("expected log/2026-08-07/top10")


def build(data_root: Path | None = None) -> dict:
    data_root = _find_data_root(data_root)
    records, audit = load_top10(data_root)
    try:
        audit["folder"] = str(Path(audit["folder"]).relative_to(ROOT))
    except (KeyError, ValueError):
        pass
    fit, validation, future = split_records(records)
    medoid = select_medoid(fit)
    payload = {
        "version": "v024-route14",
        "episode_steps": EPISODE_STEPS,
        "actions": medoid.actions,
        "memory": build_memory(medoid, medoid.seat),
        "route_features": {
            key: value for key, value in medoid.features.items()
            if key in {"max_hands", "plants", "wheat", "strawberry", "melon", "ne_day", "sw_day", "cows", "sheep", "water", "harvest", "feed", "fertilize"}
        },
    }
    manifest = {
        "version": "v024",
        "data_root": str(data_root.relative_to(ROOT)) if data_root.is_relative_to(ROOT) else str(data_root),
        "audit": audit,
        "fit_records": len(fit), "validation_records": len(validation), "future_holdout_records": len(future),
        "fit_episodes": len({record.episode for record in fit}),
        "validation_episodes": len({record.episode for record in validation}),
        "future_holdout_episodes": len({record.episode for record in future}),
        "selected_medoid": profile(medoid),
        "fit_summary": {
            "max_hands_14_records": sum(record.features.get("max_hands", 0) == 14 for record in fit),
            "mean_plants": sum(record.features.get("plants", 0) for record in fit) / max(1, len(fit)),
            "mean_harvest": sum(record.features.get("harvest", 0) for record in fit) / max(1, len(fit)),
        },
        "runtime_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "candidates": [],
    }
    for name, variant in CANDIDATES:
        manifest["candidates"].append(_write_candidate(name, variant, payload, {
            "route_features": payload["route_features"],
            "fit_episodes": manifest["fit_episodes"],
            "future_holdout_episodes": manifest["future_holdout_episodes"],
        }))
    common = ROOT / "baseline/artifacts/v024_14hands_route"
    common.mkdir(parents=True, exist_ok=True)
    (common / "route_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (common / "README.md").write_text(
        "# V024 14-hands route\n\n"
        "The four candidates use one anonymous complete route medoid selected "
        "from the chronological 70% fit window of the 2026-08-07 Top10 data. "
        "The middle 15% is option validation and the newest 15% is future holdout. "
        "The 1500-2500 and `ours` folders are not used for fitting.\n\n"
        "- `v024a_route14_control`: route plus legal market clipping and terminal liquidation.\n"
        "- `v024b_route14_weed`: adds one actor-local DIG/retry recovery.\n"
        "- `v024c_route14_order_memory`: adds high-confidence order-only memory.\n"
        "- `v024d_route14_strict_r3`: adds the strict, cooldown-limited R3 front-run gate.\n\n"
        "All submissions are self-contained and contain `main.py` at archive root. "
        "No candidate changes the repository root `main.py`.\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    result = build(args.data_root)
    print(json.dumps({
        "unique_episodes": result["audit"]["unique_episodes"],
        "unique_seats": result["audit"]["unique_seats"],
        "fit_episodes": result["fit_episodes"],
        "validation_episodes": result["validation_episodes"],
        "future_holdout_episodes": result["future_holdout_episodes"],
        "medoid": result["selected_medoid"]["features"],
        "candidates": [item["candidate"] for item in result["candidates"]],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
