"""Build V025: V024a's 14-hands route plus the V022c market overlay."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import tarfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "baseline/history/v024a_route14_control/main.py"
CANDIDATES = (
    ("v025a_route14_v022c_market", "gate"),
    ("v025b_route14_v022c_open_market", "open"),
    ("v025c_route14_v022c_mirror_market", "mirror"),
)


RUNTIME_SUFFIX = r'''
"""V025 runtime: V024a route with a V022c-compatible market-only overlay."""
import base64 as _v025_base64
import copy as _v025_copy
import json as _v025_json
import math as _v025_math
import zlib as _v025_zlib

_V025_PAYLOAD = _v025_json.loads(_v025_zlib.decompress(
    _v025_base64.b85decode("".join(_V025_B85_PARTS))
).decode("utf-8"))
_V025_ACTIONS = _V025_PAYLOAD.get("actions", []) or []
_V025_VARIANT = "__V025_VARIANT__"
_V025_EPISODE_STEPS = int(_V025_PAYLOAD.get("episode_steps", 720) or 720)
_V025_SELLABLE = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER")
_V025_PRODUCT_BY_ANIMAL = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
_V025_GLUT_WEIGHT = {
    "STRAWBERRY": 2.0, "MELON": 3.6, "MILK": 2.0, "WOOL": 3.2,
    "EGG": 1.5, "TOMATO": 1.3, "CARROT": 1.0, "WHEAT": 1.0,
    "FERTILIZER": 1.0,
}
_V025_STATE = {
    0: {"last_step": -1, "board_streak": 0, "latched": False, "divergence": 0, "mode": None},
    1: {"last_step": -1, "board_streak": 0, "latched": False, "divergence": 0, "mode": None},
}
_V025_STATS = {
    "market_overlay_calls": 0, "market_overlay_active": 0,
    "market_reorders": 0, "market_sell_units": 0,
    "mirror_latches": 0, "mirror_releases": 0,
    "terminal_liquidations": 0, "sell_clipped": 0, "errors": 0,
}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _copy_action(action):
    action = _v025_copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or []) if isinstance(item, list) and item],
    }


def _align_hands(action, obs):
    action = _copy_action(action)
    seat = _seat(obs)
    farms = _get(obs, "farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    expected = len(_get(farm, "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(item or ["PASS"]) for item in hands[:expected]]
    return action


def _public_farm_distance(left, right):
    distance = 0
    if list(_get(left, "farmer", []) or []) != list(_get(right, "farmer", []) or []):
        distance += 2
    left_hands = [tuple(item or ()) for item in (_get(left, "hands", []) or [])]
    right_hands = [tuple(item or ()) for item in (_get(right, "hands", []) or [])]
    distance += 3 * abs(len(left_hands) - len(right_hands))
    distance += sum(a != b for a, b in zip(left_hands, right_hands))
    distance += 4 * len(set(_get(left, "unlocked_quadrants", []) or []) ^ set(_get(right, "unlocked_quadrants", []) or []))
    left_tiles = _get(left, "tiles", []) or []
    right_tiles = _get(right, "tiles", []) or []
    for y in range(max(len(left_tiles), len(right_tiles))):
        left_row = left_tiles[y] if y < len(left_tiles) else []
        right_row = right_tiles[y] if y < len(right_tiles) else []
        for x in range(max(len(left_row), len(right_row))):
            a = left_row[x] if x < len(left_row) else "MISSING"
            b = right_row[x] if x < len(right_row) else "MISSING"
            ta = (a.get("kind"), a.get("crop"), a.get("animal"), a.get("yield_units")) if isinstance(a, dict) else a
            tb = (b.get("kind"), b.get("crop"), b.get("animal"), b.get("yield_units")) if isinstance(b, dict) else b
            distance += ta != tb
    return distance


def _mirror_probability(distance, money_gap, board_streak, step):
    means = (0.7754330004241974, 0.42216157898077045, 4.845066681290341, 0.2690391571999746, 0.649190802764908, 0.5422947717626129, 0.38474963508282034)
    scales = (0.9129241235454895, 0.4939040192317694, 3.256390917597153, 0.44346035797258343, 0.4421957956795104, 0.27905760741549984, 0.4865360761410145)
    coefficients = (-0.9451280735752147, -0.4680272533738784, -1.4615973335974357, 0.599421907433265, -0.1545610769891183, 0.44222011734216526, -0.025622914463126638)
    values = (
        _v025_math.log1p(max(0.0, distance)), float(distance == 0),
        _v025_math.log1p(max(0.0, money_gap)), float(money_gap <= 5.0),
        min(max(0, board_streak), 96) / 96.0, min(max(0, step), 718) / 718.0,
        float(step >= 480),
    )
    logit = 1.245742223898873 + sum(
        coefficient * ((value - mean) / scale)
        for value, mean, scale, coefficient in zip(values, means, scales, coefficients)
    )
    return 1.0 / (1.0 + _v025_math.exp(-min(35.0, max(-35.0, logit))))


def _shed_access(size):
    half = size // 2
    return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


def _projected_shed(obs, action):
    seat = _seat(obs)
    farms = _get(obs, "farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    private = _get(obs, "private", {}) or {}
    projected = {key: max(0, int(value or 0)) for key, value in dict(_get(private, "shed", {}) or {}).items()}
    inventories = _get(private, "inventories", []) or []
    positions = [_get(farm, "farmer", [0, 0]), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])]
    tiles = _get(farm, "tiles", []) or []
    access = _shed_access(len(tiles) or 10)
    for index, unit_action in enumerate(unit_actions):
        if index >= len(positions) or index >= len(inventories):
            continue
        position = positions[index]
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        x, y = int(position[0]), int(position[1])
        if (x, y) not in access or not isinstance(unit_action, list) or not unit_action:
            continue
        inventory = {key: max(0, int(value or 0)) for key, value in dict(inventories[index] or {}).items()}
        if unit_action[0] == "DROP":
            deposits = inventory.items()
        else:
            continue
        for item, quantity in deposits:
            room = max(0, 100 - sum(projected.values()))
            amount = min(max(0, int(quantity or 0)), room)
            if amount:
                projected[item] = projected.get(item, 0) + amount
    return projected


def _opponent_exposure(obs):
    seat = _seat(obs)
    farms = _get(obs, "farms", []) or []
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    exposure = {item: 0.0 for item in _V025_SELLABLE}
    for row in _get(opponent, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop", "")).upper()
            if crop in exposure:
                exposure[crop] += max(1.0, float(tile.get("yield_units", 0) or 0))
            product = _V025_PRODUCT_BY_ANIMAL.get(str(tile.get("animal", "")).upper())
            if product:
                exposure[product] += 1.0 + max(0.0, float(tile.get("yield_units", 0) or 0))
    return exposure


def _ranked_sells(obs, action, requested=None):
    shed = _projected_shed(obs, action)
    if requested is None:
        requested = {item: int(shed.get(item, 0) or 0) for item in _V025_SELLABLE}
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    exposure = _opponent_exposure(obs)
    rows = []
    for index, item in enumerate(_V025_SELLABLE):
        quantity = min(max(0, int(requested.get(item, 0) or 0)), max(0, int(shed.get(item, 0) or 0)))
        if quantity <= 0:
            continue
        score = (1.0 + exposure.get(item, 0.0)) * _V025_GLUT_WEIGHT.get(item, 1.0) * max(1.0, float(prices.get(item, 1) or 1)) * _v025_math.log1p(quantity)
        rows.append((score, -index, item, quantity))
    rows.sort(reverse=True)
    return [["SELL", item, quantity] for _, _, item, quantity in rows]


def _front_run_market(obs, action):
    market = [list(order) for order in action.get("market", []) or []]
    requests = {}
    for order in market:
        if len(order) >= 3 and str(order[0]).upper() == "SELL":
            try:
                requests[str(order[1]).upper()] = requests.get(str(order[1]).upper(), 0) + max(0, int(order[2]))
            except (TypeError, ValueError):
                continue
    sells = _ranked_sells(obs, action, requests)
    targeted = {order[1] for order in sells}
    remainder = [order for order in market if not (len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() in targeted)]
    updated = (sells + remainder)[:10]
    if updated != market[:10]:
        _V025_STATS["market_reorders"] += 1
    _V025_STATS["market_sell_units"] += sum(int(order[2]) for order in updated if len(order) >= 3 and str(order[0]).upper() == "SELL")
    action["market"] = updated
    return action


def _hybrid_market(obs, action, step):
    if step >= 718:
        return action
    seat = _seat(obs)
    state = _V025_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "board_streak": 0, "latched": False, "divergence": 0, "mode": None})
    state["last_step"] = step
    farms = _get(obs, "farms", []) or []
    distance = _public_farm_distance(farms[0], farms[1]) if len(farms) >= 2 else _v025_math.inf
    money_gap = abs(float(_get(farms[0], "money", 0) or 0) - float(_get(farms[1], "money", 0) or 0)) if len(farms) >= 2 else _v025_math.inf
    state["board_streak"] = state["board_streak"] + 1 if distance <= 2 else 0
    probability = _mirror_probability(distance, money_gap, state["board_streak"], step)
    selected = probability >= 0.8065185529227787
    if selected and not state["latched"]:
        state["latched"] = True
        _V025_STATS["mirror_latches"] += 1
    if state["latched"]:
        state["divergence"] = 0 if distance <= 2 else state["divergence"] + 1
        if state["divergence"] >= 8:
            state["latched"] = False
            state["divergence"] = 0
            _V025_STATS["mirror_releases"] += 1
    if state["mode"] is None and step >= 48:
        state["mode"] = "mirror" if selected else "open"
    if _V025_VARIANT == "open":
        active = step >= 24
    elif _V025_VARIANT == "mirror":
        active = step >= 24 and state["latched"]
    else:
        active = step >= 24 and (state["latched"] if state["mode"] == "mirror" else state["mode"] == "open")
    _V025_STATS["market_overlay_calls"] += 1
    if active:
        _V025_STATS["market_overlay_active"] += 1
        action = _front_run_market(obs, action)
    return action


def _sanitize_market(obs, action, terminal=False):
    action = _copy_action(action)
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    if terminal:
        prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
        remaining = {str(item).upper(): max(0, int(value or 0)) for item, value in shed.items()}
        action["market"] = []
        for item in sorted(_V025_SELLABLE, key=lambda key: (-int(prices.get(key, 0) or 0), key)):
            if len(action["market"]) >= 10:
                break
            if remaining.get(item, 0) > 0:
                action["market"].append(["SELL", item, remaining[item]])
        _V025_STATS["terminal_liquidations"] += 1
        return action
    remaining = {str(item).upper(): max(0, int(value or 0)) for item, value in shed.items()}
    output = []
    for order in action.get("market", []) or []:
        if len(output) >= 10:
            break
        if len(order) < 3 or str(order[0]).upper() != "SELL":
            output.append(list(order))
            continue
        item = str(order[1]).upper()
        try:
            requested = max(0, int(order[2]))
        except (TypeError, ValueError):
            requested = 0
        allowed = min(requested, remaining.get(item, 0))
        if allowed < requested:
            _V025_STATS["sell_clipped"] += 1
        if allowed > 0:
            remaining[item] = remaining.get(item, 0) - allowed
            output.append(["SELL", item, allowed])
    action["market"] = output[:10]
    return action


def agent(obs):
    try:
        step = max(0, min(_V025_EPISODE_STEPS - 1, int(_get(obs, "step", 0) or 0)))
        base = _copy_action(_V025_ACTIONS[min(step, len(_V025_ACTIONS) - 1)] if _V025_ACTIONS else {})
        base = _align_hands(base, obs)
        base = _hybrid_market(obs, base, step)
        return _sanitize_market(obs, base, terminal=step >= 718)
    except Exception as exc:
        _V025_STATS["errors"] += 1
        _V025_STATS["last_error"] = repr(exc)
        seat = _seat(obs if isinstance(obs, dict) else {})
        farms = (_get(obs, "farms", []) if isinstance(obs, dict) else []) or []
        farm = farms[seat] if seat < len(farms) else {}
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])], "market": []}
'''


def _load_payload():
    spec = importlib.util.spec_from_file_location("v024_route_payload", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module._V024_PAYLOAD)


def _encode(payload):
    encoded = base64.b85encode(zlib.compress(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")
    parts = [encoded[index:index + 120] for index in range(0, len(encoded), 120)]
    return "_V025_B85_PARTS = " + repr(parts) + "\n"


def build():
    payload = _load_payload()
    source_payload = {"version": "v025-route14-v022c-market", "episode_steps": payload.get("episode_steps", 720), "actions": payload.get("actions", [])}
    candidates = []
    for name, variant in CANDIDATES:
        source = _encode(source_payload) + RUNTIME_SUFFIX.replace("__V025_VARIANT__", variant)
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
            "parent": "v024a_route14_control", "route_actions": len(source_payload["actions"]),
        }
        (artifact / "submission_manifest.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        candidates.append(item)
    common = ROOT / "baseline/artifacts/v025_route14_v022c_market"
    common.mkdir(parents=True, exist_ok=True)
    (common / "README.md").write_text(
        "# V025 route14 + V022c market graft\n\n"
        "V025 keeps V024a's complete 14-hands production route and replaces only "
        "the market lane with the V022c public-distance / mirror-gated ranked SELL "
        "overlay. It does not import V022c's farmer, hand, crop, or livestock route.\n\n"
        "- `v025a_route14_v022c_market`: V022c's mirror/open hysteresis gate.\n"
        "- `v025b_route14_v022c_open_market`: always-on market-only ablation after step 24.\n"
        "- `v025c_route14_v022c_mirror_market`: mirror-only ablation.\n\n"
        "All archives contain a self-contained root `main.py`; root `main.py` is unchanged.\n",
        encoding="utf-8",
    )
    manifest = {"version": "v025", "parent": "v024a_route14_control", "candidates": candidates}
    (common / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
