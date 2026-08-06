"""Build V023 anonymous route candidates from the 2026-08-06 replays.

The replay files are read only by this offline builder.  The generated agents
contain compressed action routes and route checkpoints, but no episode ids,
team names, score labels, file names, seeds, notebooks, or network calls.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import tarfile
import textwrap
import zlib
from pathlib import Path

from v023_route_clusters import EPISODE_STEPS, MACRO_KEYS, RouteRecord, cluster_profiles, load_records, serializable_profile


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "v023a_high_output_14hands",
    "v023b_stable_12hands",
    "v023c_high_hands_15hands",
    "v023d_early_portfolio",
)
LABELS = ("high_output_14hands", "stable_12hands", "high_hands_15hands")
TARGETS = {
    "high_output_14hands": (14, 158, 42, 24),
    "stable_12hands": (12, 131, 44, 21),
    "high_hands_15hands": (15, 135, 41, 26),
}

README = """# V023 route portfolio

V023 is a route-refresh experiment built offline from the 2026-08-06 Top10
replays.  The Top10 folder is deduplicated by EpisodeId and clustered into
three anonymous macro-route medoids:

- `v023a_high_output_14hands`: high-output 14-hands route;
- `v023b_stable_12hands`: stable 12-hands route;
- `v023c_high_hands_15hands`: 15-hands research route with a low-confidence
  penalty in the portfolio selector;
- `v023d_early_portfolio`: common bootstrap for 48 turns, then one selection
  from the three routes using the agent's own public farm state.  The choice
  is locked through the end of the season.

Every generated submission is self-contained with `main.py` at archive root.
Runtime code does not read replay files, notebooks, team names, scores,
episode ids, seeds, or network resources.  It preserves selected field and
market actions, clips illegal SELL quantities, aligns hands dynamically, and
uses actor-local visible-WEED recovery plus terminal liquidation safeguards.

The `1500～2500` replay folder is held out by the benchmark and is not used to
build the runtime routes.  No candidate is promoted to the repository root by
this builder.
"""


def _find_data_root() -> Path:
    candidates = (
        ROOT / "log/2026-08-06",
        Path("/Users/guoziming/Desktop/比赛/kaggriculture/log/2026-08-06"),
    )
    for path in candidates:
        if (path / "top10").is_dir() and (path / "1500～2500").is_dir():
            return path
    raise FileNotFoundError("expected log/2026-08-06/top10 and log/2026-08-06/1500～2500")


def _episode_sort(records: list[RouteRecord]) -> list[RouteRecord]:
    return sorted(records, key=lambda r: (r.episode, r.seat, r.source_file))


def _split_by_episode(records: list[RouteRecord], fraction: float = 0.8) -> tuple[list[RouteRecord], list[RouteRecord]]:
    episodes = sorted({record.episode for record in records})
    cutoff = max(1, min(len(episodes), int(len(episodes) * fraction)))
    fit_episodes = set(episodes[:cutoff])
    return (
        _episode_sort([record for record in records if record.episode in fit_episodes]),
        _episode_sort([record for record in records if record.episode not in fit_episodes]),
    )


def _target_distance(record: RouteRecord, target: tuple[int, int, int, int]) -> float:
    values = (
        record.features.get("max_hands", 0),
        record.features.get("plants", 0),
        record.features.get("strawberry", 0),
        record.features.get("melon", 0),
    )
    scales = (2, 30, 12, 10)
    return sum(abs(float(value) - float(goal)) / scale for value, goal, scale in zip(values, target, scales))


def _nearest_profile(records: list[RouteRecord], label: str) -> dict:
    record = min(records, key=lambda item: (_target_distance(item, TARGETS[label]), item.episode, item.seat))
    return {
        "label": label,
        "support": 1,
        "medoid_episode": record.episode,
        "medoid_seat": record.seat,
        "medoid_source_file": record.source_file,
        "target_distance": _target_distance(record, TARGETS[label]),
        "feature_mean": {key: record.features.get(key, 0) for key in MACRO_KEYS},
        "medoid_features": record.features,
        "record": record,
    }


def _profiles_by_label(records: list[RouteRecord]) -> dict[str, dict]:
    profiles = cluster_profiles(records)
    result = {profile["label"]: profile for profile in profiles}
    for label in LABELS:
        if label not in result:
            result[label] = _nearest_profile(records, label)
    # The k-medoids implementation supplies support statistics, but a large
    # majority cluster can otherwise steal the 12-hands medoid.  Select the
    # actual route medoid by the requested macro target while retaining the
    # cluster support as an offline confidence signal.
    for label in LABELS:
        profile = result[label]
        record = min(records, key=lambda item: (_target_distance(item, TARGETS[label]), item.episode, item.seat))
        profile.update({
            "record": record,
            "medoid_episode": record.episode,
            "medoid_seat": record.seat,
            "medoid_source_file": record.source_file,
            "target_distance": _target_distance(record, TARGETS[label]),
            "medoid_features": record.features,
            "low_confidence": label == "high_hands_15hands" and record.features.get("max_hands", 0) < 15,
        })
    return result


def _json_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _majority(values: list, fallback):
    if not values:
        return fallback
    counts = collections.Counter(_json_key(value) for value in values)
    best = max(counts, key=lambda key: (counts[key], key))
    return next(value for value in values if _json_key(value) == best)


def _common_bootstrap(records: dict[str, RouteRecord], steps: int = 48) -> list[dict]:
    output = []
    for step in range(steps):
        actions = [records[label].actions[min(step, EPISODE_STEPS - 1)] for label in LABELS]
        # A majority vote can combine actions that only make sense at
        # different coordinates or inventory states.  Use consensus only when
        # all medoids agree; otherwise keep the high-output route's complete
        # action for this early, context-sensitive bootstrap.
        primary = records[LABELS[0]].actions[min(step, EPISODE_STEPS - 1)]
        farmers = [action.get("farmer", ["PASS"]) for action in actions]
        farmer = farmers[0] if all(_json_key(item) == _json_key(farmers[0]) for item in farmers[1:]) else primary.get("farmer", ["PASS"])
        max_hands = max(len(action.get("hands", []) or []) for action in actions)
        hands = []
        for index in range(max_hands):
            values = [
                (action.get("hands", []) or [])[index] if index < len(action.get("hands", []) or []) else ["PASS"]
                for action in actions
            ]
            primary_hands = primary.get("hands", []) or []
            fallback = primary_hands[index] if index < len(primary_hands) else ["PASS"]
            hands.append(values[0] if all(_json_key(item) == _json_key(values[0]) for item in values[1:]) else fallback)
        markets = [action.get("market", []) for action in actions]
        market = markets[0] if all(_json_key(item) == _json_key(markets[0]) for item in markets[1:]) else primary.get("market", [])
        output.append({"farmer": farmer, "hands": hands, "market": market})
    return output


def _runtime_payload(profiles: dict[str, dict]) -> tuple[dict, dict]:
    routes = {}
    manifest_profiles = {}
    records = {}
    for label in LABELS:
        profile = profiles[label]
        record = profile["record"]
        records[label] = record
        features = {
            key: value for key, value in record.features.items()
            if key in {
                "max_hands", "plants", "wheat", "strawberry", "melon", "carrot",
                "ne_day", "sw_day", "se_day", "cows", "sheep", "water", "harvest",
                "feed", "care", "fertilize",
            }
        }
        routes[label] = {
            "actions": record.actions,
            "checkpoint48": record.checkpoint48,
            "features": features,
            "support": int(profile.get("support", 0) or 0),
            "low_confidence": bool(profile.get("support", 0) < 6 or profile.get("low_confidence", False)),
        }
        manifest_profiles[label] = serializable_profile(profile)
    payload = {
        "version": "v023",
        "episode_steps": EPISODE_STEPS,
        "bootstrap_steps": 48,
        "routes": routes,
        "bootstrap": _common_bootstrap(records),
    }
    return payload, manifest_profiles


RUNTIME_SUFFIX = r'''

# V023 runtime: anonymous route execution, local legality guards, and one-shot
# early portfolio selection.  This suffix intentionally contains no replay
# identifiers or source-team metadata.
import base64 as _v023_base64
import copy as _v023_copy
import json as _v023_json
import zlib as _v023_zlib

_V023_PAYLOAD = _v023_json.loads(_v023_zlib.decompress(_v023_base64.b85decode("".join(_V023_B85_PARTS))).decode("utf-8"))
_V023_ROUTES = _V023_PAYLOAD["routes"]
_V023_BOOTSTRAP = _V023_PAYLOAD["bootstrap"]
_V023_VARIANT = "__V023_VARIANT__"
_V023_STATS = {
    "route_selected": "",
    "route_fallbacks": 0,
    "weed_repairs": 0,
    "weed_retries": 0,
    "weed_catchup_actions": 0,
    "terminal_liquidations": 0,
    "sell_clipped": 0,
}
_V023_STATE = {0: {"last_step": -1, "route": None, "active": {}}, 1: {"last_step": -1, "route": None, "active": {}}}
_V023_SELLABLE = ("MILK", "WOOL", "STRAWBERRY", "MELON", "WHEAT", "EGG", "TOMATO", "CARROT")


def _v023_copy_action(action):
    action = _v023_copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or []) if isinstance(item, list) and item],
    }


def _v023_seat(obs):
    return 1 if int(obs.get("player", 0) or 0) == 1 else 0


def _v023_align_hands(action, obs):
    action = _v023_copy_action(action)
    seat = _v023_seat(obs)
    farms = obs.get("farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    expected = len(farm.get("hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(item or ["PASS"]) for item in hands[:expected]]
    return action


def _v023_tile(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (farm.get("tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError, KeyError):
        return "LOCKED"


def _v023_route_action(route_name, step):
    route = _V023_ROUTES.get(route_name) or _V023_ROUTES["high_output_14hands"]
    actions = route.get("actions", []) or []
    if not actions:
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return _v023_copy_action(actions[min(max(int(step), 0), len(actions) - 1)])


def _v023_checkpoint(obs):
    seat = _v023_seat(obs)
    farms = obs.get("farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    crops = {}
    animals = {}
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            if tile.get("crop"):
                key = str(tile["crop"]).upper()
                crops[key] = crops.get(key, 0) + 1
            if tile.get("animal"):
                key = str(tile["animal"]).upper()
                animals[key] = animals.get(key, 0) + 1
    return {
        "money": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "unlocked": sorted(str(item) for item in farm.get("unlocked_quadrants", []) or []),
        "crops": crops,
        "animals": animals,
    }


def _v023_distance(checkpoint, target):
    unlocked = set(checkpoint.get("unlocked", []) or [])
    target_unlocked = set(target.get("unlocked", []) or [])
    crops = checkpoint.get("crops", {}) or {}
    target_crops = target.get("crops", {}) or {}
    animals = checkpoint.get("animals", {}) or {}
    target_animals = target.get("animals", {}) or {}
    distance = abs(int(checkpoint.get("hands", 0) or 0) - int(target.get("hands", 0) or 0)) * 1.5
    distance += abs(float(checkpoint.get("money", 0) or 0) - float(target.get("money", 0) or 0)) / 3000.0
    distance += len(unlocked ^ target_unlocked) * 2.0
    for item in ("WHEAT", "STRAWBERRY", "MELON", "TOMATO", "CARROT"):
        distance += abs(int(crops.get(item, 0) or 0) - int(target_crops.get(item, 0) or 0)) / 8.0
    for item in ("COW", "SHEEP", "GOOSE"):
        distance += abs(int(animals.get(item, 0) or 0) - int(target_animals.get(item, 0) or 0)) / 3.0
    return distance


def _v023_choose_route(obs):
    checkpoint = _v023_checkpoint(obs)
    ranked = []
    for name, route in _V023_ROUTES.items():
        score = _v023_distance(checkpoint, route.get("checkpoint48", {}))
        if route.get("low_confidence"):
            score += 1.5
        ranked.append((score, name))
    ranked.sort()
    best_score, best_name = ranked[0]
    # The high-output fallback is deliberately conservative when the current
    # state is not close to a fitted route or the margin is ambiguous.
    if _V023_VARIANT == "portfolio":
        margin = ranked[1][0] - best_score if len(ranked) > 1 else 99.0
        if best_score > 20.0 or margin < 0.5:
            best_name = "high_output_14hands"
            _V023_STATS["route_fallbacks"] += 1
    else:
        best_name = {
            "high_output": "high_output_14hands",
            "stable": "stable_12hands",
            "high_hands": "high_hands_15hands",
        }.get(_V023_VARIANT, "high_output_14hands")
    _V023_STATS["route_selected"] = best_name
    return best_name


def _v023_sanitize_market(action, obs, terminal=False):
    orders = []
    shed = (((obs.get("private", {}) or {}).get("shed", {}) or {}))
    remaining = {str(key).upper(): max(0, int(value or 0)) for key, value in shed.items()}
    for order in action.get("market", []) or []:
        if not isinstance(order, list) or not order:
            continue
        item = str(order[0]).upper()
        if item != "SELL" or len(order) < 3:
            orders.append(list(order))
            continue
        try:
            requested = max(0, int(order[2]))
        except (TypeError, ValueError):
            requested = 0
        allowed = min(requested, remaining.get(str(order[1]).upper(), 0))
        if allowed < requested:
            _V023_STATS["sell_clipped"] += 1
        if allowed > 0:
            product = str(order[1]).upper()
            remaining[product] = max(0, remaining.get(product, 0) - allowed)
            orders.append(["SELL", product, allowed])
    if terminal:
        price_map = ((obs.get("market", {}) or {}).get("prices", {}) or {})
        for product in sorted(_V023_SELLABLE, key=lambda item: (-int(price_map.get(item, 0) or 0), item)):
            if len(orders) >= 10:
                break
            quantity = remaining.get(product, 0)
            if quantity > 0:
                orders.append(["SELL", product, quantity])
    return orders[:10]


def _v023_actor_base(route_name, step, actor):
    action = _v023_route_action(route_name, step)
    if actor == "farmer":
        return action.get("farmer", ["PASS"])
    index = int(actor)
    hands = action.get("hands", []) or []
    return hands[index] if index < len(hands) else ["PASS"]


def _v023_weed_overlay(obs, action, route_name, step):
    seat = _v023_seat(obs)
    state = _V023_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        initial_route = None if _V023_VARIANT == "portfolio" and step < 48 else route_name
        state.update({"last_step": step, "route": initial_route, "active": {}, "completed": {}})
    state["last_step"] = step
    if _V023_VARIANT != "portfolio" or step >= 48 or state.get("route") is not None:
        state["route"] = route_name
    farms = obs.get("farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    positions = [farm.get("farmer"), *(farm.get("hands", []) or [])]
    unit_actions = [list(action.get("farmer") or ["PASS"]), *[list(item or ["PASS"]) for item in action.get("hands", []) or []]]
    active = state.setdefault("active", {})
    completed = state.setdefault("completed", {})
    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        age = step - int(transaction.get("start", step))
        if index >= len(unit_actions):
            active.pop(actor, None)
        elif age == 1:
            unit_actions[index] = list(transaction["intended"])
            _V023_STATS["weed_retries"] += 1
        elif 2 <= age <= 8:
            unit_actions[index] = list(_v023_actor_base(route_name, step - 1, actor))
            _V023_STATS["weed_catchup_actions"] += 1
        elif age > 8:
            active.pop(actor, None)
            completed[actor] = step
    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("PLANT", "BUILD_PASTURE"):
            continue
        tile = _v023_tile(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            completed.pop(actor, None)
            continue
        if actor in completed:
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]
        _V023_STATS["weed_repairs"] += 1
    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return action


def _v023_agent(obs):
    step = max(0, int(obs.get("step", 0) or 0))
    seat = _v023_seat(obs)
    state = _V023_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": -1, "route": None, "active": {}, "completed": {}})
    if _V023_VARIANT == "portfolio":
        if state.get("route") is None and step >= 48:
            state["route"] = _v023_choose_route(obs)
        route_name = state.get("route") or "high_output_14hands"
        base = _V023_BOOTSTRAP[min(step, len(_V023_BOOTSTRAP) - 1)] if step < 48 else _v023_route_action(route_name, step)
    else:
        route_name = {"high_output": "high_output_14hands", "stable": "stable_12hands", "high_hands": "high_hands_15hands"}.get(_V023_VARIANT, "high_output_14hands")
        base = _v023_route_action(route_name, step)
    base = _v023_align_hands(base, obs)
    base = _v023_weed_overlay(obs, base, route_name, step)
    base = _v023_align_hands(base, obs)
    base["market"] = _v023_sanitize_market(base, obs, terminal=step >= 718)
    if step >= 718:
        _V023_STATS["terminal_liquidations"] += 1
    return base


def agent(obs):
    try:
        return _v023_agent(obs)
    except Exception:
        # Keep the schema legal if an unusual replay observation is missing a
        # field; the environment will safely no-op PASS.
        seat = _v023_seat(obs if isinstance(obs, dict) else {})
        farms = (obs.get("farms", []) if isinstance(obs, dict) else []) or []
        farm = farms[seat] if seat < len(farms) else {}
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (farm.get("hands", []) or [])], "market": []}
'''


def _encode_payload(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.b85encode(zlib.compress(raw, level=9)).decode("ascii")
    parts = [encoded[index:index + 120] for index in range(0, len(encoded), 120)]
    return "_V023_B85_PARTS = " + repr(parts) + "\n"


def _write_candidate(name: str, payload: dict, metadata: dict) -> dict:
    variant = {
        "v023a_high_output_14hands": "high_output",
        "v023b_stable_12hands": "stable",
        "v023c_high_hands_15hands": "high_hands",
        "v023d_early_portfolio": "portfolio",
    }[name]
    source = _encode_payload(payload) + RUNTIME_SUFFIX.replace("__V023_VARIANT__", variant)
    history_dir = ROOT / "baseline/history" / name
    artifact_dir = ROOT / "baseline/artifacts" / name
    history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "main.py").write_text(source, encoding="utf-8")
    (artifact_dir / "main.py").write_text(source, encoding="utf-8")
    archive = artifact_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(artifact_dir / "main.py", arcname="main.py")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    manifest = {
        "candidate": name,
        "variant": variant,
        "main_sha256": digest,
        "main_bytes": len(source.encode("utf-8")),
        "archive": str(archive.relative_to(ROOT)),
        **metadata,
    }
    (artifact_dir / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build() -> dict:
    data_root = _find_data_root()
    all_records, audit = load_records(data_root)
    top10 = [record for record in all_records if record.source_file in {item["file"] for item in []}]
    # load_records returns one list and labels file counts in the audit.  Use
    # the source folder prefix from the unique file name set to separate the
    # two split directories without storing it in runtime payloads.
    top_files = {path.name for path in (data_root / "top10").glob("*.json")}
    top_records = [record for record in all_records if record.source_file in top_files]
    band_records = [record for record in all_records if record.source_file not in top_files]
    fit_records, validation_records = _split_by_episode(top_records)
    fit_profiles = _profiles_by_label(fit_records)
    final_profiles = _profiles_by_label(top_records)
    payload, profile_manifest = _runtime_payload(final_profiles)
    manifests = []
    for name in CANDIDATES:
        manifests.append(_write_candidate(name, payload, {
            "data_root": str(data_root.relative_to(ROOT)) if data_root.is_relative_to(ROOT) else str(data_root),
            "top10_records": len(top_records),
            "top10_fit_records": len(fit_records),
            "top10_validation_records": len(validation_records),
            "band_holdout_records": len(band_records),
            "route_profiles": profile_manifest,
            "audit": audit,
        }))
    artifact_root = ROOT / "baseline/artifacts/v023_route_portfolio"
    artifact_root.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v023",
        "top10_total_records": len(top_records),
        "top10_fit_records": len(fit_records),
        "top10_validation_records": len(validation_records),
        "band_holdout_records": len(band_records),
        "fit_profiles": {label: serializable_profile(profile) for label, profile in fit_profiles.items()},
        "final_profiles": profile_manifest,
        "candidates": manifests,
        "runtime_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    (artifact_root / "route_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (artifact_root / "README.md").write_text(README, encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build()
    print(json.dumps({
        "top10_records": result["top10_total_records"],
        "fit_records": result["top10_fit_records"],
        "validation_records": result["top10_validation_records"],
        "holdout_records": result["band_holdout_records"],
        "candidates": [item["candidate"] for item in result["candidates"]],
    }, ensure_ascii=False, indent=2))
