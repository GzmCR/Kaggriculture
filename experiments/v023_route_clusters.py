"""Replay normalization and deterministic macro-route clustering for V023.

The module is stdlib-only and intentionally keeps source-team metadata out of
the runtime payload.  It is used offline by ``build_v023_routes.py``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPISODE_STEPS = 720
TOP10_RELATIVE = Path("log/2026-08-06/top10")
BAND_RELATIVE = Path("log/2026-08-06/1500～2500")


def _op(action: Any) -> list:
    return list(action) if isinstance(action, list) and action else ["PASS"]


def normalize_action(action: Any) -> dict:
    if not isinstance(action, dict):
        action = {}
    hands = action.get("hands") if isinstance(action.get("hands"), list) else []
    market = action.get("market") if isinstance(action.get("market"), list) else []
    return {
        "farmer": _op(action.get("farmer")),
        "hands": [_op(item) for item in hands],
        "market": [list(item) for item in market if isinstance(item, list) and item],
    }


def _tile_stats(farm: dict) -> collections.Counter:
    result = collections.Counter()
    for row in farm.get("tiles", []) if isinstance(farm, dict) else []:
        if not isinstance(row, list):
            continue
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("crop"):
                result[f"crop:{str(tile['crop']).upper()}"] += 1
            if tile.get("animal"):
                result[f"animal:{str(tile['animal']).upper()}"] += 1
            if tile.get("kind"):
                result[f"kind:{str(tile['kind']).upper()}"] += 1
    return result


def public_checkpoint(observation: dict, seat: int) -> dict:
    farms = observation.get("farms", []) if isinstance(observation, dict) else []
    farm = farms[seat] if 0 <= seat < len(farms) else {}
    stats = _tile_stats(farm)
    return {
        "money": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "unlocked": sorted(str(x) for x in farm.get("unlocked_quadrants", []) or []),
        "crops": {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("crop:")},
        "animals": {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("animal:")},
    }


def _first_unlock(observations: list[dict], seat: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for step, observation in enumerate(observations):
        farms = observation.get("farms", []) if isinstance(observation, dict) else []
        farm = farms[seat] if 0 <= seat < len(farms) else {}
        for quadrant in farm.get("unlocked_quadrants", []) or []:
            result.setdefault(str(quadrant), step // 24)
    return result


@dataclass
class RouteRecord:
    episode: str
    seat: int
    source_file: str
    seed: int
    actions: list[dict]
    observations: list[dict]
    features: dict[str, Any]
    checkpoint48: dict[str, Any]

    @property
    def macro_vector(self) -> tuple[float, ...]:
        f = self.features
        return tuple(float(f.get(key, 0) or 0) for key in MACRO_KEYS)


MACRO_KEYS = (
    "max_hands", "plants", "wheat", "strawberry", "melon", "carrot",
    "ne_day", "sw_day", "se_day", "cows", "sheep", "water", "harvest",
    "feed", "care", "fertilize",
)
MACRO_SCALE = (2, 30, 20, 12, 10, 4, 3, 3, 5, 4, 4, 80, 40, 40, 40, 20)


def _extract_record(path: Path, episode: str, seat: int, payload: dict) -> RouteRecord:
    steps = payload.get("steps", [])
    if len(steps) < EPISODE_STEPS:
        raise ValueError(f"{path} has only {len(steps)} steps")
    observations = []
    actions = []
    counts = collections.Counter()
    plants = collections.Counter()
    sell_quantities = collections.Counter()
    sell_prices = collections.defaultdict(float)
    for step in range(EPISODE_STEPS):
        entry = steps[step][seat]
        observations.append(entry.get("observation", {}) or {})
        # Kaggle replay serialization stores the action producing observation t
        # at the next step entry.
        action = normalize_action(steps[min(step + 1, EPISODE_STEPS - 1)][seat].get("action"))
        actions.append(action)
        for unit_action in [action["farmer"], *action["hands"]]:
            name = str(unit_action[0])
            counts[name] += 1
            if name == "PLANT" and len(unit_action) >= 2:
                plants[str(unit_action[1]).upper()] += 1
        for order in action["market"]:
            if not order:
                continue
            name = str(order[0])
            counts[f"MKT_{name}"] += 1
            if name == "SELL" and len(order) >= 3:
                item = str(order[1]).upper()
                try:
                    quantity = max(0, int(order[2]))
                except (TypeError, ValueError):
                    quantity = 0
                sell_quantities[item] += quantity
                market = observations[-1].get("market", {}) or {}
                prices = market.get("prices", {}) or {}
                sell_prices[item] += quantity * float(prices.get(item, 0) or 0)
    final_checkpoint = public_checkpoint(observations[-1], seat)
    unlock = _first_unlock(observations, seat)
    final_crops = final_checkpoint.get("crops", {})
    final_animals = final_checkpoint.get("animals", {})
    features = {
        "max_hands": max(len((o.get("farms", []) or [{}])[seat].get("hands", []) or []) for o in observations),
        "plants": sum(plants.values()),
        "wheat": plants.get("WHEAT", 0),
        "strawberry": plants.get("STRAWBERRY", 0),
        "melon": plants.get("MELON", 0),
        "carrot": plants.get("CARROT", 0),
        "ne_day": unlock.get("NE", 99),
        "sw_day": unlock.get("SW", 99),
        "se_day": unlock.get("SE", 99),
        "cows": final_animals.get("COW", 0),
        "sheep": final_animals.get("SHEEP", 0),
        "water": counts.get("WATER", 0),
        "harvest": counts.get("HARVEST", 0),
        "feed": counts.get("FEED", 0),
        "care": counts.get("CARE", 0),
        "fertilize": counts.get("FERTILIZE", 0),
        "pass": counts.get("PASS", 0),
        "sell_quantities": dict(sell_quantities),
        "weighted_sell_prices": {
            item: sell_prices[item] / sell_quantities[item]
            for item in sell_quantities if sell_quantities[item]
        },
        "final_money": float(final_checkpoint.get("money", 0) or 0),
        "final_crops": final_crops,
        "final_animals": final_animals,
        "counts": dict(counts),
    }
    info = payload.get("info", {}) or {}
    return RouteRecord(
        episode=str(episode),
        seat=seat,
        source_file=path.name,
        seed=int(info.get("seed", 0) or 0),
        actions=actions,
        observations=observations,
        features=features,
        checkpoint48=public_checkpoint(observations[48], seat),
    )


def load_records(data_root: Path) -> tuple[list[RouteRecord], dict[str, Any]]:
    """Load and deduplicate both folders, returning records and audit data."""
    folders = {
        "top10": data_root / "top10",
        "band1500_2500": data_root / "1500～2500",
    }
    records: list[RouteRecord] = []
    audit = {"files": {}, "duplicates": [], "bad_files": []}
    seen_episode: dict[str, tuple[str, str]] = {}
    for split, folder in folders.items():
        files = sorted(folder.glob("*.json"))
        audit["files"][split] = len(files)
        for path in files:
            try:
                raw = path.read_bytes()
                payload = json.loads(raw.decode("utf-8"))
                info = payload.get("info", {}) or {}
                episode = str(info.get("EpisodeId", payload.get("id", path.stem)))
                digest = hashlib.sha256(raw).hexdigest()
                if episode in seen_episode:
                    audit["duplicates"].append({"episode": episode, "file": path.name, "kept": seen_episode[episode][0]})
                    continue
                seen_episode[episode] = (path.name, digest)
                for seat in (0, 1):
                    records.append(_extract_record(path, episode, seat, payload))
            except Exception as exc:  # pragma: no cover - audit path
                audit["bad_files"].append({"file": str(path), "error": repr(exc)})
    audit["unique_episodes"] = len(seen_episode)
    audit["unique_seats"] = len(records)
    return records, audit


def route_distance(left: RouteRecord, right: RouteRecord) -> float:
    macro = sum(
        abs(a - b) / scale for a, b, scale in zip(left.macro_vector, right.macro_vector, MACRO_SCALE)
    )
    action_diff = 0
    for first, second in zip(left.actions, right.actions):
        if first.get("farmer") != second.get("farmer"):
            action_diff += 1
        max_hands = max(len(first.get("hands", [])), len(second.get("hands", [])))
        for index in range(max_hands):
            a = first.get("hands", [])[index] if index < len(first.get("hands", [])) else ["PASS"]
            b = second.get("hands", [])[index] if index < len(second.get("hands", [])) else ["PASS"]
            action_diff += a != b
        if first.get("market") != second.get("market"):
            action_diff += 0.25
    return macro + action_diff / EPISODE_STEPS


TARGETS = (
    (14, 158, 42, 24),
    (12, 131, 44, 21),
    (15, 135, 41, 26),
)


def _target_distance(record: RouteRecord, target: tuple[int, int, int, int]) -> float:
    f = record.features
    values = (f["max_hands"], f["plants"], f["strawberry"], f["melon"])
    scales = (2, 30, 12, 10)
    return sum(abs(a - b) / s for a, b, s in zip(values, target, scales))


def _kmedoids(records: list[RouteRecord], k: int = 3) -> list[list[RouteRecord]]:
    if not records:
        return []
    k = min(k, len(records))
    medoid_indices = []
    remaining = set(range(len(records)))
    for target in TARGETS[:k]:
        index = min(remaining, key=lambda i: (_target_distance(records[i], target), records[i].episode, records[i].seat))
        medoid_indices.append(index)
        remaining.discard(index)
    for _ in range(5):
        clusters = [[] for _ in medoid_indices]
        for index, record in enumerate(records):
            choice = min(
                range(len(medoid_indices)),
                key=lambda j: (route_distance(record, records[medoid_indices[j]]), j),
            )
            clusters[choice].append(index)
        new_indices = []
        for cluster, old_index in zip(clusters, medoid_indices):
            if not cluster:
                new_indices.append(old_index)
                continue
            candidate = min(
                cluster,
                key=lambda i: (
                    sum(route_distance(records[i], records[j]) for j in cluster),
                    _target_distance(records[i], TARGETS[len(new_indices)]),
                    records[i].episode,
                    records[i].seat,
                ),
            )
            new_indices.append(candidate)
        if new_indices == medoid_indices:
            break
        medoid_indices = new_indices
    output = [[] for _ in medoid_indices]
    for record in records:
        choice = min(range(len(medoid_indices)), key=lambda j: (route_distance(record, records[medoid_indices[j]]), j))
        output[choice].append(record)
    return output


def cluster_profiles(records: list[RouteRecord]) -> list[dict[str, Any]]:
    clusters = _kmedoids(records, 3)
    profiles = []
    for index, cluster in enumerate(clusters):
        if not cluster:
            continue
        medoid = min(
            cluster,
            key=lambda candidate: (
                sum(route_distance(candidate, other) for other in cluster),
                candidate.episode,
                candidate.seat,
            ),
        )
        target = TARGETS[index] if index < len(TARGETS) else TARGETS[0]
        label = (
            "high_output_14hands" if index == 0 else
            "stable_12hands" if index == 1 else
            "high_hands_15hands"
        )
        feature_mean = {
            key: statistics_mean([float(record.features.get(key, 0) or 0) for record in cluster])
            for key in MACRO_KEYS
        }
        profiles.append({
            "label": label,
            "support": len(cluster),
            "medoid_episode": medoid.episode,
            "medoid_seat": medoid.seat,
            "medoid_source_file": medoid.source_file,
            "target_distance": _target_distance(medoid, target),
            "feature_mean": feature_mean,
            "medoid_features": medoid.features,
            "record": medoid,
        })
    return profiles


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def serializable_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key != "record"}
