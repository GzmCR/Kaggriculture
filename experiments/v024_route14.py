"""Offline utilities for the V024 14-hands route refresh.

The replay files are used only by the builder.  Runtime candidates receive a
compressed anonymous action route and never read this module or the replay
directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v023_route_clusters import EPISODE_STEPS, RouteRecord, _extract_record, route_distance


def _episode_key(value: str) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(text):020d}")
    except ValueError:
        return (1, text)


def load_top10(data_root: Path) -> tuple[list[RouteRecord], dict[str, Any]]:
    """Load 8/7 Top10 files, deduplicated at episode level."""
    folder = data_root / "top10"
    if not folder.is_dir():
        raise FileNotFoundError(f"missing Top10 folder: {folder}")
    files = sorted(folder.glob("*.json"))
    seen: dict[str, tuple[Path, str]] = {}
    audit: dict[str, Any] = {
        "folder": str(folder), "files": len(files), "duplicates": [],
        "bad_files": [], "episodes": [],
    }
    records: list[RouteRecord] = []
    for path in files:
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            info = payload.get("info", {}) or {}
            episode = str(info.get("EpisodeId", payload.get("id", path.stem)))
            digest = hashlib.sha256(raw).hexdigest()
            if episode in seen:
                audit["duplicates"].append({
                    "episode": episode, "file": path.name,
                    "kept": seen[episode][0].name,
                    "same_hash": digest == seen[episode][1],
                })
                continue
            seen[episode] = (path, digest)
            for seat in (0, 1):
                records.append(_extract_record(path, episode, seat, payload))
        except Exception as exc:  # pragma: no cover - audit path
            audit["bad_files"].append({"file": path.name, "error": repr(exc)})
    records.sort(key=lambda item: (_episode_key(item.episode), item.seat))
    audit["episodes"] = sorted(seen, key=_episode_key)
    audit["unique_episodes"] = len(seen)
    audit["unique_seats"] = len(records)
    return records, audit


def split_records(records: list[RouteRecord], fit_fraction=0.70, validation_fraction=0.15):
    """Chronological episode split: fit, option validation, future holdout."""
    episodes = sorted({record.episode for record in records}, key=_episode_key)
    n = len(episodes)
    fit_end = max(1, int(n * fit_fraction))
    validation_end = max(fit_end + 1, int(n * (fit_fraction + validation_fraction)))
    validation_end = min(n - 1 if n > 2 else n, validation_end)
    groups = [set(episodes[:fit_end]), set(episodes[fit_end:validation_end]), set(episodes[validation_end:])]
    return tuple(
        [record for record in records if record.episode in group]
        for group in groups
    )


def route_target_distance(record: RouteRecord) -> float:
    f = record.features
    target = {
        "max_hands": 14, "plants": 159, "wheat": 93,
        "strawberry": 42, "melon": 24, "ne_day": 7,
        "sw_day": 11, "cows": 8, "sheep": 6,
        "water": 925, "harvest": 375, "feed": 313,
        "fertilize": 80,
    }
    scales = {
        "max_hands": 1, "plants": 8, "wheat": 8, "strawberry": 4,
        "melon": 3, "ne_day": 1, "sw_day": 1, "cows": 2,
        "sheep": 2, "water": 30, "harvest": 15, "feed": 15,
        "fertilize": 8,
    }
    return sum(
        abs(float(f.get(key, 0) or 0) - value) / scales[key]
        for key, value in target.items()
    )


def select_medoid(records: list[RouteRecord]) -> RouteRecord:
    """Select the complete-route medoid from the high-output fit subset."""
    if not records:
        raise ValueError("no route records available")
    high = [
        record for record in records
        if record.features.get("max_hands", 0) >= 14
        and record.features.get("ne_day", 99) == 7
        and record.features.get("sw_day", 99) == 11
    ]
    pool = high or records
    # Limit pairwise work to records close to the intended macro structure,
    # while still using the full action route for the medoid decision.
    pool = sorted(pool, key=lambda item: (route_target_distance(item), item.episode, item.seat))[:40]
    return min(
        pool,
        key=lambda candidate: (
            sum(route_distance(candidate, other) for other in pool),
            route_target_distance(candidate), candidate.episode, candidate.seat,
        ),
    )


def public_signature(observation: dict, seat: int) -> dict[str, Any]:
    """Small, runtime-safe public state signature for order-only memory."""
    farms = observation.get("farms", []) if isinstance(observation, dict) else []
    farm = farms[seat] if 0 <= seat < len(farms) else {}
    tile_counts: dict[str, int] = {}
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "")).upper()
            crop = str(tile.get("crop", "")).upper()
            animal = str(tile.get("animal", "")).upper()
            token = crop or animal or kind
            if token:
                tile_counts[token] = tile_counts.get(token, 0) + 1
    return {
        "money": int(float(farm.get("money", 0) or 0) // 250),
        "farmer": list(farm.get("farmer", [0, 0]) or [0, 0]),
        "hands": [list(item) for item in (farm.get("hands", []) or [])],
        "unlocked": sorted(str(item) for item in (farm.get("unlocked_quadrants", []) or [])),
        "tiles": tile_counts,
    }


def build_memory(record: RouteRecord, source_seat: int) -> list[dict[str, Any]]:
    """Use only the fit medoid's public opponent view and market plan."""
    opponent = 1 - source_seat
    memory = []
    for step, observation in enumerate(record.observations):
        action = record.actions[min(step, len(record.actions) - 1)]
        premium = []
        for order in action.get("market", []) or []:
            if len(order) >= 3 and str(order[0]).upper() == "SELL":
                item = str(order[1]).upper()
                if item in {"MILK", "WOOL", "STRAWBERRY", "MELON"}:
                    premium.append(item)
        memory.append({
            "step": step,
            "signature": public_signature(observation, opponent),
            "sell_order": list(dict.fromkeys(premium)),
        })
    return memory


def profile(record: RouteRecord) -> dict[str, Any]:
    return {
        "episode": record.episode,
        "seat": record.seat,
        "source_file": record.source_file,
        "seed": record.seed,
        "features": record.features,
        "checkpoint48": record.checkpoint48,
    }
