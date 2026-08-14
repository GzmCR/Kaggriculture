"""Build anonymous V032 opponent route profiles from offline replays.

The input replays must be generated offline from the notebook agents.  This
builder deliberately ignores team names, scores, filenames and seed values in
the embedded runtime profiles.  The nowinlog directory is not an input by
default; pass a separate directory only when building an explicitly marked
diagnostic profile set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (96, 144, 192, 240, 288)
PREMIUM = ("MILK", "STRAWBERRY", "WOOL", "MELON")


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _farm_signature(farm):
    counts = {"COW": 0, "SHEEP": 0, "GOOSE": 0, "PASTURE": 0,
              "COOP": 0, "WHEAT": 0, "STRAWBERRY": 0, "MELON": 0,
              "TOMATO": 0, "CARROT": 0, "WEED": 0}
    for row in (farm or {}).get("tiles", []) or []:
        if not isinstance(row, list):
            continue
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "")).upper()
            if kind == "PLANT":
                key = str(tile.get("crop", "")).upper()
            elif kind in ("COOP", "PASTURE"):
                key = str(tile.get("animal", "")).upper() or kind
                counts[kind] += 1
            else:
                key = kind
            if key in counts:
                counts[key] += 1
    return {
        "hands": len((farm or {}).get("hands", []) or []),
        "unlocked": sorted(str(x) for x in ((farm or {}).get("unlocked_quadrants", []) or [])),
        "counts": counts,
    }


def _action_sells(action):
    result = {item: 0 for item in PREMIUM}
    for order in (action or {}).get("market", []) or []:
        if not isinstance(order, list) or len(order) < 3:
            continue
        if str(order[0]).upper() == "SELL" and str(order[1]).upper() in result:
            result[str(order[1]).upper()] += max(0, _int(order[2]))
    return result


def _profile_id(payload, seat, path):
    # Route identity is anonymous and content-derived.  Episode IDs, names and
    # scores remain outside the generated runtime profile.
    parts = []
    steps = payload.get("steps", [])
    for cp in CHECKPOINTS:
        if cp + 1 >= len(steps):
            continue
        obs = (steps[cp + 1][seat] or {}).get("observation", {}) or {}
        farms = obs.get("farms", []) or []
        farm = farms[seat] if seat < len(farms) else {}
        parts.append(_farm_signature(farm))
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def extract_profile(path: Path, seat: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", []) or []
    if len(steps) < 300:
        raise ValueError(f"incomplete replay: {path}")
    checkpoints = {}
    for cp in CHECKPOINTS:
        if cp + 1 >= len(steps):
            continue
        state = steps[cp + 1][seat] or {}
        obs = state.get("observation", {}) or {}
        farms = obs.get("farms", []) or []
        if seat < len(farms):
            checkpoints[str(cp)] = _farm_signature(farms[seat])

    # Use direct offline actions to estimate supply bands.  At runtime these
    # are only compared against the public market residual, never replayed.
    sells = {item: [] for item in PREMIUM}
    for turn in range(min(719, len(steps) - 1)):
        state = steps[turn + 1][seat] or {}
        row = _action_sells(state.get("action", {}) or {})
        for item in PREMIUM:
            sells[item].append(row[item])
    market_bands = {}
    for item in PREMIUM:
        values = [value for value in sells[item] if value > 0]
        upper = max(values + [0])
        # The public residual includes town/shop consumption and timing noise,
        # so use a tolerant upper band rather than an exact action quantity.
        market_bands[item] = {"low": 0, "high": max(12, upper + 12)}

    forecasts = {}
    for item in PREMIUM:
        forecasts[item] = {"default": statistics.mean(sells[item]) if sells[item] else 0.0}
        for turn, value in enumerate(sells[item]):
            if value:
                forecasts[item][str(turn)] = value

    profile = {
        "profile_id": _profile_id(payload, seat, path),
        "checkpoints": checkpoints,
        "route_distance": 8,
        "market_bands": market_bands,
        "supply_forecast": forecasts,
        "source_kind": "offline_replay",
    }
    return profile


def build(replay_dir: Path, output: Path, source_manifest: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    manifest = {"input_dir": str(replay_dir), "files": [], "profiles": [], "source_manifest": str(source_manifest or "")}
    seat_by_file = {}
    if source_manifest and Path(source_manifest).exists():
        source = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
        for game in source.get("games", []) or []:
            seat_by_file[str(game.get("file", ""))] = int(game.get("seat", 0) or 0)
    for path in sorted(Path(replay_dir).glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name in seat_by_file:
                seats = (seat_by_file[path.name],)
            else:
                # Without a source manifest this is intentionally a diagnostic
                # fallback; callers should provide the manifest for fitting.
                agents = payload.get("info", {}).get("Agents", []) or []
                seats = range(min(2, len(agents))) if agents else (0, 1)
            for seat in seats:
                profile = extract_profile(path, int(seat))
                # Keep one profile per anonymous route signature.
                if any(x["profile_id"] == profile["profile_id"] for x in profiles):
                    continue
                profiles.append(profile)
                manifest["profiles"].append({"profile_id": profile["profile_id"], "seat": int(seat)})
            manifest["files"].append(path.name)
        except Exception as exc:
            manifest.setdefault("errors", []).append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    (output / "profiles.json").write_text(json.dumps(profiles, separators=(",", ":")) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v032_route_conditioned_timing/profiles")
    args = parser.parse_args()
    print(json.dumps(build(args.replay_dir, args.output, args.source_manifest), indent=2))
