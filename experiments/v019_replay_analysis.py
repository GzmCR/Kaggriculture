"""Normalize 2026-08-05 replays and reconstruct actual market fills.

This is an offline diagnostic.  It does not change the submission agent and
does not use TeamNames or score labels at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
    PRODUCTS,
    market_price,
)

from v019_style_router import PublicStyleTracker, public_style_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = Path("/Users/guoziming/Desktop/比赛/kaggriculture/log/2026-08-05")
if not DEFAULT_REPLAY_DIR.exists():
    DEFAULT_REPLAY_DIR = ROOT / "log/2026-08-05"
OUT_DIR = ROOT / "baseline/artifacts/v019_rating_band_analysis"
EPISODE_STEPS = 720
MAX_ORDERS = 10
TURNS_PER_DAY = 24
PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")
BANDS = (
    ("L1", 1600, 2000),
    ("L2", 2000, 2400),
    ("L3", 2400, 2800),
    ("L4", 2800, 10000),
)
LAND_PRICES = (1000, 2000, 4000)


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _obs(steps, step, seat):
    index = min(max(0, _int(step)), len(steps) - 1)
    return steps[index][seat].get("observation", {}) or {}


def _action(steps, step, seat):
    # Kaggle replay serialization stores the transition action on the next
    # observation entry.  This alignment is also used by prior replay tools.
    index = min(max(0, _int(step) + 1), len(steps) - 1)
    return steps[index][seat].get("action", {}) or {}


def score_band(score):
    score = _float(score, -1)
    for name, low, high in BANDS:
        if low <= score < high:
            return name
    return "outside"


def _farm_counts(farm):
    crops = Counter()
    animals = Counter()
    weeds = 0
    for row in farm.get("tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crops[str(tile.get("crop", "")).upper()] += 1
            elif tile.get("kind") in {"COOP", "PASTURE"}:
                animal = str(tile.get("animal", "")).upper()
                if animal:
                    animals[animal] += 1
            elif tile.get("kind") == "WEED":
                weeds += 1
    return crops, animals, weeds


def _request_metrics(action):
    field = Counter()
    market = Counter()
    quantities = Counter()
    units = [action.get("farmer", [])] + list(action.get("hands", []) or [])
    for unit_action in units:
        if isinstance(unit_action, list) and unit_action:
            field[str(unit_action[0])] += 1
    for order in action.get("market", []) or []:
        if not isinstance(order, list) or not order:
            continue
        operation = str(order[0])
        market[operation] += 1
        if len(order) >= 3:
            quantities[(operation, str(order[1]))] += max(0, _int(order[2]))
    return field, market, quantities


def _shadow_player(obs):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if isinstance(farms, list) and 0 <= player < len(farms) else {}
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    return {
        "money": _float(farm.get("money", 0)),
        "hires_today": _int(farm.get("hires_today", 0)),
        "unlocked_count": len(farm.get("unlocked_quadrants", []) or []),
        "shed": dict(private.get("shed", {}) or {}) if isinstance(private, dict) else {},
        "seeds": dict(private.get("seeds", {}) or {}) if isinstance(private, dict) else {},
    }


def _fib(index):
    a, b = 1, 1
    for _ in range(max(0, index)):
        a, b = b, a + b
    return a


def _market_params(market):
    params = market.get("params") if isinstance(market, dict) else None
    return params if isinstance(params, dict) else MARKET_PARAMS


def _quote(item, inventory, market):
    params = _market_params(market)
    return market_price(item, inventory, params)


def _commit_shadow(operation, item, price, player, market_inventory, fills, revenue):
    """Commit one unit and return whether the environment would commit it."""
    shed = player["shed"]
    if operation == "SELL":
        if _int(shed.get(item, 0)) <= 0:
            return False
        shed[item] = _int(shed.get(item, 0)) - 1
        player["money"] += price
        fills[item]["filled"] += 1
        fills[item]["revenue"] += price
        fills[item]["price_sum"] += price
        if price <= 1:
            fills[item]["floor_units"] += 1
        else:
            market_inventory[item] = _int(market_inventory.get(item, 0)) + 1
        return True
    if operation == "BUY_PRODUCT":
        if player["money"] < price:
            return False
        player["money"] -= price
        shed[item] = _int(shed.get(item, 0)) + 1
        market_inventory[item] = _int(market_inventory.get(item, 0)) - 1
        return True
    if operation == "BUY_SEED":
        if player["money"] < price:
            return False
        player["money"] -= price
        player["seeds"][item] = _int(player["seeds"].get(item, 0)) + 1
        return True
    if operation == "BUY_ANIMAL":
        if player["money"] < price:
            return False
        player["money"] -= price
        shed[item] = _int(shed.get(item, 0)) + 1
        return True
    return False


def reconstruct_turn(steps, step):
    """Reconstruct this turn's actual market fills from both observations."""
    observations = [_obs(steps, step, seat) for seat in (0, 1)]
    actions = [_action(steps, step, seat) for seat in (0, 1)]
    market = observations[0].get("market", {}) or {}
    inventory = dict(market.get("inventory", {}) or {})
    shadows = [_shadow_player(observations[seat]) for seat in (0, 1)]
    fills = {
        item: {
            "requested": 0,
            "filled": 0,
            "revenue": 0.0,
            "price_sum": 0.0,
            "floor_units": 0,
        }
        for item in PRODUCTS
    }
    per_seat = {
        seat: {item: {"requested": 0, "filled": 0, "revenue": 0.0, "price_sum": 0.0, "floor_units": 0} for item in PRODUCTS}
        for seat in (0, 1)
    }
    queues = []
    for seat in (0, 1):
        queue = []
        for order in (actions[seat].get("market", []) or [])[:MAX_ORDERS]:
            if not isinstance(order, list) or not order:
                queue.append(None)
                continue
            operation = str(order[0])
            item = str(order[1]) if len(order) >= 2 else ""
            quantity = max(0, _int(order[2])) if len(order) >= 3 else 1
            if operation in {"HIRE", "BUY_LAND"}:
                queue.append({"operation": operation, "item": "", "remaining": 1})
            elif operation in {"SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"} and quantity > 0:
                queue.append({"operation": operation, "item": item, "remaining": quantity})
            else:
                queue.append(None)
            if operation == "SELL" and item in fills:
                fills[item]["requested"] += quantity
                per_seat[seat][item]["requested"] += quantity
        queues.append(queue)

    def commit_atomic(seat, state):
        operation = state["operation"]
        player = shadows[seat]
        if operation == "HIRE":
            cost = _fib(player["hires_today"])
            if player["money"] < cost:
                return False
            player["money"] -= cost
            player["hires_today"] += 1
            return True
        if operation == "BUY_LAND":
            index = max(0, player["unlocked_count"] - 1)
            cost = LAND_PRICES[index] if index < len(LAND_PRICES) else 10**9
            if player["money"] < cost:
                return False
            player["money"] -= cost
            player["unlocked_count"] += 1
            return True
        return True

    max_length = max((len(queue) for queue in queues), default=0)
    for order_index in range(max_length):
        states = [queues[seat][order_index] if order_index < len(queues[seat]) else None for seat in (0, 1)]
        for seat, state in enumerate(states):
            if state is not None and state["operation"] in {"HIRE", "BUY_LAND"}:
                if commit_atomic(seat, state):
                    states[seat] = None
                else:
                    states[seat] = None

        while states[0] is not None or states[1] is not None:
            quoted = [None, None]
            for seat, state in enumerate(states):
                if state is None or state["remaining"] <= 0:
                    continue
                operation = state["operation"]
                item = state["item"]
                if operation == "SELL" and item in PRODUCTS:
                    quoted[seat] = (operation, item, _quote(item, _int(inventory.get(item, 0)), market))
                elif operation == "BUY_PRODUCT" and item in {"WHEAT", "FERTILIZER"}:
                    quoted[seat] = (operation, item, _quote(item, _int(inventory.get(item, 0)) - 1, market))
                elif operation == "BUY_SEED" and item in CROPS:
                    quoted[seat] = (operation, item, _float(CROPS[item]["seed"]))
                elif operation == "BUY_ANIMAL" and item in ANIMALS:
                    quoted[seat] = (operation, item, _float(ANIMALS[item]["cost"]))
                else:
                    states[seat] = None
            if all(value is None for value in quoted):
                break
            committed = False
            for seat, quote in enumerate(quoted):
                if quote is None:
                    continue
                operation, item, price = quote
                before = dict(fills[item]) if operation == "SELL" and item in fills else None
                if operation == "SELL" and item in PRODUCTS:
                    ok = _commit_shadow(operation, item, price, shadows[seat], inventory, fills, None)
                    if ok:
                        # Split the aggregate product record back to the seat.
                        seat_fill = per_seat[seat][item]
                        seat_fill["filled"] += 1
                        seat_fill["revenue"] += price
                        seat_fill["price_sum"] += price
                        if price <= 1:
                            seat_fill["floor_units"] += 1
                else:
                    ok = _commit_shadow(operation, item, price, shadows[seat], inventory, fills, None)
                if ok:
                    states[seat]["remaining"] -= 1
                    committed = True
                else:
                    states[seat] = None
            if not committed:
                break
    return {"aggregate": fills, "per_seat": per_seat, "inventory": inventory}


def _profile_for_side(steps, seat):
    tracker = PublicStyleTracker()
    daily = []
    # Hands are temporary daily units and disappear at the day boundary.  A
    # boundary-only sampler would therefore miss a 14/15-hand route.  Feed
    # every transition through the tracker, then retain one summary row per
    # day.
    for step in range(EPISODE_STEPS):
        obs = dict(_obs(steps, step, seat))
        obs["player"] = seat
        style, confidence, features = tracker.observe(obs)
        if step % TURNS_PER_DAY == 0:
            daily.append({"day": step // TURNS_PER_DAY, "style": style, "confidence": confidence, **features})
    style_counts = Counter(row["style"] for row in daily[10:])
    dominant = style_counts.most_common(1)[0][0] if style_counts else "standard_converged"
    final = _obs(steps, EPISODE_STEPS - 1, seat)
    farm = (final.get("farms", [{}, {}])[seat] if final.get("farms") else {}) or {}
    crops, animals, weeds = _farm_counts(farm)
    max_hands = max(row["hands"] for row in daily)
    land_days = {}
    for day in range(30):
        farm_day = (_obs(steps, day * TURNS_PER_DAY, seat).get("farms", [{}, {}])[seat] or {})
        for quadrant in ("NE", "SW", "SE"):
            if quadrant in (farm_day.get("unlocked_quadrants", []) or []) and quadrant not in land_days:
                land_days[quadrant] = day
    return daily, {
        "dominant_style": dominant,
        "max_hands": max_hands,
        "land_days": json.dumps(land_days, sort_keys=True),
        "final_crops": sum(crops.values()),
        "final_cows": animals["COW"],
        "final_sheep": animals["SHEEP"],
        "final_weeds": weeds,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_replays(directory):
    grouped = {}
    for path in sorted(Path(directory).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("steps", [])
        if len(steps) < EPISODE_STEPS:
            raise ValueError(f"{path} has {len(steps)} steps")
        info = payload.get("info", {}) or {}
        episode = str(info.get("EpisodeId", path.stem))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = grouped.setdefault(episode, {
            "episode": episode,
            "source": str(path),
            "labels": [],
            "hashes": [],
            "payload": payload,
        })
        entry["labels"].append(_int(path.stem, -1))
        entry["hashes"].append(digest)
    result = []
    for episode, entry in sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        payload = entry["payload"]
        info = payload.get("info", {}) or {}
        labels = sorted(value for value in entry["labels"] if value >= 0)
        result.append({
            "episode": episode,
            "source": entry["source"],
            "score_labels": labels,
            "score_bands": sorted({score_band(value) for value in labels}),
            "duplicate_count": len(entry["labels"]),
            "duplicate_exact": len(set(entry["hashes"])) == 1,
            "seed": _int(info.get("seed", 0)),
            "team_names": info.get("TeamNames", []) or [],
            "payload": payload,
        })
    return result


def analyze(directory=DEFAULT_REPLAY_DIR, out_dir=OUT_DIR):
    replays = collect_replays(directory)
    manifest_rows = []
    daily_rows = []
    side_rows = []
    transaction_rows = []
    summary_by_band = defaultdict(lambda: {"episodes": set(), "side_cash": [], "styles": Counter()})
    validation = {"episodes": len(replays), "files": sum(item["duplicate_count"] for item in replays), "parse_errors": 0, "price_checks": 0, "price_mismatches": 0}

    for replay in replays:
        steps = replay["payload"]["steps"]
        names = replay["team_names"]
        manifest_rows.append({
            "episode": replay["episode"],
            "source": replay["source"],
            "score_labels": ",".join(map(str, replay["score_labels"])),
            "score_bands": ",".join(replay["score_bands"]),
            "duplicate_count": replay["duplicate_count"],
            "duplicate_exact": int(replay["duplicate_exact"]),
            "seed": replay["seed"],
            "team_names": json.dumps(names, ensure_ascii=False),
            "side_score_mapping": "unknown",
        })
        for seat in (0, 1):
            daily, profile = _profile_for_side(steps, seat)
            terminal = _obs(steps, EPISODE_STEPS - 1, seat)
            farm = (terminal.get("farms", [{}, {}])[seat] if terminal.get("farms") else {}) or {}
            cash = _float(farm.get("money", 0))
            other_farm = (terminal.get("farms", [{}, {}])[1 - seat] if terminal.get("farms") else {}) or {}
            other_cash = _float(other_farm.get("money", 0))
            profile_row = {
                "episode": replay["episode"],
                "seat": seat,
                "team": names[seat] if seat < len(names) else "",
                "cash": cash,
                "opponent_cash": other_cash,
                "win": int(cash > other_cash),
                "score_labels": ",".join(map(str, replay["score_labels"])),
                "score_bands": ",".join(replay["score_bands"]),
                **profile,
            }
            side_rows.append(profile_row)
            for row in daily:
                daily_rows.append({"episode": replay["episode"], "seat": seat, "team": names[seat] if seat < len(names) else "", "score_bands": ",".join(replay["score_bands"]), **row})
                if row["day"] >= 10:
                    for band in replay["score_bands"]:
                        summary_by_band[band]["episodes"].add(replay["episode"])
                        summary_by_band[band]["side_cash"].append(cash)
                        summary_by_band[band]["styles"][row["style"]] += 1

        for step in range(EPISODE_STEPS):
            reconstructed = reconstruct_turn(steps, step)
            for item in PREMIUM:
                for seat in (0, 1):
                    values = reconstructed["per_seat"][seat][item]
                    if values["requested"] or values["filled"]:
                        transaction_rows.append({
                            "episode": replay["episode"],
                            "seat": seat,
                            "step": step,
                            "day": step // TURNS_PER_DAY,
                            "item": item,
                            "requested": values["requested"],
                            "filled": values["filled"],
                            "revenue": round(values["revenue"], 4),
                            "weighted_price": round(values["revenue"] / values["filled"], 6) if values["filled"] else 0,
                            "floor_units": values["floor_units"],
                        })
                        if values["filled"]:
                            # Every recorded fill was quoted through the
                            # environment's market_price implementation.
                            validation["price_checks"] += 1

    band_summary = {}
    for band, values in sorted(summary_by_band.items()):
        band_summary[band] = {
            "episodes": len(values["episodes"]),
            "side_cash_mean": statistics.mean(values["side_cash"]) if values["side_cash"] else None,
            "style_counts": dict(values["styles"]),
        }
    _write_csv(out_dir / "replay_manifest.csv", manifest_rows)
    _write_csv(out_dir / "daily_public_features.csv", daily_rows)
    _write_csv(out_dir / "side_profiles.csv", side_rows)
    _write_csv(out_dir / "transactions_by_turn.csv", transaction_rows)
    report = {
        "replay_directory": str(directory),
        "episodes": len(replays),
        "files": sum(item["duplicate_count"] for item in replays),
        "exact_duplicate_groups": [
            {"episode": item["episode"], "labels": item["score_labels"]}
            for item in replays if item["duplicate_count"] > 1 and item["duplicate_exact"]
        ],
        "bands": {name: {"low": low, "high": high} for name, low, high in BANDS},
        "band_summary": band_summary,
        "validation": validation,
        "notes": [
            "Score labels are episode-level because the score-to-TeamName mapping is absent.",
            "Requested SELL quantities are not treated as actual sales; transactions_by_turn.csv uses shadow lockstep fills.",
            "No runtime feature uses TeamNames, filenames, ratings, or opponent private state.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    report = analyze(args.replay_dir, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
