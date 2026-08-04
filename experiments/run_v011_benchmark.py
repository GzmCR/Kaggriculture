"""Benchmark V011 crop-maintenance and unit-scheduling ablations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from run_v008_benchmark import (
    DEFAULT_SEEDS,
    _counter_json,
    _quantity_json,
    _valid_shape,
    load_opponents,
    percentile,
    summarize as summarize_base,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
HIGH_VALUE_CROPS = {"MELON", "STRAWBERRY", "TOMATO"}
DEFAULT_OPPONENTS = ("baseline", "starter", "random", "hamburger", "frontier")
DEFAULT_CANDIDATES = {
    "control": ROOT / "main.py",
    "v011a_water_guard": ROOT / "baseline/history/v011a_water_guard/main.py",
    "v011b_harvest_storage": ROOT / "baseline/history/v011b_harvest_storage/main.py",
    "v011c_priority_scheduler": ROOT / "baseline/history/v011c_priority_scheduler/main.py",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} must define agent(obs, config=None)")
    return module


def public_crop_metrics(obs, player):
    farms = obs.get("farms", []) or []
    if not (0 <= player < len(farms)):
        return 0, 0, 0
    unwatered = 0
    weeds = 0
    max_consecutive = 0
    for row in farms[player].get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "WEED":
                weeds += 1
            if tile.get("kind") == "PLANT" and tile.get("crop") in HIGH_VALUE_CROPS:
                if not bool(tile.get("watered_today", False)):
                    unwatered += 1
                max_consecutive = max(
                    max_consecutive,
                    int(tile.get("consecutive_unwatered", 0) or 0),
                )
    return unwatered, weeds, max_consecutive


def action_crop(obs, player, unit_index):
    farms = obs.get("farms", []) or []
    if not (0 <= player < len(farms)):
        return None
    farm = farms[player]
    positions = [farm.get("farmer"), *(farm.get("hands", []) or [])]
    if not (0 <= unit_index < len(positions)) or positions[unit_index] is None:
        return None
    x, y = positions[unit_index]
    tiles = farm.get("tiles", []) or []
    if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
        return None
    tile = tiles[y][x]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop")
        if crop in HIGH_VALUE_CROPS:
            return str(crop)
    return None


class CandidateProbe:
    def __init__(self, module):
        self.module = module
        self.calls = 0
        self.errors = 0
        self.invalid = 0
        self.times_ms = []
        self.field_counts = Counter()
        self.high_value_counts = Counter()
        self.market_counts = Counter()
        self.market_quantities = Counter()
        self.eod_unwatered = []
        self.eod_weeds = []
        self.max_consecutive_unwatered = 0

    def __call__(self, obs, config=None):
        player = int(obs.get("player", 0) or 0)
        unwatered, weeds, consecutive = public_crop_metrics(obs, player)
        self.max_consecutive_unwatered = max(
            self.max_consecutive_unwatered, consecutive
        )
        if int(obs.get("hour", 0) or 0) == 23:
            self.eod_unwatered.append(unwatered)
            self.eod_weeds.append(weeds)

        started = time.perf_counter_ns()
        try:
            action = self.module.agent(obs, config)
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        self.calls += 1
        if not _valid_shape(action, obs, config):
            self.invalid += 1
        if isinstance(action, dict):
            operations = [
                action.get("farmer", []),
                *(action.get("hands", []) or []),
            ]
            for unit_index, operation in enumerate(operations):
                if isinstance(operation, list) and operation:
                    op = str(operation[0])
                    self.field_counts[op] += 1
                    crop = action_crop(obs, player, unit_index)
                    if crop is not None and op in {"WATER", "HARVEST"}:
                        self.high_value_counts[f"{op}:{crop}"] += 1
                        self.high_value_counts[f"{op}:HIGH_VALUE"] += 1
            for order in action.get("market", []) or []:
                if not isinstance(order, list) or not order:
                    continue
                operation = str(order[0])
                self.market_counts[operation] += 1
                if len(order) >= 3 and operation in {
                    "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL",
                }:
                    self.market_quantities[(operation, str(order[1]))] += int(order[2] or 0)
        return action

    def timing(self):
        return {
            "p50": percentile(self.times_ms, 0.50),
            "p95": percentile(self.times_ms, 0.95),
            "p99": percentile(self.times_ms, 0.99),
            "max": max(self.times_ms or [0.0]),
        }


def diagnostics(module):
    state = getattr(module, "V011_STATS", {})
    return json.dumps(state if isinstance(state, dict) else {}, sort_keys=True)


def diagnostic_value(row, key):
    try:
        values = json.loads(row.get("v011_stats", "{}") or "{}")
    except (TypeError, ValueError):
        values = {}
    return float(values.get(key, 0) or 0)


def run_game(module, opponent, seed, seat):
    candidate = CandidateProbe(module)
    players = [candidate, opponent] if seat == 0 else [opponent, candidate]
    started = time.perf_counter()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    candidate_state = final[seat]
    opponent_state = final[1 - seat]
    candidate_money = float(candidate_state.observation["farms"][seat]["money"])
    opponent_money = float(opponent_state.observation["farms"][1 - seat]["money"])
    margin = candidate_money - opponent_money
    result = "win" if margin > 0 else "loss" if margin < 0 else "tie"
    timing = candidate.timing()
    return {
        "seed": seed,
        "seat": seat,
        "candidate_money": candidate_money,
        "opponent_money": opponent_money,
        "margin": margin,
        "result": result,
        "candidate_status": candidate_state.status,
        "opponent_status": opponent_state.status,
        "game_done": int(candidate_state.status == "DONE" and opponent_state.status == "DONE"),
        "action_calls": candidate.calls,
        "agent_errors": candidate.errors,
        "invalid_action_shapes": candidate.invalid,
        "runtime_p50_ms": timing["p50"],
        "runtime_p95_ms": timing["p95"],
        "runtime_p99_ms": timing["p99"],
        "runtime_max_ms": timing["max"],
        "wall_seconds": time.perf_counter() - started,
        "route_fallbacks": 0,
        "route_switches": 0,
        "route_history": "[]",
        "daily_cash": "{}",
        "field_counts": _counter_json(candidate.field_counts),
        "high_value_counts": _counter_json(candidate.high_value_counts),
        "market_counts": _counter_json(candidate.market_counts),
        "market_quantities": _quantity_json(candidate.market_quantities),
        "eod_unwatered_mean": statistics.mean(candidate.eod_unwatered or [0]),
        "eod_unwatered_max": max(candidate.eod_unwatered or [0]),
        "eod_weeds_mean": statistics.mean(candidate.eod_weeds or [0]),
        "eod_weeds_max": max(candidate.eod_weeds or [0]),
        "max_consecutive_unwatered": candidate.max_consecutive_unwatered,
        "v011_stats": diagnostics(module),
    }


def summarize(rows):
    summary = summarize_base(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["candidate"], row["opponent"])].append(row)
    for item in summary:
        group = groups[(item["candidate"], item["opponent"])]
        high_value = [
            json.loads(row.get("high_value_counts", "{}") or "{}")
            for row in group
        ]
        high_value_water = [
            float(values.get("WATER:HIGH_VALUE", 0) or 0)
            for values in high_value
        ]
        high_value_harvest = [
            float(values.get("HARVEST:HIGH_VALUE", 0) or 0)
            for values in high_value
        ]
        field_counts = [
            json.loads(row.get("field_counts", "{}") or "{}")
            for row in group
        ]
        item.update({
            "mean_eod_unwatered": statistics.mean(row["eod_unwatered_mean"] for row in group),
            "max_eod_unwatered": max(row["eod_unwatered_max"] for row in group),
            "mean_eod_weeds": statistics.mean(row["eod_weeds_mean"] for row in group),
            "max_eod_weeds": max(row["eod_weeds_max"] for row in group),
            "max_consecutive_unwatered": max(row["max_consecutive_unwatered"] for row in group),
            "mean_high_value_water": statistics.mean(high_value_water),
            "mean_high_value_harvest": statistics.mean(high_value_harvest),
            "mean_pass": statistics.mean(float(values.get("PASS", 0) or 0) for values in field_counts),
            "mean_movement": statistics.mean(
                sum(float(values.get(direction, 0) or 0) for direction in ("NORTH", "SOUTH", "EAST", "WEST"))
                for values in field_counts
            ),
            "mean_feed": statistics.mean(float(values.get("FEED", 0) or 0) for values in field_counts),
            "mean_care": statistics.mean(float(values.get("CARE", 0) or 0) for values in field_counts),
            "mean_drop": statistics.mean(float(values.get("DROP", 0) or 0) for values in field_counts),
            "mean_unassigned_urgent": statistics.mean(
                diagnostic_value(row, "unassigned_urgent_jobs") for row in group
            ),
            "mean_extra_water_assignments": statistics.mean(
                diagnostic_value(row, "extra_water_assignments") for row in group
            ),
            "mean_extra_harvest_assignments": statistics.mean(
                diagnostic_value(row, "extra_harvest_assignments") for row in group
            ),
            "mean_extra_drop_assignments": statistics.mean(
                diagnostic_value(row, "extra_drop_assignments") for row in group
            ),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "baseline/artifacts/v011_crop_scheduler")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    args = parser.parse_args()

    modules = {
        name: load_module(path, f"v011_{name}")
        for name, path in DEFAULT_CANDIDATES.items()
        if name in args.candidates
    }
    unknown = sorted(set(args.candidates) - set(modules))
    if unknown:
        raise ValueError(f"Unknown V011 candidates: {unknown}")
    opponents = load_opponents(args.opponents)
    total = len(modules) * len(opponents) * len(args.seeds) * 2
    rows = []
    index = 0
    for candidate_name, module in modules.items():
        for opponent_name, opponent in opponents.items():
            for seed in args.seeds:
                for seat in (0, 1):
                    row = run_game(module, opponent, seed, seat)
                    row.update({"candidate": candidate_name, "opponent": opponent_name})
                    rows.append(row)
                    index += 1
                    print(
                        f"[{index}/{total}] {candidate_name} vs {opponent_name} "
                        f"seed={seed} seat={seat} money={row['candidate_money']:.0f} "
                        f"status={row['candidate_status']}",
                        flush=True,
                    )
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_csv(args.out / "v011_raw.csv", rows)
    write_csv(args.out / "v011_summary.csv", summary)
    (args.out / "v011_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
