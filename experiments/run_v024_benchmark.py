"""Run V024 smoke, local matrix, and fixed-replay counterfactual tests.

Examples:
  python experiments/run_v024_benchmark.py --smoke
  python experiments/run_v024_benchmark.py --ours
  python experiments/run_v024_benchmark.py --matrix --candidate v024b_route14_weed --opponent starter
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
SEEDS = (17, 42, 2026, 217, 317, 733)
CANDIDATES = (
    "v022c_control",
    "v024a_route14_control", "v024b_route14_weed",
    "v024c_route14_order_memory", "v024d_route14_strict_r3",
    "v025a_route14_v022c_market", "v025b_route14_v022c_open_market",
    "v025c_route14_v022c_mirror_market",
)
OPPONENTS = ("v022c", "v012", "root", "starter", "random")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_path(name):
    if name == "v022c_control":
        return ROOT / "baseline/history/v022c_medoid_recovery/main.py"
    if name.startswith("v025"):
        return ROOT / "baseline/history" / name / "main.py"
    return ROOT / "baseline/history" / name / "main.py"


def _action(value):
    if not isinstance(value, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (value.get("hands") or [])],
        "market": [list(item) for item in (value.get("market") or []) if isinstance(item, list) and item],
    }


class Probe:
    def __init__(self, function):
        self.function = function
        self.times = []
        self.errors = 0
        self.actions = Counter()
        self.market = Counter()

    def __call__(self, obs, config=None):
        start = time.perf_counter_ns()
        try:
            try:
                value = self.function(obs, config)
            except TypeError:
                value = self.function(obs)
        except Exception:
            self.errors += 1
            value = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times.append((time.perf_counter_ns() - start) / 1_000_000)
        value = _action(value)
        for operation in [value["farmer"], *value["hands"]]:
            if operation:
                self.actions[str(operation[0])] += 1
        for order in value["market"]:
            self.market[str(order[0])] += 1
            if len(order) >= 3 and str(order[0]).upper() == "SELL":
                try:
                    self.market[f"SELL_{str(order[1]).upper()}"] += max(0, int(order[2]))
                except (TypeError, ValueError):
                    pass
        return value

    def metrics(self):
        ordered = sorted(self.times)
        def pct(q):
            return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))] if ordered else 0.0
        return {
            "calls": len(ordered), "p50_ms": pct(.50), "p95_ms": pct(.95),
            "p99_ms": pct(.99), "max_ms": max(ordered or [0.0]),
            "actions": dict(self.actions), "market": dict(self.market),
        }


class ReplayAgent:
    def __init__(self, actions):
        self.actions = actions

    def __call__(self, obs, config=None):
        step = max(0, min(EPISODE_STEPS - 1, int(obs.get("step", 0) or 0)))
        return _action(self.actions[step])


def _module_agent(path, tag):
    return load_module(path, f"v024_{tag}_{time.time_ns()}").agent


def opponent(name):
    if name in {"starter", "random"}:
        return name
    paths = {
        "v022c": ROOT / "baseline/history/v022c_medoid_recovery/main.py",
        "v012": ROOT / "baseline/history/v012_top5_replaced_v18/main.py",
        "root": ROOT / "main.py",
    }
    if name not in paths:
        raise ValueError(name)
    return _module_agent(paths[name], name)


def play(agent, other, seed, seat, candidate, opponent_name, source):
    probe = Probe(agent)
    players = [probe, other] if seat == 0 else [other, probe]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine = final[seat]
    theirs = final[1 - seat]
    mine_money = float(mine.observation["farms"][seat]["money"])
    other_money = float(theirs.observation["farms"][1 - seat]["money"])
    row = {
        "candidate": candidate, "opponent": opponent_name, "source": source,
        "seed": seed, "seat": seat, "candidate_money": mine_money,
        "opponent_money": other_money, "margin": mine_money - other_money,
        "result": "win" if mine_money > other_money else "loss" if mine_money < other_money else "tie",
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "agent_errors": getattr(import_module_for_probe(probe), "_V024_STATS", {}).get("errors", 0),
        **probe.metrics(),
    }
    # The module diagnostics are attached by the caller after play; this field
    # remains useful for runners that only consume the CSV.
    return row


def import_module_for_probe(probe):
    # Probe keeps the bound method so its module can be recovered without a
    # global registry.  It is intentionally defensive for string opponents.
    function = getattr(probe.function, "__self__", None)
    return function if function is not None and hasattr(function, "_V024_STATS") else getattr(probe.function, "_V024_MODULE", object())


def _run_one(candidate, other, seed, seat, opponent_name, source):
    module = load_module(candidate_path(candidate), f"v024_run_{candidate}_{time.time_ns()}")
    probe = Probe(module.agent)
    players = [probe, other] if seat == 0 else [other, probe]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine, theirs = final[seat], final[1 - seat]
    mine_money = float(mine.observation["farms"][seat]["money"])
    other_money = float(theirs.observation["farms"][1 - seat]["money"])
    diagnostics = getattr(module, "_V024_STATS", None)
    if diagnostics is None:
        diagnostics = getattr(module, "_V025_STATS", {})
    return {
        "candidate": candidate, "opponent": opponent_name, "source": source,
        "seed": seed, "seat": seat, "candidate_money": mine_money,
        "opponent_money": other_money, "margin": mine_money - other_money,
        "result": "win" if mine_money > other_money else "loss" if mine_money < other_money else "tie",
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "candidate_status": mine.status, "opponent_status": theirs.status,
        "agent_errors": diagnostics.get("errors", 0),
        **{f"diag_{key}": value for key, value in diagnostics.items()},
        **probe.metrics(),
    }


def _unique_replays(folder):
    result = {}
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        info = payload.get("info", {}) or {}
        episode = str(info.get("EpisodeId", payload.get("id", path.stem)))
        result.setdefault(episode, (path, payload))
    return list(result.values())


def _fixed_actions(payload, side):
    steps = payload.get("steps", [])
    return [
        _action(steps[min(step + 1, len(steps) - 1)][side].get("action") if steps else None)
        for step in range(EPISODE_STEPS)
    ]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    flat = []
    for row in rows:
        item = dict(row)
        for key in ("actions", "market"):
            if key in item:
                item[key] = repr(item[key])
        flat.append(item)
    fields = sorted({key for row in flat for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)


def summary(rows, keys=("candidate", "opponent")):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for group_key, group in sorted(groups.items()):
        results = Counter(row["result"] for row in group)
        output.append({
            **dict(zip(keys, group_key)), "games": len(group),
            "mean_money": statistics.mean(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "min_money": min(row["candidate_money"] for row in group),
            "wins": results["win"], "ties": results["tie"], "losses": results["loss"],
            "win_rate": results["win"] / len(group),
            "all_done": int(all(row["done"] for row in group)),
            "errors": sum(row.get("agent_errors", 0) for row in group),
            "p99_ms": max(row.get("p99_ms", 0) for row in group),
        })
    return output


def run_matrix(candidates, opponents, seeds, output):
    rows = []
    total = len(candidates) * len(opponents) * len(seeds) * 2
    index = 0
    for candidate in candidates:
        for opponent_name in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    print(f"[{index}/{total}] {candidate} vs {opponent_name} seed={seed} seat={seat}", flush=True)
                    rows.append(_run_one(candidate, opponent(opponent_name), seed, seat, opponent_name, "local"))
    write_csv(output / "matrix_raw.csv", rows)
    write_csv(output / "matrix_summary.csv", summary(rows))
    return rows


def run_fixed(candidates, folder, output, tail=None):
    rows = []
    pairs = _unique_replays(folder)
    if tail is not None:
        pairs = pairs[-int(tail):]
    total = len(candidates) * len(pairs) * 2
    index = 0
    for candidate in candidates:
        for path, payload in pairs:
            info = payload.get("info", {}) or {}
            seed = int(info.get("seed", 0) or 0)
            for seat in (0, 1):
                index += 1
                print(f"[{index}/{total}] {candidate} fixed={path.name} seat={seat}", flush=True)
                other = ReplayAgent(_fixed_actions(payload, 1 - seat))
                rows.append(_run_one(candidate, other, seed, seat, "fixed_replay", path.name))
    write_csv(output / "ours_counterfactual_raw.csv", rows)
    write_csv(output / "ours_counterfactual_summary.csv", summary(rows, ("candidate",)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--ours", action="store_true")
    mode.add_argument("--future", action="store_true")
    mode.add_argument("--matrix", action="store_true")
    parser.add_argument("--candidate", action="append", choices=CANDIDATES)
    parser.add_argument("--opponent", action="append", choices=OPPONENTS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v024_14hands_route")
    args = parser.parse_args()
    candidates = tuple(args.candidate or CANDIDATES)
    if args.smoke:
        rows = run_matrix(candidates, ("starter",), (17,), args.output)
    elif args.ours:
        rows = run_fixed(candidates, ROOT / "log/2026-08-07/ours", args.output)
    elif args.future:
        # The newest 15% is the frozen future-holdout slice (7 episodes from
        # the current 44-episode Top10 set).
        rows = run_fixed(candidates, ROOT / "log/2026-08-07/top10", args.output, tail=7)
    else:
        rows = run_matrix(candidates, tuple(args.opponent or OPPONENTS), tuple(args.seed or SEEDS), args.output)
    print(f"V024 benchmark complete: {len(rows)} games")


if __name__ == "__main__":
    main()
