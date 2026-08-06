"""Run V022 smoke tests, replay counterfactuals, and paired local games."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from run_v006_benchmark import load_hamburger_agent
from run_v008_benchmark import load_notebook_agent
from run_v012_top5_vs_v18 import load_v18_agent


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
SEEDS = (17, 42, 2026, 217, 317, 733)
CANDIDATES = {
    name: ROOT / "baseline/artifacts" / name / "main.py"
    for name in (
        "v022a_weed_recovery",
        "v022b_fresh_medoid",
        "v022c_medoid_recovery",
        "v022d_medoid_recovery_tactical",
    )
}
CONTROL = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
ROOT_BASELINE = ROOT / "main.py"
OPPONENTS = ("control", "baseline", "v18", "hamburger", "frontier", "starter", "random")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
    return values[index]


class Probe:
    def __init__(self, function):
        self.function = function
        self.times = []
        self.errors = 0
        self.invalid = 0
        self.actions = Counter()

    def __call__(self, obs, config=None):
        start = time.perf_counter_ns()
        try:
            try:
                action = self.function(obs, config)
            except TypeError:
                action = self.function(obs)
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times.append((time.perf_counter_ns() - start) / 1_000_000)
        if not isinstance(action, dict) or not isinstance(action.get("market", []), list):
            self.invalid += 1
            return {"farmer": ["PASS"], "hands": [], "market": []}
        for op in [action.get("farmer", []), *(action.get("hands", []) or [])]:
            if isinstance(op, list) and op:
                self.actions[op[0]] += 1
        return action

    def metrics(self):
        return {
            "calls": len(self.times),
            "errors": self.errors,
            "invalid": self.invalid,
            "p50_ms": percentile(self.times, 0.50),
            "p95_ms": percentile(self.times, 0.95),
            "p99_ms": percentile(self.times, 0.99),
            "max_ms": max(self.times or [0.0]),
            "actions": dict(self.actions),
        }


class PassAgent:
    def __call__(self, obs, config=None):
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0) or 0)
        hands = len((farms[player] if player < len(farms) else {}).get("hands", []) or [])
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(hands)], "market": []}


def _opponent(name):
    if name == "random":
        return "random"
    if name == "starter":
        return "starter"
    if name == "control":
        return load_module(CONTROL, f"v022_control_opponent_{time.time_ns()}").agent
    if name == "baseline":
        return load_module(ROOT_BASELINE, f"v022_root_opponent_{time.time_ns()}").agent
    if name == "v18":
        return load_v18_agent()
    if name == "hamburger":
        return load_hamburger_agent(ROOT / "baseline/kaggriculture-hamburger.ipynb")
    if name == "frontier":
        return load_notebook_agent(
            ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb",
            f"v022_frontier_{time.time_ns()}",
        )
    raise ValueError(f"unknown opponent: {name}")


def run_game(candidate, opponent, seed, seat):
    probe = Probe(candidate)
    players = [probe, opponent] if seat == 0 else [opponent, probe]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine, other = final[seat], final[1 - seat]
    money = float(mine.observation["farms"][seat]["money"])
    other_money = float(other.observation["farms"][1 - seat]["money"])
    margin = money - other_money
    metrics = probe.metrics()
    return {
        "seed": seed,
        "seat": seat,
        "candidate_money": money,
        "opponent_money": other_money,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "candidate_status": mine.status,
        "opponent_status": other.status,
        "done": int(mine.status == "DONE" and other.status == "DONE"),
        **metrics,
    }


def flatten(row):
    output = dict(row)
    output["actions"] = repr(output.get("actions", {}))
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    rows = [flatten(row) for row in rows]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["opponent"])].append(row)
    summary = []
    for (candidate, opponent), group in sorted(grouped.items()):
        outcomes = Counter(row["result"] for row in group)
        summary.append({
            "candidate": candidate,
            "opponent": opponent,
            "games": len(group),
            "mean_money": statistics.mean(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "min_money": min(row["candidate_money"] for row in group),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(group),
            "all_done": int(all(row["done"] for row in group)),
            "errors": sum(row["errors"] for row in group),
            "invalid": sum(row["invalid"] for row in group),
            "p99_ms": max(row["p99_ms"] for row in group),
        })
    return summary


def run_matrix(names, opponents, seeds, output):
    rows = []
    raw_path = output / "matrix_raw.csv"
    total = len(names) * len(opponents) * len(seeds) * 2
    index = 0
    for candidate_name in names:
        for opponent_name in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    module = load_module(
                        CONTROL if candidate_name == "control" else CANDIDATES[candidate_name],
                        f"v022_{candidate_name}_{index}_{time.time_ns()}",
                    )
                    opponent = _opponent(opponent_name)
                    print(f"[{index}/{total}] {candidate_name} vs {opponent_name} seed={seed} seat={seat}", flush=True)
                    row = run_game(module.agent, opponent, seed, seat)
                    diagnostics = getattr(module, "_V022_DIAGNOSTICS", {})
                    for key, value in diagnostics.items():
                        row[f"v022_{key}"] = value
                    row.update({"candidate": candidate_name, "opponent": opponent_name})
                    rows.append(row)
                    if index % 10 == 0:
                        write_csv(raw_path, rows)
    write_csv(raw_path, rows)
    write_csv(output / "matrix_summary.csv", summarize(rows))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", choices=("control", *CANDIDATES), dest="candidates")
    parser.add_argument("--opponent", action="append", choices=OPPONENTS, dest="opponents")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v022_fresh_route")
    args = parser.parse_args()
    candidates = tuple(args.candidates or ("control", *CANDIDATES))
    opponents = tuple(args.opponents or OPPONENTS)
    seeds = tuple(args.seeds or SEEDS)
    rows = run_matrix(candidates, opponents, seeds, args.output)
    print(f"V022 matrix complete: {len(rows)} games")


if __name__ == "__main__":
    main()
