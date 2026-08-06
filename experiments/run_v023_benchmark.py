"""Run V023 route candidates against fixed replays and local opponents.

Examples (use the competition conda environment):

  python experiments/run_v023_benchmark.py --smoke
  python experiments/run_v023_benchmark.py --holdout
  python experiments/run_v023_benchmark.py --matrix --candidate v023d_early_portfolio --opponent frontier

The benchmark never passes replay metadata into a candidate.  Replay files
are used only by the offline ``ReplayAgent`` in the harness.
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

from run_v006_benchmark import load_hamburger_agent
from run_v008_benchmark import load_notebook_agent
from run_v012_top5_vs_v18 import load_v18_agent


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
SEEDS = (17, 42, 2026, 217, 317, 733)
CANDIDATE_NAMES = (
    "control",
    "v023a_high_output_14hands",
    "v023b_stable_12hands",
    "v023c_high_hands_15hands",
    "v023d_early_portfolio",
)
CANDIDATES = {
    "control": ROOT / "baseline/history/v012_top5_replaced_v18/main.py",
    **{
        name: ROOT / "baseline/artifacts" / name / "main.py"
        for name in CANDIDATE_NAMES[1:]
    },
}
OPPONENTS = (
    "control", "v022d", "v18", "hamburger", "frontier", "baseline", "starter", "random"
)
DATA_ROOT = ROOT / "log/2026-08-06"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _normalize_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or []) if isinstance(item, list) and item],
    }


class Probe:
    def __init__(self, function):
        self.function = function
        self.times = []
        self.errors = 0
        self.invalid = 0
        self.actions = Counter()
        self.market = Counter()

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
                self.actions[str(op[0])] += 1
        for order in action.get("market", []) or []:
            if isinstance(order, list) and order:
                self.market[str(order[0])] += 1
                if order[0] == "SELL" and len(order) >= 3:
                    self.market[f"SELL_{str(order[1]).upper()}"] += max(0, int(order[2] or 0))
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
            "market": dict(self.market),
        }


class ReplayAgent:
    def __init__(self, actions, side):
        self.actions = actions
        self.side = side

    def __call__(self, obs, config=None):
        step = max(0, min(EPISODE_STEPS - 1, int(obs.get("step", 0) or 0)))
        return _normalize_action(self.actions[step])


def _opponent(name):
    if name in ("random", "starter"):
        return name
    if name == "control":
        return load_module(CANDIDATES["control"], f"v023_control_{time.time_ns()}").agent
    if name == "v022d":
        return load_module(ROOT / "baseline/artifacts/v022d_medoid_recovery_tactical/main.py", f"v023_v022d_{time.time_ns()}").agent
    if name == "v18":
        return load_v18_agent()
    if name == "hamburger":
        return load_hamburger_agent(ROOT / "baseline/kaggriculture-hamburger.ipynb")
    if name == "frontier":
        return load_notebook_agent(ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb", f"v023_frontier_{time.time_ns()}")
    if name == "baseline":
        return load_module(ROOT / "main.py", f"v023_root_{time.time_ns()}").agent
    raise ValueError(name)


def _final_row(probe, final, seat, source, candidate, opponent):
    mine, other = final[seat], final[1 - seat]
    mine_money = float(mine.observation["farms"][seat]["money"])
    other_money = float(other.observation["farms"][1 - seat]["money"])
    margin = mine_money - other_money
    return {
        "candidate": candidate,
        "opponent": opponent,
        "source": source,
        "seat": seat,
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "candidate_status": mine.status,
        "opponent_status": other.status,
        "done": int(mine.status == "DONE" and other.status == "DONE"),
        **probe.metrics(),
    }


def run_game(agent, opponent, seed, seat, candidate, opponent_name, source="local"):
    probe = Probe(agent)
    players = [probe, opponent] if seat == 0 else [opponent, probe]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)}, debug=False)
    env.run(players)
    return _final_row(probe, env.steps[-1], seat, source, candidate, opponent_name)


def _replay_files(folder: Path):
    by_episode = {}
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        info = payload.get("info", {}) or {}
        episode = str(info.get("EpisodeId", payload.get("id", path.stem)))
        by_episode.setdefault(episode, (path, payload))
    return list(by_episode.values())


def run_replay_counterfactual(candidates, folder: Path, output: Path):
    rows = []
    pairs = _replay_files(folder)
    total = len(candidates) * len(pairs) * 2
    index = 0
    for candidate in candidates:
        for path, payload in pairs:
            steps = payload.get("steps", [])
            info = payload.get("info", {}) or {}
            seed = int(info.get("seed", 0) or 0)
            actions_by_side = []
            for side in (0, 1):
                actions_by_side.append([
                    steps[min(step + 1, len(steps) - 1)][side].get("action") if steps else {"farmer": ["PASS"], "hands": [], "market": []}
                    for step in range(EPISODE_STEPS)
                ])
            for seat in (0, 1):
                index += 1
                module = load_module(CANDIDATES[candidate], f"v023_cf_{candidate}_{index}_{time.time_ns()}")
                replay_opponent = ReplayAgent(actions_by_side[1 - seat], 1 - seat)
                print(f"[{index}/{total}] {candidate} fixed={path.stem} seat={seat}", flush=True)
                row = run_game(module.agent, replay_opponent, seed, seat, candidate, "fixed_replay", path.stem)
                diagnostics = getattr(module, "_V023_STATS", {})
                for key, value in diagnostics.items():
                    row[f"v023_{key}"] = value
                rows.append(row)
    write_csv(output / "replay_counterfactual_raw.csv", rows)
    write_csv(output / "replay_counterfactual_summary.csv", summarize(rows, ("candidate",)))
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    flattened = []
    for row in rows:
        item = dict(row)
        for key in ("actions", "market"):
            if key in item:
                item[key] = repr(item[key])
        flattened.append(item)
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def summarize(rows, group_keys=("candidate", "opponent")):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for group_key, group in sorted(grouped.items()):
        outcomes = Counter(row["result"] for row in group)
        row = dict(zip(group_keys, group_key))
        row.update({
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
        output.append(row)
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
                    module = load_module(CANDIDATES[candidate], f"v023_m_{candidate}_{index}_{time.time_ns()}")
                    opponent = _opponent(opponent_name)
                    print(f"[{index}/{total}] {candidate} vs {opponent_name} seed={seed} seat={seat}", flush=True)
                    row = run_game(module.agent, opponent, seed, seat, candidate, opponent_name)
                    for key, value in getattr(module, "_V023_STATS", {}).items():
                        row[f"v023_{key}"] = value
                    rows.append(row)
                    if index % 10 == 0:
                        write_csv(output / "matrix_raw.csv", rows)
    write_csv(output / "matrix_raw.csv", rows)
    write_csv(output / "matrix_summary.csv", summarize(rows))
    return rows


def run_smoke(output):
    rows = run_matrix(CANDIDATE_NAMES[1:], ("starter",), (17,), output)
    return rows


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--holdout", action="store_true")
    mode.add_argument("--matrix", action="store_true")
    parser.add_argument("--candidate", action="append", choices=CANDIDATE_NAMES)
    parser.add_argument("--opponent", action="append", choices=OPPONENTS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v023_route_portfolio")
    args = parser.parse_args()
    candidates = tuple(args.candidate or CANDIDATE_NAMES)
    if args.smoke:
        rows = run_smoke(args.output)
    elif args.holdout:
        rows = run_replay_counterfactual(candidates, DATA_ROOT / "1500～2500", args.output)
    else:
        rows = run_matrix(candidates, tuple(args.opponent or OPPONENTS), tuple(args.seed or SEEDS), args.output)
    print(f"V023 benchmark complete: {len(rows)} games")


if __name__ == "__main__":
    main()
