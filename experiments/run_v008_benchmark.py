"""Benchmark V008 hybrid-router candidates against all required opponents."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from run_v006_benchmark import load_hamburger_agent, percentile


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
DEFAULT_SEEDS = (17, 42, 2026, 217, 317, 733)
DEFAULT_OPPONENTS = ("starter", "random", "hamburger", "builder", "frontier")
DEFAULT_CANDIDATES = (
    "v008_current",
    "v008_q25",
    "v008_q50",
    "v008_q75",
    "v008_q90",
    "v008_q100",
    "v008_frontier",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} must define agent(obs, config=None)")
    return module


def load_notebook_agent(path: Path, name: str):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "AGENT_SOURCE" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                getattr(target, "id", None) == "AGENT_SOURCE"
                for target in node.targets
            ):
                continue
            source = ast.literal_eval(node.value)
            namespace = {"__name__": name}
            exec(compile(source, str(path), "exec"), namespace)
            agent = namespace.get("agent")
            if not callable(agent):
                raise AttributeError(f"{path} source did not define agent")
            return agent
    raise ValueError(f"AGENT_SOURCE not found in {path}")


def _counter_json(counter):
    return json.dumps(dict(sorted(counter.items())), sort_keys=True)


def _quantity_json(counter):
    values = {
        f"{operation}:{item}": quantity
        for (operation, item), quantity in sorted(counter.items())
    }
    return json.dumps(values, sort_keys=True)


def _valid_shape(action, obs, config):
    if not isinstance(action, dict):
        return False
    if not isinstance(action.get("farmer"), list):
        return False
    hands = action.get("hands", [])
    if not isinstance(hands, list):
        return False
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    expected_hands = (
        len(farms[player].get("hands", []) or [])
        if 0 <= player < len(farms)
        else 0
    )
    if len(hands) != expected_hands:
        return False
    market = action.get("market", [])
    if not isinstance(market, list):
        return False
    for operation in [action.get("farmer", []), *(action.get("hands", []) or [])]:
        if not isinstance(operation, list) or not operation:
            return False
    max_orders = int((config or {}).get("maxMarketOrdersPerTurn", 10) or 10)
    if len(market) > max_orders:
        return False
    for order in market:
        if not isinstance(order, list) or not order:
            return False
    return True


class CandidateProbe:
    def __init__(self, module):
        self.module = module
        self.calls = 0
        self.errors = 0
        self.invalid = 0
        self.times_ms = []
        self.field_counts = Counter()
        self.market_counts = Counter()
        self.market_quantities = Counter()
        self.route_history = []
        self.daily_cash = {}
        self._route_seen = 0

    def __call__(self, obs, config=None):
        step = int(obs.get("step", 0) or 0)
        if step == 0:
            self.route_history = []
            self.daily_cash = {}
            self._route_seen = 0
        if step % 24 == 23:
            player = int(obs.get("player", 0) or 0)
            farms = obs.get("farms", []) or []
            if 0 <= player < len(farms):
                self.daily_cash[str(int(obs.get("day", step // 24) or 0))] = float(
                    farms[player].get("money", 0)
                )

        started = time.perf_counter_ns()
        try:
            action = self.module.agent(obs, config)
        except Exception:
            self.errors += 1
            action = {
                "farmer": ["PASS"],
                "hands": [],
                "market": [],
            }
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        self.calls += 1
        self.times_ms.append(elapsed)
        if not _valid_shape(action, obs, config):
            self.invalid += 1
        if isinstance(action, dict):
            operations = [action.get("farmer", [])]
            operations.extend(action.get("hands", []) or [])
            for operation in operations:
                if isinstance(operation, list) and operation:
                    self.field_counts[str(operation[0])] += 1
            market = action.get("market", []) or []
            if isinstance(market, list):
                for order in market:
                    if not isinstance(order, list) or not order:
                        continue
                    operation = str(order[0])
                    self.market_counts[operation] += 1
                    if len(order) >= 3 and operation in {
                        "BUY_SEED",
                        "BUY_PRODUCT",
                        "BUY_ANIMAL",
                        "SELL",
                    }:
                        try:
                            quantity = int(order[2] or 0)
                        except (TypeError, ValueError):
                            quantity = 0
                        self.market_quantities[(operation, str(order[1]))] += quantity

        history = getattr(self.module, "V008_ROUTE_HISTORY", [])
        if isinstance(history, list) and len(history) > self._route_seen:
            self.route_history.extend(history[self._route_seen :])
            self._route_seen = len(history)
        return action

    def timing(self):
        return {
            "p50": percentile(self.times_ms, 0.50),
            "p95": percentile(self.times_ms, 0.95),
            "p99": percentile(self.times_ms, 0.99),
            "max": max(self.times_ms or [0.0]),
        }


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
    elapsed = time.perf_counter() - started
    final = env.steps[-1]
    candidate_state = final[seat]
    opponent_state = final[1 - seat]
    candidate_money = float(candidate_state.observation["farms"][seat]["money"])
    opponent_money = float(
        opponent_state.observation["farms"][1 - seat]["money"]
    )
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
        "game_done": int(
            candidate_state.status == "DONE" and opponent_state.status == "DONE"
        ),
        "action_calls": candidate.calls,
        "agent_errors": candidate.errors,
        "invalid_action_shapes": candidate.invalid,
        "runtime_p50_ms": timing["p50"],
        "runtime_p95_ms": timing["p95"],
        "runtime_p99_ms": timing["p99"],
        "runtime_max_ms": timing["max"],
        "wall_seconds": elapsed,
        "route_fallbacks": int(getattr(module, "V008_FALLBACKS", 0) or 0),
        "route_switches": int(getattr(module, "V008_ROUTE_SWITCHES", 0) or 0),
        "route_history": json.dumps(
            getattr(module, "V008_ROUTE_HISTORY", []), sort_keys=True
        ),
        "daily_cash": json.dumps(candidate.daily_cash, sort_keys=True),
        "field_counts": _counter_json(candidate.field_counts),
        "market_counts": _counter_json(candidate.market_counts),
        "market_quantities": _quantity_json(candidate.market_quantities),
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["opponent"])].append(row)
    summary = []
    for (candidate, opponent), group in sorted(grouped.items()):
        results = Counter(row["result"] for row in group)
        summary.append(
            {
                "candidate": candidate,
                "opponent": opponent,
                "games": len(group),
                "mean_money": statistics.mean(row["candidate_money"] for row in group),
                "min_money": min(row["candidate_money"] for row in group),
                "max_money": max(row["candidate_money"] for row in group),
                "mean_margin": statistics.mean(row["margin"] for row in group),
                "wins": results["win"],
                "ties": results["tie"],
                "losses": results["loss"],
                "win_rate": results["win"] / len(group),
                "all_done": int(all(row["game_done"] for row in group)),
                "agent_errors": sum(row["agent_errors"] for row in group),
                "invalid_action_shapes": sum(
                    row["invalid_action_shapes"] for row in group
                ),
                "mean_fallbacks": statistics.mean(
                    row["route_fallbacks"] for row in group
                ),
                "max_fallbacks": max(row["route_fallbacks"] for row in group),
                "mean_route_switches": statistics.mean(
                    row["route_switches"] for row in group
                ),
                "mean_calls": statistics.mean(row["action_calls"] for row in group),
                "p50_ms": percentile(
                    [row["runtime_p50_ms"] for row in group], 0.50
                ),
                "p95_ms": percentile(
                    [row["runtime_p95_ms"] for row in group], 0.95
                ),
                "p99_ms": max(row["runtime_p99_ms"] for row in group),
                "max_ms": max(row["runtime_max_ms"] for row in group),
            }
        )
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_opponents(names):
    paths = {
        "hamburger": ROOT / "baseline/kaggriculture-hamburger.ipynb",
        "builder": ROOT / "baseline/kaggriculture-agent-builder.ipynb",
        "frontier": ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb",
    }
    loaded = {}
    for name in names:
        if name in {"starter", "random"}:
            loaded[name] = name
        else:
            path = paths[name]
            if not path.exists():
                raise FileNotFoundError(f"Opponent notebook missing: {path}")
            if name == "hamburger":
                loaded[name] = load_hamburger_agent(path)
            else:
                loaded[name] = load_notebook_agent(path, f"v008_{name}")
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "baseline/artifacts/v008_hybrid_router")
    parser.add_argument("--candidate-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    args = parser.parse_args()

    candidate_dir = args.candidate_dir or args.out / "candidates"
    candidates = {
        name: candidate_dir / f"{name}.py" for name in args.candidates
    }
    modules = {
        name: load_module(path, f"v008_{name}")
        for name, path in candidates.items()
    }
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

    summary = summarize(rows)
    write_csv(args.out / "v008_raw.csv", rows)
    write_csv(args.out / "v008_summary.csv", summary)
    (args.out / "v008_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
