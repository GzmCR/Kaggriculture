"""Run the V006 ablation matrix and write machine-readable results.

The Hamburger opponent is reconstructed from the notebook's compressed
benchmark source.  Its fixed trace is used only as an opponent; no trace is
imported by any candidate agent.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import gzip
import importlib.util
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (17, 42, 2026, 217, 317, 733)
DEFAULT_OPPONENTS = ("starter", "random", "hamburger")
EPISODE_STEPS = 720


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} must define agent(obs, config=None)")
    return module


def _notebook_blob(notebook_text: str, name: str) -> str:
    match = re.search(name + r" = '([^']+)'", notebook_text)
    if match is None:
        raise ValueError(f"Missing {name} in Hamburger notebook")
    return match.group(1)


def load_hamburger_agent(notebook_path: Path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    anchor = gzip.decompress(
        base64.b64decode(_notebook_blob(cell_text, "ANCHOR_BLOB"))
    ).decode("utf-8")
    wrapper = gzip.decompress(
        base64.b64decode(_notebook_blob(cell_text, "WRAPPER_BLOB"))
    ).decode("utf-8")
    core_match = re.search(
        r"CORE_ACTIONS = (.*?)\nCANDIDATE_SPECS", cell_text, re.S
    )
    if core_match is None:
        raise ValueError("Missing Hamburger CORE_ACTIONS")
    core_actions = ast.literal_eval(core_match.group(1).strip())

    tree = ast.parse(anchor)
    trace = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "TRACE_ACTIONS"
        ):
            trace = ast.literal_eval(node.value)
            break
    if not isinstance(trace, list) or len(trace) != EPISODE_STEPS:
        raise ValueError("Hamburger anchor trace is not 720 steps")

    source = anchor.rsplit("\ndef agent(obs, config=None):", 1)[0]
    wrapper = (
        wrapper.replace("__CORE_ACTIONS__", repr(core_actions))
        .replace("__ADAPTIVE_TRIAD__", "True")
        .replace("__TREASURY_START__", "710")
        .replace("__TREASURY_FLUSH__", "718")
        .replace("__TERMINAL_WORK_START__", "704")
        .replace("__CLONE_FRONT_RUN__", "True")
        .replace("__FRONT_RUN_HORIZON__", "1")
        .replace(
            "__FRONT_RUN_ITEMS__",
            repr(("MELON", "STRAWBERRY", "MILK", "WOOL")),
        )
    )
    namespace = {"__name__": "hamburger_v24_benchmark"}
    exec(source + "\n" + wrapper, namespace)
    agent = namespace.get("agent")
    if not callable(agent):
        raise AttributeError("Hamburger source did not define agent")
    return agent


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


class TracedAgent:
    def __init__(self, module):
        self.module = module
        self.calls = 0
        self.errors = 0
        self.invalid = 0
        self.times_ms = []
        self.field_counts = Counter()
        self.market_counts = Counter()
        self.market_quantities = Counter()

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        try:
            action = self.module.agent(obs, config)
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        self.calls += 1
        self.times_ms.append(elapsed)
        if not isinstance(action, dict):
            self.invalid += 1
            return action
        farmer = action.get("farmer", [])
        hands = action.get("hands", []) or []
        market = action.get("market", []) or []
        if not isinstance(farmer, list) or not isinstance(hands, list):
            self.invalid += 1
        for operation in [farmer, *hands]:
            if isinstance(operation, list) and operation:
                self.field_counts[str(operation[0])] += 1
        if not isinstance(market, list):
            self.invalid += 1
        else:
            for order in market:
                if not isinstance(order, list) or not order:
                    self.invalid += 1
                    continue
                operation = str(order[0])
                self.market_counts[operation] += 1
                if len(order) >= 3 and operation in {
                    "BUY_SEED",
                    "BUY_PRODUCT",
                    "BUY_ANIMAL",
                    "SELL",
                }:
                    self.market_quantities[
                        (operation, str(order[1]))
                    ] += int(order[2] or 0)
        return action

    def timing(self):
        return {
            "p50": percentile(self.times_ms, 0.50),
            "p95": percentile(self.times_ms, 0.95),
            "p99": percentile(self.times_ms, 0.99),
            "max": max(self.times_ms or [0.0]),
        }


def _counter_json(counter):
    return json.dumps(dict(sorted(counter.items())), sort_keys=True)


def _quantity_json(counter):
    values = {
        f"{operation}:{item}": quantity
        for (operation, item), quantity in sorted(counter.items())
    }
    return json.dumps(values, sort_keys=True)


def run_game(module, opponent, hamburger, seed, seat):
    candidate = TracedAgent(module)
    if opponent == "hamburger":
        opponent_agent = hamburger
    else:
        opponent_agent = opponent
    agents = [candidate, opponent_agent] if seat == 0 else [
        opponent_agent,
        candidate,
    ]
    started = time.perf_counter()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
        debug=False,
    )
    env.run(agents)
    elapsed = time.perf_counter() - started
    final = env.steps[-1]
    candidate_state = final[seat]
    opponent_state = final[1 - seat]
    candidate_money = float(
        candidate_state.observation["farms"][seat]["money"]
    )
    opponent_money = float(
        opponent_state.observation["farms"][1 - seat]["money"]
    )
    margin = candidate_money - opponent_money
    if margin > 0:
        result = "win"
    elif margin < 0:
        result = "loss"
    else:
        result = "tie"
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
            candidate_state.status == "DONE"
            and opponent_state.status == "DONE"
        ),
        "action_calls": candidate.calls,
        "agent_errors": candidate.errors,
        "invalid_action_shapes": candidate.invalid,
        "runtime_p50_ms": timing["p50"],
        "runtime_p95_ms": timing["p95"],
        "runtime_p99_ms": timing["p99"],
        "runtime_max_ms": timing["max"],
        "wall_seconds": elapsed,
        "field_counts": _counter_json(candidate.field_counts),
        "market_counts": _counter_json(candidate.market_counts),
        "market_quantities": _quantity_json(candidate.market_quantities),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
                "mean_money": statistics.mean(
                    row["candidate_money"] for row in group
                ),
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
                "mean_calls": statistics.mean(
                    row["action_calls"] for row in group
                ),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "baseline/artifacts/v006_hamburger_transplant",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=DEFAULT_OPPONENTS,
        default=list(DEFAULT_OPPONENTS),
    )
    parser.add_argument("--candidates", nargs="+", default=None)
    args = parser.parse_args()

    candidates = {
        "current": ROOT / "main.py",
        "v006a_fertilizer_terminal": ROOT
        / "baseline/history/v006a_fertilizer_terminal/main.py",
        "v006b_livestock_wheat": ROOT
        / "baseline/history/v006b_livestock_wheat/main.py",
        "v006c_combined": ROOT
        / "baseline/history/v006c_combined/main.py",
        # Diagnostic-only livestock/wheat ablations; these are not promotion
        # candidates in the V006 gate.
        "v006b_livestock_only": ROOT
        / "baseline/history/v006b_livestock_only/main.py",
        "v006b_wheat_only": ROOT
        / "baseline/history/v006b_wheat_only/main.py",
    }
    if args.candidates:
        candidates = {
            name: candidates[name] for name in args.candidates
        }
    modules = {
        name: load_module(path, "v006_" + name.replace("-", "_"))
        for name, path in candidates.items()
    }
    hamburger = load_hamburger_agent(
        ROOT / "baseline/kaggriculture-hamburger.ipynb"
    )

    rows = []
    total = len(modules) * len(args.opponents) * len(args.seeds) * 2
    index = 0
    for candidate_name, module in modules.items():
        for opponent in args.opponents:
            for seed in args.seeds:
                for seat in (0, 1):
                    index += 1
                    print(
                        f"[{index}/{total}] {candidate_name} vs "
                        f"{opponent} seed={seed} seat={seat}",
                        flush=True,
                    )
                    row = run_game(module, opponent, hamburger, seed, seat)
                    row.update({"candidate": candidate_name, "opponent": opponent})
                    rows.append(row)

    row_fields = [
        "candidate", "opponent", "seed", "seat", "candidate_money",
        "opponent_money", "margin", "result", "candidate_status",
        "opponent_status", "game_done", "action_calls", "agent_errors",
        "invalid_action_shapes", "runtime_p50_ms", "runtime_p95_ms",
        "runtime_p99_ms", "runtime_max_ms", "wall_seconds", "field_counts",
        "market_counts", "market_quantities",
    ]
    summary = summarize(rows)
    summary_fields = list(summary[0]) if summary else []
    write_csv(args.out / "v006_results.csv", rows, row_fields)
    write_csv(args.out / "v006_summary.csv", summary, summary_fields)
    (args.out / "v006_results.json").write_text(json.dumps(rows, indent=2))
    (args.out / "v006_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"WROTE {args.out / 'v006_results.csv'}")
    print(f"WROTE {args.out / 'v006_summary.csv'}")


if __name__ == "__main__":
    main()
