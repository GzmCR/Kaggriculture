"""Replace v18's embedded experts with five current replay routes and compare.

The board executor is locked to the strongest standalone route found in V012
(automatylicza).  The five replay bundles form the v18-style daily market
expert bank.  This preserves route coherence while testing whether the newer
market policies improve on the original four-expert v18 artifact.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import json
import statistics
import time
import zlib
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make

from run_v012_top5_vs_v18 import (
    DEFAULT_SEEDS,
    EPISODE_STEPS,
    LOG_DIR,
    ROOT,
    SELECTED_SPECS,
    load_replay_trace,
)


V18_NOTEBOOK = ROOT / "baseline/40-53-top-10-future-holdout-v18-closed-loop.ipynb"
OUT_DIR = ROOT / "baseline/artifacts/v012_top5_replaced_v18"
BOARD_ROUTE = "automatylicza"


def load_v18_namespace():
    notebook = json.loads(V18_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "payload =" not in text or "b85decode" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id != "payload":
                continue
            packed = "".join(ast.literal_eval(node.value.args[0]))
            source = zlib.decompress(base64.b85decode(packed)).decode("utf-8")
            namespace = {"__name__": "v012_replaced_v18"}
            exec(compile(source, str(V18_NOTEBOOK), "exec"), namespace)
            return namespace
    raise RuntimeError("Could not decode the embedded v18 source")


def build_replaced_namespace(routes):
    namespace = load_v18_namespace()
    original_runtime = namespace["_V18_RUNTIME"]
    experts = {}
    for name, route in routes.items():
        features = [namespace["_v18_state_features"](obs) for obs in route["observations"]]
        experts[name] = {
            "actions": route["actions"],
            "prototypes_by_day": [features[min(day * 24, EPISODE_STEPS - 1)] for day in range(30)],
            "board_prototype_at_fork": features[min(632, EPISODE_STEPS - 1)],
        }

    runtime = copy.deepcopy(original_runtime)
    runtime["experts"] = experts
    runtime["board_by_seat"] = {"0": BOARD_ROUTE, "1": BOARD_ROUTE}
    runtime["board_distance_strength"] = 0.0
    runtime["distance_strength"] = 0.5
    runtime["stay_bonus"] = 0.5
    # The standalone V012-v18 comparison is the outcome-training split for
    # this local experiment.  Without a prior, all day-0 prototypes are
    # nearly tied and the lexicographic gate picked navazsh_fathi in every
    # game.  Give the strongest complete route a modest prior; the public
    # state-distance term can still move the market lane later.
    route_bias = {name: 0.0 for name in experts}
    route_bias[BOARD_ROUTE] = 0.75
    runtime["market_bias_by_seat"] = {"0": dict(route_bias), "1": dict(route_bias)}
    runtime["board_bias_by_seat"] = {"0": dict(route_bias), "1": dict(route_bias)}
    namespace["_V18_RUNTIME"] = runtime
    namespace["STRATEGY"]["v18_closed_loop_board"] = True
    namespace["STRATEGY"]["v18_closed_loop_market"] = True
    return namespace


class CandidateProbe:
    def __init__(self, namespace):
        self.namespace = namespace
        self.agent = namespace["agent"]
        self.calls = 0
        self.errors = 0
        self.times_ms = []
        self.route_history = []
        self._last_route = None

    def __call__(self, obs, config=None):
        step = int(obs.get("step", 0) or 0)
        if step == 0:
            self.calls = 0
            self.errors = 0
            self.times_ms = []
            self.route_history = []
            self._last_route = None
        started = time.perf_counter_ns()
        try:
            action = self.agent(obs)
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        self.calls += 1
        player = int(obs.get("player", 0) or 0)
        selected = (self.namespace.get("_V18_SELECTED_MARKET") or {}).get(player)
        if selected != self._last_route:
            self.route_history.append({"step": step, "day": int(obs.get("day", 0) or 0), "route": selected})
            self._last_route = selected
        return action


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def run_game(candidate_namespace, opponent_namespace, seed, seat):
    candidate = CandidateProbe(candidate_namespace)
    opponent = opponent_namespace["agent"]
    players = [candidate, opponent] if seat == 0 else [opponent, candidate]
    started = time.perf_counter()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
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
    return {
        "seed": int(seed),
        "seat": int(seat),
        "candidate_money": candidate_money,
        "opponent_money": opponent_money,
        "margin": margin,
        "result": result,
        "candidate_status": candidate_state.status,
        "opponent_status": opponent_state.status,
        "game_done": int(candidate_state.status == "DONE" and opponent_state.status == "DONE"),
        "action_calls": candidate.calls,
        "agent_errors": candidate.errors,
        "route_switches": max(0, len(candidate.route_history) - 1),
        "route_history": json.dumps(candidate.route_history, sort_keys=True),
        "runtime_p50_ms": percentile(candidate.times_ms, 0.50),
        "runtime_p95_ms": percentile(candidate.times_ms, 0.95),
        "runtime_p99_ms": percentile(candidate.times_ms, 0.99),
        "runtime_max_ms": max(candidate.times_ms or [0.0]),
        "wall_seconds": time.perf_counter() - started,
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["candidate"]].append(row)
    summary = []
    for name, group in groups.items():
        wins = sum(row["result"] == "win" for row in group)
        summary.append({
            "candidate": name,
            "opponent": "original_v18",
            "games": len(group),
            "mean_cash": statistics.mean(row["candidate_money"] for row in group),
            "mean_opponent_cash": statistics.mean(row["opponent_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "min_cash": min(row["candidate_money"] for row in group),
            "wins": wins,
            "ties": sum(row["result"] == "tie" for row in group),
            "losses": sum(row["result"] == "loss" for row in group),
            "win_rate": wins / len(group) if group else 0.0,
            "done_rate": statistics.mean(row["game_done"] for row in group),
            "agent_errors": sum(row["agent_errors"] for row in group),
            "max_route_switches": max(row["route_switches"] for row in group),
            "max_p99_ms": max(row["runtime_p99_ms"] for row in group),
            "max_runtime_ms": max(row["runtime_max_ms"] for row in group),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    routes = {spec["name"]: load_replay_trace(spec) for spec in SELECTED_SPECS}
    candidate_namespace = build_replaced_namespace(routes)
    opponent_namespace = load_v18_namespace()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "experts.json").write_text(
        json.dumps({
            "board_route": BOARD_ROUTE,
            "market_experts": list(routes),
            "seeds": list(args.seeds),
            "seats": [0, 1],
        }, indent=2),
        encoding="utf-8",
    )

    rows = []
    total = len(args.seeds) * 2
    completed = 0
    for seed in args.seeds:
        for seat in (0, 1):
            row = run_game(candidate_namespace, opponent_namespace, seed, seat)
            row.update({"candidate": "top5_replaced_v18", "route": BOARD_ROUTE})
            rows.append(row)
            completed += 1
            print(
                f"[{completed}/{total}] top5_replaced_v18 vs original_v18 "
                f"seed={seed} seat={seat} cash={row['candidate_money']:.0f} "
                f"margin={row['margin']:.0f} route_switches={row['route_switches']} "
                f"status={row['candidate_status']}",
                flush=True,
            )

    summary = summarize(rows)
    write_csv(args.out / "v012_replaced_raw.csv", rows)
    write_csv(args.out / "v012_replaced_summary.csv", summary)
    (args.out / "v012_replaced_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
