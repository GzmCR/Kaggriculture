"""Compare one representative replay route per selected player against v18.

The replay routes are intentionally kept as complete bundles.  This first
experiment does not splice farmer/hand actions from one replay with market
orders from another; it tells us whether any of the five source routes is a
stronger board-level candidate before building a router.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import copy
import json
import statistics
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "log/2026-08-04"
V18_NOTEBOOK = ROOT / "baseline/40-53-top-10-future-holdout-v18-closed-loop.ipynb"
OUT_DIR = ROOT / "baseline/artifacts/v012_top5_replay"
EPISODE_STEPS = 720
DEFAULT_SEEDS = (17, 42, 2026, 217, 317, 733)

# The five identities are selected from the available 15 replays by aggregate
# observed terminal cash.  The selected replay for each identity is the
# medoid of that player's own field+market traces; see README.md.
SELECTED_SPECS = (
    {"name": "mohit", "file": "89817349", "seat": 0},
    {"name": "automatylicza", "file": "89830916", "seat": 0},
    {"name": "manual_player", "file": "89820316", "seat": 0},
    {"name": "navazsh_fathi", "file": "89830910", "seat": 0},
    {"name": "lucien_de_rubempre", "file": "89822684", "seat": 1},
)


def load_replay_trace(spec):
    path = LOG_DIR / f"{spec['file']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    seat = int(spec["seat"])
    actions = []
    observations = []
    for index in range(EPISODE_STEPS):
        # Kaggle replay serialization stores the action that produced the
        # current observation on the following step entry.  The step-0 action
        # is an initial placeholder; use step+1 for executable transitions.
        observation_entry = steps[index][seat]
        action_entry = steps[min(index + 1, EPISODE_STEPS - 1)][seat]
        actions.append(copy.deepcopy(action_entry.get("action") or {}))
        observations.append(copy.deepcopy(observation_entry.get("observation") or {}))
    if len(actions) != EPISODE_STEPS:
        raise ValueError(f"{path} contains {len(actions)} actions, expected {EPISODE_STEPS}")
    return {
        **spec,
        "path": str(path),
        "actions": actions,
        "observations": observations,
        "terminal_cash": float(observations[-1]["farms"][seat]["money"]),
    }


def _pass():
    return ["PASS"]


def _prepare_action(action, obs, stats):
    """Preserve replay actions; record the serialized hand-list mismatch."""
    copied = copy.deepcopy(action) if isinstance(action, dict) else {}
    farmer = copied.get("farmer")
    copied["farmer"] = farmer if isinstance(farmer, list) and farmer else _pass()

    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    expected = 0
    if 0 <= player < len(farms):
        expected = len(farms[player].get("hands", []) or [])
    hands = copied.get("hands")
    hands = hands if isinstance(hands, list) else []
    if len(hands) != expected:
        stats["hand_shape_mismatches"] += 1
    copied["hands"] = [item if isinstance(item, list) and item else _pass() for item in hands]
    market = copied.get("market")
    copied["market"] = list(market[:10]) if isinstance(market, list) else []
    return copied


class ReplayAgent:
    def __init__(self, route):
        self.route = route
        self.stats = Counter()
        self.times_ms = []

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        step = max(0, min(int(obs.get("step", 0) or 0), EPISODE_STEPS - 1))
        if step == 0:
            self.stats.clear()
            self.times_ms.clear()
        self.stats["calls"] += 1
        action = _prepare_action(self.route["actions"][step], obs, self.stats)
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        return action


def load_v18_agent():
    """Decode the exact self-contained v18 source embedded in its notebook."""
    notebook = json.loads(V18_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "payload =" not in text or "b85decode" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "payload":
                continue
            value = node.value
            chunks = ast.literal_eval(value.args[0])
            packed = "".join(chunks)
            source = zlib.decompress(base64.b85decode(packed)).decode("utf-8")
            namespace = {"__name__": "v18_embedded"}
            exec(compile(source, str(V18_NOTEBOOK), "exec"), namespace)
            agent = namespace.get("agent")
            if not callable(agent):
                raise RuntimeError("Decoded v18 source does not define agent")
            return agent
    raise RuntimeError("Could not find embedded v18 payload")


def _valid_shape(action, obs):
    if not isinstance(action, dict):
        return False
    if not isinstance(action.get("farmer"), list) or not action["farmer"]:
        return False
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    expected = len(farms[player].get("hands", []) or []) if 0 <= player < len(farms) else 0
    hands = action.get("hands", [])
    if not isinstance(hands, list) or len(hands) != expected:
        return False
    market = action.get("market", [])
    if not isinstance(market, list) or len(market) > 10:
        return False
    return all(isinstance(item, list) and item for item in [action["farmer"], *hands, *market])


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def run_game(route, v18_agent, seed, seat):
    candidate = ReplayAgent(route)
    players = [candidate, v18_agent] if seat == 0 else [v18_agent, candidate]
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
        "action_calls": candidate.stats["calls"],
        "hand_shape_mismatches": candidate.stats["hand_shape_mismatches"],
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
    result = []
    for candidate, group in groups.items():
        wins = sum(row["result"] == "win" for row in group)
        ties = sum(row["result"] == "tie" for row in group)
        losses = sum(row["result"] == "loss" for row in group)
        result.append({
            "candidate": candidate,
            "games": len(group),
            "mean_cash": statistics.mean(row["candidate_money"] for row in group),
            "mean_opponent_cash": statistics.mean(row["opponent_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "min_cash": min(row["candidate_money"] for row in group),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": wins / len(group) if group else 0.0,
            "done_rate": statistics.mean(row["game_done"] for row in group),
            "max_hand_shape_mismatches": max(row["hand_shape_mismatches"] for row in group),
            "max_p99_ms": max(row["runtime_p99_ms"] for row in group),
            "max_runtime_ms": max(row["runtime_max_ms"] for row in group),
        })
    return sorted(result, key=lambda row: (-row["win_rate"], -row["mean_margin"], row["candidate"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    routes = {spec["name"]: load_replay_trace(spec) for spec in SELECTED_SPECS}
    args.out.mkdir(parents=True, exist_ok=True)
    selection = [
        {
            "name": route["name"],
            "file": route["file"],
            "source_seat": route["seat"],
            "terminal_cash_in_source_replay": route["terminal_cash"],
            "action_count": len(route["actions"]),
        }
        for route in routes.values()
    ]
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    v18_agent = load_v18_agent()
    total = len(routes) * len(args.seeds) * 2
    rows = []
    completed = 0
    for name, route in routes.items():
        for seed in args.seeds:
            for seat in (0, 1):
                row = run_game(route, v18_agent, seed, seat)
                row["candidate"] = name
                row["opponent"] = "v18"
                row["route_file"] = route["file"]
                rows.append(row)
                completed += 1
                print(
                    f"[{completed}/{total}] {name} vs v18 seed={seed} seat={seat} "
                    f"cash={row['candidate_money']:.0f} margin={row['margin']:.0f} "
                    f"status={row['candidate_status']}",
                    flush=True,
                )

    summary = summarize(rows)
    write_csv(args.out / "v012_raw.csv", rows)
    write_csv(args.out / "v012_summary.csv", summary)
    (args.out / "v012_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
