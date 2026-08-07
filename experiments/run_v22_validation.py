"""Validate the embedded 44-46 v22 agent against the local controls.

This runner does not modify the root agent and does not expose replay files to
the candidate at runtime.  The v22 notebook is loaded only to extract its
embedded self-contained ``agent`` source.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import time
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
SEEDS = (17, 42, 2026, 217, 317, 733)
OPPONENTS = (
    "v22", "v22_route_only", "v13_r3", "v022c", "v21_1", "root", "v012",
    "v024a", "v025b", "starter", "random",
)


def _load_file_agent(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f"v22_validation_{tag}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def _load_notebook_agent(path: Path, tag: str, disable_impact: bool = False):
    import ast
    import base64
    import json
    import types
    import zlib

    notebook = json.loads(path.read_text(encoding="utf-8"))
    payloads = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "_AGENT_B85_PARTS" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "_AGENT_B85_PARTS"
                       for target in node.targets):
                continue
            parts = ast.literal_eval(node.value)
            payloads.append(zlib.decompress(base64.b85decode("".join(parts).encode("ascii"))))
    if not payloads:
        raise ValueError(f"No embedded agent payload in {path}")
    module = types.ModuleType(f"v22_validation_notebook_{tag}_{time.time_ns()}")
    module.__file__ = str(path)
    raw = max(payloads, key=len)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if disable_impact:
        # Keep the exact embedded route and WEED recovery, but remove only the
        # in-place ranking of existing SELL slots.  The agent resolves this
        # global at call time, so this is a clean route-only ablation.
        module._impact_slots = lambda obs, action: action
    return module.agent


def _fresh(name: str):
    if name in {"starter", "random"}:
        return name
    if name in {"v22", "v22_route_only"}:
        return _load_notebook_agent(
            ROOT / "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb",
            name,
            disable_impact=name == "v22_route_only",
        )
    if name == "v13_r3":
        return _load_notebook_agent(
            ROOT / "baseline/v13-r3-top-meta-order-safe-premium-control.ipynb", name
        )
    if name == "v21_1":
        return _load_notebook_agent(
            ROOT / "baseline/177-180-fresh-top-30-v21-1-conditional-memory.ipynb", name
        )
    paths = {
        "v022c": ROOT / "baseline/history/v022c_medoid_recovery/main.py",
        "root": ROOT / "main.py",
        "v012": ROOT / "baseline/history/v012_top5_replaced_v18/main.py",
        "v024a": ROOT / "baseline/history/v024a_route14_control/main.py",
        "v025b": ROOT / "baseline/history/v025b_route14_v022c_open_market/main.py",
    }
    return _load_file_agent(paths[name], name)


def _normalize(value):
    if not isinstance(value, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (value.get("hands") or [])],
        "market": [list(item) for item in (value.get("market") or [])
                   if isinstance(item, list) and item],
    }


class Probe:
    def __init__(self, function):
        self.function = function
        self.times = []
        self.errors = 0
        self.actions = defaultdict(int)
        self.market = defaultdict(int)

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        try:
            try:
                value = self.function(obs, config)
            except TypeError:
                value = self.function(obs)
        except Exception:
            self.errors += 1
            value = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times.append((time.perf_counter_ns() - started) / 1_000_000)
        value = _normalize(value)
        for operation in [value["farmer"], *value["hands"]]:
            if operation:
                self.actions[str(operation[0]).upper()] += 1
        for order in value["market"]:
            self.market[str(order[0]).upper()] += 1
        return value

    def metrics(self):
        ordered = sorted(self.times)

        def pct(q):
            if not ordered:
                return 0.0
            return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]

        return {
            "calls": len(ordered),
            "agent_errors": self.errors,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "max_ms": max(ordered or [0.0]),
            "actions": dict(self.actions),
            "market": dict(self.market),
        }


def _run_one(candidate: str, opponent: str, seed: int, seat: int):
    mine = Probe(_fresh(candidate))
    other = _fresh(opponent)
    players = [mine, other] if seat == 0 else [other, mine]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine_state, other_state = final[seat], final[1 - seat]
    mine_money = float(mine_state.observation["farms"][seat]["money"])
    other_money = float(other_state.observation["farms"][1 - seat]["money"])
    margin = mine_money - other_money
    return {
        "candidate": candidate,
        "opponent": opponent,
        "seed": seed,
        "seat": seat,
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "done": int(mine_state.status == "DONE" and other_state.status == "DONE"),
        "candidate_status": str(mine_state.status),
        "opponent_status": str(other_state.status),
        **mine.metrics(),
    }


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["candidate"], row["opponent"])].append(row)
    result = []
    for (candidate, opponent), group in sorted(groups.items()):
        wins = sum(row["result"] == "win" for row in group)
        ties = sum(row["result"] == "tie" for row in group)
        losses = sum(row["result"] == "loss" for row in group)
        result.append({
            "candidate": candidate,
            "opponent": opponent,
            "games": len(group),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": wins / len(group),
            "score_rate": (wins + 0.5 * ties) / len(group),
            "mean_money": statistics.mean(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "min_money": min(row["candidate_money"] for row in group),
            "all_done": int(all(row["done"] for row in group)),
            "errors": sum(row["agent_errors"] for row in group),
            "p99_ms_max": max(row["p99_ms"] for row in group),
        })
    return result


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = []
    for row in rows:
        item = dict(row)
        for key in ("actions", "market"):
            if key in item:
                item[key] = repr(item[key])
        serial.append(item)
    fields = sorted({key for row in serial for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serial)


def run(candidates, opponents, seeds, output):
    rows = []
    total = len(candidates) * len(opponents) * len(seeds) * 2
    index = 0
    for candidate in candidates:
        for opponent in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    print(f"[{index}/{total}] {candidate} vs {opponent} seed={seed} seat={seat}", flush=True)
                    row = _run_one(candidate, opponent, seed, seat)
                    rows.append(row)
                    print(f"  {row['result']} margin={row['margin']:.0f} cash={row['candidate_money']:.0f}", flush=True)
    _write_csv(output / "raw.csv", rows)
    _write_csv(output / "summary.csv", _summary(rows))
    print(f"Wrote {len(rows)} games to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v22_validation")
    parser.add_argument("--opponent", action="append", choices=OPPONENTS)
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args()
    run(("v22",), tuple(args.opponent or OPPONENTS), tuple(args.seed or SEEDS), args.output)
