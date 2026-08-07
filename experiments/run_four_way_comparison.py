"""Compare V025b, V022c, notebook v21.1, and notebook V13-R3.

The two notebook candidates are loaded from their embedded, hash-checked
payloads in memory.  No replay, score, identity, or network input is exposed
to the agents during a game.

Examples:
  python experiments/run_four_way_comparison.py --smoke
  python experiments/run_four_way_comparison.py --matrix
  python experiments/run_four_way_comparison.py --matrix --opponent v21_1
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import types
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
SEEDS = (17, 42, 2026, 217, 317, 733)
CANDIDATES = ("v025b", "v022c", "v21_1", "v13_r3")
OPPONENTS = (
    "root", "v012", "v022c", "v024a", "v025b",
    "v21_1", "v13_r3", "starter", "random",
)


def _load_file_agent(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(
        f"four_way_{tag}_{time.time_ns()}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def _notebook_payload(path: Path) -> bytes:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
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
            if not any(isinstance(t, ast.Name) and t.id == "_AGENT_B85_PARTS" for t in node.targets):
                continue
            parts = ast.literal_eval(node.value)
            if not isinstance(parts, list) or not parts:
                continue
            raw = zlib.decompress(base64.b85decode("".join(parts).encode("ascii")))
            candidates.append(raw)
    if not candidates:
        raise ValueError(f"No embedded agent payload in {path}")
    return max(candidates, key=len)


def _load_notebook_agent(path: Path, tag: str):
    raw = _notebook_payload(path)
    module = types.ModuleType(f"four_way_notebook_{tag}_{time.time_ns()}")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "agent", None)):
        raise ValueError(f"Embedded payload has no agent: {path}")
    return module.agent


def _fresh_agent(name: str):
    paths = {
        "v025b": ROOT / "baseline/history/v025b_route14_v022c_open_market/main.py",
        "v022c": ROOT / "baseline/history/v022c_medoid_recovery/main.py",
        "root": ROOT / "main.py",
        "v012": ROOT / "baseline/history/v012_top5_replaced_v18/main.py",
        "v024a": ROOT / "baseline/history/v024a_route14_control/main.py",
        "v025b": ROOT / "baseline/history/v025b_route14_v022c_open_market/main.py",
    }
    if name in {"starter", "random"}:
        return name
    if name in {"v21_1", "v13_r3"}:
        notebook = {
            "v21_1": ROOT / "baseline/177-180-fresh-top-30-v21-1-conditional-memory.ipynb",
            "v13_r3": ROOT / "baseline/v13-r3-top-meta-order-safe-premium-control.ipynb",
        }[name]
        return _load_notebook_agent(notebook, name)
    return _load_file_agent(paths[name], name)


def _normalize(value):
    if not isinstance(value, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    hands = value.get("hands") or []
    market = value.get("market") or []
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in hands],
        "market": [list(item) for item in market if isinstance(item, list) and item],
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
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "max_ms": max(ordered or [0.0]),
            "agent_errors": self.errors,
            "actions": dict(self.actions),
            "market": dict(self.market),
        }


def _run_one(candidate: str, opponent: str, seed: int, seat: int):
    mine = Probe(_fresh_agent(candidate))
    other = _fresh_agent(opponent)
    players = [mine, other] if seat == 0 else [other, mine]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine_state, other_state = final[seat], final[1 - seat]
    mine_money = float(mine_state.observation["farms"][seat]["money"])
    other_money = float(other_state.observation["farms"][1 - seat]["money"])
    margin = mine_money - other_money
    result = "win" if margin > 0 else "loss" if margin < 0 else "tie"
    metrics = mine.metrics()
    return {
        "candidate": candidate,
        "opponent": opponent,
        "seed": seed,
        "seat": seat,
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": margin,
        "result": result,
        "done": int(mine_state.status == "DONE" and other_state.status == "DONE"),
        "candidate_status": str(mine_state.status),
        "opponent_status": str(other_state.status),
        **metrics,
    }


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


def run(candidates, opponents, seeds, output):
    rows = []
    total = len(candidates) * len(opponents) * len(seeds) * 2
    index = 0
    for candidate in candidates:
        for opponent in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    print(
                        f"[{index}/{total}] {candidate} vs {opponent} "
                        f"seed={seed} seat={seat}",
                        flush=True,
                    )
                    row = _run_one(candidate, opponent, seed, seat)
                    rows.append(row)
                    print(
                        f"  {row['result']} margin={row['margin']:.0f} "
                        f"cash={row['candidate_money']:.0f}",
                        flush=True,
                    )
    _write_csv(output / "raw.csv", rows)
    _write_csv(output / "summary.csv", _summary(rows))
    print(f"Wrote {len(rows)} games to {output}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--candidate", action="append", choices=CANDIDATES)
    parser.add_argument("--opponent", action="append", choices=OPPONENTS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "baseline/artifacts/four_way_comparison",
    )
    args = parser.parse_args()
    candidates = tuple(args.candidate or CANDIDATES)
    opponents = tuple(args.opponent or OPPONENTS)
    seeds = tuple(args.seed or ((17,) if args.smoke else SEEDS))
    if args.smoke:
        opponents = tuple(args.opponent or ("starter", "v21_1", "v13_r3"))
    run(candidates, opponents, seeds, args.output)


if __name__ == "__main__":
    main()
