"""Pairwise ablation: V022c as submitted, V022c with its dormant hybrid layer,
and notebook V13-R3.

The original V022c file is not modified.  The activated candidate loads the
same module and calls its existing ``_hybrid_action`` after the same WEED
repair step, making the only change the final agent wiring.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
V022C_PATH = ROOT / "baseline/history/v022c_medoid_recovery/main.py"
V13_PATH = ROOT / "baseline/v13-r3-top-meta-order-safe-premium-control.ipynb"
SEEDS = (17, 42, 2026, 217, 317, 733)
EPISODE_STEPS = 720
CANDIDATES = ("v022c", "v022c_hybrid", "v13_r3")


def _load_module(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f"v022c_ablation_{tag}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_payload(path: Path) -> bytes:
    import ast
    import base64
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
            if not any(isinstance(t, ast.Name) and t.id == "_AGENT_B85_PARTS" for t in node.targets):
                continue
            parts = ast.literal_eval(node.value)
            raw = zlib.decompress(base64.b85decode("".join(parts).encode("ascii")))
            payloads.append(raw)
    if not payloads:
        raise ValueError(f"No embedded agent payload in {path}")
    return max(payloads, key=len)


def _load_v13():
    import types

    module = types.ModuleType(f"v13_ablation_{time.time_ns()}")
    module.__file__ = str(V13_PATH)
    raw = _notebook_payload(V13_PATH)
    exec(compile(raw, str(V13_PATH), "exec"), module.__dict__)
    return module.agent


def _load_v022c(hybrid: bool):
    module = _load_module(V022C_PATH, "hybrid" if hybrid else "control")
    if not hybrid:
        return module.agent

    def agent(obs, config=None):
        try:
            step = min(max(0, int(module._get(obs, "step", 0) or 0)), len(module._ACTIONS) - 1)
            action = module._weed_repair_action(
                obs, module._copy_action(module._ACTIONS[step]), step
            )
            return module._hybrid_action(obs, action, step)
        except Exception:
            return module.agent(obs)

    return agent


def _fresh(name: str):
    if name == "v022c":
        return _load_v022c(False)
    if name == "v022c_hybrid":
        return _load_v022c(True)
    if name == "v13_r3":
        return _load_v13()
    raise KeyError(name)


def _normalize(value):
    if not isinstance(value, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(x or ["PASS"]) for x in (value.get("hands") or [])],
        "market": [list(x) for x in (value.get("market") or []) if isinstance(x, list) and x],
    }


class Probe:
    def __init__(self, fn):
        self.fn = fn
        self.errors = 0
        self.times = []

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        try:
            try:
                result = self.fn(obs, config)
            except TypeError:
                result = self.fn(obs)
        except Exception:
            self.errors += 1
            result = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times.append((time.perf_counter_ns() - started) / 1_000_000)
        return _normalize(result)

    def metrics(self):
        values = sorted(self.times)

        def percentile(q):
            if not values:
                return 0.0
            return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]

        return {
            "calls": len(values),
            "agent_errors": self.errors,
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "max_ms": max(values or [0.0]),
        }


def _run_one(candidate, opponent, seed, seat):
    mine = Probe(_fresh(candidate))
    other = Probe(_fresh(opponent))
    players = [mine, other] if seat == 0 else [other, mine]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine_state, other_state = final[seat], final[1 - seat]
    mine_money = float(mine_state.observation["farms"][seat]["money"])
    other_money = float(other_state.observation["farms"][1 - seat]["money"])
    margin = mine_money - other_money
    result = "win" if margin > 0 else "loss" if margin < 0 else "tie"
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
        **mine.metrics(),
    }


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["candidate"], row["opponent"])].append(row)
    output = []
    for (candidate, opponent), group in sorted(groups.items()):
        wins = sum(x["result"] == "win" for x in group)
        ties = sum(x["result"] == "tie" for x in group)
        losses = sum(x["result"] == "loss" for x in group)
        output.append({
            "candidate": candidate,
            "opponent": opponent,
            "games": len(group),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": wins / len(group),
            "score_rate": (wins + 0.5 * ties) / len(group),
            "mean_money": statistics.mean(x["candidate_money"] for x in group),
            "mean_margin": statistics.mean(x["margin"] for x in group),
            "min_money": min(x["candidate_money"] for x in group),
            "all_done": int(all(x["done"] for x in group)),
            "errors": sum(x["agent_errors"] for x in group),
            "p99_ms_max": max(x["p99_ms"] for x in group),
        })
    return output


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path):
    rows = []
    pairs = [(left, right) for left in CANDIDATES for right in CANDIDATES if left != right]
    total = len(pairs) * len(SEEDS) * 2
    index = 0
    for candidate, opponent in pairs:
        for seed in SEEDS:
            for seat in (0, 1):
                index += 1
                print(f"[{index}/{total}] {candidate} vs {opponent} seed={seed} seat={seat}", flush=True)
                row = _run_one(candidate, opponent, seed, seat)
                rows.append(row)
                print(f"  {row['result']} margin={row['margin']:.0f}", flush=True)
    _write_csv(output / "raw.csv", rows)
    _write_csv(output / "summary.csv", _summary(rows))
    print(f"Wrote {len(rows)} games to {output}")


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "baseline/artifacts/v022c_hybrid_ablation"
    run(output)
