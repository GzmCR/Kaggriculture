"""Benchmark v22 control against V026 recovery candidates."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import gzip
import importlib.util
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
import types

from kaggle_environments import make

from run_v022_fresh_route import (
    ROOT,
    SEEDS,
    _opponent as _legacy_opponent,
    load_module,
    load_hamburger_agent,
    load_notebook_agent,
    load_v18_agent,
)
from run_v22_validation import _fresh as _v22_fresh


EPISODE_STEPS = 720
OPPONENTS = (
    "v22", "v022c", "v13_r3", "v21_1", "v012",
    "v024a", "v025b", "hamburger", "frontier", "root", "starter", "random",
)
CANDIDATES = {
    "v026a": ROOT / "baseline/artifacts/v026_v22_v022c_recovery/v026a_v22_single_retry/main.py",
    "v026b": ROOT / "baseline/artifacts/v026_v22_v022c_recovery/v026b_v22_single_retry_guard/main.py",
}


def _load_candidate(path, tag):
    return load_module(path, f"v026_{tag}_{time.time_ns()}")


def _opponent(name):
    if name in {"v22", "v022c", "v13_r3", "v21_1", "v012", "v024a", "v025b", "root"}:
        return _v22_fresh({"v22": "v22", "root": "root"}.get(name, name))
    if name == "hamburger":
        return _load_hamburger_anchor(ROOT / "baseline/kaggriculture-hamburger.ipynb")
    if name == "frontier":
        return load_notebook_agent(
            ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb",
            f"v026_frontier_{time.time_ns()}",
        )
    if name in {"starter", "random"}:
        return name
    raise ValueError(name)


def _load_hamburger_anchor(path):
    """Load the current notebook's embedded Anchor New Strategy.

    The repository's older helper expects WRAPPER_BLOB, while this notebook
    stores its public anchor in ANCHOR_BLOB.  Only the self-contained anchor
    source is executed; notebook evaluation code is not run.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "ANCHOR_BLOB" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "ANCHOR_BLOB" for target in node.targets):
                continue
            blob = ast.literal_eval(node.value)
            raw = gzip.decompress(base64.b64decode(blob.encode("ascii"))).decode("utf-8")
            module = types.ModuleType(f"v026_hamburger_{time.time_ns()}")
            module.__file__ = str(path)
            exec(compile(raw, str(path), "exec"), module.__dict__)
            return module.agent
    raise ValueError(f"Missing ANCHOR_BLOB in {path}")


def _normalize(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in action.get("hands", []) or []],
        "market": [list(item) for item in action.get("market", []) or [] if isinstance(item, list)],
    }


class Probe:
    def __init__(self, function, shadow=None):
        self.function = function
        self.shadow = shadow
        self.times = []
        self.errors = 0
        self.invalid = 0
        self.actions = Counter()
        self.action_diff_calls = 0
        self.action_diff_farmer = 0
        self.action_diff_hands = 0
        self.action_diff_market = 0

    def __call__(self, obs, config=None):
        start = time.perf_counter_ns()
        try:
            try:
                raw = self.function(obs, config)
            except TypeError:
                raw = self.function(obs)
        except Exception:
            self.errors += 1
            raw = {"farmer": ["PASS"], "hands": [], "market": []}
        action = _normalize(raw)
        if self.shadow is not None:
            try:
                try:
                    shadow_raw = self.shadow(obs, config)
                except TypeError:
                    shadow_raw = self.shadow(obs)
                shadow = _normalize(shadow_raw)
                if action != shadow:
                    self.action_diff_calls += 1
                    self.action_diff_farmer += int(action["farmer"] != shadow["farmer"])
                    self.action_diff_hands += int(action["hands"] != shadow["hands"])
                    self.action_diff_market += int(action["market"] != shadow["market"])
            except Exception:
                pass
        self.times.append((time.perf_counter_ns() - start) / 1_000_000)
        for operation in [action["farmer"], *action["hands"]]:
            if operation:
                self.actions[str(operation[0]).upper()] += 1
        if len(action["market"]) > 10:
            self.invalid += 1
        return action

    def metrics(self):
        ordered = sorted(self.times)

        def pct(q):
            return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))] if ordered else 0.0

        return {
            "calls": len(ordered),
            "errors": self.errors,
            "invalid": self.invalid,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "max_ms": max(ordered or [0.0]),
            "actions": dict(self.actions),
            "action_diff_calls": self.action_diff_calls,
            "action_diff_farmer": self.action_diff_farmer,
            "action_diff_hands": self.action_diff_hands,
            "action_diff_market": self.action_diff_market,
        }


def _run_one(candidate_name, opponent_name, seed, seat):
    if candidate_name == "v22":
        candidate_module = None
        candidate = _v22_fresh("v22")
    else:
        candidate_module = _load_candidate(CANDIDATES[candidate_name], candidate_name)
        candidate = candidate_module.agent
    shadow = None if candidate_name == "v22" else _v22_fresh("v22")
    probe = Probe(candidate, shadow=shadow)
    other = _opponent(opponent_name)
    players = [probe, other] if seat == 0 else [other, probe]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine, other_state = final[seat], final[1 - seat]
    money = float(mine.observation["farms"][seat]["money"])
    opponent_money = float(other_state.observation["farms"][1 - seat]["money"])
    row = {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "seed": seed,
        "seat": seat,
        "candidate_money": money,
        "opponent_money": opponent_money,
        "margin": money - opponent_money,
        "result": "win" if money > opponent_money else "loss" if money < opponent_money else "tie",
        "done": int(mine.status == "DONE" and other_state.status == "DONE"),
        "candidate_status": str(mine.status),
        "opponent_status": str(other_state.status),
        **probe.metrics(),
    }
    if candidate_module is not None:
        for key, value in getattr(candidate_module, "_V026_STATS", {}).items():
            row[f"recovery_{key}"] = value
        for key, value in getattr(candidate_module, "_V026_GUARD_STATS", {}).items():
            row[f"guard_{key}"] = value
    return row


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = []
    for row in rows:
        item = dict(row)
        for key in ("actions",):
            if key in item:
                value = item[key]
                if isinstance(value, dict):
                    item[key] = repr(value)
                elif isinstance(value, str):
                    # Resume runs read CSV fields back as strings.  Unwrap
                    # older repr layers so a checkpoint never double-encodes
                    # the action counter.
                    parsed = value
                    for _ in range(3):
                        if not isinstance(parsed, str):
                            break
                        try:
                            parsed = ast.literal_eval(parsed)
                        except (SyntaxError, ValueError):
                            break
                    item[key] = repr(parsed) if isinstance(parsed, dict) else value
        serial.append(item)
    fields = sorted({key for row in serial for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serial)


def _summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["opponent"])].append(row)
    output = []
    for (candidate, opponent), group in sorted(grouped.items()):
        outcomes = Counter(row["result"] for row in group)
        number = lambda row, key: float(row[key])
        integer = lambda row, key: int(float(row[key]))
        output.append({
            "candidate": candidate,
            "opponent": opponent,
            "games": len(group),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(group),
            "score_rate": (outcomes["win"] + 0.5 * outcomes["tie"]) / len(group),
            "mean_money": statistics.mean(number(row, "candidate_money") for row in group),
            "mean_margin": statistics.mean(number(row, "margin") for row in group),
            "min_money": min(number(row, "candidate_money") for row in group),
            "all_done": int(all(integer(row, "done") for row in group)),
            "errors": sum(integer(row, "errors") for row in group),
            "invalid": sum(integer(row, "invalid") for row in group),
            "max_action_diff_calls": max(integer(row, "action_diff_calls") for row in group),
            "p99_ms_max": max(number(row, "p99_ms") for row in group),
        })
    return output


def run(candidates, opponents, seeds, output):
    raw_path = output / "raw.csv"
    rows = []
    completed = set()
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed = {
            (row.get("candidate"), row.get("opponent"), int(row["seed"]), int(row["seat"]))
            for row in rows
        }
    total = len(candidates) * len(opponents) * len(seeds) * 2
    index = len(completed)
    for candidate in candidates:
        for opponent in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    key = (candidate, opponent, int(seed), int(seat))
                    if key in completed:
                        continue
                    index += 1
                    print(f"[{index}/{total}] {candidate} vs {opponent} seed={seed} seat={seat}", flush=True)
                    rows.append(_run_one(candidate, opponent, seed, seat))
                    if index % 10 == 0:
                        _write_csv(raw_path, rows)
    _write_csv(raw_path, rows)
    _write_csv(output / "summary.csv", _summary(rows))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", choices=("v22", *CANDIDATES))
    parser.add_argument("--opponent", action="append", choices=OPPONENTS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v026_v22_v022c_recovery/full_matrix")
    args = parser.parse_args()
    rows = run(
        tuple(args.candidate or ("v22", *CANDIDATES)),
        tuple(args.opponent or OPPONENTS),
        tuple(args.seed or SEEDS),
        args.output,
    )
    print(f"V026 benchmark complete: {len(rows)} games")
