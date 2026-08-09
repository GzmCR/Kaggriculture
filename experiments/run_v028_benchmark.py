"""Benchmark V028 order-permutation candidates against the local pool."""

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

from run_v026_v22_v022c_recovery import EPISODE_STEPS, OPPONENTS as DEFAULT_OPPONENTS, ROOT, _opponent, _v22_fresh


SEEDS = (17, 42, 2026, 217, 317, 733)
CANDIDATES = {
    "v22": None,
    "v028a": ROOT / "baseline/artifacts/v028_order_search/v028a_marginal_order/main.py",
    "v028b": ROOT / "baseline/artifacts/v028_order_search/v028b_safe_order/main.py",
    "v028c": ROOT / "baseline/artifacts/v028_order_search/v028c_robust_order/main.py",
}


def _load_module(path, tag):
    spec = importlib.util.spec_from_file_location(f"v028_{tag}_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize(value):
    if not isinstance(value, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (value.get("hands") or [])],
        "market": [list(item) for item in (value.get("market") or []) if isinstance(item, list)],
    }


class Probe:
    def __init__(self, function, shadow=None):
        self.function = function
        self.shadow = shadow
        self.times_ms = []
        self.errors = 0
        self.invalid = 0
        self.action_diff_calls = 0
        self.action_diff_farmer = 0
        self.action_diff_hands = 0
        self.action_diff_market = 0
        self.market_counts = Counter()

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        try:
            try:
                raw = self.function(obs, config)
            except TypeError:
                raw = self.function(obs)
        except Exception:
            self.errors += 1
            raw = {"farmer": ["PASS"], "hands": [], "market": []}
        action = _normalize(raw)
        if len(action["market"]) > 10:
            self.invalid += 1
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
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        for order in action["market"]:
            if len(order) >= 3 and str(order[0]).upper() == "SELL":
                self.market_counts[f"SELL_{str(order[1]).upper()}"] += max(0, int(order[2]))
        return action

    def metrics(self):
        ordered = sorted(self.times_ms)

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
            "action_diff_calls": self.action_diff_calls,
            "action_diff_farmer": self.action_diff_farmer,
            "action_diff_hands": self.action_diff_hands,
            "action_diff_market": self.action_diff_market,
            "market_counts": dict(self.market_counts),
        }


def _copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=True))


def _run_one(candidate_name, opponent_name, seed, seat):
    if candidate_name == "v22":
        module = None
        candidate = _v22_fresh("v22")
        shadow = None
    else:
        module = _load_module(CANDIDATES[candidate_name], candidate_name)
        candidate = module.agent
        shadow = _v22_fresh("v22")
    probe = Probe(candidate, shadow=shadow)
    other = _opponent(opponent_name)
    players = [probe, other] if seat == 0 else [other, probe]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine, theirs = final[seat], final[1 - seat]
    mine_money = float(mine.observation["farms"][seat]["money"])
    other_money = float(theirs.observation["farms"][1 - seat]["money"])
    diagnostics = {}
    if module is not None:
        stats = getattr(module, "_V030_STATS", None)
        if stats is None:
            stats = getattr(module, "_V028_STATS", None)
        if stats is None:
            stats = getattr(module, "_V029_STATS", {})
        diagnostics = _copy_json(stats)
        timing_runtime = getattr(module, "_RL003_RUNTIME", None)
        if timing_runtime is None:
            timing_runtime = getattr(module, "_RL004_RUNTIME", None)
        if timing_runtime is None:
            timing_runtime = getattr(module, "_RL005_RUNTIME", None)
        if timing_runtime is None:
            timing_runtime = getattr(module, "_RL006_RUNTIME", None)
        if timing_runtime is not None:
            runtime_errors = int(getattr(timing_runtime, "errors", 0))
            diagnostics["timing_runtime_errors"] = runtime_errors
            diagnostics["timing_last_error"] = str(getattr(timing_runtime, "last_error", ""))
            diagnostics["errors"] = max(int(diagnostics.get("errors", 0)), runtime_errors)
            diagnostics["timing_changed_calls"] = int(getattr(timing_runtime, "changed_calls", 0))
            diagnostics["timing_changed_units"] = int(getattr(timing_runtime, "changed_units", 0))
            diagnostics["timing_delayed_orders"] = int(getattr(timing_runtime, "delayed_orders", 0))
            diagnostics["timing_preempt_units"] = int(getattr(timing_runtime, "preempt_units", 0))
            diagnostics["timing_delay_units"] = int(getattr(timing_runtime, "delay_units", 0))
            diagnostics["timing_fallbacks"] = int(getattr(timing_runtime, "fallbacks", 0))
            diagnostics["timing_decisions"] = list(getattr(timing_runtime, "decisions", []))
            diagnostics["changed_calls"] = diagnostics["timing_changed_calls"]
            diagnostics["decision_points"] = len(diagnostics["timing_decisions"])
            diagnostics["evaluations"] = len(diagnostics["timing_decisions"])
    return {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "seed": int(seed),
        "seat": int(seat),
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": mine_money - other_money,
        "result": "win" if mine_money > other_money else "loss" if mine_money < other_money else "tie",
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "candidate_status": str(mine.status),
        "opponent_status": str(theirs.status),
        "errors": int(probe.errors + diagnostics.get("errors", 0)),
        "invalid": int(probe.invalid),
        "changed_calls": int(diagnostics.get("changed_calls", 0)),
        "decision_points": int(diagnostics.get("decision_points", 0)),
        "evaluations": int(diagnostics.get("evaluations", 0)),
        "predicted_delta": float(diagnostics.get("predicted_delta", 0.0)),
        "diagnostics": diagnostics,
        **probe.metrics(),
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, (dict, list)):
                item[key] = repr(value)
        serial.append(item)
    fields = sorted({key for row in serial for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(serial)


def _summary(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for values_key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        outcomes = Counter(row["result"] for row in group)
        result.append({
            **dict(zip(keys, values_key)),
            "games": len(group),
            "mean_money": statistics.mean(row["candidate_money"] for row in group),
            "min_money": min(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(group),
            "all_done": int(all(row["done"] for row in group)),
            "errors": sum(row["errors"] for row in group),
            "invalid": sum(row["invalid"] for row in group),
            "mean_changed_calls": statistics.mean(row["changed_calls"] for row in group),
            "mean_decision_points": statistics.mean(row["decision_points"] for row in group),
            "mean_evaluations": statistics.mean(row["evaluations"] for row in group),
            "mean_predicted_delta": statistics.mean(row["predicted_delta"] for row in group),
            "max_action_diff_calls": max(row["action_diff_calls"] for row in group),
            "max_action_diff_farmer": max(row["action_diff_farmer"] for row in group),
            "max_action_diff_hands": max(row["action_diff_hands"] for row in group),
            "max_action_diff_market": max(row["action_diff_market"] for row in group),
            "p99_ms_max": max(row["p99_ms"] for row in group),
        })
    return result


def run(candidates, opponents, seeds, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(candidates) * len(opponents) * len(seeds) * 2
    index = 0
    for candidate in candidates:
        for opponent in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    print(f"[{index}/{total}] {candidate} vs {opponent} seed={seed} seat={seat}", flush=True)
                    rows.append(_run_one(candidate, opponent, seed, seat))
    _write_jsonl(output / "matrix_raw.jsonl", rows)
    _write_csv(output / "matrix_raw.csv", rows)
    summaries = {
        "by_candidate_opponent": _summary(rows, ("candidate", "opponent")),
        "overall": _summary(rows, ("candidate",)),
    }
    (output / "matrix_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "matrix_summary.csv", summaries["by_candidate_opponent"])
    return rows, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baseline/artifacts/v028_order_search/matrix",
    )
    args = parser.parse_args()
    run(tuple(args.candidates), tuple(args.opponents), tuple(args.seeds), args.output)
