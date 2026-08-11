"""Run paired cross-graft validation for RL-010.

By default this runs the 36 core mechanism x route candidates against a
fresh V27 mechanism opponent for seeds 17, 42 and 2026.  The opponent and
candidate subset are configurable so the same runner can execute the second
stage and RL-010 comparisons without changing code.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import types

from kaggle_environments import make

from build_cross_graft_validation import (
    ARTIFACT_ROOT,
    MECHANISM_NOTEBOOKS,
    ROOT,
    decode_mechanism,
    load_archive_routes,
)
from rl_010_opponents import OPPONENT_SPECS, load_spec


EPISODE_STEPS = 720
DEFAULT_SEEDS = (17, 42, 2026)
_ARCHIVE_ROUTE_METADATA = None
RL_ARTIFACT_ROOT = ROOT / "baseline/artifacts/rl_010_milk_bidirectional"
RL_VARIANT_NAMES = {
    "rl010a_delay_only",
    "rl010b_bidirectional_no_opp",
    "rl010c_bidirectional_opp",
}
_RL_OPPONENT_SPECS = {row["name"]: row for row in OPPONENT_SPECS}


def _load_module(path, tag):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"cross_{tag}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_mechanism_module(name):
    source, _ = decode_mechanism(name)
    module = types.ModuleType(f"cross_mechanism_{name}_{time.time_ns()}")
    module.__file__ = str(MECHANISM_NOTEBOOKS[name])
    sys.modules[module.__name__] = module
    exec(compile(source, str(MECHANISM_NOTEBOOKS[name]), "exec"), module.__dict__)
    if not callable(getattr(module, "agent", None)):
        raise ValueError(f"mechanism {name} has no agent")
    return module


def _load_mechanism(name):
    return _load_mechanism_module(name).agent


def _fixed_route(route):
    route = copy.deepcopy(route)

    def agent(obs, config=None):
        step = int(obs.get("step", 0) or 0)
        return copy.deepcopy(route[min(max(step, 0), len(route) - 1)])

    return agent


def _normalize(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(value or ["PASS"]) for value in action.get("hands", []) or []],
        "market": [list(value) for value in action.get("market", []) or [] if isinstance(value, list)],
    }


class Probe:
    def __init__(self, function, reference_route=None):
        self.function = function
        self.reference_route = reference_route
        self.errors = 0
        self.invalid = 0
        self.times_ms = []
        self.ops = Counter()
        self.sell_units = Counter()
        self.sell_quote_value = Counter()
        self.actions = []

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
        self.actions.append(action)
        if len(action["market"]) > 10:
            self.invalid += 1
        for unit_action in [action["farmer"], *action["hands"]]:
            if unit_action:
                self.ops[str(unit_action[0]).upper()] += 1
        for order in action["market"]:
            if len(order) >= 3 and str(order[0]).upper() == "SELL":
                item = str(order[1]).upper()
                quantity = max(0, int(order[2]))
                self.sell_units[item] += quantity
                if item == "MILK":
                    market = obs.get("market", {}) or {}
                    prices = market.get("prices", {}) or {}
                    self.sell_quote_value[item] += quantity * float(prices.get(item, 0) or 0)
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        return action

    def metrics(self):
        values = sorted(self.times_ms)

        def percentile(q):
            if not values:
                return 0.0
            return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]

        result = {
            "calls": len(values),
            "errors": self.errors,
            "invalid": self.invalid,
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "max_ms": max(values or [0.0]),
            "ops": dict(self.ops),
            "sell_units": dict(self.sell_units),
            "sell_quote_value": dict(self.sell_quote_value),
        }
        if self.reference_route is not None:
            field_diff = 0
            hands_diff = 0
            market_diff = 0
            changed_calls = 0
            for step, action in enumerate(self.actions):
                if step >= len(self.reference_route):
                    changed_calls += 1
                    field_diff += 1
                    hands_diff += 1
                    market_diff += 1
                    continue
                reference = _normalize(self.reference_route[step])
                field_changed = action["farmer"] != reference["farmer"]
                hands_changed = action["hands"] != reference["hands"]
                market_changed = action["market"] != reference["market"]
                field_diff += int(field_changed)
                hands_diff += int(hands_changed)
                market_diff += int(market_changed)
                changed_calls += int(field_changed or hands_changed or market_changed)
            result.update({
                "field_action_diff": field_diff,
                "hands_action_diff": hands_diff,
                "market_action_diff": market_diff,
                "changed_action_calls": changed_calls,
            })
        return result


def _load_candidate(name):
    if name == "v27_original":
        module = _load_mechanism_module("v27")
        return module.agent, module
    if name == "starter" or name == "random":
        return name, None
    if name in {"v27_control", "rl010", "rl010_bidirectional"}:
        if name == "v27_control":
            path = ROOT / "baseline/history/v031_route_market_combo/v27_order_only/main.py"
        else:
            path = RL_ARTIFACT_ROOT / "main.py"
        module = _load_module(path, name)
        return module.agent, module
    if name in RL_VARIANT_NAMES:
        path = RL_ARTIFACT_ROOT / name / "main.py"
        if not path.exists():
            raise FileNotFoundError(path)
        module = _load_module(path, name)
        return module.agent, module
    path = ARTIFACT_ROOT / name / "main.py"
    if not path.exists():
        raise FileNotFoundError(path)
    module = _load_module(path, name)
    return module.agent, module


def _candidate_metadata(name, module):
    """Return auditable mechanism/route identity for a player."""
    global _ARCHIVE_ROUTE_METADATA
    manifest_path = ARTIFACT_ROOT / name / "manifest.json"
    if not manifest_path.exists():
        manifest_path = RL_ARTIFACT_ROOT / name / "manifest.json"
    if not manifest_path.exists() and name in {"rl010", "rl010_bidirectional"}:
        manifest_path = RL_ARTIFACT_ROOT / "manifest.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _ARCHIVE_ROUTE_METADATA is None:
            _ARCHIVE_ROUTE_METADATA = load_archive_routes()[1]
        route_meta = _ARCHIVE_ROUTE_METADATA.get(metadata.get("route"), {})
        counts = route_meta.get("action_counts", {})
        max_hands = int(route_meta.get("max_hands", 0))
        if max_hands >= 15:
            macro = "15_hands"
        elif max_hands >= 14:
            macro = "14_hands"
        elif max_hands >= 12:
            macro = "12_hands"
        else:
            macro = f"{max_hands}_hands"
        return {
            "mechanism": metadata.get("mechanism", name),
            "route": metadata.get("route", "v27_order_only"),
            "mechanism_sha256": metadata.get("mechanism_sha256", ""),
            "route_sha256": metadata.get("route_sha256", ""),
            "route_macro": macro if route_meta else "v27_14_hands",
            "route_plant_count": int(counts.get("PLANT", 0)),
            "route_harvest_count": int(counts.get("HARVEST", 0)),
            "route_water_count": int(counts.get("WATER", 0)),
            "route_max_hands": max_hands,
        }
    if name in MECHANISM_NOTEBOOKS:
        _, metadata = decode_mechanism(name)
        return {
            "mechanism": name,
            "route": "native",
            "mechanism_sha256": metadata["source_sha256"],
            "route_sha256": "",
            "route_macro": "native",
            "route_plant_count": 0,
            "route_harvest_count": 0,
            "route_water_count": 0,
            "route_max_hands": 0,
        }
    if name in {"rl010", "rl010_bidirectional"} or name in RL_VARIANT_NAMES:
        return {
            "mechanism": "rl010",
            "route": "v27_order_only",
            "mechanism_sha256": "",
            "route_sha256": "",
            "route_macro": "v27_14_hands",
            "route_plant_count": 0,
            "route_harvest_count": 0,
            "route_water_count": 0,
            "route_max_hands": 0,
        }
    return {
        "mechanism": name,
        "route": "builtin_or_external",
        "mechanism_sha256": "",
        "route_sha256": "",
        "route_macro": "unknown",
        "route_plant_count": 0,
        "route_harvest_count": 0,
        "route_water_count": 0,
        "route_max_hands": 0,
    }


def _load_opponent(name):
    if name in MECHANISM_NOTEBOOKS:
        module = _load_mechanism_module(name)
        return module.agent, module
    if name in _RL_OPPONENT_SPECS:
        agent, _ = load_spec(_RL_OPPONENT_SPECS[name])
        return agent, None
    return _load_candidate(name)


def _result(candidate_name, opponent_name, seed, seat):
    candidate_function, candidate_module = _load_candidate(candidate_name)
    opponent_function, opponent_module = _load_opponent(opponent_name)
    reference_route = None
    if candidate_module is not None:
        reference_route = getattr(candidate_module, "_CROSS_GRAFT_ROUTE", None)
        if reference_route is None and candidate_name in RL_VARIANT_NAMES | {"rl010", "rl010_bidirectional"}:
            reference_route = getattr(candidate_module, "_ACTIONS", None)
    candidate = Probe(candidate_function, reference_route=reference_route) if callable(candidate_function) else candidate_function
    opponent = Probe(opponent_function) if callable(opponent_function) else opponent_function
    players = [candidate, opponent] if int(seat) == 0 else [opponent, candidate]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine, theirs = final[int(seat)], final[1 - int(seat)]
    mine_money = float(mine.observation["farms"][int(seat)]["money"])
    other_money = float(theirs.observation["farms"][1 - int(seat)]["money"])
    margin = mine_money - other_money
    candidate_metrics = candidate.metrics() if isinstance(candidate, Probe) else {}
    opponent_metrics = opponent.metrics() if isinstance(opponent, Probe) else {}
    diagnostics = {}
    runtime = getattr(candidate_module, "_RL010_RUNTIME", None) if candidate_module else None
    if runtime is not None:
            diagnostics = {
            "interventions": int(getattr(runtime, "interventions", 0)),
            "advance_units": int(getattr(runtime, "advance_units", 0)),
            "delay_units": int(getattr(runtime, "delay_units", 0)),
            "repayment_successes": int(getattr(runtime, "repayment_successes", 0)),
            "repayment_failures": int(getattr(runtime, "repayment_failures", 0)),
            "fallbacks": int(getattr(runtime, "fallbacks", 0)),
                "runtime_errors": int(getattr(runtime, "errors", 0)),
                "decision_count": len(getattr(runtime, "decisions", []) or []),
                "selected_actions": dict(
                    Counter(
                        row.get("selected", "CONTROL")
                        for row in (getattr(runtime, "decisions", []) or [])
                    )
                ),
            }
    mine_private = mine.observation.get("private", {}) or {}
    mine_shed = mine_private.get("shed", {}) or {}
    return {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "seed": int(seed),
        "seat": int(seat),
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": margin,
        "result": "W" if margin > 0 else "L" if margin < 0 else "T",
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "candidate_status": str(mine.status),
        "opponent_status": str(theirs.status),
        "candidate_probe": candidate_metrics if isinstance(candidate, Probe) else {},
        "opponent_probe": opponent_metrics if isinstance(opponent, Probe) else {},
        "diagnostics": diagnostics,
        "candidate_final_milk": int(mine_shed.get("MILK", 0) or 0),
        **_candidate_metadata(candidate_name, candidate_module),
    }


def _write_rows(output, rows):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    serial = []
    for row in rows:
        value = dict(row)
        for key in ("candidate_probe", "opponent_probe", "diagnostics"):
            value[key] = repr(value.get(key, {}))
        serial.append(value)
    fields = sorted({key for row in serial for key in row})
    with (output / "raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(serial)


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["candidate"], row["opponent"])].append(row)
    result = []
    for (candidate, opponent), group in sorted(groups.items()):
        outcomes = Counter(row["result"] for row in group)
        result.append({
            "candidate": candidate,
            "opponent": opponent,
            "games": len(group),
            "wins": outcomes["W"],
            "ties": outcomes["T"],
            "losses": outcomes["L"],
            "win_rate": outcomes["W"] / len(group),
            "mean_money": statistics.mean(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "min_margin": min(row["margin"] for row in group),
            "all_done": int(all(row["done"] for row in group)),
            "errors": sum(row["candidate_probe"]["errors"] for row in group),
            "invalid": sum(row["candidate_probe"]["invalid"] for row in group),
            "p99_ms_max": max(row["candidate_probe"]["p99_ms"] for row in group),
            "repayment_failures": sum(row["diagnostics"].get("repayment_failures", 0) for row in group),
            "interventions": sum(row["diagnostics"].get("interventions", 0) for row in group),
            "advance_units": sum(row["diagnostics"].get("advance_units", 0) for row in group),
            "delay_units": sum(row["diagnostics"].get("delay_units", 0) for row in group),
            "candidate_milk_sell_units": sum(
                row["candidate_probe"].get("sell_units", {}).get("MILK", 0)
                for row in group
            ),
            "candidate_final_milk": sum(row.get("candidate_final_milk", 0) for row in group),
            "field_action_diff": sum(
                row["candidate_probe"].get("field_action_diff", 0) for row in group
            ),
            "hands_action_diff": sum(
                row["candidate_probe"].get("hands_action_diff", 0) for row in group
            ),
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
                    print(f"[{index}/{total}] {candidate} vs {opponent} seed={seed} seat={seat}", flush=True)
                    rows.append(_result(candidate, opponent, seed, seat))
    _write_rows(Path(output), rows)
    summary = _summary(rows)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (Path(output) / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in summary for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "benchmark_core")
    args = parser.parse_args()
    routes, _ = load_archive_routes()
    default_candidates = [f"{mechanism}_x_{route}" for mechanism in MECHANISM_NOTEBOOKS for route in routes]
    summary = run(
        args.candidate or default_candidates,
        args.opponent or ["v27_original"],
        tuple(args.seed or DEFAULT_SEEDS),
        args.output,
    )
    print(json.dumps(summary, indent=2))
