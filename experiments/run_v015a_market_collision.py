"""Run V015a replay counterfactuals and local paired benchmarks."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from v015a_market_overlay import (
    MAX_MARKET_ORDERS,
    MarketCollisionOverlay,
    PREMIUM_PRODUCTS,
    prepare_observation,
)


ROOT = Path(__file__).resolve().parents[1]
V012_PATH = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
REPLAY_DIR = ROOT / "baseline/history/v012_top5_replaced_v18/log"
V18_NOTEBOOK = ROOT / "baseline/40-53-top-10-future-holdout-v18-closed-loop.ipynb"
HAMBURGER_NOTEBOOK = ROOT / "baseline/kaggriculture-hamburger.ipynb"
FRONTIER_NOTEBOOK = ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb"
OUT_DIR = ROOT / "baseline/artifacts/v015a_market_collision"
EPISODE_STEPS = 720
DEFAULT_SEEDS = (17, 42, 2026, 217, 317, 733)
DEFAULT_OPPONENTS = ("v012", "baseline", "v18", "hamburger", "frontier", "starter", "random")
VARIANT_THRESHOLDS = {
    "default": {"single_drop_ratio": 0.20, "two_step_drop_ratio": 0.30, "recent_median_ratio": 0.70},
    "sensitive": {"single_drop_ratio": 0.15, "two_step_drop_ratio": 0.25, "recent_median_ratio": 0.80},
    "conservative": {"single_drop_ratio": 0.25, "two_step_drop_ratio": 0.35, "recent_median_ratio": 0.60},
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} has no agent")
    return module


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(float(value) for value in values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def _counter_json(counter):
    return json.dumps(dict(sorted(counter.items())), sort_keys=True)


def _quantity_json(counter):
    return json.dumps(
        {f"{op}:{item}": n for (op, item), n in sorted(counter.items())},
        sort_keys=True,
    )


def valid_shape(action, obs):
    if not isinstance(action, dict):
        return False
    farmer = action.get("farmer")
    hands = action.get("hands", [])
    market = action.get("market", [])
    if not isinstance(farmer, list) or not farmer:
        return False
    if not isinstance(hands, list) or not isinstance(market, list):
        return False
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    expected = len(farms[player].get("hands", []) or []) if player < len(farms) else 0
    # A HIRE order is processed after the field actions for the current
    # transition.  Kaggle's observation can therefore already expose the
    # newly hired hand while the action still contains the previous hand
    # count (replay serialization can show the inverse at the same boundary).
    # Accept that one-unit boundary case; it is inherited by V012 and is
    # tracked identically for control and candidate.
    return (
        len(hands) in {expected, max(0, expected - 1), expected + 1}
        and len(market) <= MAX_MARKET_ORDERS
        and all(isinstance(op, list) and op for op in [farmer, *hands, *market])
    )


class Probe:
    def __init__(self, call):
        self.call = call
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
            try:
                action = self.call(obs, config)
            except TypeError:
                action = self.call(obs)
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        self.calls += 1
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        if not valid_shape(action, obs):
            self.invalid += 1
        if isinstance(action, dict):
            for op in [action.get("farmer", []), *(action.get("hands", []) or [])]:
                if isinstance(op, list) and op:
                    self.field_counts[str(op[0])] += 1
            for order in action.get("market", []) or []:
                if isinstance(order, list) and order:
                    name = str(order[0])
                    self.market_counts[name] += 1
                    if len(order) >= 3:
                        try:
                            self.market_quantities[(name, str(order[1]))] += int(order[2])
                        except (TypeError, ValueError):
                            pass
        return action

    def metrics(self):
        return {
            "action_calls": self.calls,
            "agent_errors": self.errors,
            "invalid_action_shapes": self.invalid,
            "runtime_p50_ms": percentile(self.times_ms, 0.50),
            "runtime_p95_ms": percentile(self.times_ms, 0.95),
            "runtime_p99_ms": percentile(self.times_ms, 0.99),
            "runtime_max_ms": max(self.times_ms or [0.0]),
            "field_counts": _counter_json(self.field_counts),
            "market_counts": _counter_json(self.market_counts),
            "market_quantities": _quantity_json(self.market_quantities),
        }


class OverlayAgent:
    def __init__(self, variant="default"):
        self.variant = variant
        self.module = load_module(V012_PATH, f"v015a_base_{id(self)}")
        self.overlay = MarketCollisionOverlay(**VARIANT_THRESHOLDS[variant])
        self.field_changed = 0
        self.nonpremium_changed = 0

    @staticmethod
    def _nonpremium(orders):
        return [
            order for order in orders
            if not (
                isinstance(order, list) and len(order) >= 2
                and order[0] == "SELL" and order[1] in PREMIUM_PRODUCTS
            )
        ]

    def __call__(self, obs, config=None):
        prepared = prepare_observation(obs, self.overlay)
        try:
            base = self.module.agent(prepared, config)
        except TypeError:
            base = self.module.agent(prepared)
        output = self.overlay.apply(prepared, base)
        if output.get("farmer") != base.get("farmer") or output.get("hands") != base.get("hands"):
            self.field_changed += 1
        if self._nonpremium(output.get("market", [])) != self._nonpremium(base.get("market", [])):
            self.nonpremium_changed += 1
        return output

    def diagnostics(self, player):
        result = self.overlay.diagnostics(player)
        result.update({"field_changed": self.field_changed, "nonpremium_changed": self.nonpremium_changed})
        return result


class ReplayAgent:
    def __init__(self, actions):
        self.actions = actions

    def __call__(self, obs, config=None):
        step = max(0, min(int(obs.get("step", 0) or 0), EPISODE_STEPS - 1))
        action = copy.deepcopy(self.actions[step])
        return action if isinstance(action, dict) else {"farmer": ["PASS"], "hands": [], "market": []}


def load_replay(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    if len(steps) < EPISODE_STEPS:
        raise ValueError(f"{path} has {len(steps)} steps")
    info = payload.get("info", {}) or {}
    names = info.get("TeamNames", []) or info.get("teamNames", []) or []
    seat = next((i for i, name in enumerate(names[:2]) if "GzmCR" in str(name) or "632" in str(name)), 0)
    actions = [copy.deepcopy(steps[min(i + 1, EPISODE_STEPS - 1)][seat].get("action") or {}) for i in range(EPISODE_STEPS)]
    return {"path": str(path), "seat": seat, "seed": int(info.get("seed", 0) or 0), "actions": actions}


def run_game(candidate, opponent, seed, seat):
    probe = Probe(candidate)
    players = [probe, opponent] if seat == 0 else [opponent, probe]
    started = time.perf_counter()
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine, other = final[seat], final[1 - seat]
    my_money = float(mine.observation["farms"][seat]["money"])
    other_money = float(other.observation["farms"][1 - seat]["money"])
    margin = my_money - other_money
    row = {
        "seed": int(seed), "seat": int(seat), "candidate_money": my_money,
        "opponent_money": other_money, "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "candidate_status": mine.status, "opponent_status": other.status,
        "game_done": int(mine.status == "DONE" and other.status == "DONE"),
        "wall_seconds": time.perf_counter() - started,
    }
    shed = (mine.observation.get("private", {}) or {}).get("shed", {}) or {}
    for item in PREMIUM_PRODUCTS:
        try:
            row[f"final_shed_{item}"] = int(shed.get(item, 0) or 0)
        except (TypeError, ValueError):
            row[f"final_shed_{item}"] = 0
    row.update(probe.metrics())
    if isinstance(candidate, OverlayAgent):
        row.update({f"overlay_{k}": v for k, v in candidate.diagnostics(seat).items()})
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path):
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    numeric = {
        "seed", "seat", "candidate_money", "opponent_money", "margin",
        "game_done", "agent_errors", "invalid_action_shapes", "runtime_p99_ms",
    }
    for row in rows:
        for field in numeric:
            if field not in row or row[field] == "":
                continue
            try:
                row[field] = int(row[field]) if field in {"seed", "seat", "game_done", "agent_errors", "invalid_action_shapes"} else float(row[field])
            except (TypeError, ValueError):
                pass
    return rows


def summarize(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        outcomes = Counter(row["result"] for row in group)
        item = dict(zip(fields, key))
        item.update({
            "games": len(group),
            "mean_money": statistics.mean(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "min_money": min(row["candidate_money"] for row in group),
            "wins": outcomes["win"], "ties": outcomes["tie"], "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(group),
            "all_done": int(all(row["game_done"] for row in group)),
            "agent_errors": sum(row["agent_errors"] for row in group),
            "invalid_action_shapes": sum(row["invalid_action_shapes"] for row in group),
            "p50_ms": percentile([row["runtime_p50_ms"] for row in group], 0.50),
            "p95_ms": percentile([row["runtime_p95_ms"] for row in group], 0.95),
            "p99_ms": max(row["runtime_p99_ms"] for row in group),
            "max_ms": max(row["runtime_max_ms"] for row in group),
        })
        for metric in ("overlay_price_shocks", "overlay_delayed_units", "overlay_released_units", "overlay_terminal_flush_units", "overlay_field_changed", "overlay_nonpremium_changed"):
            if any(metric in row for row in group):
                item[metric] = sum(float(row.get(metric, 0) or 0) for row in group)
        output.append(item)
    return output


def load_v18_agent():
    from run_v012_top5_vs_v18 import load_v18_agent as loader
    return loader()


def load_hamburger_agent():
    from run_v006_benchmark import load_hamburger_agent as loader
    return loader(HAMBURGER_NOTEBOOK)


def load_frontier_agent():
    from run_v008_benchmark import load_notebook_agent
    return load_notebook_agent(FRONTIER_NOTEBOOK, f"v015a_frontier_{time.time_ns()}")


def load_opponent(name):
    if name in {"starter", "random"}:
        return name
    if name == "v012":
        return load_module(V012_PATH, f"v015a_opp_v012_{time.time_ns()}").agent
    if name == "baseline":
        return load_module(ROOT / "main.py", f"v015a_opp_baseline_{time.time_ns()}").agent
    if name == "v18":
        return load_v18_agent()
    if name == "hamburger":
        return load_hamburger_agent()
    if name == "frontier":
        return load_frontier_agent()
    raise ValueError(f"unknown opponent: {name}")


def run_replay_counterfactual(variants):
    rows = []
    for path in sorted(REPLAY_DIR.glob("*.json")):
        route = load_replay(path)
        opponent = ReplayAgent(route["actions"])
        for variant in variants:
            control = load_module(V012_PATH, f"v015a_replay_control_{time.time_ns()}").agent
            control_row = run_game(control, opponent, route["seed"], route["seat"])
            control_row.update({"replay": path.stem, "variant": "control", "source": str(path)})
            rows.append(control_row)
            candidate = OverlayAgent(variant)
            candidate_row = run_game(candidate, opponent, route["seed"], route["seat"])
            candidate_row.update({"replay": path.stem, "variant": variant, "source": str(path)})
            rows.append(candidate_row)
            delta = candidate_row["margin"] - control_row["margin"]
            print(f"[replay] {path.stem} {variant} delta={delta:.1f}", flush=True)
    write_csv(OUT_DIR / "replay_counterfactual_raw.csv", rows)
    write_csv(OUT_DIR / "replay_counterfactual_summary.csv", summarize(rows, ("variant",)))
    return rows


def run_matrix(variants, opponents, seeds):
    rows = []
    total = (1 + len(variants)) * len(opponents) * len(seeds) * 2
    index = 0
    for variant in ("control", *variants):
        for opponent_name in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    if variant == "control":
                        candidate = load_module(V012_PATH, f"v015a_control_{time.time_ns()}").agent
                    else:
                        candidate = OverlayAgent(variant)
                    opponent = load_opponent(opponent_name)
                    print(f"[{index}/{total}] {variant} vs {opponent_name} seed={seed} seat={seat}", flush=True)
                    row = run_game(candidate, opponent, seed, seat)
                    row.update({"variant": variant, "opponent": opponent_name})
                    rows.append(row)
    write_csv(OUT_DIR / "matrix_raw.csv", rows)
    write_csv(OUT_DIR / "matrix_summary.csv", summarize(rows, ("variant", "opponent")))
    return rows


def build_gate_report(matrix_rows, replay_rows):
    summary = summarize(matrix_rows, ("variant", "opponent"))
    control = [row for row in summary if row.get("variant") == "control"]
    control_mean = statistics.mean([row["mean_money"] for row in control]) if control else 0.0
    control_min = min([row["min_money"] for row in control], default=0.0)
    control_wins = sum(row["wins"] for row in control)
    control_games = sum(row["games"] for row in control)
    report = {"variants": {}, "control_mean_money": control_mean, "control_min_money": control_min}
    for variant in sorted({row.get("variant") for row in summary if row.get("variant") not in {None, "control"}}):
        group = [row for row in summary if row.get("variant") == variant]
        raw_group = [row for row in matrix_rows if row.get("variant") == variant]
        raw_control = {
            (row.get("opponent"), row.get("seed"), row.get("seat")): row
            for row in matrix_rows if row.get("variant") == "control"
        }
        inherited_shape_ok = True
        for row in raw_group:
            control_row = raw_control.get((row.get("opponent"), row.get("seed"), row.get("seat")))
            if control_row is None:
                inherited_shape_ok = False
                break
            if row.get("agent_errors", 0) > control_row.get("agent_errors", 0):
                inherited_shape_ok = False
                break
            if row.get("invalid_action_shapes", 0) > control_row.get("invalid_action_shapes", 0):
                inherited_shape_ok = False
                break
        games = sum(row["games"] for row in group)
        wins = sum(row["wins"] for row in group)
        mean_money = statistics.mean([row["mean_money"] for row in group]) if group else 0.0
        min_money = min([row["min_money"] for row in group], default=0.0)
        p99 = max([row["p99_ms"] for row in group], default=0.0)
        replay_candidates = [row for row in replay_rows if row.get("variant") == variant]
        replay_controls = {row["replay"]: row["margin"] for row in replay_rows if row.get("variant") == "control"}
        deltas = [row["margin"] - replay_controls[row["replay"]] for row in replay_candidates if row["replay"] in replay_controls]
        checks = {
            "all_done": bool(group) and all(row["all_done"] for row in group),
            # Some opponents expose a hand-count boundary that V012 already
            # produces (notably the Hamburger trace).  The candidate must not
            # add errors/invalid shapes relative to the same control game.
            "no_new_errors_or_invalid": bool(group) and inherited_shape_ok,
            "mean_cash_not_lower": mean_money >= control_mean,
            "min_cash_not_below_97pct": control_min <= 0 or min_money >= control_min * 0.97,
            "win_rate_not_lower": games > 0 and wins / games >= (control_wins / control_games if control_games else 0.0),
            "p99_under_1000ms": p99 < 1000.0,
            "field_unchanged": all(row.get("overlay_field_changed", 0) == 0 for row in group),
            "nonpremium_unchanged": all(row.get("overlay_nonpremium_changed", 0) == 0 for row in group),
            "replay_mean_delta_nonnegative": bool(deltas) and statistics.mean(deltas) >= -1e-6,
        }
        report["variants"][variant] = {
            "checks": checks,
            "pass": all(checks.values()),
            "mean_money": mean_money,
            "min_money": min_money,
            "replay_mean_delta": statistics.mean(deltas) if deltas else None,
            "replay_improved_games": sum(delta > 0 for delta in deltas),
            "replay_games": len(deltas),
        }
    return report


def write_submission(path=None):
    """Write V012 plus overlay as an independent submission artifact."""
    target = path or (OUT_DIR / "main.py")
    overlay = (ROOT / "experiments/v015a_market_overlay.py").read_text(encoding="utf-8")
    suffix = """

# V015a market collision overlay
_V015A_OVERLAY = MarketCollisionOverlay()
_V015A_BASE_AGENT = agent
def agent(obs, config=None):
    _prepared = prepare_observation(obs, _V015A_OVERLAY)
    try:
        _base_action = _V015A_BASE_AGENT(_prepared, config)
    except TypeError:
        _base_action = _V015A_BASE_AGENT(_prepared)
    return _V015A_OVERLAY.apply(_prepared, _base_action)
"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(V012_PATH.read_text(encoding="utf-8") + "\n\n" + overlay + suffix, encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("replay", "matrix", "all"), default="all")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_THRESHOLDS), default=["default"])
    parser.add_argument("--opponents", nargs="+", choices=DEFAULT_OPPONENTS, default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--write-submission", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    replay_rows = []
    matrix_rows = []
    if args.stage in {"replay", "all"}:
        replay_rows = run_replay_counterfactual(args.variants)
    if args.stage in {"matrix", "all"}:
        matrix_rows = run_matrix(args.variants, args.opponents, args.seeds)
        if not replay_rows:
            replay_rows = read_csv_rows(OUT_DIR / "replay_counterfactual_raw.csv")
    report = build_gate_report(matrix_rows, replay_rows) if matrix_rows else {"variants": {}}
    (OUT_DIR / "gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write_submission:
        passed = [name for name, item in report.get("variants", {}).items() if item.get("pass")]
        if len(passed) == 1:
            print(f"submission artifact: {write_submission()}")
        elif not passed:
            print("no V015a variant passed; no submission artifact written")
        else:
            print(f"multiple passing variants {passed}; choose one before writing")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
