"""Evaluate V018 wave/MPC market candidates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from v015a_market_overlay import MAX_MARKET_ORDERS, PREMIUM_PRODUCTS
from run_v015a_market_collision import (
    DEFAULT_SEEDS,
    EPISODE_STEPS,
    FRONTIER_NOTEBOOK,
    HAMBURGER_NOTEBOOK,
    REPLAY_DIR,
    V012_PATH,
    OverlayAgent,
    Probe,
    ReplayAgent,
    load_module,
    load_opponent,
    load_replay,
    percentile,
    read_csv_rows,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "baseline/artifacts/v018_market_wave"
ARTIFACTS = {
    "fixed_wave": OUT_DIR / "v018a_fixed_wave.py",
    "daily_mpc": OUT_DIR / "v018b_daily_mpc.py",
    "robust_mpc": OUT_DIR / "v018c_robust_mpc.py",
}
VARIANTS = ("fixed_wave", "daily_mpc", "robust_mpc")
MATRIX_OPPONENTS = ("v012", "baseline", "v18", "hamburger", "frontier")


def _nonpremium(orders):
    return [
        order for order in (orders or [])
        if not (
            isinstance(order, list)
            and len(order) >= 2
            and order[0] == "SELL"
            and order[1] in PREMIUM_PRODUCTS
        )
    ]


class V018Agent:
    """Run the exact generated artifact and expose invariant diagnostics."""

    def __init__(self, variant):
        self.variant = variant
        self.module = load_module(ARTIFACTS[variant], f"v018_{variant}_{time.time_ns()}")
        self.field_changed = 0
        self.nonpremium_changed = 0
        self.base_module = load_module(V012_PATH, f"v018_base_{variant}_{time.time_ns()}")

    def __call__(self, obs, config=None):
        overlay = self.module._V018_OVERLAY
        prepared = self.module.prepare_observation(obs, overlay)
        try:
            # Use an independent V012 module for the invariant comparison.
            # Calling _V018_BASE_AGENT and then module.agent would execute the
            # same stateful baseline twice in one transition.
            base = self.base_module.agent(prepared, config)
        except TypeError:
            base = self.base_module.agent(prepared)
        try:
            output = self.module.agent(prepared, config)
        except TypeError:
            output = self.module.agent(prepared)
        if output.get("farmer") != base.get("farmer") or output.get("hands") != base.get("hands"):
            self.field_changed += 1
        # V018's wrapper applies the V015a overlay after the wave controller;
        # non-premium orders must be identical to the V012 action.
        if _nonpremium(output.get("market")) != _nonpremium(base.get("market")):
            self.nonpremium_changed += 1
        return output

    def diagnostics(self, player):
        controller = self.module._V018_CONTROLLER.diagnostics(player)
        overlay = self.module._V018_OVERLAY.diagnostics(player)
        result = {f"wave_{key}": value for key, value in controller.items()}
        result.update({f"overlay_{key}": value for key, value in overlay.items()})
        result.update({"field_changed": self.field_changed, "nonpremium_changed": self.nonpremium_changed})
        return result


def build_candidate(name):
    if name == "control_v012":
        return load_module(V012_PATH, f"v018_control_v012_{time.time_ns()}").agent
    if name == "control_v015a":
        return OverlayAgent("default")
    if name in VARIANTS:
        return V018Agent(name)
    raise ValueError(name)


def run_game(candidate, opponent, seed, seat):
    probe = Probe(candidate)
    players = [probe, opponent] if seat == 0 else [opponent, probe]
    started = time.perf_counter()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine, other = final[seat], final[1 - seat]
    my_money = float(mine.observation["farms"][seat]["money"])
    other_money = float(other.observation["farms"][1 - seat]["money"])
    margin = my_money - other_money
    row = {
        "seed": int(seed),
        "seat": int(seat),
        "candidate_money": my_money,
        "opponent_money": other_money,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "candidate_status": mine.status,
        "opponent_status": other.status,
        "game_done": int(mine.status == "DONE" and other.status == "DONE"),
        "wall_seconds": time.perf_counter() - started,
    }
    shed = (mine.observation.get("private", {}) or {}).get("shed", {}) or {}
    for item in PREMIUM_PRODUCTS:
        row[f"final_shed_{item}"] = int(shed.get(item, 0) or 0)
    row.update(probe.metrics())
    if callable(getattr(candidate, "diagnostics", None)):
        diagnostics = candidate.diagnostics(seat)
        for key, value in diagnostics.items():
            if key in {"history", "ratios", "items", "pending"} or key.endswith("_history") or key.endswith("_ratios"):
                row[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
    return row


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def summarize(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        outcomes = Counter(row.get("result") for row in group)
        item = dict(zip(fields, key))
        item.update({
            "games": len(group),
            "mean_money": statistics.mean(_as_float(row.get("candidate_money")) for row in group),
            "mean_margin": statistics.mean(_as_float(row.get("margin")) for row in group),
            "min_money": min(_as_float(row.get("candidate_money")) for row in group),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(group),
            "all_done": int(all(int(row.get("game_done", 0) or 0) for row in group)),
            "agent_errors": sum(int(row.get("agent_errors", 0) or 0) for row in group),
            "invalid_action_shapes": sum(int(row.get("invalid_action_shapes", 0) or 0) for row in group),
            "p50_ms": percentile([_as_float(row.get("runtime_p50_ms")) for row in group], 0.50),
            "p95_ms": percentile([_as_float(row.get("runtime_p95_ms")) for row in group], 0.95),
            "p99_ms": max(_as_float(row.get("runtime_p99_ms")) for row in group),
            "max_ms": max(_as_float(row.get("runtime_max_ms")) for row in group),
        })
        for metric in (
            "wave_replans", "wave_changed_market_actions", "wave_guard_skips",
            "wave_reduced_MELON", "wave_reduced_STRAWBERRY", "wave_reduced_MILK", "wave_reduced_WOOL",
            "overlay_price_shocks", "overlay_delayed_units", "overlay_released_units",
            "overlay_terminal_flush_units", "field_changed", "nonpremium_changed",
        ):
            if any(metric in row for row in group):
                item[f"sum_{metric}"] = sum(_as_float(row.get(metric)) for row in group)
        output.append(item)
    return output


def run_replay_counterfactual():
    rows = []
    for path in sorted(REPLAY_DIR.glob("*.json")):
        route = load_replay(path)
        opponent = ReplayAgent(route["actions"])
        for name in ("control_v012", "control_v015a", *VARIANTS):
            candidate = build_candidate(name)
            row = run_game(candidate, opponent, route["seed"], route["seat"])
            row.update({"replay": path.stem, "variant": name, "source": str(path)})
            rows.append(row)
            print(f"[replay] {path.stem} {name} margin={row['margin']:.1f}", flush=True)
    write_csv(OUT_DIR / "replay_counterfactual_raw.csv", rows)
    write_csv(OUT_DIR / "replay_counterfactual_summary.csv", summarize(rows, ("variant",)))
    return rows


def run_matrix(opponents, seeds):
    rows = []
    candidates = ("control_v012", "control_v015a", *VARIANTS)
    total = len(candidates) * len(opponents) * len(seeds) * 2
    index = 0
    for name in candidates:
        for opponent_name in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    candidate = build_candidate(name)
                    opponent = load_opponent(opponent_name)
                    print(f"[{index}/{total}] {name} vs {opponent_name} seed={seed} seat={seat}", flush=True)
                    row = run_game(candidate, opponent, seed, seat)
                    row.update({"variant": name, "opponent": opponent_name})
                    rows.append(row)
    write_csv(OUT_DIR / "matrix_raw.csv", rows)
    write_csv(OUT_DIR / "matrix_summary.csv", summarize(rows, ("variant", "opponent")))
    return rows


def _mean(rows, field):
    values = [_as_float(row.get(field)) for row in rows]
    return statistics.mean(values) if values else 0.0


def build_gate_report(matrix_rows, replay_rows):
    control = [row for row in matrix_rows if row.get("variant") == "control_v015a"]
    control_mean = _mean(control, "candidate_money")
    control_min = min((_as_float(row.get("candidate_money")) for row in control), default=0.0)
    control_wins = sum(row.get("result") == "win" for row in control)
    control_games = len(control)
    report = {
        "control": "control_v015a",
        "control_mean_money": control_mean,
        "control_min_money": control_min,
        "variants": {},
    }
    replay_control = {
        row.get("replay"): _as_float(row.get("margin"))
        for row in replay_rows if row.get("variant") == "control_v015a"
    }
    control_by_game = {
        (row.get("opponent"), row.get("seed"), row.get("seat")): row
        for row in control
    }
    for name in VARIANTS:
        rows = [row for row in matrix_rows if row.get("variant") == name]
        replay_rows_for_variant = [row for row in replay_rows if row.get("variant") == name]
        deltas = [
            _as_float(row.get("margin")) - replay_control[row.get("replay")]
            for row in replay_rows_for_variant if row.get("replay") in replay_control
        ]
        no_new_errors = bool(rows)
        for row in rows:
            control_row = control_by_game.get((row.get("opponent"), row.get("seed"), row.get("seat")))
            if control_row is None:
                no_new_errors = False
                break
            if int(row.get("agent_errors", 0) or 0) > int(control_row.get("agent_errors", 0) or 0):
                no_new_errors = False
                break
            if int(row.get("invalid_action_shapes", 0) or 0) > int(control_row.get("invalid_action_shapes", 0) or 0):
                no_new_errors = False
                break
        wins = sum(row.get("result") == "win" for row in rows)
        hf = [row for row in rows if row.get("opponent") in {"hamburger", "frontier"}]
        control_hf = [row for row in control if row.get("opponent") in {"hamburger", "frontier"}]
        p99 = max((_as_float(row.get("runtime_p99_ms")) for row in rows), default=0.0)
        checks = {
            "all_done": bool(rows) and all(int(row.get("game_done", 0) or 0) for row in rows),
            "no_new_errors_or_invalid": no_new_errors,
            "mean_cash_plus_0_5pct": _mean(rows, "candidate_money") >= control_mean * 1.005,
            "min_cash_at_least_95pct": min((_as_float(row.get("candidate_money")) for row in rows), default=0.0) >= control_min * 0.95,
            "win_rate_not_lower": bool(rows) and wins / len(rows) >= (control_wins / control_games if control_games else 0.0),
            "hamburger_frontier_at_least_95pct": _mean(hf, "candidate_money") >= _mean(control_hf, "candidate_money") * 0.95,
            "replay_mean_delta_nonnegative": bool(deltas) and statistics.mean(deltas) >= 0.0,
            "replay_at_least_6_not_lower": sum(delta >= 0.0 for delta in deltas) >= 6,
            "p99_under_1000ms": p99 < 1000.0,
            "field_unchanged": all(_as_float(row.get("field_changed")) == 0 for row in rows),
            "nonpremium_unchanged": all(_as_float(row.get("nonpremium_changed")) == 0 for row in rows),
        }
        report["variants"][name] = {
            "pass": all(checks.values()),
            "checks": checks,
            "mean_money": _mean(rows, "candidate_money"),
            "min_money": min((_as_float(row.get("candidate_money")) for row in rows), default=0.0),
            "wins": wins,
            "games": len(rows),
            "replay_mean_delta": statistics.mean(deltas) if deltas else None,
            "replay_not_lower": sum(delta >= 0.0 for delta in deltas),
            "replay_games": len(deltas),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "replay", "matrix", "all"), default="all")
    parser.add_argument("--opponents", nargs="+", default=list(MATRIX_OPPONENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        for name in VARIANTS:
            row = run_game(build_candidate(name), "starter", 17, 0)
            print(json.dumps({
                "variant": name,
                "candidate_money": row.get("candidate_money"),
                "game_done": row.get("game_done"),
                "agent_errors": row.get("agent_errors"),
                "invalid_action_shapes": row.get("invalid_action_shapes"),
                "runtime_p99_ms": row.get("runtime_p99_ms"),
            }, ensure_ascii=False))
        return
    replay_rows = run_replay_counterfactual() if args.stage in {"replay", "all"} else read_csv_rows(OUT_DIR / "replay_counterfactual_raw.csv")
    matrix_rows = run_matrix(args.opponents, args.seeds) if args.stage in {"matrix", "all"} else read_csv_rows(OUT_DIR / "matrix_raw.csv")
    report = build_gate_report(matrix_rows, replay_rows) if matrix_rows and replay_rows else {"variants": {}}
    (OUT_DIR / "gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
