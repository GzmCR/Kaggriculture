"""Run V020 replay counterfactuals and local paired benchmarks."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from v020_value_aware_market import ValueAwareMarketController
from run_v015a_market_collision import (
    DEFAULT_OPPONENTS,
    DEFAULT_SEEDS,
    EPISODE_STEPS,
    OUT_DIR as V015A_OUT_DIR,
    REPLAY_DIR as DEFAULT_REPLAY_DIR,
    V012_PATH,
    Probe,
    ReplayAgent,
    load_module,
    load_opponent,
    run_game,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "baseline/artifacts/v020_value_aware_market"
REPLAY_DIR = ROOT / "baseline/history/v015a_market_collision/log"
VARIANTS = ("balanced", "sensitive", "conservative")
V015A_PATH = ROOT / "baseline/artifacts/v015a_market_collision/main.py"
ALL_OPPONENTS = tuple(dict.fromkeys((*DEFAULT_OPPONENTS, "v015a")))


class V020Agent:
    def __init__(self, variant):
        self.variant = variant
        self.module = load_module(V012_PATH, f"v020_base_{variant}_{time.time_ns()}")
        runtime = getattr(self.module, "_V012_RUNTIME", {})
        self.controller = ValueAwareMarketController(variant=variant, runtime=runtime)
        self.field_changed = 0
        self.nonpremium_changed = 0

    @staticmethod
    def _nonpremium(orders):
        return [
            list(order) for order in (orders or [])
            if not (
                isinstance(order, list)
                and len(order) >= 2
                and order[0] == "SELL"
                and order[1] in {"MELON", "STRAWBERRY", "MILK", "WOOL"}
            )
        ]

    def __call__(self, obs, config=None):
        try:
            try:
                base = self.module.agent(obs, config)
            except TypeError:
                base = self.module.agent(obs)
            output = self.controller.apply(obs, base)
            if output.get("farmer") != base.get("farmer") or output.get("hands") != base.get("hands"):
                self.field_changed += 1
            if self._nonpremium(output.get("market")) != self._nonpremium(base.get("market")):
                self.nonpremium_changed += 1
            return output
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}

    def diagnostics(self, player):
        result = self.controller.diagnostics(player)
        result.update({
            "field_changed": self.field_changed,
            "nonpremium_changed": self.nonpremium_changed,
        })
        return result


def load_recorded_opponent(path):
    """Load the actual opponent seat from a replay, not our own history."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    if len(steps) < EPISODE_STEPS:
        raise ValueError(f"{path} has {len(steps)} steps")
    info = payload.get("info", {}) or {}
    names = info.get("TeamNames", []) or info.get("teamNames", []) or []
    our_seat = next(
        (index for index, name in enumerate(names[:2]) if "GzmCR" in str(name) or "632" in str(name)),
        0,
    )
    opponent_seat = 1 - our_seat
    actions = [
        copy.deepcopy(steps[min(index + 1, EPISODE_STEPS - 1)][opponent_seat].get("action") or {})
        for index in range(EPISODE_STEPS)
    ]
    return {
        "path": str(path),
        "seat": our_seat,
        "seed": _int(info.get("seed", 0)),
        "actions": actions,
    }


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def run_v020_game(candidate, opponent, seed, seat):
    row = run_game(candidate, opponent, seed, seat)
    if isinstance(candidate, V020Agent):
        diagnostics = candidate.diagnostics(seat)
        for key, value in diagnostics.items():
            if isinstance(value, (dict, list)):
                row[f"v020_{key}"] = json.dumps(value, sort_keys=True, ensure_ascii=False)
            else:
                row[f"v020_{key}"] = value
    return row


def _control_agent():
    return load_module(V012_PATH, f"v020_control_{time.time_ns()}").agent


def _load_opponent(name):
    if name == "v015a":
        return load_module(V015A_PATH, f"v020_opp_v015a_{time.time_ns()}").agent
    return load_opponent(name)


def run_replay_counterfactual(variants):
    rows = []
    paths = sorted(REPLAY_DIR.glob("*.json"))
    for path in paths:
        route = load_recorded_opponent(path)
        opponent = ReplayAgent(route["actions"])
        control = _control_agent()
        control_row = run_v020_game(control, opponent, route["seed"], route["seat"])
        control_row.update({"replay": path.stem, "variant": "control", "source": str(path)})
        rows.append(control_row)
        for variant in variants:
            candidate = V020Agent(variant)
            candidate_row = run_v020_game(candidate, opponent, route["seed"], route["seat"])
            candidate_row.update({"replay": path.stem, "variant": variant, "source": str(path)})
            rows.append(candidate_row)
            print(
                f"[replay] {path.stem} {variant} "
                f"delta={candidate_row['margin'] - control_row['margin']:.1f}",
                flush=True,
            )
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
                    candidate = _control_agent() if variant == "control" else V020Agent(variant)
                    opponent = _load_opponent(opponent_name)
                    print(
                        f"[{index}/{total}] {variant} vs {opponent_name} "
                        f"seed={seed} seat={seat}",
                        flush=True,
                    )
                    row = run_v020_game(candidate, opponent, seed, seat)
                    row.update({"variant": variant, "opponent": opponent_name})
                    rows.append(row)
    write_csv(OUT_DIR / "matrix_raw.csv", rows)
    write_csv(OUT_DIR / "matrix_summary.csv", summarize(rows, ("variant", "opponent")))
    return rows


def _mean(values):
    return statistics.mean(values) if values else 0.0


def build_gate_report(matrix_rows, replay_rows):
    summary = summarize(matrix_rows, ("variant", "opponent"))
    control_summary = [row for row in summary if row.get("variant") == "control"]
    control_mean = _mean([row["mean_money"] for row in control_summary])
    control_min = min([row["min_money"] for row in control_summary], default=0.0)
    control_wins = sum(row["wins"] for row in control_summary)
    control_games = sum(row["games"] for row in control_summary)
    report = {
        "replay_files": len(list(REPLAY_DIR.glob("*.json"))),
        "matrix_games": len(matrix_rows),
        "control_mean_money": control_mean,
        "control_min_money": control_min,
        "variants": {},
    }
    raw_control = {
        (row.get("opponent"), row.get("seed"), row.get("seat")): row
        for row in matrix_rows if row.get("variant") == "control"
    }
    replay_controls = {
        row.get("replay"): float(row.get("margin", 0.0) or 0.0)
        for row in replay_rows if row.get("variant") == "control"
    }
    variant_names = sorted({row.get("variant") for row in matrix_rows if row.get("variant") not in {None, "control"}})
    for variant in variant_names:
        group = [row for row in summary if row.get("variant") == variant]
        raw_group = [row for row in matrix_rows if row.get("variant") == variant]
        inherited_shape_ok = True
        for row in raw_group:
            control = raw_control.get((row.get("opponent"), row.get("seed"), row.get("seat")))
            if control is None or row.get("agent_errors", 0) > control.get("agent_errors", 0):
                inherited_shape_ok = False
                break
            if row.get("invalid_action_shapes", 0) > control.get("invalid_action_shapes", 0):
                inherited_shape_ok = False
                break
        games = sum(row["games"] for row in group)
        wins = sum(row["wins"] for row in group)
        mean_money = _mean([row["mean_money"] for row in group])
        min_money = min([row["min_money"] for row in group], default=0.0)
        p99 = max([row["p99_ms"] for row in group], default=0.0)
        replay_rows_variant = [row for row in replay_rows if row.get("variant") == variant]
        deltas = [
            float(row.get("margin", 0.0) or 0.0) - replay_controls[row["replay"]]
            for row in replay_rows_variant
            if row.get("replay") in replay_controls
        ]
        checks = {
            "all_done": bool(group) and all(row["all_done"] for row in group),
            "no_new_errors_or_invalid": bool(group) and inherited_shape_ok,
            "mean_cash_not_lower": mean_money >= control_mean,
            "min_cash_not_below_97pct": control_min <= 0 or min_money >= control_min * 0.97,
            "win_rate_not_lower": games > 0 and wins / games >= (control_wins / control_games if control_games else 0.0),
            "p99_under_1000ms": p99 < 1000.0,
            "field_unchanged": all(float(row.get("v020_field_changed", 0) or 0) == 0 for row in raw_group),
            "nonpremium_unchanged": all(float(row.get("v020_nonpremium_changed", 0) or 0) == 0 for row in raw_group),
            "replay_mean_delta_nonnegative": bool(deltas) and _mean(deltas) >= -1e-6,
        }
        report["variants"][variant] = {
            "checks": checks,
            "pass": all(checks.values()),
            "mean_money": mean_money,
            "min_money": min_money,
            "replay_mean_delta": _mean(deltas) if deltas else None,
            "replay_improved_games": sum(delta > 0 for delta in deltas),
            "replay_games": len(deltas),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("replay", "matrix", "all"), default="all")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--opponents", nargs="+", choices=ALL_OPPONENTS, default=list(ALL_OPPONENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    replay_rows = []
    matrix_rows = []
    if args.stage in {"replay", "all"}:
        replay_rows = run_replay_counterfactual(args.variants)
    if args.stage in {"matrix", "all"}:
        matrix_rows = run_matrix(args.variants, args.opponents, args.seeds)
    if not replay_rows:
        replay_path = OUT_DIR / "replay_counterfactual_raw.csv"
        if replay_path.exists():
            import csv
            with replay_path.open(newline="", encoding="utf-8") as handle:
                replay_rows = list(csv.DictReader(handle))
    report = build_gate_report(matrix_rows, replay_rows) if matrix_rows else {"variants": {}}
    (OUT_DIR / "gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
