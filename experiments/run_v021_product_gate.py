"""Run V021 replay counterfactuals and paired local matrix benchmarks."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from v021_product_gate import V021ProductGateController
from run_v015a_market_collision import (
    DEFAULT_SEEDS,
    EPISODE_STEPS,
    Probe,
    ReplayAgent,
    load_module,
    load_opponent,
    run_game,
    write_csv,
)
from run_v020_value_aware import load_recorded_opponent


ROOT = Path(__file__).resolve().parents[1]
V012_PATH = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
V015A_PATH = ROOT / "baseline/artifacts/v015a_market_collision/main.py"
REPLAY_DIR = ROOT / "baseline/history/v015a_market_collision/log"
OUT_DIR = ROOT / "baseline/artifacts/v021_product_gate"
VARIANTS = ("safety_patch", "product_gate", "win_guard")
OPPONENTS = ("v012", "v015a", "baseline", "v18", "hamburger", "frontier", "starter", "random")


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class V021Agent:
    def __init__(self, variant):
        self.variant = variant
        self.module = load_module(V012_PATH, f"v021_base_{variant}_{time.time_ns()}")
        runtime = getattr(self.module, "_V012_RUNTIME", {})
        self.controller = V021ProductGateController(variant=variant, runtime=runtime)
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


def _load_opponent(name):
    if name == "v015a":
        return load_module(V015A_PATH, f"v021_opp_v015a_{time.time_ns()}").agent
    return load_opponent(name)


def _control_agent():
    return load_module(V012_PATH, f"v021_control_{time.time_ns()}").agent


def _run(candidate, opponent, seed, seat):
    # run_v015a's runner already probes calls, status, invalid shapes and
    # runtime.  V021 diagnostics are attached after the game completes.
    row = run_game(candidate, opponent, seed, seat)
    if isinstance(candidate, V021Agent):
        diagnostics = candidate.diagnostics(seat)
        for key, value in diagnostics.items():
            if isinstance(value, (dict, list)):
                row[f"v021_{key}"] = json.dumps(value, sort_keys=True)
            else:
                row[f"v021_{key}"] = value
    return row


def run_replay(variants, replay_dir=REPLAY_DIR):
    rows = []
    paths = sorted(Path(replay_dir).glob("*.json"))
    for path in paths:
        route = load_recorded_opponent(path)
        opponent = ReplayAgent(route["actions"])
        control = _run(_control_agent(), opponent, route["seed"], route["seat"])
        control.update({"replay": path.stem, "variant": "control", "source": str(path)})
        rows.append(control)
        for variant in variants:
            candidate = V021Agent(variant)
            row = _run(candidate, opponent, route["seed"], route["seat"])
            row.update({"replay": path.stem, "variant": variant, "source": str(path)})
            rows.append(row)
            print(
                f"[replay] {path.stem} {variant} "
                f"delta={row['margin'] - control['margin']:.1f}",
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
                    candidate = _control_agent() if variant == "control" else V021Agent(variant)
                    opponent = _load_opponent(opponent_name)
                    print(
                        f"[{index}/{total}] {variant} vs {opponent_name} "
                        f"seed={seed} seat={seat}",
                        flush=True,
                    )
                    row = _run(candidate, opponent, seed, seat)
                    row.update({"variant": variant, "opponent": opponent_name})
                    rows.append(row)
    write_csv(OUT_DIR / "matrix_raw.csv", rows)
    write_csv(OUT_DIR / "matrix_summary.csv", summarize(rows, ("variant", "opponent")))
    return rows


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
            "mean_money": statistics.mean(_num(row.get("candidate_money")) for row in group),
            "mean_margin": statistics.mean(_num(row.get("margin")) for row in group),
            "min_money": min(_num(row.get("candidate_money")) for row in group),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(group),
            "all_done": int(all(int(_num(row.get("game_done"))) for row in group)),
            "agent_errors": sum(int(_num(row.get("agent_errors"))) for row in group),
            "invalid_action_shapes": sum(int(_num(row.get("invalid_action_shapes"))) for row in group),
            "p50_ms": statistics.median(_num(row.get("runtime_p50_ms")) for row in group),
            "p95_ms": max(_num(row.get("runtime_p95_ms")) for row in group),
            "p99_ms": max(_num(row.get("runtime_p99_ms")) for row in group),
            "max_ms": max(_num(row.get("runtime_max_ms")) for row in group),
        })
        for metric in (
            "v021_deferred_MILK", "v021_deferred_WOOL", "v021_deferred_STRAWBERRY",
            "v021_deferred_MELON", "v021_terminal_passthrough", "v021_changed_market_actions",
        ):
            if any(metric in row for row in group):
                item[metric] = sum(_num(row.get(metric)) for row in group)
        output.append(item)
    return output


def build_gate_report(matrix_rows, replay_rows):
    summary = summarize(matrix_rows, ("variant", "opponent"))
    controls = [row for row in summary if row.get("variant") == "control"]
    control_mean = statistics.mean(row["mean_money"] for row in controls) if controls else 0.0
    control_min = min((row["min_money"] for row in controls), default=0.0)
    control_wins = sum(row["wins"] for row in controls)
    control_games = sum(row["games"] for row in controls)
    control_by_target = {
        (row.get("opponent"), row.get("seed"), row.get("seat")): row
        for row in matrix_rows if row.get("variant") == "control"
    }
    # Matrix raw rows have one result per target; use the same key without
    # relying on summary aggregates for shape comparisons.
    control_raw = {
        (row.get("opponent"), row.get("seed"), row.get("seat")): row
        for row in matrix_rows if row.get("variant") == "control"
    }
    replay_controls = {
        row.get("replay"): _num(row.get("margin"))
        for row in replay_rows if row.get("variant") == "control"
    }
    report = {
        "replay_files": len({row.get("replay") for row in replay_rows if row.get("replay")}),
        "matrix_games": len(matrix_rows),
        "control_mean_money": control_mean,
        "control_min_money": control_min,
        "control_win_rate": control_wins / control_games if control_games else 0.0,
        "variants": {},
    }
    for variant in VARIANTS:
        group = [row for row in summary if row.get("variant") == variant]
        raw_group = [row for row in matrix_rows if row.get("variant") == variant]
        shape_ok = bool(group)
        for row in raw_group:
            key = (row.get("opponent"), row.get("seed"), row.get("seat"))
            control = control_raw.get(key)
            if control is None:
                shape_ok = False
                continue
            shape_ok = shape_ok and int(_num(row.get("agent_errors"))) <= int(_num(control.get("agent_errors")))
            shape_ok = shape_ok and int(_num(row.get("invalid_action_shapes"))) <= int(_num(control.get("invalid_action_shapes")))
        games = sum(row["games"] for row in group)
        wins = sum(row["wins"] for row in group)
        mean_money = statistics.mean(row["mean_money"] for row in group) if group else 0.0
        min_money = min((row["min_money"] for row in group), default=0.0)
        p99 = max((row["p99_ms"] for row in group), default=0.0)
        replay_candidates = [row for row in replay_rows if row.get("variant") == variant]
        replay_deltas = [
            _num(row.get("margin")) - replay_controls[row.get("replay")]
            for row in replay_candidates if row.get("replay") in replay_controls
        ]
        combined = {"v012", "v18"}
        control_combined = sum(
            1 for row in matrix_rows
            if row.get("variant") == "control" and row.get("opponent") in combined and row.get("result") == "win"
        )
        candidate_combined = sum(
            1 for row in raw_group
            if row.get("opponent") in combined and row.get("result") == "win"
        )
        checks = {
            "all_done": bool(group) and all(row["all_done"] for row in group),
            "no_new_errors_or_invalid": shape_ok,
            "mean_cash_not_lower": mean_money >= control_mean,
            "min_cash_not_below_97pct": min_money >= control_min * 0.97 if control_min else True,
            "win_rate_not_lower": games > 0 and wins / games >= (control_wins / control_games if control_games else 0.0),
            "v012_v18_wins_not_lower": candidate_combined >= control_combined,
            "p99_under_1000ms": p99 < 1000.0,
            "field_unchanged": all(_num(row.get("v021_field_changed")) == 0 for row in raw_group),
            "nonpremium_unchanged": all(_num(row.get("v021_nonpremium_changed")) == 0 for row in raw_group),
            "replay_mean_delta_nonnegative": bool(replay_deltas) and statistics.mean(replay_deltas) >= -1e-6,
            "replay_at_least_6_not_lower": sum(delta >= -1e-6 for delta in replay_deltas) >= 6,
        }
        report["variants"][variant] = {
            "checks": checks,
            "pass": all(checks.values()),
            "mean_money": mean_money,
            "min_money": min_money,
            "wins": wins,
            "games": games,
            "replay_mean_delta": statistics.mean(replay_deltas) if replay_deltas else None,
            "replay_not_lower_games": sum(delta >= -1e-6 for delta in replay_deltas),
            "replay_games": len(replay_deltas),
            "v012_v18_wins": candidate_combined,
            "control_v012_v18_wins": control_combined,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("replay", "matrix", "all"), default="all")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--replay-dir", type=Path, default=REPLAY_DIR)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    replay_rows = run_replay(args.variants, args.replay_dir) if args.stage in {"replay", "all"} else []
    matrix_rows = run_matrix(args.variants, args.opponents, args.seeds) if args.stage in {"matrix", "all"} else []
    if not replay_rows and (OUT_DIR / "replay_counterfactual_raw.csv").exists():
        with (OUT_DIR / "replay_counterfactual_raw.csv").open(newline="", encoding="utf-8") as handle:
            replay_rows = list(csv.DictReader(handle))
    if matrix_rows:
        report = build_gate_report(matrix_rows, replay_rows)
        (OUT_DIR / "gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
