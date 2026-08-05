"""Evaluate V016 market-expert selection against V012 and V015a.

The default matrix is 3 candidates x 5 opponents x 6 seeds x 2 seats,
plus V012 and V015a controls.  Use ``--stage smoke`` for a quick local check.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from v015a_market_overlay import MAX_MARKET_ORDERS, MarketCollisionOverlay, PREMIUM_PRODUCTS, prepare_observation
from v016_market_selector import MarketValueSelector
from run_v015a_market_collision import (
    DEFAULT_SEEDS,
    EPISODE_STEPS,
    FRONTIER_NOTEBOOK,
    HAMBURGER_NOTEBOOK,
    REPLAY_DIR,
    V012_PATH,
    load_module,
    load_opponent,
    load_replay,
    OverlayAgent,
    Probe,
    ReplayAgent,
    percentile,
    read_csv_rows,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "baseline/artifacts/v016_market_value_selector"
VARIANTS = ("value_only", "collision_hedged", "aggressive_value")
MATRIX_OPPONENTS = ("v012", "baseline", "v18", "hamburger", "frontier")


class V016Agent:
    """V012 board route + V016 market selector + V015a overlay."""

    def __init__(self, variant):
        self.variant = variant
        self.module = load_module(V012_PATH, f"v016_base_{variant}_{time.time_ns()}")
        self.selector = MarketValueSelector(
            variant,
            runtime=self.module._V18_RUNTIME,
            pipeline_fn=self.module._farm_pipeline,
        )
        self.overlay = MarketCollisionOverlay()
        self.field_changed = 0
        self.nonpremium_changed = 0

    @staticmethod
    def _nonpremium(orders):
        return [
            order for order in (orders or [])
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
        selected = self.selector.apply(prepared, base)
        output = self.overlay.apply(prepared, selected)
        if output.get("farmer") != base.get("farmer") or output.get("hands") != base.get("hands"):
            self.field_changed += 1
        # Expert switching is allowed to select a different complete market
        # schedule.  The V015a overlay itself must not alter that expert's
        # non-premium orders, so compare against the selected action here.
        if self._nonpremium(output.get("market")) != self._nonpremium(selected.get("market")):
            self.nonpremium_changed += 1
        return output

    def diagnostics(self, player):
        result = self.selector.diagnostics(player)
        overlay = self.overlay.diagnostics(player)
        result.update({f"overlay_{key}": value for key, value in overlay.items()})
        result.update({"field_changed": self.field_changed, "nonpremium_changed": self.nonpremium_changed})
        return result


def build_candidate(variant):
    if variant == "control":
        return load_module(V012_PATH, f"v016_control_{time.time_ns()}").agent
    if variant == "v015a":
        return OverlayAgent("default")
    if variant in VARIANTS:
        return V016Agent(variant)
    raise ValueError(variant)


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
    row = {
        "seed": int(seed),
        "seat": int(seat),
        "candidate_money": my_money,
        "opponent_money": other_money,
        "margin": my_money - other_money,
        "result": "win" if my_money > other_money else "loss" if my_money < other_money else "tie",
        "candidate_status": mine.status,
        "opponent_status": other.status,
        "game_done": int(mine.status == "DONE" and other.status == "DONE"),
        "wall_seconds": time.perf_counter() - started,
    }
    shed = (mine.observation.get("private", {}) or {}).get("shed", {}) or {}
    for item in PREMIUM_PRODUCTS:
        row[f"final_shed_{item}"] = int(shed.get(item, 0) or 0)
    row.update(probe.metrics())
    diagnostics = candidate.diagnostics(seat) if callable(getattr(candidate, "diagnostics", None)) else {}
    for key, value in diagnostics.items():
        if key in {"selection_history", "score_snapshot"}:
            row[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    return row


def _summary(rows, fields):
    result = summarize(rows, fields)
    # summarize() already provides the standard cash/W-T-L/runtime columns.
    # Add aggregate diagnostics that are specific to V016.
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    by_key = {tuple(item.get(field) for field in fields): item for item in result}
    for key, group in grouped.items():
        item = by_key[key]
        for metric in (
            "selection_count", "market_replacements", "overlay_price_shocks",
            "overlay_delayed_units", "overlay_released_units", "overlay_terminal_flush_units",
            "field_changed", "nonpremium_changed",
        ):
            values = [float(row.get(metric, 0) or 0) for row in group]
            if values:
                item[f"sum_{metric}"] = sum(values)
        experts = Counter(str(row.get("selected_expert")) for row in group if row.get("selected_expert"))
        item["selected_experts"] = json.dumps(dict(sorted(experts.items())), sort_keys=True)
    return result


def run_replay_counterfactual():
    rows = []
    replay_paths = sorted(REPLAY_DIR.glob("*.json"))
    for path in replay_paths:
        route = load_replay(path)
        opponent = ReplayAgent(route["actions"])
        for variant in ("control", "v015a", *VARIANTS):
            candidate = build_candidate(variant)
            row = run_game(candidate, opponent, route["seed"], route["seat"])
            row.update({"replay": path.stem, "variant": variant, "source": str(path)})
            rows.append(row)
            print(f"[replay] {path.stem} {variant} margin={row['margin']:.1f}", flush=True)
    write_csv(OUT_DIR / "replay_counterfactual_raw.csv", rows)
    write_csv(OUT_DIR / "replay_counterfactual_summary.csv", _summary(rows, ("variant",)))
    return rows


def run_matrix(opponents, seeds):
    rows = []
    candidates = ("control", "v015a", *VARIANTS)
    total = len(candidates) * len(opponents) * len(seeds) * 2
    index = 0
    for variant in candidates:
        for opponent_name in opponents:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    candidate = build_candidate(variant)
                    opponent = load_opponent(opponent_name)
                    print(f"[{index}/{total}] {variant} vs {opponent_name} seed={seed} seat={seat}", flush=True)
                    row = run_game(candidate, opponent, seed, seat)
                    row.update({"variant": variant, "opponent": opponent_name})
                    rows.append(row)
    write_csv(OUT_DIR / "matrix_raw.csv", rows)
    write_csv(OUT_DIR / "matrix_summary.csv", _summary(rows, ("variant", "opponent")))
    return rows


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.mean(values) if values else 0.0


def build_gate_report(matrix_rows, replay_rows):
    report = {"variants": {}, "control_mean_money": 0.0, "control_min_money": 0.0}
    control_rows = [row for row in matrix_rows if row.get("variant") == "control"]
    report["control_mean_money"] = _mean(control_rows, "candidate_money")
    report["control_min_money"] = min((float(row["candidate_money"]) for row in control_rows), default=0.0)
    control_wins = sum(row.get("result") == "win" for row in control_rows)
    control_games = len(control_rows)
    replay_control = {row["replay"]: float(row["margin"]) for row in replay_rows if row.get("variant") == "control"}
    for variant in VARIANTS:
        rows = [row for row in matrix_rows if row.get("variant") == variant]
        replay_rows_variant = [row for row in replay_rows if row.get("variant") == variant]
        deltas = [float(row["margin"]) - replay_control[row["replay"]] for row in replay_rows_variant if row.get("replay") in replay_control]
        by_game = {
            (row.get("opponent"), row.get("seed"), row.get("seat")): row
            for row in matrix_rows if row.get("variant") == "control"
        }
        no_new_errors = True
        for row in rows:
            control = by_game.get((row.get("opponent"), row.get("seed"), row.get("seat")))
            if control is None or row.get("agent_errors", 0) > control.get("agent_errors", 0) or row.get("invalid_action_shapes", 0) > control.get("invalid_action_shapes", 0):
                no_new_errors = False
                break
        money = _mean(rows, "candidate_money")
        min_money = min((float(row["candidate_money"]) for row in rows), default=0.0)
        wins = sum(row.get("result") == "win" for row in rows)
        p99 = max((float(row.get("runtime_p99_ms", 0) or 0) for row in rows), default=0.0)
        hamburger_frontier = [row for row in rows if row.get("opponent") in {"hamburger", "frontier"}]
        control_hf = [row for row in control_rows if row.get("opponent") in {"hamburger", "frontier"}]
        checks = {
            "all_done": bool(rows) and all(row.get("game_done") for row in rows),
            "no_new_errors_or_invalid": bool(rows) and no_new_errors,
            "mean_cash_plus_1pct": money >= report["control_mean_money"] * 1.01,
            "min_cash_at_least_90pct": min_money >= report["control_min_money"] * 0.90,
            "win_rate_not_lower": bool(rows) and wins / len(rows) >= (control_wins / control_games if control_games else 0.0),
            "hamburger_frontier_at_least_90pct": _mean(hamburger_frontier, "candidate_money") >= _mean(control_hf, "candidate_money") * 0.90,
            "p99_under_1000ms": p99 < 1000.0,
            "field_unchanged": all(float(row.get("field_changed", 0) or 0) == 0 for row in rows),
            "nonpremium_unchanged": all(float(row.get("nonpremium_changed", 0) or 0) == 0 for row in rows),
            "replay_mean_delta_nonnegative": bool(deltas) and statistics.mean(deltas) >= -1e-6,
        }
        report["variants"][variant] = {
            "pass": all(checks.values()),
            "checks": checks,
            "mean_money": money,
            "min_money": min_money,
            "wins": wins,
            "games": len(rows),
            "replay_mean_delta": statistics.mean(deltas) if deltas else None,
            "replay_improved_games": sum(delta > 0 for delta in deltas),
            "replay_games": len(deltas),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("replay", "matrix", "smoke", "all"), default="all")
    parser.add_argument("--opponents", nargs="+", default=list(MATRIX_OPPONENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        rows = []
        for variant in VARIANTS:
            candidate = build_candidate(variant)
            row = run_game(candidate, "starter", 17, 0)
            row["variant"] = variant
            rows.append(row)
            print(json.dumps({key: row.get(key) for key in ("variant", "candidate_money", "game_done", "agent_errors", "invalid_action_shapes", "runtime_p99_ms", "selected_expert")}, ensure_ascii=False))
        return
    replay_rows = run_replay_counterfactual() if args.stage in {"replay", "all"} else read_csv_rows(OUT_DIR / "replay_counterfactual_raw.csv")
    matrix_rows = run_matrix(args.opponents, args.seeds) if args.stage in {"matrix", "all"} else read_csv_rows(OUT_DIR / "matrix_raw.csv")
    report = build_gate_report(matrix_rows, replay_rows) if matrix_rows and replay_rows else {"variants": {}}
    (OUT_DIR / "gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
