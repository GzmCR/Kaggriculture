"""Evaluate V017 product-level market rollout candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from v015a_market_overlay import MAX_MARKET_ORDERS, MarketCollisionOverlay, PREMIUM_PRODUCTS, prepare_observation
from v017_market_rollout import MarketRolloutController
from run_v015a_market_collision import (
    DEFAULT_SEEDS,
    EPISODE_STEPS,
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
OUT_DIR = ROOT / "baseline/artifacts/v017_market_rollout"
VARIANTS = ("curve_only", "opponent_aware", "robust_quota")
MATRIX_OPPONENTS = ("v012", "baseline", "v18", "hamburger", "frontier")


class V017Agent:
    def __init__(self, variant):
        self.variant = variant
        self.module = load_module(V012_PATH, f"v017_base_{variant}_{time.time_ns()}")
        self.controller = MarketRolloutController(
            variant=variant,
            runtime=self.module._V18_RUNTIME,
            selected_state=self.module._V18_SELECTED_MARKET,
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
        rollout = self.controller.apply(prepared, base)
        output = self.overlay.apply(prepared, rollout)
        if output.get("farmer") != base.get("farmer") or output.get("hands") != base.get("hands"):
            self.field_changed += 1
        if self._nonpremium(output.get("market")) != self._nonpremium(rollout.get("market")):
            self.nonpremium_changed += 1
        return output

    def diagnostics(self, player):
        result = self.controller.diagnostics(player)
        result.update({f"overlay_{key}": value for key, value in self.overlay.diagnostics(player).items()})
        result.update({"field_changed": self.field_changed, "nonpremium_changed": self.nonpremium_changed})
        return result


def build_candidate(variant):
    if variant == "control_v012":
        return load_module(V012_PATH, f"v017_control_v012_{time.time_ns()}").agent
    if variant == "control_v015a":
        return OverlayAgent("default")
    if variant in VARIANTS:
        return V017Agent(variant)
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
        if key in {"history", "items"}:
            row[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    return row


def summarize_rows(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda pair: tuple(str(value) for value in pair[0])):
        outcomes = Counter(row["result"] for row in group)
        item = dict(zip(fields, key))
        item.update({
            "games": len(group),
            "mean_money": statistics.mean(float(row["candidate_money"]) for row in group),
            "mean_margin": statistics.mean(float(row["margin"]) for row in group),
            "min_money": min(float(row["candidate_money"]) for row in group),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(group),
            "all_done": int(all(row.get("game_done") for row in group)),
            "agent_errors": sum(int(row.get("agent_errors", 0) or 0) for row in group),
            "invalid_action_shapes": sum(int(row.get("invalid_action_shapes", 0) or 0) for row in group),
            "p50_ms": percentile([float(row.get("runtime_p50_ms", 0) or 0) for row in group], 0.50),
            "p95_ms": percentile([float(row.get("runtime_p95_ms", 0) or 0) for row in group], 0.95),
            "p99_ms": max(float(row.get("runtime_p99_ms", 0) or 0) for row in group),
            "max_ms": max(float(row.get("runtime_max_ms", 0) or 0) for row in group),
        })
        for metric in (
            "adjusted_items", "adjusted_units", "changed_market_actions",
            "overlay_price_shocks", "overlay_delayed_units", "overlay_released_units",
            "overlay_terminal_flush_units", "field_changed", "nonpremium_changed",
        ):
            item[f"sum_{metric}"] = sum(float(row.get(metric, 0) or 0) for row in group)
        experts = Counter()
        for row in group:
            try:
                history = json.loads(row.get("history", "[]"))
                experts.update(str(entry.get("expert")) for entry in history if entry.get("expert"))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        item["expert_counts"] = json.dumps(dict(sorted(experts.items())), sort_keys=True)
        output.append(item)
    return output


def run_replay_counterfactual():
    rows = []
    for path in sorted(REPLAY_DIR.glob("*.json")):
        route = load_replay(path)
        opponent = ReplayAgent(route["actions"])
        for variant in ("control_v012", "control_v015a", *VARIANTS):
            candidate = build_candidate(variant)
            row = run_game(candidate, opponent, route["seed"], route["seat"])
            row.update({"replay": path.stem, "variant": variant, "source": str(path)})
            rows.append(row)
            print(f"[replay] {path.stem} {variant} margin={row['margin']:.1f}", flush=True)
    write_csv(OUT_DIR / "replay_counterfactual_raw.csv", rows)
    write_csv(OUT_DIR / "replay_counterfactual_summary.csv", summarize_rows(rows, ("variant",)))
    return rows


def run_matrix(opponents, seeds):
    rows = []
    candidates = ("control_v012", "control_v015a", *VARIANTS)
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
    write_csv(OUT_DIR / "matrix_summary.csv", summarize_rows(rows, ("variant", "opponent")))
    return rows


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.mean(values) if values else 0.0


def build_gate_report(matrix_rows, replay_rows):
    report = {"control": "control_v015a", "variants": {}}
    control = [row for row in matrix_rows if row.get("variant") == "control_v015a"]
    control_mean = _mean(control, "candidate_money")
    control_min = min((float(row["candidate_money"]) for row in control), default=0.0)
    control_wins = sum(row.get("result") == "win" for row in control)
    control_games = len(control)
    report.update({"control_mean_money": control_mean, "control_min_money": control_min})

    replay_control = {row["replay"]: float(row["margin"]) for row in replay_rows if row.get("variant") == "control_v015a"}
    for variant in VARIANTS:
        rows = [row for row in matrix_rows if row.get("variant") == variant]
        replay_variant = [row for row in replay_rows if row.get("variant") == variant]
        replay_deltas = [float(row["margin"]) - replay_control[row["replay"]] for row in replay_variant if row.get("replay") in replay_control]
        control_by_game = {
            (row.get("opponent"), row.get("seed"), row.get("seat")): row
            for row in control
        }
        no_new_errors = bool(rows)
        for row in rows:
            same = control_by_game.get((row.get("opponent"), row.get("seed"), row.get("seat")))
            if same is None or row.get("agent_errors", 0) > same.get("agent_errors", 0) or row.get("invalid_action_shapes", 0) > same.get("invalid_action_shapes", 0):
                no_new_errors = False
                break
        wins = sum(row.get("result") == "win" for row in rows)
        p99 = max((float(row.get("runtime_p99_ms", 0) or 0) for row in rows), default=0.0)
        hf = [row for row in rows if row.get("opponent") in {"hamburger", "frontier"}]
        control_hf = [row for row in control if row.get("opponent") in {"hamburger", "frontier"}]
        checks = {
            "all_done": bool(rows) and all(row.get("game_done") for row in rows),
            "no_new_errors_or_invalid": no_new_errors,
            "mean_cash_plus_0_5pct": _mean(rows, "candidate_money") >= control_mean * 1.005,
            "min_cash_at_least_95pct": min((float(row["candidate_money"]) for row in rows), default=0.0) >= control_min * 0.95,
            "win_rate_not_lower": bool(rows) and wins / len(rows) >= (control_wins / control_games if control_games else 0.0),
            "hamburger_frontier_at_least_95pct": _mean(hf, "candidate_money") >= _mean(control_hf, "candidate_money") * 0.95,
            "replay_mean_delta_nonnegative": bool(replay_deltas) and statistics.mean(replay_deltas) >= -1e-6,
            "replay_at_least_6_not_lower": sum(delta >= -1e-6 for delta in replay_deltas) >= 6,
            "p99_under_1000ms": p99 < 1000.0,
            "field_unchanged": all(float(row.get("field_changed", 0) or 0) == 0 for row in rows),
            "nonpremium_unchanged": all(float(row.get("nonpremium_changed", 0) or 0) == 0 for row in rows),
        }
        report["variants"][variant] = {
            "pass": all(checks.values()),
            "checks": checks,
            "mean_money": _mean(rows, "candidate_money"),
            "min_money": min((float(row["candidate_money"]) for row in rows), default=0.0),
            "wins": wins,
            "games": len(rows),
            "replay_mean_delta": statistics.mean(replay_deltas) if replay_deltas else None,
            "replay_not_lower": sum(delta >= -1e-6 for delta in replay_deltas),
            "replay_games": len(replay_deltas),
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
        for variant in VARIANTS:
            row = run_game(build_candidate(variant), "starter", 17, 0)
            print(json.dumps({key: row.get(key) for key in ("candidate_money", "game_done", "agent_errors", "invalid_action_shapes", "runtime_p99_ms", "adjusted_units")}, ensure_ascii=False))
        return
    replay_rows = run_replay_counterfactual() if args.stage in {"replay", "all"} else read_csv_rows(OUT_DIR / "replay_counterfactual_raw.csv")
    matrix_rows = run_matrix(args.opponents, args.seeds) if args.stage in {"matrix", "all"} else read_csv_rows(OUT_DIR / "matrix_raw.csv")
    report = build_gate_report(matrix_rows, replay_rows) if matrix_rows and replay_rows else {"variants": {}}
    (OUT_DIR / "gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

