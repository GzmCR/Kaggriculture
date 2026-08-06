"""Run the V022c control versus V022e adaptive recovery experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_v022_fresh_route import (
    ROOT,
    SEEDS,
    load_module,
    run_game,
    summarize,
    write_csv,
)


V022C = ROOT / "baseline/artifacts/v022c_medoid_recovery/main.py"
V022E = ROOT / "baseline/artifacts/v022e_adaptive_recovery/main.py"
ROOT_BASELINE = ROOT / "main.py"
OPPONENTS = ("v012", "v022d", "v18", "hamburger", "frontier", "baseline", "starter", "random")
CANDIDATES = {
    "v022c_control": V022C,
    "v022e_adaptive_recovery": V022E,
}


def _opponent(name):
    from run_v022_fresh_route import _opponent as old_opponent

    if name == "v012":
        return old_opponent("control")
    if name == "v022d":
        return load_module(
            ROOT / "baseline/artifacts/v022d_medoid_recovery_tactical/main.py",
            f"v022e_v022d_{time.time_ns()}",
        ).agent
    return old_opponent(name)


def _flat_actions(row):
    actions = row.get("actions", {}) or {}
    return {f"action_{key}": value for key, value in actions.items()}


def _gate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    control_rows = grouped["v022c_control"]
    control_mean = statistics.mean(row["candidate_money"] for row in control_rows)
    control_min = min(row["candidate_money"] for row in control_rows)
    control_wins = sum(row["result"] == "win" for row in control_rows)
    control_rate = control_wins / len(control_rows)
    report = {
        "control": {
            "games": len(control_rows),
            "mean_money": control_mean,
            "min_money": control_min,
            "wins": control_wins,
            "win_rate": control_rate,
        },
        "candidates": {},
    }
    for name, candidate_rows in sorted(grouped.items()):
        wins = sum(row["result"] == "win" for row in candidate_rows)
        mean_money = statistics.mean(row["candidate_money"] for row in candidate_rows)
        min_money = min(row["candidate_money"] for row in candidate_rows)
        checks = {
            "all_done": all(row["done"] for row in candidate_rows),
            "no_errors": sum(row["errors"] for row in candidate_rows) == 0,
            "no_invalid": sum(row["invalid"] for row in candidate_rows) == 0,
            "mean_cash_at_least_99_5pct_control": mean_money >= control_mean * 0.995,
            "min_cash_at_least_97pct_control": min_money >= control_min * 0.97,
            "win_rate_not_lower": wins / len(candidate_rows) >= control_rate,
            "p99_under_1000ms": max(row["p99_ms"] for row in candidate_rows) < 1000.0,
        }
        report["candidates"][name] = {
            "games": len(candidate_rows),
            "mean_money": mean_money,
            "min_money": min_money,
            "wins": wins,
            "ties": sum(row["result"] == "tie" for row in candidate_rows),
            "losses": sum(row["result"] == "loss" for row in candidate_rows),
            "win_rate": wins / len(candidate_rows),
            "p99_ms": max(row["p99_ms"] for row in candidate_rows),
            "checks": checks,
            "passed": all(checks.values()),
        }
    return report


def run_matrix(candidate_names, opponent_names, seeds, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(candidate_names) * len(opponent_names) * len(seeds) * 2
    index = 0
    for candidate_name in candidate_names:
        for opponent_name in opponent_names:
            for seed in seeds:
                for seat in (0, 1):
                    index += 1
                    module = load_module(
                        CANDIDATES[candidate_name],
                        f"v022e_{candidate_name}_{index}_{time.time_ns()}",
                    )
                    opponent = _opponent(opponent_name)
                    print(f"[{index}/{total}] {candidate_name} vs {opponent_name} seed={seed} seat={seat}", flush=True)
                    row = run_game(module.agent, opponent, seed, seat)
                    row.update(_flat_actions(row))
                    diagnostics = getattr(module, "_V022E_STATS", {})
                    for key, value in diagnostics.items():
                        row[f"v022e_{key}"] = value
                    row.update({"candidate": candidate_name, "opponent": opponent_name})
                    rows.append(row)
                    if index % 10 == 0:
                        write_csv(output / "matrix_raw.csv", rows)
    write_csv(output / "matrix_raw.csv", rows)
    write_csv(output / "matrix_summary.csv", summarize(rows))
    (output / "gate_report.json").write_text(
        json.dumps(_gate(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", choices=tuple(CANDIDATES))
    parser.add_argument("--opponent", action="append", choices=OPPONENTS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baseline/artifacts/v022e_adaptive_recovery/full_matrix",
    )
    args = parser.parse_args()
    candidates = tuple(args.candidate or CANDIDATES)
    opponents = tuple(args.opponent or OPPONENTS)
    seeds = tuple(args.seed or SEEDS)
    rows = run_matrix(candidates, opponents, seeds, args.output)
    print(f"V022e benchmark complete: {len(rows)} games")


if __name__ == "__main__":
    main()
