"""Evaluate the V006 promotion gate from the merged per-game CSV."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "baseline/artifacts/v006_hamburger_transplant"
MAIN = ("current", "v006a_fertilizer_terminal", "v006b_livestock_wheat", "v006c_combined")


def mean_money(rows):
    return statistics.mean(float(row["candidate_money"]) for row in rows)


def main():
    with (ARTIFACTS / "v006_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    current = [row for row in rows if row["candidate"] == "current"]
    current_mean = mean_money(current)
    current_min = min(float(row["candidate_money"]) for row in current)
    current_wins = sum(row["result"] == "win" for row in current)
    current_win_rate = current_wins / len(current)
    report = {
        "matrix_games": len(rows),
        "control_mean_money": current_mean,
        "control_min_money": current_min,
        "control_win_rate": current_win_rate,
        "thresholds": {
            "mean_gain_pct": 0.005,
            "min_cash_ratio": 0.97,
            "p99_ms": 1000.0,
            "hamburger_regression_ratio": 0.95,
        },
        "candidates": {},
    }
    for candidate in MAIN:
        group = [row for row in rows if row["candidate"] == candidate]
        candidate_mean = mean_money(group)
        candidate_min = min(float(row["candidate_money"]) for row in group)
        candidate_wins = sum(row["result"] == "win" for row in group)
        hamburger = [row for row in group if row["opponent"] == "hamburger"]
        hamburger_control = [
            row for row in current if row["opponent"] == "hamburger"
        ]
        metrics = {
            "mean_money": candidate_mean,
            "mean_gain_pct": candidate_mean / current_mean - 1.0,
            "min_money": candidate_min,
            "min_cash_ratio": candidate_min / current_min,
            "win_rate": candidate_wins / len(group),
            "all_done": all(row["game_done"] == "1" for row in group),
            "agent_errors": sum(int(row["agent_errors"]) for row in group),
            "invalid_action_shapes": sum(
                int(row["invalid_action_shapes"]) for row in group
            ),
            "p99_ms_max": max(float(row["runtime_p99_ms"]) for row in group),
            "hamburger_mean_money": mean_money(hamburger),
            "hamburger_ratio": mean_money(hamburger)
            / mean_money(hamburger_control),
        }
        metrics["pass"] = all(
            (
                metrics["all_done"],
                metrics["agent_errors"] == 0,
                metrics["invalid_action_shapes"] == 0,
                metrics["mean_gain_pct"] >= 0.005,
                metrics["min_cash_ratio"] >= 0.97,
                metrics["win_rate"] >= current_win_rate,
                metrics["p99_ms_max"] < 1000.0,
                metrics["hamburger_ratio"] >= 0.95,
            )
        )
        report["candidates"][candidate] = metrics
    (ARTIFACTS / "v006_gate.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
