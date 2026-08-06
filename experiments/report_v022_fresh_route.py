"""Create a machine-readable V022 gate report from the completed matrix."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "baseline/artifacts/v022_fresh_route/full_matrix_v2"
SUMMARY = OUT / "matrix_summary.csv"
RAW = OUT / "matrix_raw.csv"


def _num(value):
    return float(value or 0)


def main():
    summary = list(csv.DictReader(SUMMARY.open(encoding="utf-8")))
    raw = list(csv.DictReader(RAW.open(encoding="utf-8")))
    grouped = {}
    for row in summary:
        grouped.setdefault(row["candidate"], []).append(row)
    control = grouped["control"]
    control_games = sum(int(row["games"]) for row in control)
    control_mean = sum(_num(row["mean_money"]) * int(row["games"]) for row in control) / control_games
    control_min = min(_num(row["min_money"]) for row in control)
    control_wins = sum(int(row["wins"]) for row in control)

    report = {
        "matrix_games": len(raw),
        "control": {
            "mean_money": control_mean,
            "min_money": control_min,
            "wins": control_wins,
            "games": control_games,
        },
        "variants": {},
    }
    for candidate, rows in sorted(grouped.items()):
        games = sum(int(row["games"]) for row in rows)
        wins = sum(int(row["wins"]) for row in rows)
        mean_money = sum(_num(row["mean_money"]) * int(row["games"]) for row in rows) / games
        min_money = min(_num(row["min_money"]) for row in rows)
        p99 = max(_num(row["p99_ms"]) for row in rows)
        checks = {
            "all_done": all(int(row["all_done"]) for row in rows),
            "no_errors": sum(int(row["errors"]) for row in rows) == 0,
            "no_invalid": sum(int(row["invalid"]) for row in rows) == 0,
            "mean_cash_not_lower": mean_money >= control_mean,
            "min_cash_not_below_97pct": min_money >= control_min * 0.97,
            "win_rate_not_lower": wins / games >= control_wins / control_games,
            "p99_under_1000ms": p99 < 1000.0,
        }
        report["variants"][candidate] = {
            "games": games,
            "mean_money": mean_money,
            "min_money": min_money,
            "wins": wins,
            "win_rate": wins / games,
            "p99_ms": p99,
            "checks": checks,
            "passed": all(checks.values()),
        }
    (OUT / "gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
