"""Merge per-candidate V007 benchmark runs into public artifact files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from run_v006_benchmark import summarize, write_csv


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "baseline/artifacts/v007_trajectory_safe"
RAW = ARTIFACTS / "raw"
MAIN_CANDIDATES = {
    "current",
    "v007a_terminal_safe",
    "v007b_idle_fertilizer",
    "v007c_combined",
}


def _coerce(rows):
    numeric = {
        "seed", "seat", "candidate_money", "opponent_money", "margin",
        "game_done", "action_calls", "agent_errors",
        "invalid_action_shapes", "runtime_p50_ms", "runtime_p95_ms",
        "runtime_p99_ms", "runtime_max_ms", "wall_seconds",
    }
    for row in rows:
        for key in numeric:
            if key not in row:
                continue
            value = row[key]
            row[key] = float(value) if "." in value else int(value)
    return rows


def main():
    rows = []
    for result_path in sorted(RAW.glob("*/v007_results.csv")):
        with result_path.open(newline="") as handle:
            loaded = _coerce(list(csv.DictReader(handle)))
        if result_path.parent.name in MAIN_CANDIDATES:
            rows.extend(loaded)

    rows.sort(
        key=lambda row: (
            row["candidate"], row["opponent"],
            int(row["seed"]), int(row["seat"]),
        )
    )
    summary = summarize(rows)
    row_fields = [
        "candidate", "opponent", "seed", "seat", "candidate_money",
        "opponent_money", "margin", "result", "candidate_status",
        "opponent_status", "game_done", "action_calls", "agent_errors",
        "invalid_action_shapes", "runtime_p50_ms", "runtime_p95_ms",
        "runtime_p99_ms", "runtime_max_ms", "wall_seconds", "field_counts",
        "field_item_counts", "market_counts", "market_quantities",
    ]
    summary_fields = list(summary[0]) if summary else []
    write_csv(ARTIFACTS / "v007_results.csv", rows, row_fields)
    write_csv(ARTIFACTS / "v007_summary.csv", summary, summary_fields)
    (ARTIFACTS / "v007_results.json").write_text(json.dumps(rows, indent=2))
    (ARTIFACTS / "v007_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Merged {len(rows)} rows")


if __name__ == "__main__":
    main()
