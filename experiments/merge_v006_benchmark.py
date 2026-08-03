"""Merge per-candidate V006 benchmark runs into the public artifact files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from run_v006_benchmark import summarize, write_csv


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "baseline/artifacts/v006_hamburger_transplant"
RAW = ARTIFACTS / "raw"
MAIN_CANDIDATES = {
    "current",
    "v006a_fertilizer_terminal",
    "v006b_livestock_wheat",
    "v006c_combined",
}


def main():
    rows = []
    diagnostic_rows = []
    for result_path in sorted(RAW.glob("*/v006_results.csv")):
        with result_path.open(newline="") as handle:
            loaded = list(csv.DictReader(handle))
        if result_path.parent.name in MAIN_CANDIDATES:
            rows.extend(loaded)
        else:
            diagnostic_rows.extend(loaded)
    numeric = {
        "seed", "seat", "candidate_money", "opponent_money", "margin",
        "game_done", "action_calls", "agent_errors",
        "invalid_action_shapes", "runtime_p50_ms", "runtime_p95_ms",
        "runtime_p99_ms", "runtime_max_ms", "wall_seconds",
    }
    for row in [*rows, *diagnostic_rows]:
        for key in numeric:
            if key in row:
                row[key] = float(row[key]) if "." in row[key] else int(row[key])
    rows.sort(
        key=lambda row: (
            row["candidate"], row["opponent"], int(row["seed"]), int(row["seat"])
        )
    )
    summary = summarize(rows)
    row_fields = [
        "candidate", "opponent", "seed", "seat", "candidate_money",
        "opponent_money", "margin", "result", "candidate_status",
        "opponent_status", "game_done", "action_calls", "agent_errors",
        "invalid_action_shapes", "runtime_p50_ms", "runtime_p95_ms",
        "runtime_p99_ms", "runtime_max_ms", "wall_seconds", "field_counts",
        "market_counts", "market_quantities",
    ]
    summary_fields = list(summary[0]) if summary else []
    write_csv(ARTIFACTS / "v006_results.csv", rows, row_fields)
    write_csv(ARTIFACTS / "v006_summary.csv", summary, summary_fields)
    (ARTIFACTS / "v006_results.json").write_text(json.dumps(rows, indent=2))
    (ARTIFACTS / "v006_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    diagnostic_summary = summarize(diagnostic_rows)
    if diagnostic_summary:
        write_csv(
            ARTIFACTS / "v006_ablation_summary.csv",
            diagnostic_summary,
            list(diagnostic_summary[0]),
        )
        (ARTIFACTS / "v006_ablation_summary.json").write_text(
            json.dumps(diagnostic_summary, indent=2)
        )
    print(f"Merged {len(rows)} rows")


if __name__ == "__main__":
    main()
