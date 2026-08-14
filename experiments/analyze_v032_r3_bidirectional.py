"""Summarize V032-R3 JSONL output without rerunning games."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from run_v032_r3_bidirectional import summarize


def load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def detailed_report(rows):
    report = summarize(rows)
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "EVALUATED":
            continue
        groups[(row.get("item"), row.get("kind"), row.get("horizon"), row.get("transfer"))].append(row)
    detail = []
    for key, group in sorted(groups.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        safe = [row for row in group if row.get("safe")]
        detail.append({
            "item": key[0], "kind": key[1], "horizon": key[2], "transfer": key[3],
            "rows": len(group), "safe_rows": len(safe),
            "mean_predicted_local_margin": (
                sum(float(row["predicted_local_margin_delta"]) for row in safe) / len(safe)
                if safe else None
            ),
            "mean_actual_interval_margin": (
                sum(float(row["actual_interval_margin_delta"]) for row in safe) / len(safe)
                if safe else None
            ),
            "mean_actual_final_margin": (
                sum(float(row["actual_final_margin_delta"]) for row in safe) / len(safe)
                if safe else None
            ),
            "positive_prediction_negative_actual": sum(
                1 for row in safe
                if float(row["predicted_local_margin_delta"]) > 0
                and float(row["actual_interval_margin_delta"]) < 0
            ),
        })
    report["detail"] = detail
    report["status_counts"] = dict(Counter(row.get("status") for row in rows))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = detailed_report(load_rows(args.input))
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
