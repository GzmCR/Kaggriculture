"""Summarize V032-R2 interval diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _corr(left, right):
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def summarize(rows):
    safe = [row for row in rows if row.get("safe")]
    def one(group):
        errors = [float(row["predicted_local_margin_delta"]) - float(row["actual_interval_margin_delta"]) for row in group]
        pred = [float(row["predicted_local_margin_delta"]) for row in group]
        actual = [float(row["actual_interval_margin_delta"]) for row in group]
        sign_rows = [row for row in group if float(row["predicted_local_margin_delta"]) != 0 and float(row["actual_interval_margin_delta"]) != 0]
        return {
            "rows": len(group),
            "mae": statistics.mean(abs(value) for value in errors) if errors else None,
            "bias": statistics.mean(errors) if errors else None,
            "correlation": _corr(pred, actual),
            "sign_accuracy": (
                sum(int((float(row["predicted_local_margin_delta"]) > 0) == (float(row["actual_interval_margin_delta"]) > 0)) for row in sign_rows)
                / len(sign_rows)
                if sign_rows else None
            ),
            "positive_pred_negative_actual": sum(1 for row in group if float(row["predicted_local_margin_delta"]) > 0 and float(row["actual_interval_margin_delta"]) < 0),
            "mean_final_margin_delta": statistics.mean(float(row["actual_final_margin_delta"]) for row in group) if group else None,
        }
    reasons = {}
    for row in rows:
        for reason in row.get("safety_reasons", []) or []:
            reasons[reason] = reasons.get(reason, 0) + 1
    result = {
        "rows": len(rows),
        "safe_rows": len(safe),
        "safe_rate": len(safe) / len(rows) if rows else 0.0,
        "safety_reasons": reasons,
        "all_rows": one(rows),
        "safe_rows_metrics": one(safe),
        "by_item": {item: one([row for row in safe if row.get("item") == item])
                     for item in sorted({str(row.get("item")) for row in safe})},
        "by_ratio": {str(ratio): one([row for row in safe if float(row.get("ratio", -1)) == float(ratio)])
                     for ratio in sorted({float(row.get("ratio", -1)) for row in safe})},
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(_rows(args.input))
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
