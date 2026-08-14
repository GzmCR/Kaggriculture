"""Summarize old V032 prediction error and fit a conservative R1 correction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _median(values):
    return float(statistics.median(values)) if values else 0.0


def _features(row):
    # Keep the compact model deliberately stable across route sources.  The
    # intercept is represented by the residual median and these five centered
    # features carry only broad price/quantity/horizon information.
    raw = float(row.get("raw_gain") or 0.0)
    return [
        float(row.get("transfer", 0) or 0) / 10.0,
        float(row.get("step", 0) or 0) / 720.0,
        float(row.get("due", 0) or 0) - float(row.get("step", 0) or 0),
        1.0 if str(row.get("mode")) == "advance" else -1.0,
        raw / 100.0,
    ]


def _ridge(x, y, alpha=10.0):
    try:
        import numpy as np
    except ImportError:
        return None
    if not x:
        return None
    matrix = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    matrix = np.column_stack([np.ones(len(matrix)), matrix])
    gram = matrix.T @ matrix
    gram += float(alpha) * np.eye(gram.shape[0])
    gram[0, 0] -= float(alpha)
    try:
        return np.linalg.solve(gram, matrix.T @ target).tolist()
    except np.linalg.LinAlgError:
        return None


def analyze(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (str(row.get("item")), str(row.get("mode")))
        groups[key].append(row)
    summary = []
    calibration = {"version": "v032-r1-residual-v1", "items": {}, "global": {}}
    residuals = []
    for key, group in sorted(groups.items()):
        residual = [float(row.get("actual_margin_delta", 0) or 0) - float(row.get("raw_gain", 0) or 0) for row in group]
        residuals.extend(residual)
        positives = sum(float(row.get("raw_gain", 0) or 0) > 0 and float(row.get("actual_margin_delta", 0) or 0) < 0 for row in group)
        summary.append({"item": key[0], "mode": key[1], "events": len(group),
                        "support_groups": len({(row.get("source_hash"), row.get("seed"), row.get("seat")) for row in group}),
                        "raw_mean": statistics.mean(float(row.get("raw_gain", 0) or 0) for row in group),
                        "actual_mean": statistics.mean(float(row.get("actual_margin_delta", 0) or 0) for row in group),
                        "residual_median": _median(residual),
                        "residual_mae": statistics.mean(abs(x) for x in residual),
                        "positive_predicted_negative_actual": positives})
        x = [_features(row) for row in group]
        coef = _ridge(x, residual)
        calibration["items"][f"{key[0]}:{key[1]}"] = {
            "support_groups": len({(row.get("source_hash"), row.get("seed"), row.get("seat")) for row in group}),
            "median_residual": _median(residual),
            "coefficients": coef[1:] if coef else [],
        }
    calibration["global"] = {
        "support_groups": len({(row.get("source_hash"), row.get("seed"), row.get("seat")) for row in rows}),
        "median_residual": _median(residuals), "coefficients": [],
    }
    actual = [float(row.get("actual_margin_delta", 0) or 0) for row in rows]
    raw = [float(row.get("raw_gain", 0) or 0) for row in rows]
    calibrated = []
    for row in rows:
        item = f"{row.get('item')}:{row.get('mode')}"
        model = calibration["items"].get(item, calibration["global"])
        prediction = float(row.get("raw_gain", 0) or 0) + float(model.get("median_residual", 0) or 0)
        calibrated.append(prediction)
    report = {
        "events": len(rows),
        "positive_predicted_negative_actual": sum(a > 0 and b < 0 for a, b in zip(raw, actual)),
        "raw_mae": statistics.mean(abs(a - b) for a, b in zip(raw, actual)) if rows else 0.0,
        "calibrated_mae": statistics.mean(abs(a - b) for a, b in zip(calibrated, actual)) if rows else 0.0,
        "raw_mean_bias": statistics.mean(a - b for a, b in zip(raw, actual)) if rows else 0.0,
        "calibrated_mean_bias": statistics.mean(a - b for a, b in zip(calibrated, actual)) if rows else 0.0,
        "groups": summary,
        "calibration": calibration,
    }
    # Source-isolated diagnostics: a residual learned from one source must
    # never be scored on that same source in the reported holdout metric.
    loo = []
    source_values = sorted({str(row.get("source_hash", "")) for row in rows})
    for held_out in source_values:
        train = [row for row in rows if str(row.get("source_hash", "")) != held_out]
        test = [row for row in rows if str(row.get("source_hash", "")) == held_out]
        train_residuals = [float(row.get("actual_margin_delta", 0) or 0) - float(row.get("raw_gain", 0) or 0) for row in train]
        correction = _median(train_residuals)
        loo_pred = [float(row.get("raw_gain", 0) or 0) + correction for row in test]
        loo_actual = [float(row.get("actual_margin_delta", 0) or 0) for row in test]
        loo.append({"held_out_source": held_out, "train_events": len(train), "test_events": len(test),
                    "raw_mae": statistics.mean(abs(float(row.get("raw_gain", 0) or 0) - actual) for row, actual in zip(test, loo_actual)) if test else 0.0,
                    "calibrated_mae": statistics.mean(abs(pred - actual) for pred, actual in zip(loo_pred, loo_actual)) if test else 0.0,
                    "train_residual_median": correction})
    report["leave_one_source_out"] = loo
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v032_route_conditioned_timing_r1/gain_diagnostics")
    args = parser.parse_args()
    rows = _rows(args.events)
    report = analyze(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "calibration.json").write_text(json.dumps(report["calibration"], indent=2) + "\n", encoding="utf-8")
    with (args.output / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row}) or ["item"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ("events", "raw_mae", "calibrated_mae", "raw_mean_bias", "calibrated_mean_bias", "positive_predicted_negative_actual")}, indent=2))


if __name__ == "__main__":
    main()
