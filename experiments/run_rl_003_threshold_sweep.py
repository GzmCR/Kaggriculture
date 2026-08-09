"""Sweep the conservative threshold of RL-003 on the fixed v22 route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_rl_003_trade_timing as builder
import run_v028_benchmark as harness


harness.CANDIDATES = {
    "v22": None,
    "rl003": harness.ROOT / "baseline/artifacts/rl_003_trade_timing/main.py",
}


def run(thresholds, seeds, output):
    output.mkdir(parents=True, exist_ok=True)
    base_path = builder.OUT_ARTIFACT / "fit_v2" / "weights.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    reports = []
    for threshold in thresholds:
        weights = dict(base)
        weights["threshold"] = float(threshold)
        weight_path = output / f"weights_{threshold:g}.json"
        weight_path.write_text(json.dumps(weights, ensure_ascii=True), encoding="utf-8")
        builder.build(weight_path)
        run_dir = output / f"benchmark_{threshold:g}"
        harness.run(("rl003",), ("v22",), tuple(seeds), run_dir)
        summary = json.loads((run_dir / "matrix_summary.json").read_text(encoding="utf-8"))
        reports.append({"threshold": float(threshold), "summary": summary})
    (output / "summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0, 5, 7.5, 10, 12])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026, 217, 317, 733])
    parser.add_argument(
        "--output",
        type=Path,
        default=harness.ROOT / "baseline/artifacts/rl_003_trade_timing/threshold_sweep",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.thresholds, args.seeds, args.output), indent=2))
