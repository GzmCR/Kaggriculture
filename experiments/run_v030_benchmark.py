"""Benchmark V030 adaptive market variants against local opponents."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_v028_benchmark as harness


harness.CANDIDATES = {
    "v22": None,
    "v029a": harness.ROOT / "baseline/artifacts/v029_milk_schedule/v029a_milk_safe_schedule/main.py",
    "v030a": harness.ROOT / "baseline/artifacts/v030_adaptive_market/v030a_milk_momentum_gate/main.py",
    "v030b": harness.ROOT / "baseline/artifacts/v030_adaptive_market/v030b_cross_product_guard/main.py",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=["v22", "v029a", "v030a", "v030b"])
    parser.add_argument("--opponents", nargs="+", default=["v22"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026, 217, 317, 733])
    parser.add_argument(
        "--output",
        type=Path,
        default=harness.ROOT / "baseline/artifacts/v030_adaptive_market/benchmark",
    )
    args = parser.parse_args()
    harness.run(tuple(args.candidates), tuple(args.opponents), tuple(args.seeds), args.output)
