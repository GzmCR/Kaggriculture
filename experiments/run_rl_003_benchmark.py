"""Benchmark RL-003 timing policy with the existing matrix harness."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_v028_benchmark as harness


harness.CANDIDATES = {
    "v22": None,
    "rl003": harness.ROOT / "baseline/artifacts/rl_003_trade_timing/main.py",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=["v22", "rl003"])
    parser.add_argument("--opponents", nargs="+", default=["v22"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026, 217, 317, 733])
    parser.add_argument(
        "--output",
        type=Path,
        default=harness.ROOT / "baseline/artifacts/rl_003_trade_timing/benchmark",
    )
    args = parser.parse_args()
    harness.run(tuple(args.candidates), tuple(args.opponents), tuple(args.seeds), args.output)
