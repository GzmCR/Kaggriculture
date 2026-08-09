"""Run RL-003 timing against the stronger local opponent pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_v026_v22_v022c_recovery as harness


harness.CANDIDATES["rl003"] = harness.ROOT / "baseline/artifacts/rl_003_trade_timing/main.py"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opponent",
        nargs="+",
        default=["v22", "v022c", "v13_r3", "v21_1", "hamburger", "frontier"],
    )
    parser.add_argument("--seed", nargs="+", type=int, default=[17, 217, 733])
    parser.add_argument(
        "--output",
        type=Path,
        default=harness.ROOT / "baseline/artifacts/rl_003_trade_timing/pool_3seed",
    )
    args = parser.parse_args()
    harness.run(("rl003",), tuple(args.opponent), tuple(args.seed), args.output)
