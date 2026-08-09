"""Benchmark RL-005 against the local 2026-08-09 opponent pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_v028_benchmark as harness

from top10_opponents import TOP10_NOTEBOOKS, load_top10_agent


ORIGINAL_OPPONENT = harness._opponent
harness.CANDIDATES = {
    "v22": None,
    "rl005": harness.ROOT / "baseline/artifacts/rl_005_multi_opponent/main.py",
}


def _opponent(name):
    if name in TOP10_NOTEBOOKS:
        return load_top10_agent(name)[0]
    return ORIGINAL_OPPONENT(name)


harness._opponent = _opponent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=["v22", "rl005"])
    parser.add_argument("--opponents", nargs="+", default=list(TOP10_NOTEBOOKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[811, 919, 1021])
    parser.add_argument(
        "--output",
        type=Path,
        default=harness.ROOT / "baseline/artifacts/rl_005_multi_opponent/benchmark_pool",
    )
    args = parser.parse_args()
    harness.run(tuple(args.candidates), tuple(args.opponents), tuple(args.seeds), args.output)
