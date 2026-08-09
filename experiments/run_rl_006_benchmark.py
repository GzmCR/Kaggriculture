"""Benchmark RL-006 ablations against deduplicated 2026-08-09 opponents."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_v028_benchmark as harness

from top10_opponents import TOP10_NOTEBOOKS, load_top10_agent, unique_top10_names


harness.CANDIDATES = {
    "v22": None,
    "rl005": harness.ROOT / "baseline/artifacts/rl_005_multi_opponent/main.py",
    "rl006_preempt": harness.ROOT / "baseline/artifacts/rl_006_bidirectional_timing/preempt_only/main.py",
    "rl006_delay": harness.ROOT / "baseline/artifacts/rl_006_bidirectional_timing/delay_only/main.py",
    "rl006_bidirectional": harness.ROOT / "baseline/artifacts/rl_006_bidirectional_timing/bidirectional/main.py",
}

_original_opponent = harness._opponent


def _opponent(name):
    if name in TOP10_NOTEBOOKS:
        return load_top10_agent(name)[0]
    return _original_opponent(name)


harness._opponent = _opponent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=["v22", "rl005", "rl006_preempt", "rl006_delay", "rl006_bidirectional"])
    parser.add_argument("--opponents", nargs="+", default=unique_top10_names())
    parser.add_argument("--seeds", nargs="+", type=int, default=[811, 919, 1021])
    parser.add_argument(
        "--output",
        type=Path,
        default=harness.ROOT / "baseline/artifacts/rl_006_bidirectional_timing/benchmark_holdout",
    )
    args = parser.parse_args()
    harness.run(tuple(args.candidates), tuple(args.opponents), tuple(args.seeds), args.output)
