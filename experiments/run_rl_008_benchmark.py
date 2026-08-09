"""Benchmark RL-008 variants against deduplicated current opponents."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_v028_benchmark as harness

from top10_opponents import TOP10_NOTEBOOKS, load_top10_agent, unique_top10_names


ARTIFACT = harness.ROOT / "baseline/artifacts/rl_008_small_shift_timing"
harness.CANDIDATES = {
    "v22": None,
    "gated_preempt": ARTIFACT / "gated_preempt/main.py",
    "ungated_preempt": ARTIFACT / "ungated_preempt/main.py",
    "gated_bidirectional": ARTIFACT / "gated_bidirectional/main.py",
    "ungated_bidirectional": ARTIFACT / "ungated_bidirectional/main.py",
}

_original_opponent = harness._opponent


def _opponent(name):
    if name in TOP10_NOTEBOOKS:
        return load_top10_agent(name)[0]
    return _original_opponent(name)


harness._opponent = _opponent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["v22", "gated_preempt", "ungated_preempt", "gated_bidirectional", "ungated_bidirectional"],
    )
    parser.add_argument("--opponents", nargs="+", default=unique_top10_names())
    parser.add_argument("--seeds", nargs="+", type=int, default=[811, 919, 1021])
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT / "benchmark_holdout",
    )
    args = parser.parse_args()
    harness.run(tuple(args.candidates), tuple(args.opponents), tuple(args.seeds), args.output)
