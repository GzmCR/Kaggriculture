"""Benchmark V27 control and the three RL-010 ablations.

The runner reuses the paired game machinery from the cross-graft evaluator,
but defaults to the explicit RL-010 validation seed split and opponent catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_cross_graft_validation import ROOT
from build_rl_010_bidirectional import VARIANTS
from build_rl_010_data import TRAIN_SEEDS, VALIDATION_SEEDS
from run_cross_graft_validation import run


DEFAULT_CANDIDATES = (
    "v27_control",
    "rl010a_delay_only",
    "rl010b_bidirectional_no_opp",
    "rl010c_bidirectional_opp",
)
DEFAULT_OPPONENTS = (
    "v27_current",
    "v14_public",
    "adaptive_replay",
    "v13_r3",
    "v21_1",
    "strong_barnyard",
    "hamburger",
    "frontier_current",
    "starter",
    "random",
)
DEFAULT_OUTPUT = ROOT / "baseline/artifacts/rl_010_milk_bidirectional/benchmark_validation"


def benchmark(candidates, opponents, seeds, output):
    output = Path(output).resolve()
    summary = run(tuple(candidates), tuple(opponents), tuple(seeds), output)
    manifest = {
        "candidates": list(candidates),
        "opponents": list(opponents),
        "seeds": list(seeds),
        "episode_steps": 720,
        "variants": sorted(VARIANTS),
        "control": "v27_control",
        "root_main_modified": False,
    }
    (output / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--split", choices=("train", "validation", "val"), default="validation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.seed:
        seeds = tuple(args.seed)
    else:
        seeds = TRAIN_SEEDS if args.split == "train" else VALIDATION_SEEDS
    summary = benchmark(
        args.candidate or DEFAULT_CANDIDATES,
        args.opponent or DEFAULT_OPPONENTS,
        seeds,
        args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
