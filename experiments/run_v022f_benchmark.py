"""Run the V022c control versus the V022f single-retry ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_v022e_benchmark import CANDIDATES, OPPONENTS, ROOT, SEEDS, run_matrix


CANDIDATES["v022f_single_retry"] = ROOT / "baseline/artifacts/v022f_single_retry/main.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", choices=("v022c_control", "v022f_single_retry"))
    parser.add_argument("--opponent", action="append", choices=OPPONENTS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baseline/artifacts/v022f_single_retry/full_matrix",
    )
    args = parser.parse_args()
    candidates = tuple(args.candidate or ("v022c_control", "v022f_single_retry"))
    opponents = tuple(args.opponent or OPPONENTS)
    seeds = tuple(args.seed or SEEDS)
    rows = run_matrix(candidates, opponents, seeds, args.output)
    print(f"V022f benchmark complete: {len(rows)} games")


if __name__ == "__main__":
    main()
