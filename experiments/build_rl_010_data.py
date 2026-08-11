"""Build the RL-010 training or validation data split.

The collector itself is shared with the earlier smoke entry point.  This
wrapper makes the seed split explicit so a validation run cannot accidentally
reuse the training seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_rl_010_data import (
    DEFAULT_EVENTS,
    DEFAULT_OUTPUT,
    collect,
)


TRAIN_SEEDS = (17, 42, 2026, 217, 317, 733)
VALIDATION_SEEDS = (811, 919, 1021, 1123, 1229, 1337)


def build_split(split, output=None, seeds=None, events=None, opponents=None):
    split = str(split).lower()
    if split not in {"train", "validation", "val"}:
        raise ValueError("split must be train or validation")
    if seeds is None:
        seeds = TRAIN_SEEDS if split == "train" else VALIDATION_SEEDS
    if output is None:
        output = DEFAULT_OUTPUT.parent / f"data_{'validation' if split != 'train' else 'train'}"
    report = collect(
        Path(output),
        tuple(int(seed) for seed in seeds),
        tuple(int(event) for event in (events or DEFAULT_EVENTS)),
        tuple(opponents or []),
    )
    report["split"] = "validation" if split != "train" else "train"
    report["seed_policy"] = {
        "train": list(TRAIN_SEEDS),
        "validation": list(VALIDATION_SEEDS),
    }
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "split_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation", "val"), default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--event", action="append", type=int, default=None)
    parser.add_argument("--opponent", action="append", default=None)
    args = parser.parse_args()
    report = build_split(
        args.split,
        args.output,
        args.seed,
        args.event,
        args.opponent,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
