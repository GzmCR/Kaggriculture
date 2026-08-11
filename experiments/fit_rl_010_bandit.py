"""Fit and package all RL-010 ablations from a fixed training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_rl_010_bidirectional import (
    DEFAULT_SAMPLES,
    OUT_ARTIFACT,
    VARIANTS,
    build,
    build_all,
)


def fit(samples, output, variant=None):
    if variant is None:
        return build_all(samples, output)
    return {variant: build(samples, output / variant, variant)[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=OUT_ARTIFACT)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    args = parser.parse_args()
    result = fit(args.samples, args.output, args.variant)
    print(json.dumps(result, indent=2, ensure_ascii=True))
