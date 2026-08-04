"""Benchmark the V010 crop-mix candidate with a direct baseline opponent."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from run_v008_benchmark import (
    DEFAULT_SEEDS,
    load_opponents,
    run_game,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPPONENTS = ("baseline", "starter", "random", "hamburger", "frontier")
DEFAULT_CANDIDATES = {
    "control": ROOT / "main.py",
    "v010a_crop_mix": ROOT / "baseline/history/v010a_crop_mix/main.py",
    "v010b_carrot_half": ROOT / "baseline/history/v010b_carrot_half/main.py",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} must define agent(obs, config=None)")
    return module


def load_v010_opponents(names):
    names = list(names)
    loaded = load_opponents([name for name in names if name != "baseline"])
    if "baseline" in names:
        baseline = load_module(ROOT / "main.py", "v010_baseline_opponent")
        loaded["baseline"] = baseline.agent
    return {name: loaded[name] for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "baseline/artifacts/v010_crop_mix",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    args = parser.parse_args()

    modules = {
        name: load_module(path, f"v010_{name}")
        for name, path in DEFAULT_CANDIDATES.items()
        if name in args.candidates
    }
    unknown = sorted(set(args.candidates) - set(modules))
    if unknown:
        raise ValueError(f"Unknown V010 candidates: {unknown}")
    opponents = load_v010_opponents(args.opponents)

    total = len(modules) * len(opponents) * len(args.seeds) * 2
    rows = []
    index = 0
    for candidate_name, module in modules.items():
        for opponent_name, opponent in opponents.items():
            for seed in args.seeds:
                for seat in (0, 1):
                    row = run_game(module, opponent, seed, seat)
                    row.update({
                        "candidate": candidate_name,
                        "opponent": opponent_name,
                    })
                    rows.append(row)
                    index += 1
                    print(
                        f"[{index}/{total}] {candidate_name} vs {opponent_name} "
                        f"seed={seed} seat={seat} "
                        f"money={row['candidate_money']:.0f} "
                        f"status={row['candidate_status']}",
                        flush=True,
                    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_csv(args.out / "v010_raw.csv", rows)
    write_csv(args.out / "v010_summary.csv", summary)
    (args.out / "v010_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
