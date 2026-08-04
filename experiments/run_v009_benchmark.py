"""Run the V009 market-memory and public-meta candidates."""

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
# The repository currently contains no builder notebook. Keep the available
# matrix runnable; pass --opponents ... builder after adding that asset.
DEFAULT_OPPONENTS = ("baseline", "starter", "random", "hamburger", "frontier")
DEFAULT_CANDIDATES = {
    "current": ROOT / "main.py",
    "v009a_market_memory": ROOT / "baseline/history/v009a_market_memory/main.py",
    "v009b_public_meta_counter": ROOT / "baseline/history/v009b_public_meta_counter/main.py",
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


def diagnostics(module):
    for name in ("V009A_STATE", "V009B_STATE"):
        state = getattr(module, name, None)
        if isinstance(state, dict):
            return json.dumps(state, sort_keys=True)
    return "{}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "baseline/artifacts/v009_market_counter")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    args = parser.parse_args()

    modules = {}
    for name in args.candidates:
        if name not in DEFAULT_CANDIDATES:
            raise ValueError(f"Unknown V009 candidate: {name}")
        modules[name] = load_module(DEFAULT_CANDIDATES[name], f"v009_{name}")
    opponents = load_opponents(args.opponents)
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
                        "v009_state": diagnostics(module),
                    })
                    rows.append(row)
                    index += 1
                    print(
                        f"[{index}/{total}] {candidate_name} vs {opponent_name} "
                        f"seed={seed} seat={seat} money={row['candidate_money']:.0f} "
                        f"status={row['candidate_status']}",
                        flush=True,
                    )

    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "v009_raw.csv", rows)
    write_csv(args.out / "v009_summary.csv", summary)
    (args.out / "v009_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
