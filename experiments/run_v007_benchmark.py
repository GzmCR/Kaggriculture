"""Run the V007 trajectory-preserving ablation matrix.

The control is the unchanged root agent.  The three V007 candidates are
loaded from their historical copies.  Hamburger is reconstructed only as an
opponent, exactly as in the V006 benchmark; its fixed trace is not part of a
candidate submission.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from run_v006_benchmark import (
    DEFAULT_OPPONENTS,
    DEFAULT_SEEDS,
    EPISODE_STEPS,
    load_hamburger_agent,
    load_module,
    run_game,
    summarize,
    write_csv,
)


class ActionTrackingModule:
    """Proxy a candidate module and retain item-specific field statistics."""

    def __init__(self, module):
        self.module = module
        self.field_item_counts = Counter()

    def agent(self, obs, config=None):
        action = self.module.agent(obs, config)
        if isinstance(action, dict):
            operations = [
                action.get("farmer", []),
                *(action.get("hands", []) or []),
            ]
            for operation in operations:
                if (
                    isinstance(operation, list)
                    and len(operation) >= 2
                    and operation
                ):
                    self.field_item_counts[
                        f"{operation[0]}:{operation[1]}"
                    ] += 1
        return action


def _counter_json(counter):
    return json.dumps(dict(sorted(counter.items())), sort_keys=True)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "baseline/artifacts/v007_trajectory_safe"

CANDIDATES = {
    "current": ROOT / "main.py",
    "v007a_terminal_safe": ROOT
    / "baseline/history/v007a_terminal_safe/main.py",
    "v007b_idle_fertilizer": ROOT
    / "baseline/history/v007b_idle_fertilizer/main.py",
    "v007c_combined": ROOT / "baseline/history/v007c_combined/main.py",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=DEFAULT_OPPONENTS,
        default=list(DEFAULT_OPPONENTS),
    )
    parser.add_argument("--candidates", nargs="+", default=None)
    args = parser.parse_args()

    selected = CANDIDATES
    if args.candidates:
        unknown = sorted(set(args.candidates) - set(CANDIDATES))
        if unknown:
            parser.error(f"unknown candidates: {', '.join(unknown)}")
        selected = {name: CANDIDATES[name] for name in args.candidates}

    modules = {
        name: load_module(path, "v007_" + name.replace("-", "_"))
        for name, path in selected.items()
    }
    hamburger = load_hamburger_agent(
        ROOT / "baseline/kaggriculture-hamburger.ipynb"
    )

    rows = []
    total = len(modules) * len(args.opponents) * len(args.seeds) * 2
    index = 0
    for candidate_name, module in modules.items():
        for opponent in args.opponents:
            for seed in args.seeds:
                for seat in (0, 1):
                    index += 1
                    print(
                        f"[{index}/{total}] {candidate_name} vs "
                        f"{opponent} seed={seed} seat={seat}",
                        flush=True,
                    )
                    tracked = ActionTrackingModule(module)
                    row = run_game(tracked, opponent, hamburger, seed, seat)
                    row.update({
                        "candidate": candidate_name,
                        "opponent": opponent,
                        "field_item_counts": _counter_json(
                            tracked.field_item_counts
                        ),
                    })
                    rows.append(row)

    row_fields = [
        "candidate", "opponent", "seed", "seat", "candidate_money",
        "opponent_money", "margin", "result", "candidate_status",
        "opponent_status", "game_done", "action_calls", "agent_errors",
        "invalid_action_shapes", "runtime_p50_ms", "runtime_p95_ms",
        "runtime_p99_ms", "runtime_max_ms", "wall_seconds", "field_counts",
        "field_item_counts", "market_counts", "market_quantities",
    ]
    summary = summarize(rows)
    summary_fields = list(summary[0]) if summary else []
    write_csv(args.out / "v007_results.csv", rows, row_fields)
    write_csv(args.out / "v007_summary.csv", summary, summary_fields)
    (args.out / "v007_results.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "v007_results.json").write_text(json.dumps(rows, indent=2))
    (args.out / "v007_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"WROTE {args.out / 'v007_results.csv'}")
    print(f"WROTE {args.out / 'v007_summary.csv'}")


if __name__ == "__main__":
    main()
