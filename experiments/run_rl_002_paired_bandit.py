"""Paired causal intervention study for RL-001 market overlays.

For each seed/seat/opponent, run V022c control first. Then rerun the same
episode with exactly one 48-turn window forced to mode 1, 2, or 3. The field
route and every other market window remain the control route. The resulting
cash difference is a much cleaner training target than replay cash alone.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_001_selector import ACTION_COUNT, FEATURE_DIM, SelectorRuntime
from run_rl_001 import (
    DEFAULT_OPPONENTS,
    EPISODE_STEPS,
    HOLDOUT_START,
    V022C_PATH,
    load_module,
    opponent_factory,
    run_episode,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOCKS = (288, 432, 576)
DEFAULT_SEEDS = (17, 42)
DEFAULT_OPPONENTS_PILOT = ("starter", "random", "v022c", "v18")


def run_matrix(seeds, opponents, blocks, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(seeds) * len(opponents) * 2 * (1 + len(blocks) * 3)
    completed = 0
    for seed in seeds:
        for opponent_name in opponents:
            for seat in (0, 1):
                base_module = load_module(V022C_PATH, f"rl002_base_{time.time_ns()}")
                control_runtime = SelectorRuntime(training=False, seed=seed + seat)
                control = run_episode(
                    base_module.agent,
                    getattr(base_module, "_ACTIONS", None),
                    opponent_factory(opponent_name)(),
                    seed,
                    seat,
                    control_runtime,
                    opponent_name,
                    "control",
                    len(rows),
                )
                control["intervention_block"] = None
                control["intervention_mode"] = 0
                control["control_cash"] = control["candidate_money"]
                control["cash_delta"] = 0.0
                rows.append(control)
                completed += 1
                print(f"{completed}/{total} control seed={seed} opponent={opponent_name} seat={seat} cash={control['candidate_money']:.0f}", flush=True)

                for block in blocks:
                    for mode in range(1, ACTION_COUNT):
                        base_module = load_module(V022C_PATH, f"rl002_intervention_{time.time_ns()}")
                        runtime = SelectorRuntime(training=False, seed=seed + seat + block + mode)
                        runtime.forced_modes = {int(block): int(mode)}
                        result = run_episode(
                            base_module.agent,
                            getattr(base_module, "_ACTIONS", None),
                            opponent_factory(opponent_name)(),
                            seed,
                            seat,
                            runtime,
                            opponent_name,
                            "intervention",
                            len(rows),
                        )
                        result["intervention_block"] = int(block)
                        result["intervention_mode"] = int(mode)
                        result["control_cash"] = control["candidate_money"]
                        result["cash_delta"] = result["candidate_money"] - control["candidate_money"]
                        result["context_features"] = runtime.boundary_features.get(int(block), np.zeros(FEATURE_DIM)).tolist()
                        rows.append(result)
                        completed += 1
                        print(f"{completed}/{total} mode={mode} block={block} seed={seed} opponent={opponent_name} seat={seat} delta={result['cash_delta']:.0f}", flush=True)
    with (output / "intervention_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = summarize(rows)
    (output / "intervention_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("intervention_mode", 0):
            grouped[(row["intervention_mode"], row["intervention_block"])].append(row["cash_delta"])
    by_cell = {
        f"mode_{mode}_block_{block}": {
            "games": len(values),
            "mean_delta": statistics.mean(values),
            "median_delta": statistics.median(values),
            "positive_rate": sum(value > 0 for value in values) / len(values),
            "min_delta": min(values),
            "max_delta": max(values),
        }
        for (mode, block), values in sorted(grouped.items())
    }
    by_mode = {}
    for mode in range(1, ACTION_COUNT):
        values = [row["cash_delta"] for row in rows if row.get("intervention_mode") == mode]
        if values:
            by_mode[str(mode)] = {
                "games": len(values),
                "mean_delta": statistics.mean(values),
                "median_delta": statistics.median(values),
                "positive_rate": sum(value > 0 for value in values) / len(values),
                "min_delta": min(values),
                "max_delta": max(values),
            }
    return {"games": len(rows), "intervention_cells": by_cell, "by_mode": by_mode}


def fit_bandit(rows, output):
    samples = [row for row in rows if row.get("intervention_mode", 0) and row.get("context_features")]
    weights = {"q_a": [[0.0] * FEATURE_DIM for _ in range(ACTION_COUNT)],
               "q_b": [[0.0] * FEATURE_DIM for _ in range(ACTION_COUNT)]}
    reports = {}
    for mode in range(1, ACTION_COUNT):
        selected = [row for row in samples if int(row["intervention_mode"]) == mode]
        if not selected:
            continue
        matrix = np.asarray([row["context_features"] for row in selected], dtype=np.float64)
        target = np.asarray([row["cash_delta"] for row in selected], dtype=np.float64)
        ridge = matrix.T @ matrix + np.eye(FEATURE_DIM) * 100.0
        beta = np.linalg.solve(ridge, matrix.T @ target)
        weights["q_a"][mode] = beta.tolist()
        weights["q_b"][mode] = beta.tolist()
        reports[str(mode)] = {
            "samples": len(selected),
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "beta_norm": float(np.linalg.norm(beta)),
        }
    (output / "bandit_weights.json").write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    (output / "bandit_fit.json").write_text(json.dumps({"models": reports, "action_0_is_control": True}, indent=2) + "\n", encoding="utf-8")
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline" / "artifacts" / "rl_002_paired_bandit")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS_PILOT))
    parser.add_argument("--blocks", nargs="+", type=int, default=list(DEFAULT_BLOCKS))
    parser.add_argument("--skip-fit", action="store_true")
    args = parser.parse_args()
    rows = run_matrix(tuple(args.seeds), tuple(args.opponents), tuple(args.blocks), args.output)
    if not args.skip_fit:
        fit_bandit(rows, args.output)


if __name__ == "__main__":
    main()
