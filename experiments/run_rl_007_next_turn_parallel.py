"""Parallel paired counterfactual collector for RL-007."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

from rl_007_next_turn_preemption import RL007_ACTION_NAMES, RL007_FEATURE_DIM, rl007_shift_quantity
from run_rl_007_next_turn_data import (
    DEFAULT_SEEDS,
    DEFAULT_TARGETS,
    NextTurnPreemptAgent,
    _available_opponents,
    _fresh_opponent,
    _run,
    _select_opportunities,
)
from run_v026_v22_v022c_recovery import ROOT, _v22_fresh


def _control_task(task):
    opponent, seed, seat = task
    return opponent, int(seed), int(seat), _run(_v22_fresh("v22"), _fresh_opponent(opponent), seed, seat)


def _candidate_task(task):
    opponent, opportunity, action_id, seed, seat, source_sha = task
    candidate = NextTurnPreemptAgent(opportunity, action_id)
    result = _run(candidate, _fresh_opponent(opponent), seed, seat)
    return {
        **opportunity,
        "action_id": int(action_id),
        "action": RL007_ACTION_NAMES[action_id],
        "moved_quantity": rl007_shift_quantity(opportunity),
        "seed": int(seed),
        "seat": int(seat),
        "opponent": opponent,
        "opponent_source_sha256": source_sha,
        "features": candidate.features.tolist() if candidate.features is not None else [],
        "snapshot": candidate.snapshot,
        "candidate_money": result["candidate_money"],
        "candidate_margin": result["margin"],
        "candidate_done": result["done"],
        "changed_calls": len(candidate.changed),
        "shift_applied": int(bool(candidate.changed) and candidate.future_repaid),
        "future_repaid": int(candidate.future_repaid),
    }


def _write(output, rows, controls, targets, opponents, skipped, inspection, seeds):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    control_rows = [{"opponent": opponent, "seed": seed, "seat": seat, **row}
                    for (opponent, seed, seat), row in sorted(controls.items())]
    fields = sorted({key for row in control_rows for key in row})
    with (output / "control.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(control_rows)
    report = {
        "targets": list(targets),
        "seeds": list(seeds),
        "opponents": opponents,
        "skipped_opponents": skipped,
        "samples": len(rows),
        "applied_samples": int(sum(row["shift_applied"] for row in rows)),
        "controls": len(control_rows),
        "feature_dim": RL007_FEATURE_DIM,
        "actions": RL007_ACTION_NAMES,
        "all_done": int(all(row["candidate_done"] and row["control_done"] for row in rows)),
        "all_repaid": int(all((not row["shift_applied"]) or row["future_repaid"] for row in rows)),
        "inspection": list(inspection.values()),
    }
    (output / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def collect_parallel(output, seeds, targets, requested_opponents, workers=4):
    output = Path(output).resolve()
    opportunities = _select_opportunities(targets)
    opponents, skipped, inspection = _available_opponents(requested_opponents)
    control_tasks = [(name, int(seed), seat) for name in opponents for seed in seeds for seat in (0, 1)]
    candidate_tasks = [
        (name, opportunity, action_id, int(seed), seat, inspection[name]["source_sha256"])
        for name in opponents
        for opportunity in opportunities
        for action_id in (1, 2, 3)
        for seed in seeds
        for seat in (0, 1)
    ]
    controls = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
        for opponent, seed, seat, result in executor.map(_control_task, control_tasks):
            controls[(opponent, seed, seat)] = result
        rows = []
        for row in executor.map(_candidate_task, candidate_tasks):
            control = controls[(row["opponent"], row["seed"], row["seat"])]
            if not row["features"] or len(row["features"]) != RL007_FEATURE_DIM:
                raise RuntimeError(f"invalid RL-007 features: {row}")
            row["control_money"] = control["candidate_money"]
            row["cash_delta"] = row["candidate_money"] - control["candidate_money"]
            row["control_margin"] = control["margin"]
            row["margin_delta"] = row["candidate_margin"] - control["margin"]
            row["control_done"] = control["done"]
            rows.append(row)
    rows.sort(key=lambda row: (row["opponent"], row["item"], row["future_step"], row["horizon"], row["action_id"], row["seed"], row["seat"]))
    return _write(output, rows, controls, targets, opponents, skipped, inspection, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/rl_007_next_turn_preemption/data_train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--target", action="append", default=None, help="ITEM:FUTURE_STEP")
    args = parser.parse_args()
    targets = []
    for value in args.target or []:
        item, step = value.split(":", 1)
        targets.append((item, int(step)))
    print(json.dumps(collect_parallel(
        args.output,
        tuple(args.seed or DEFAULT_SEEDS),
        tuple(targets or DEFAULT_TARGETS),
        tuple(args.opponent or ()),
        args.workers,
    ), indent=2))
