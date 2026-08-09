"""Parallel paired counterfactual collector for RL-006."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

from rl_006_bidirectional_timing import (
    RL006_ACTION_NAMES,
    RL006_FEATURE_DIM,
    rl006_action_direction,
    rl006_action_quantity,
    rl006_route_opportunities,
)
from run_rl_006_bidirectional_data import (
    DEFAULT_EVENTS,
    DEFAULT_SEEDS,
    SingleEventShiftAgent,
    _available_opponents,
    _fresh_opponent,
    _run,
    _select_opportunities,
)
from run_v026_v22_v022c_recovery import ROOT, _v22_fresh


def _control_task(task):
    opponent, seed, seat = task
    result = _run(_v22_fresh("v22"), _fresh_opponent(opponent), seed, seat)
    return opponent, int(seed), int(seat), result


def _candidate_task(task):
    opponent, opportunity, action_id, seed, seat, source_sha = task
    candidate = SingleEventShiftAgent(opportunity, action_id)
    result = _run(candidate, _fresh_opponent(opponent), seed, seat)
    return {
        "item": opportunity["item"],
        "current_step": opportunity["current_step"],
        "future_step": opportunity["future_step"],
        "current_quantity": opportunity["current_quantity"],
        "future_quantity": opportunity["future_quantity"],
        "gap": opportunity["gap"],
        "action_id": int(action_id),
        "action": RL006_ACTION_NAMES[action_id],
        "direction": rl006_action_direction(action_id),
        "moved_quantity": rl006_action_quantity(action_id, opportunity),
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
        "future_repaid": int(candidate.future_repaid),
    }


def _write_results(output, rows, controls, opportunities, opponents, skipped, inspection, seeds):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    control_rows = []
    for (opponent, seed, seat), row in sorted(controls.items()):
        control_rows.append({"opponent": opponent, "seed": seed, "seat": seat, **row})
    fields = sorted({key for row in control_rows for key in row})
    with (output / "control.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(control_rows)
    support = {}
    for opportunity in opportunities:
        for action_id in range(1, 7):
            key = "{}|{}|{}|{}".format(
                opportunity["item"], opportunity["current_step"], opportunity["future_step"], action_id
            )
            support[key] = len({
                (row["seed"], row["seat"], row["opponent_source_sha256"])
                for row in rows
                if row["item"] == opportunity["item"]
                and row["current_step"] == opportunity["current_step"]
                and row["future_step"] == opportunity["future_step"]
                and row["action_id"] == action_id
            })
    report = {
        "seeds": list(seeds),
        "events": opportunities,
        "opponents": opponents,
        "skipped_opponents": skipped,
        "samples": len(rows),
        "controls": len(control_rows),
        "feature_dim": RL006_FEATURE_DIM,
        "actions": RL006_ACTION_NAMES,
        "all_done": int(all(row["candidate_done"] and row["control_done"] for row in rows)),
        "all_repaid": int(all(row["future_repaid"] for row in rows)),
        "unique_support": support,
        "inspection": list(inspection.values()),
    }
    (output / "collection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return report


def collect_parallel(output, seeds, requested_events, requested_opponents, workers):
    output = Path(output).resolve()
    opportunities = _select_opportunities(requested_events)
    opponents, skipped, inspection = _available_opponents(requested_opponents)
    if not opponents:
        raise RuntimeError("no loadable unique top10 opponents")

    controls = {}
    control_tasks = [(name, int(seed), seat) for name in opponents for seed in seeds for seat in (0, 1)]
    candidate_tasks = [
        (name, opportunity, action_id, int(seed), seat, inspection[name]["source_sha256"])
        for name in opponents
        for opportunity in opportunities
        for action_id in range(1, 7)
        for seed in seeds
        for seat in (0, 1)
    ]
    total = len(control_tasks) + len(candidate_tasks)
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(_control_task, task) for task in control_tasks]
        for future in concurrent.futures.as_completed(futures):
            opponent, seed, seat, result = future.result()
            controls[(opponent, seed, seat)] = result
            completed += 1
            print(f"[{completed}/{total}] control {opponent} seed={seed} seat={seat}", flush=True)

        futures = [executor.submit(_candidate_task, task) for task in candidate_tasks]
        rows = []
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            control = controls[(row["opponent"], row["seed"], row["seat"])]
            if not row["features"] or len(row["features"]) != RL006_FEATURE_DIM:
                raise RuntimeError(f"invalid features for {row}")
            row["control_money"] = control["candidate_money"]
            row["cash_delta"] = row["candidate_money"] - control["candidate_money"]
            row["control_margin"] = control["margin"]
            row["margin_delta"] = row["candidate_margin"] - control["margin"]
            row["control_done"] = control["done"]
            rows.append(row)
            completed += 1
            print(
                f"[{completed}/{total}] {row['opponent']} {row['action']} "
                f"{row['item']} {row['current_step']}->{row['future_step']} "
                f"seed={row['seed']} seat={row['seat']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["opponent"], row["item"], row["current_step"], row["action_id"], row["seed"], row["seat"]))
    return _write_results(output, rows, controls, opportunities, opponents, skipped, inspection, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/rl_006_bidirectional_timing/data_train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--event", action="append", default=None, help="ITEM:STEP")
    args = parser.parse_args()
    events = []
    for value in args.event or []:
        item, step = value.split(":", 1)
        events.append((item, int(step)))
    report = collect_parallel(
        args.output,
        tuple(args.seed or DEFAULT_SEEDS),
        tuple(events or DEFAULT_EVENTS),
        tuple(args.opponent or ()),
        args.workers,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
