"""Collect paired counterfactual data for RL-006 bidirectional timing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kaggle_environments import make

from rl_006_bidirectional_timing import (
    RL006_ACTION_NAMES,
    RL006_FEATURE_DIM,
    RL006History,
    rl006_action_direction,
    rl006_action_quantity,
    rl006_adjust_sell,
    rl006_features,
    rl006_normalize_action,
    rl006_route_opportunities,
    rl006_step,
)
from run_v026_v22_v022c_recovery import EPISODE_STEPS, ROOT, _v22_fresh
from top10_opponents import inspect_top10, load_top10_agent, unique_top10_names


DEFAULT_SEEDS = (17, 42, 2026)
DEFAULT_EVENTS = (
    ("MILK", 215),
    ("MILK", 310),
    ("MILK", 452),
    ("STRAWBERRY", 480),
    ("MELON", 264),
    ("WOOL", 450),
)


def _call(agent, obs, config=None):
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


def _fresh_opponent(name):
    return load_top10_agent(name)[0]


def _action_delta(action_id, opportunity):
    direction = rl006_action_direction(action_id)
    quantity = rl006_action_quantity(action_id, opportunity)
    return quantity if direction == "PREEMPT" else -quantity


class SingleEventShiftAgent:
    """V22 control with exactly one signed timing shift."""

    def __init__(self, opportunity, action_id):
        self.opportunity = dict(opportunity)
        self.action_id = int(action_id)
        self.base = _v22_fresh("v22")
        self.history = RL006History()
        self.pending = 0
        self.last_step = -1
        self.features = None
        self.snapshot = {}
        self.changed = []
        self.future_repaid = False

    def __call__(self, obs, config=None):
        step = rl006_step(obs)
        if step == 0 or step < self.last_step:
            self.history = RL006History()
            self.pending = 0
            self.features = None
            self.snapshot = {}
            self.changed = []
            self.future_repaid = False
        base = rl006_normalize_action(_call(self.base, obs, config))
        self.history.observe(obs)
        changed = 0
        if step == int(self.opportunity["current_step"]):
            self.features = rl006_features(obs, self.opportunity, self.history, base)
            market = obs.get("market", {}) or {}
            prices = market.get("prices", {}) or {}
            inventory = market.get("inventory", {}) or {}
            farms = obs.get("farms", []) or []
            player = int(obs.get("player", 0) or 0)
            other = farms[1 - player] if len(farms) == 2 and player in (0, 1) else {}
            self.snapshot = {
                "price": prices.get(self.opportunity["item"], 0),
                "market_inventory": inventory.get(self.opportunity["item"], 0),
                "market_orders": len(base.get("market", []) or []),
                "opponent_money": other.get("money", 0),
            }
            if self.action_id:
                delta = _action_delta(self.action_id, self.opportunity)
                changed = rl006_adjust_sell(base, self.opportunity["item"], delta)
                expected = abs(delta)
                if changed == expected:
                    self.pending = changed
                else:
                    self.pending = 0
        elif step == int(self.opportunity["future_step"]) and self.pending:
            direction = rl006_action_direction(self.action_id)
            delta = -self.pending if direction == "PREEMPT" else self.pending
            changed = rl006_adjust_sell(base, self.opportunity["item"], delta)
            self.future_repaid = changed == self.pending
            self.pending = 0
        if changed:
            self.changed.append({"step": step, "quantity": int(changed)})
        self.last_step = step
        return base


def _run(agent, opponent, seed, seat):
    players = [agent, opponent] if seat == 0 else [opponent, agent]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine, theirs = final[seat], final[1 - seat]
    mine_money = float(mine.observation["farms"][seat]["money"])
    other_money = float(theirs.observation["farms"][1 - seat]["money"])
    return {
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": mine_money - other_money,
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "candidate_status": str(mine.status),
        "opponent_status": str(theirs.status),
    }


def _select_opportunities(requested):
    module = _v22_fresh("v22")
    all_rows = rl006_route_opportunities(module.__globals__.get("_ACTIONS", []))
    index = {(row["item"], row["current_step"]): row for row in all_rows}
    selected = []
    for item, step in requested:
        key = (str(item).upper(), int(step))
        if key not in index:
            raise ValueError(f"missing V22 route opportunity: {key}")
        selected.append(index[key])
    return selected


def _available_opponents(requested):
    inspection = {row["name"]: row for row in inspect_top10()}
    names = list(requested or unique_top10_names())
    available = []
    skipped = []
    seen = set()
    for name in names:
        row = inspection.get(name)
        if not row or row.get("load_error"):
            skipped.append(row or {"name": name, "load_error": "unknown opponent"})
            continue
        source_sha = row.get("source_sha256") or name
        if source_sha in seen:
            continue
        seen.add(source_sha)
        available.append(name)
    return available, skipped, inspection


def collect(output, seeds, requested_events, requested_opponents):
    output.mkdir(parents=True, exist_ok=True)
    opportunities = _select_opportunities(requested_events)
    opponents, skipped, inspection = _available_opponents(requested_opponents)
    if not opponents:
        raise RuntimeError("no loadable unique top10 opponents")

    controls = {}
    for opponent_name in opponents:
        for seed in seeds:
            for seat in (0, 1):
                print(f"control opponent={opponent_name} seed={seed} seat={seat}", flush=True)
                controls[(opponent_name, int(seed), int(seat))] = _run(
                    _v22_fresh("v22"), _fresh_opponent(opponent_name), seed, seat
                )

    rows = []
    for opponent_name in opponents:
        for opportunity in opportunities:
            for action_id in range(1, 7):
                for seed in seeds:
                    for seat in (0, 1):
                        print(
                            f"{opponent_name} {opportunity['item']} "
                            f"{opportunity['current_step']}->{opportunity['future_step']} "
                            f"{RL006_ACTION_NAMES[action_id]} seed={seed} seat={seat}",
                            flush=True,
                        )
                        candidate = SingleEventShiftAgent(opportunity, action_id)
                        result = _run(candidate, _fresh_opponent(opponent_name), seed, seat)
                        control = controls[(opponent_name, int(seed), int(seat))]
                        if candidate.features is None:
                            raise RuntimeError(f"event was not observed: {opportunity}")
                        if len(candidate.features) != RL006_FEATURE_DIM:
                            raise RuntimeError("unexpected RL006 feature vector")
                        rows.append({
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
                            "opponent": opponent_name,
                            "opponent_source_sha256": inspection[opponent_name]["source_sha256"],
                            "features": candidate.features.tolist(),
                            "snapshot": candidate.snapshot,
                            "candidate_money": result["candidate_money"],
                            "control_money": control["candidate_money"],
                            "cash_delta": result["candidate_money"] - control["candidate_money"],
                            "candidate_margin": result["margin"],
                            "control_margin": control["margin"],
                            "margin_delta": result["margin"] - control["margin"],
                            "candidate_done": result["done"],
                            "control_done": control["done"],
                            "changed_calls": len(candidate.changed),
                            "future_repaid": int(candidate.future_repaid),
                        })

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/rl_006_bidirectional_timing/data_train")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--event", action="append", default=None, help="ITEM:STEP")
    args = parser.parse_args()
    events = []
    for value in args.event or []:
        item, step = value.split(":", 1)
        events.append((item, int(step)))
    report = collect(
        args.output,
        tuple(args.seed or DEFAULT_SEEDS),
        tuple(events or DEFAULT_EVENTS),
        tuple(args.opponent or unique_top10_names()),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
