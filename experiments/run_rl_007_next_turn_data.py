"""Collect paired counterfactuals for RL-007 turn-level preemption."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kaggle_environments import make

from rl_006_bidirectional_timing import (
    RL006History,
    rl006_features,
    rl006_normalize_action,
    rl006_private_inventory,
    rl006_step,
)
from rl_007_next_turn_preemption import (
    RL007_ACTION_NAMES,
    RL007_FEATURE_DIM,
    rl007_append_sell,
    rl007_reduce_sell,
    rl007_route_opportunities,
    rl007_shift_quantity,
)
from run_v026_v22_v022c_recovery import EPISODE_STEPS, ROOT, _v22_fresh
from top10_opponents import inspect_top10, load_top10_agent, unique_top10_names


DEFAULT_SEEDS = (17, 42, 2026)
DEFAULT_TARGETS = (
    ("MILK", 260),
    ("MILK", 336),
    ("MILK", 473),
    ("STRAWBERRY", 503),
    ("MELON", 281),
    ("WOOL", 470),
)


def _call(agent, obs, config=None):
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


def _fresh_opponent(name):
    return load_top10_agent(name)[0]


class NextTurnPreemptAgent:
    """V022c with exactly one H1/H2/H3 premium preemption."""

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
                "horizon": self.opportunity["horizon"],
            }
            quantity = rl007_shift_quantity(self.opportunity)
            if (
                self.action_id
                and len(base.get("market", []) or []) < 10
                and rl006_private_inventory(obs, self.opportunity["item"]) >= quantity
            ):
                changed = rl007_append_sell(base, self.opportunity["item"], quantity)
                if changed == quantity:
                    self.pending = changed
        elif step == int(self.opportunity["future_step"]) and self.pending:
            changed = rl007_reduce_sell(base, self.opportunity["item"], self.pending)
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
    all_rows = rl007_route_opportunities(module.__globals__.get("_ACTIONS", []))
    index = {(row["item"], row["future_step"], row["horizon"]): row for row in all_rows}
    selected = []
    for item, future_step in requested:
        for horizon in (1, 2, 3):
            key = (str(item).upper(), int(future_step), horizon)
            if key not in index:
                raise ValueError(f"missing V022 route sale: {key}")
            selected.append(index[key])
    return selected


def _available_opponents(requested):
    inspection = {row["name"]: row for row in inspect_top10()}
    names = list(requested or unique_top10_names())
    available, skipped, seen = [], [], set()
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


def collect(output, seeds, requested_targets, requested_opponents):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    opportunities = _select_opportunities(requested_targets)
    opponents, skipped, inspection = _available_opponents(requested_opponents)
    controls = {}
    for opponent in opponents:
        for seed in seeds:
            for seat in (0, 1):
                print(f"control {opponent} seed={seed} seat={seat}", flush=True)
                controls[(opponent, int(seed), int(seat))] = _run(
                    _v22_fresh("v22"), _fresh_opponent(opponent), seed, seat
                )
    rows = []
    for opponent in opponents:
        for opportunity in opportunities:
            for action_id in (1, 2, 3):
                for seed in seeds:
                    for seat in (0, 1):
                        print(
                            f"{opponent} {opportunity['item']} target={opportunity['future_step']} "
                            f"H{opportunity['horizon']} seed={seed} seat={seat}", flush=True
                        )
                        candidate = NextTurnPreemptAgent(opportunity, action_id)
                        result = _run(candidate, _fresh_opponent(opponent), seed, seat)
                        control = controls[(opponent, int(seed), int(seat))]
                        if candidate.features is None or len(candidate.features) != RL007_FEATURE_DIM:
                            raise RuntimeError(f"invalid features for {opportunity}")
                        rows.append({
                            **opportunity,
                            "action_id": int(action_id),
                            "action": RL007_ACTION_NAMES[action_id],
                            "moved_quantity": rl007_shift_quantity(opportunity),
                            "seed": int(seed),
                            "seat": int(seat),
                            "opponent": opponent,
                            "opponent_source_sha256": inspection[opponent]["source_sha256"],
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
                            "shift_applied": int(bool(candidate.changed) and candidate.future_repaid),
                            "future_repaid": int(candidate.future_repaid),
                        })
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
        "targets": list(requested_targets),
        "seeds": list(seeds),
        "opponents": opponents,
        "skipped_opponents": skipped,
        "samples": len(rows),
        "controls": len(control_rows),
        "feature_dim": RL007_FEATURE_DIM,
        "actions": RL007_ACTION_NAMES,
        "all_done": int(all(row["candidate_done"] and row["control_done"] for row in rows)),
        "applied_samples": int(sum(row["shift_applied"] for row in rows)),
        "all_repaid": int(all((not row["shift_applied"]) or row["future_repaid"] for row in rows)),
        "inspection": list(inspection.values()),
    }
    (output / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/rl_007_next_turn_preemption/data_train")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--target", action="append", default=None, help="ITEM:FUTURE_STEP")
    args = parser.parse_args()
    targets = []
    for value in args.target or []:
        item, step = value.split(":", 1)
        targets.append((item, int(step)))
    result = collect(
        args.output,
        tuple(args.seed or DEFAULT_SEEDS),
        tuple(targets or DEFAULT_TARGETS),
        tuple(args.opponent or unique_top10_names()),
    )
    print(json.dumps(result, indent=2))
