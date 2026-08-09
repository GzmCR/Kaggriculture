"""Collect paired counterfactuals for RL-008 small quantity timing."""

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
from rl_008_small_shift_timing import (
    RL008_ACTION_NAMES,
    RL008_DELAY_ACTIONS,
    RL008_FEATURE_DIM,
    RL008_PREEMPT_ACTIONS,
    rl008_action_kind,
    rl008_action_matches,
    rl008_action_quantity,
    rl008_append_sell,
    rl008_delay_opportunities,
    rl008_preempt_opportunities,
    rl008_shift_sell,
)
from run_v026_v22_v022c_recovery import EPISODE_STEPS, ROOT, _v22_fresh
from top10_opponents import inspect_top10, load_top10_agent, unique_top10_names


DEFAULT_SEEDS = (17, 42, 2026, 217, 317, 733)
DEFAULT_TARGETS = (
    ("MILK", 215, 260),
    ("MILK", 310, 336),
    ("MILK", 452, 473),
    ("STRAWBERRY", 480, 503),
    ("MELON", 264, 281),
    ("WOOL", 450, 470),
)


def _call(agent, obs, config=None):
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


def _fresh_opponent(name):
    return load_top10_agent(name)[0]


def _route_actions():
    module = _v22_fresh("v22")
    return module.__globals__.get("_ACTIONS", [])


def _select_opportunities(targets):
    actions = _route_actions()
    selected = []
    for item, current_step, future_step in targets:
        item = str(item).upper()
        preempt = [
            row for row in rl008_preempt_opportunities(actions, [(item, future_step)])
            if row["future_step"] == int(future_step)
        ]
        if len(preempt) != 3:
            raise ValueError(f"expected H1/H2/H3 route sales for {(item, current_step, future_step)}")
        delays = [
            row for row in rl008_delay_opportunities(actions)
            if row["item"] == item
            and row["current_step"] == int(current_step)
            and row["future_step"] == int(future_step)
        ]
        if len(delays) != 1:
            raise ValueError(f"expected delay opportunity for {(item, current_step, future_step)}")
        selected.extend(sorted(preempt, key=lambda row: row["horizon"]))
        selected.extend(delays)
    return selected


class SingleEventShiftAgent:
    """V022c with one small bidirectional shift and exact repayment."""

    def __init__(self, opportunity, action_id):
        self.opportunity = dict(opportunity)
        self.action_id = int(action_id)
        self.base = _v22_fresh("v22")
        self.history = RL006History()
        self.pending = None
        self.last_step = -1
        self.features = None
        self.snapshot = {}
        self.changed = []
        self.future_repaid = False
        self.shift_applied = False
        self.failure_reason = ""

    def __call__(self, obs, config=None):
        step = rl006_step(obs)
        if step == 0 or step < self.last_step:
            self.history = RL006History()
            self.pending = None
            self.last_step = -1
            self.features = None
            self.snapshot = {}
            self.changed = []
            self.future_repaid = False
            self.shift_applied = False
            self.failure_reason = ""

        base = rl006_normalize_action(_call(self.base, obs, config))
        self.history.observe(obs)
        changed = 0

        if self.pending and step == int(self.pending["future_step"]):
            debt = self.pending
            delta = -int(debt["quantity"]) if debt["kind"] == "PREEMPT" else int(debt["quantity"])
            moved = rl008_shift_sell(base, debt["item"], delta)
            self.future_repaid = moved == abs(delta)
            if not self.future_repaid:
                self.failure_reason = "repayment_failed"
            changed += moved
            self.changed.append({"step": int(step), "kind": "REPAY", "quantity": int(moved)})
            self.pending = None

        if step == int(self.opportunity["current_step"]):
            self.features = rl006_features(obs, self.opportunity, self.history, base, {})
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
                "kind": self.opportunity["kind"],
                "horizon": self.opportunity.get("horizon", 0),
                "clone_distance": self._clone_distance(obs),
            }
            quantity = rl008_action_quantity(self.action_id, self.opportunity)
            if self.action_id == 0 or quantity <= 0:
                self.failure_reason = "control_or_zero_quantity"
            elif len(base.get("market", []) or []) >= 10:
                self.failure_reason = "market_full"
            elif self.opportunity["kind"] == "PREEMPT" and rl006_private_inventory(obs, self.opportunity["item"]) < quantity:
                self.failure_reason = "inventory_short"
            elif self.opportunity["kind"] == "DELAY" and self.opportunity["current_quantity"] < quantity:
                self.failure_reason = "current_order_short"
            else:
                if self.opportunity["kind"] == "PREEMPT":
                    moved = rl008_append_sell(base, self.opportunity["item"], quantity)
                else:
                    moved = rl008_shift_sell(base, self.opportunity["item"], -quantity)
                if moved != quantity:
                    self.failure_reason = "initial_shift_failed"
                else:
                    self.pending = {
                        "item": str(self.opportunity["item"]).upper(),
                        "quantity": int(quantity),
                        "kind": str(self.opportunity["kind"]).upper(),
                        "future_step": int(self.opportunity["future_step"]),
                    }
                    self.shift_applied = True
                    changed += moved
                    self.changed.append({"step": int(step), "kind": "SHIFT", "quantity": int(moved)})

        self.last_step = step
        return base

    @staticmethod
    def _clone_distance(obs):
        farms = list(obs.get("farms", []) or [])
        if len(farms) != 2:
            return 10**9
        left = farms[0]
        right = farms[1]
        from rl_006_bidirectional_timing import rl006_clone_distance
        return float(rl006_clone_distance(left, right))


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


def collect(output, seeds, targets, requested_opponents):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    opportunities = _select_opportunities(targets)
    opponents, skipped, inspection = _available_opponents(requested_opponents)
    if not opponents:
        raise RuntimeError("no loadable unique top10 opponents")

    controls = {}
    for opponent in opponents:
        for seed in seeds:
            for seat in (0, 1):
                print(f"control opponent={opponent} seed={seed} seat={seat}", flush=True)
                controls[(opponent, int(seed), int(seat))] = _run(
                    _v22_fresh("v22"), _fresh_opponent(opponent), seed, seat
                )

    rows = []
    action_ids = tuple(range(1, 10))
    for opponent in opponents:
        for opportunity in opportunities:
            for action_id in action_ids:
                if not rl008_action_matches(action_id, opportunity):
                    continue
                for seed in seeds:
                    for seat in (0, 1):
                        print(
                            f"{opponent} {opportunity['kind']} {opportunity['item']} "
                            f"{opportunity['current_step']}->{opportunity['future_step']} "
                            f"{RL008_ACTION_NAMES[action_id]} seed={seed} seat={seat}",
                            flush=True,
                        )
                        candidate = SingleEventShiftAgent(opportunity, action_id)
                        result = _run(candidate, _fresh_opponent(opponent), seed, seat)
                        control = controls[(opponent, int(seed), int(seat))]
                        if candidate.features is None:
                            raise RuntimeError(f"event was not observed: {opportunity}")
                        if len(candidate.features) != RL008_FEATURE_DIM:
                            raise RuntimeError("unexpected RL-008 feature vector")
                        rows.append({
                            **opportunity,
                            "action_id": int(action_id),
                            "action": RL008_ACTION_NAMES[action_id],
                            "direction": rl008_action_kind(action_id),
                            "moved_quantity": rl008_action_quantity(action_id, opportunity),
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
                            "shift_applied": int(candidate.shift_applied),
                            "future_repaid": int(candidate.future_repaid),
                            "failure_reason": candidate.failure_reason,
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
        "targets": [list(row) for row in targets],
        "seeds": list(seeds),
        "opponents": opponents,
        "skipped_opponents": skipped,
        "samples": len(rows),
        "controls": len(control_rows),
        "feature_dim": RL008_FEATURE_DIM,
        "actions": RL008_ACTION_NAMES,
        "all_done": int(all(row["candidate_done"] and row["control_done"] for row in rows)),
        "applied_samples": int(sum(row["shift_applied"] for row in rows)),
        "repaid_applied": int(sum(row["shift_applied"] and row["future_repaid"] for row in rows)),
        "inspection": list(inspection.values()),
    }
    (output / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/rl_008_small_shift_timing/data_train")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument("--target", action="append", default=None, help="ITEM:CURRENT_STEP:FUTURE_STEP")
    args = parser.parse_args()
    targets = []
    for value in args.target or []:
        item, current, future = value.split(":", 2)
        targets.append((item, int(current), int(future)))
    result = collect(
        args.output,
        tuple(args.seed or DEFAULT_SEEDS),
        tuple(targets or DEFAULT_TARGETS),
        tuple(args.opponent or unique_top10_names()),
    )
    print(json.dumps(result, indent=2))
