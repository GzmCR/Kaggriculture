"""Collect paired single-event data for RL-010.

Each treatment changes exactly one MILK event and then returns to the V27
order-only route.  This keeps the first training set causal and easy to audit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

from kaggle_environments import make

from rl_010_milk_bidirectional import (
    RL010_ACTIONS,
    RL010History,
    RL010_RATIOS,
    rl010_adjust_sell,
    rl010_features,
    rl010_int,
    rl010_normalize_action,
    rl010_opportunity_key,
    rl010_private_inventory,
    rl010_route_opportunities,
    rl010_round_half_up,
    rl010_sell_quantity,
    rl010_step,
)
from rl_010_opponents import ROOT, inspect_opponents, load_spec, unique_loadable_specs


EPISODE_STEPS = 720
DEFAULT_SEEDS = (17, 42, 2026)
DEFAULT_EVENTS = (216, 264, 336, 406, 455, 480, 527, 597)
DEFAULT_OUTPUT = ROOT / "baseline/artifacts/rl_010_milk_bidirectional/data_train"
V27_PATH = ROOT / "baseline/history/v031_route_market_combo/v27_order_only/main.py"


def _load_v27():
    spec = importlib.util.spec_from_file_location(f"rl010_v27_{time.time_ns()}", V27_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(function, obs, config=None):
    try:
        return function(obs, config)
    except TypeError:
        return function(obs)


def _run_game(candidate, opponent, seed, seat):
    players = [candidate, opponent] if int(seat) == 0 else [opponent, candidate]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine, theirs = final[int(seat)], final[1 - int(seat)]
    mine_money = float(mine.observation["farms"][int(seat)]["money"])
    other_money = float(theirs.observation["farms"][1 - int(seat)]["money"])
    return {
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": mine_money - other_money,
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "candidate_status": str(mine.status),
        "opponent_status": str(theirs.status),
    }


class SingleEventBidirectional:
    def __init__(self, opportunity, action_name):
        self.opportunity = dict(opportunity)
        self.action_name = str(action_name).upper()
        self.base_module = _load_v27()
        self.base = self.base_module.agent
        self.history = RL010History()
        self.pending = False
        self.last_step = -1
        self.features = None
        self.snapshot = {}
        self.changed = []
        self.repayment_success = False
        self.repayment_failure = False
        self.route_action_at_event = None
        self.rl_adjusted_action_at_event = None
        self.final_action_at_event = None

    def _route_action(self, obs):
        """V27 route + WEED layer, before V27 market ranking."""
        step = rl010_step(obs)
        action = self.base_module._v031_copy_action(self.base_module._ACTIONS[step])
        action = self.base_module._v031_weed_action(obs, action, step)
        return self.base_module._v031_align_hands(action, obs)

    def _finalize(self, obs, action):
        """Apply the original V27 order-only ranking after the intervention."""
        action = self.base_module._v031_reorder_existing(obs, action)
        return self.base_module._v031_align_hands(action, obs)

    def _units(self, action, obs):
        if self.action_name not in RL010_RATIOS:
            return 0
        ratio = RL010_RATIOS[self.action_name]
        if self.action_name.startswith("ADVANCE"):
            desired = rl010_round_half_up(self.opportunity["future_quantity"] * ratio)
            available = max(0, rl010_private_inventory(obs) - rl010_sell_quantity(action))
            return max(0, min(desired, self.opportunity["future_quantity"], available))
        desired = rl010_round_half_up(self.opportunity["current_quantity"] * ratio)
        return max(0, min(desired, rl010_sell_quantity(action)))

    def __call__(self, obs, config=None):
        step = rl010_step(obs)
        if step == 0 or step < self.last_step:
            self.history.reset()
            self.pending = False
            self.features = None
            self.snapshot = {}
            self.changed = []
            self.repayment_success = False
            self.repayment_failure = False
            self.route_action_at_event = None
            self.rl_adjusted_action_at_event = None
            self.final_action_at_event = None
        base = rl010_normalize_action(self._route_action(obs))
        self.history.observe(obs)
        changed = 0
        if step == int(self.opportunity["future_step"]) and self.pending:
            delta = -self.pending_delta
            trial = json.loads(json.dumps(base))
            if rl010_adjust_sell(trial, delta):
                base = trial
                changed = abs(delta)
                self.repayment_success = True
            else:
                self.repayment_failure = True
            self.pending = False
        if step == int(self.opportunity["current_step"]):
            self.route_action_at_event = json.loads(json.dumps(base))
            self.features = rl010_features(obs, self.opportunity, self.history, base)
            market = obs.get("market", {}) or {}
            prices = market.get("prices", {}) or {}
            inventory = market.get("inventory", {}) or {}
            self.snapshot = {
                "price": prices.get("MILK", 0),
                "market_inventory": inventory.get("MILK", 0),
                "current_sell": rl010_sell_quantity(base),
                "own_inventory": rl010_private_inventory(obs),
                "market_orders": len(base.get("market", []) or []),
            }
            units = self._units(base, obs)
            if units > 0:
                delta = units if self.action_name.startswith("ADVANCE") else -units
                trial = json.loads(json.dumps(base))
                if rl010_adjust_sell(trial, delta):
                    base = trial
                    self.pending_delta = delta
                    self.pending = True
                    changed = units
                    self.changed.append({"step": step, "units": units, "delta": delta})
            self.rl_adjusted_action_at_event = json.loads(json.dumps(base))
        if changed:
            self.changed.append({"step": step, "units": changed, "delta": 0})
        self.last_step = step
        final = self._finalize(obs, base)
        if step == int(self.opportunity["current_step"]):
            self.final_action_at_event = json.loads(json.dumps(final))
        return final


def _select_opportunities(events):
    module = _load_v27()
    rows = rl010_route_opportunities(module._ACTIONS)
    by_step = {int(row["current_step"]): row for row in rows}
    selected = []
    for step in events:
        if int(step) not in by_step:
            raise ValueError(f"no V27 MILK opportunity at step {step}")
        selected.append(by_step[int(step)])
    return selected


def _select_opponents(names=None):
    available, inspection = unique_loadable_specs()
    by_name = {row["name"]: row for row in available}
    if names:
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"opponents unavailable: {missing}")
        selected = [by_name[name] for name in names]
    else:
        selected = available
    return selected, inspection


def collect(output, seeds, events, opponent_names=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    opportunities = _select_opportunities(events)
    opponents, inspection = _select_opponents(opponent_names)
    if not opponents:
        raise RuntimeError("no executable opponents")

    controls = {}
    for opponent_meta in opponents:
        for seed in seeds:
            for seat in (0, 1):
                opponent, _ = load_spec(opponent_meta)
                control = _load_v27().agent
                print(f"control {opponent_meta['name']} seed={seed} seat={seat}", flush=True)
                controls[(opponent_meta["name"], int(seed), int(seat))] = _run_game(
                    control, opponent, seed, seat
                )

    rows = []
    treatment_actions = [name for name in RL010_ACTIONS if name != "CONTROL"]
    total = len(opponents) * len(opportunities) * len(treatment_actions) * len(seeds) * 2
    done_count = 0
    for opponent_meta in opponents:
        for opportunity in opportunities:
            for action_name in treatment_actions:
                for seed in seeds:
                    for seat in (0, 1):
                        done_count += 1
                        print(
                            f"[{done_count}/{total}] {opponent_meta['name']} "
                            f"{action_name} {opportunity['current_step']} "
                            f"seed={seed} seat={seat}",
                            flush=True,
                        )
                        candidate = SingleEventBidirectional(opportunity, action_name)
                        opponent, _ = load_spec(opponent_meta)
                        result = _run_game(candidate, opponent, seed, seat)
                        control = controls[(opponent_meta["name"], int(seed), int(seat))]
                        if candidate.features is None:
                            raise RuntimeError(
                                f"event not observed: {opportunity['current_step']}"
                            )
                        margin_delta = result["margin"] - control["margin"]
                        control_win = control["margin"] > 0
                        candidate_win = result["margin"] > 0
                        bad_outcome = bool(
                            not result["done"]
                            or candidate.repayment_failure
                            or (control_win and not candidate_win)
                        )
                        rows.append({
                            "item": "MILK",
                            "current_step": opportunity["current_step"],
                            "future_step": opportunity["future_step"],
                            "current_quantity": opportunity["current_quantity"],
                            "future_quantity": opportunity["future_quantity"],
                            "gap": opportunity["gap"],
                            "action": action_name,
                            "seed": int(seed),
                            "seat": int(seat),
                            "opponent": opponent_meta["name"],
                            "opponent_family": opponent_meta.get("family", "unknown"),
                            "opponent_source_sha256": opponent_meta["source_sha256"],
                            "features": candidate.features.tolist(),
                            "snapshot": candidate.snapshot,
                            "route_action": candidate.route_action_at_event,
                            "rl_adjusted_action": candidate.rl_adjusted_action_at_event,
                            "final_ranked_action": candidate.final_action_at_event,
                            "candidate_money": result["candidate_money"],
                            "control_money": control["candidate_money"],
                            "cash_delta": result["candidate_money"] - control["candidate_money"],
                            "candidate_margin": result["margin"],
                            "control_margin": control["margin"],
                            "margin_delta": margin_delta,
                            "candidate_done": result["done"],
                            "control_done": control["done"],
                            "repayment_success": int(candidate.repayment_success),
                            "repayment_failure": int(candidate.repayment_failure),
                            "changed_units": sum(item.get("units", 0) for item in candidate.changed),
                            "bad_outcome": int(bad_outcome),
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
    (output / "source_manifest.json").write_text(
        json.dumps(inspection, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    report = {
        "engine": "kaggle-environments",
        "episode_steps": EPISODE_STEPS,
        "seeds": list(seeds),
        "events": opportunities,
        "opponents": [row["name"] for row in opponents],
        "samples": len(rows),
        "controls": len(control_rows),
        "all_done": int(all(row["candidate_done"] and row["control_done"] for row in rows)),
        "repayment_failures": sum(row["repayment_failure"] for row in rows),
        "actions": treatment_actions,
    }
    (output / "collection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--event", action="append", type=int, default=None)
    parser.add_argument("--opponent", action="append", default=None)
    args = parser.parse_args()
    report = collect(
        args.output,
        tuple(args.seed or DEFAULT_SEEDS),
        tuple(args.event or DEFAULT_EVENTS),
        tuple(args.opponent or []),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
