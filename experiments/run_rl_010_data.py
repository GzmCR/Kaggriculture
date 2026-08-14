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

import kaggle_environments
from kaggle_environments import make

from rl_010_milk_bidirectional import (
    RL010_ACTIONS,
    RL010History,
    RL010_RATIOS,
    rl010_adjust_sell,
    rl010_delay_guard,
    rl010_features,
    rl010_int,
    rl010_normalize_action,
    rl010_opportunity_key,
    rl010_market_shed_state,
    rl010_route_opportunities,
    rl010_round_half_up,
    rl010_sell_quantity,
    rl010_step,
)
from rl_010_opponents import ROOT, inspect_opponents, load_spec, unique_loadable_specs


EPISODE_STEPS = 720
DEFAULT_SEEDS = (17, 42, 2026)
# Omit --event to collect every adjacent MILK event in the frozen route.
# Keeping an explicit event flag is useful for a quick pilot/smoke run.
DEFAULT_EVENTS = ()
DEFAULT_OUTPUT = ROOT / "baseline/artifacts/rl_010_milk_bidirectional/data_train"
V27_PATH = ROOT / "baseline/history/v031_route_market_combo/v27_order_only/main.py"

# The downloaded replays used for this study are from the rebalance regime.
# Keeping only episodeSteps/seed here silently switches the local engine to
# the legacy town-center cadence, which changes the route timing and can make
# otherwise valid MILK events appear to have no inventory.  Keep this config
# in one place so data collection and later smoke checks use the same rules.
GAME_CONFIGURATION = {
    "episodeSteps": EPISODE_STEPS,
    "turnsPerDay": 24,
    "boardSize": 10,
    "startingMoney": 3000,
    "farmHandCostMult": 1,
    "shedCapacity": 100,
    "maxMarketOrdersPerTurn": 10,
    "weedSpawnChance": 0.005,
    "townCenterSellInterval": 24,
    "townShopSellInterval": 4,
    "townShopUnlockInterval": 3,
}

REQUIRED_ENGINE_VERSION = "1.32.6"


def _version_tuple(value):
    parts = []
    for piece in str(value or "0").split(".")[:3]:
        digits = "".join(character for character in piece if character.isdigit())
        parts.append(int(digits or 0))
    return tuple((parts + [0, 0, 0])[:3])


def _engine_version():
    return str(getattr(kaggle_environments, "__version__", "unknown"))


def _check_engine_version(allow_engine_mismatch=False):
    actual = _engine_version()
    compatible = _version_tuple(actual) == _version_tuple(REQUIRED_ENGINE_VERSION)
    if not compatible and not allow_engine_mismatch:
        raise RuntimeError(
            "RL-010 data collection requires kaggle-environments "
            f"{REQUIRED_ENGINE_VERSION}; found {actual}. "
            "Install the required engine or pass --allow-engine-mismatch "
            "for diagnostic-only data that must not be fitted."
        )
    return actual, compatible


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


class _TraceAgent:
    """Record live pre-market MILK availability at selected route events."""

    def __init__(self, function, opportunities):
        self.function = function
        self.steps = {
            int(row["current_step"]): row for row in opportunities or []
        }
        self.steps.update({
            int(row["future_step"]): row for row in opportunities or []
        })
        self.trace = {}

    def __call__(self, obs, config=None):
        action = rl010_normalize_action(_call(self.function, obs, config))
        step = rl010_step(obs)
        if step in self.steps:
            state = rl010_market_shed_state(obs, action, config=config)
            self.trace[int(step)] = {
                "requested_sell": int(rl010_sell_quantity(action)),
                "available_after_units": int(state["shed_after_actions"]),
                "action": action,
            }
        return action


def _run_game(candidate, opponent, seed, seat, trace_opportunities=None):
    players = [candidate, opponent] if int(seat) == 0 else [opponent, candidate]
    tracer = None
    if trace_opportunities:
        tracer = _TraceAgent(candidate, trace_opportunities)
        players = [tracer, opponent] if int(seat) == 0 else [opponent, tracer]
    configuration = dict(GAME_CONFIGURATION)
    configuration["seed"] = int(seed)
    env = make(
        "kaggriculture",
        configuration=configuration,
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
        "trace": tracer.trace if tracer is not None else {},
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
        self.event_units = 0
        self.pending_delta = 0
        self.activation_reason = "not_event"

    def _route_action(self, obs, config=None):
        """Use the complete V27 runtime route before MILK intervention."""
        return self.base_module.agent(obs, config)

    def _finalize(self, obs, action):
        """Apply the original V27 order-only ranking after the intervention."""
        action = self.base_module._v031_reorder_existing(obs, action)
        return self.base_module._v031_align_hands(action, obs)

    def _units(self, action, obs, config=None):
        if self.action_name not in RL010_RATIOS:
            return 0
        ratio = RL010_RATIOS[self.action_name]
        if self.action_name.startswith("ADVANCE"):
            desired = rl010_round_half_up(self.opportunity["future_quantity"] * ratio)
            state = rl010_market_shed_state(obs, action, config=config)
            return max(0, min(desired, self.opportunity["future_quantity"], state["sellable_extra"]))
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
            self.event_units = 0
            self.pending_delta = 0
            self.activation_reason = "not_event"
        base = rl010_normalize_action(self._route_action(obs, config))
        self.history.observe(obs)
        changed = 0
        if step == int(self.opportunity["future_step"]) and self.pending:
            delta = -self.pending_delta
            trial = json.loads(json.dumps(base))
            valid = rl010_adjust_sell(trial, delta)
            if valid:
                state = rl010_market_shed_state(obs, trial, config=config)
                valid = state["shed_after_actions"] >= rl010_sell_quantity(trial)
            if valid:
                base = trial
                changed = abs(delta)
                self.repayment_success = True
                self.activation_reason = "repayment_success"
            else:
                self.repayment_failure = True
                self.activation_reason = "repayment_inventory_short"
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
                "own_inventory": rl010_market_shed_state(obs, base, config=config)["shed_after_actions"],
                "shed_total_after_actions": rl010_market_shed_state(obs, base, config=config)["shed_total_after_actions"],
                "market_orders": len(base.get("market", []) or []),
            }
            future_available = self.opportunity.get("control_future_available")
            future_quantity = int(self.opportunity.get("future_quantity", 0))
            if (
                self.action_name.startswith("DELAY")
                and future_available is not None
                and int(future_available) < future_quantity
            ):
                units = 0
                self.activation_reason = "control_future_not_sellable"
            else:
                units = self._units(base, obs, config=config)
            if self.action_name.startswith("ADVANCE") and units > 0:
                state = rl010_market_shed_state(obs, base, config=config)
                if rl010_sell_quantity(base) + units > state["shed_after_actions"]:
                    units = 0
            if self.action_name.startswith("DELAY") and units > 0:
                allowed, reason = rl010_delay_guard(
                    obs, base, self.opportunity, units, config=config
                )
                if not allowed:
                    units = 0
                    self.activation_reason = reason
            if units > 0:
                delta = units if self.action_name.startswith("ADVANCE") else -units
                trial = json.loads(json.dumps(base))
                if rl010_adjust_sell(trial, delta):
                    base = trial
                    self.pending_delta = delta
                    self.pending = True
                    changed = units
                    self.event_units = units
                    self.activation_reason = "legal_intervention"
                    self.changed.append({"step": step, "units": units, "delta": delta})
            elif self.activation_reason == "not_event":
                self.activation_reason = "inventory_or_route_guard"
            self.rl_adjusted_action_at_event = json.loads(json.dumps(base))
        if changed and step != int(self.opportunity["current_step"]):
            self.changed.append({"step": step, "units": changed, "delta": 0})
        self.last_step = step
        final = self._finalize(obs, base)
        if step == int(self.opportunity["current_step"]):
            self.final_action_at_event = json.loads(json.dumps(final))
        return final


def _select_opportunities(events=None):
    module = _load_v27()
    # Empty means all route opportunities.  A non-empty tuple is an explicit
    # focused subset for smoke tests or a small pilot collection.
    rows = rl010_route_opportunities(module._ACTIONS, events or None)
    if not events:
        return rows
    by_step = {int(row["current_step"]): row for row in rows}
    selected = []
    for step in events:
        if int(step) not in by_step:
            raise ValueError(f"no V27 MILK opportunity at step {step}")
        selected.append(by_step[int(step)])
    return selected


def _filter_opportunities(opportunities, min_step=None, max_step=None, max_events=None):
    rows = list(opportunities)
    if min_step is not None:
        rows = [row for row in rows if int(row["current_step"]) >= int(min_step)]
    if max_step is not None:
        rows = [row for row in rows if int(row["current_step"]) <= int(max_step)]
    if max_events is not None:
        rows = rows[:max(0, int(max_events))]
    return rows


def _trace_at(control, step):
    trace = control.get("trace", {}) or {}
    value = trace.get(str(int(step)), trace.get(int(step), {}))
    return value if isinstance(value, dict) else {}


def _control_allows_action(opportunity, action_name, control):
    """Cheap preflight before launching a 720-turn counterfactual.

    This deliberately mirrors the conservative legality used by
    SingleEventBidirectional.  It is only a skip optimization: the candidate
    still performs its full live safety checks.
    """
    current = _trace_at(control, opportunity["current_step"])
    future = _trace_at(control, opportunity["future_step"])
    current_available = int(current.get("available_after_units", 0) or 0)
    future_available = int(future.get("available_after_units", 0) or 0)
    current_quantity = int(opportunity["current_quantity"])
    future_quantity = int(opportunity["future_quantity"])
    action_name = str(action_name).upper()
    if action_name.startswith("ADVANCE"):
        ratio = RL010_RATIOS[action_name]
        units = min(
            rl010_round_half_up(future_quantity * ratio),
            future_quantity,
        )
        return units > 0 and current_available >= current_quantity + units
    if action_name.startswith("DELAY"):
        ratio = RL010_RATIOS[action_name]
        units = min(
            rl010_round_half_up(current_quantity * ratio),
            max(0, current_quantity - 1),
        )
        # The runtime conservatively requires the unmodified next event to be
        # sellable.  Holding units cannot rescue a broken base route.
        return (
            units > 0
            and current_available >= current_quantity
            and future_available >= future_quantity
            and int(opportunity["future_step"]) < 648
        )
    return False


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


def collect(
    output,
    seeds,
    events,
    opponent_names=None,
    allow_engine_mismatch=False,
    treatment_actions=None,
    min_step=None,
    max_step=None,
    max_events=None,
):
    engine_version, engine_compatible = _check_engine_version(allow_engine_mismatch)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    opportunities = _filter_opportunities(
        _select_opportunities(events),
        min_step=min_step,
        max_step=max_step,
        max_events=max_events,
    )
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
                    control, opponent, seed, seat, trace_opportunities=opportunities
                )

    rows = []
    treatment_actions = list(treatment_actions or [
        name for name in RL010_ACTIONS if name != "CONTROL"
    ])
    valid_actions = {name for name in RL010_ACTIONS if name != "CONTROL"}
    invalid_actions = [name for name in treatment_actions if name not in valid_actions]
    if invalid_actions:
        raise ValueError(f"invalid treatment actions: {invalid_actions}")

    # Build only jobs whose control trace has a real sellable window.  This is
    # the main speedup: an omitted job saves one full 720-turn environment run.
    jobs = []
    eligible_counts = {}
    skipped_jobs = 0
    for opponent_meta in opponents:
        for opportunity in opportunities:
            control_by_seed_seat = {
                (int(seed), int(seat)): controls[(opponent_meta["name"], int(seed), int(seat))]
                for seed in seeds
                for seat in (0, 1)
            }
            for action_name in treatment_actions:
                for seed in seeds:
                    for seat in (0, 1):
                        eligible = _control_allows_action(
                            opportunity,
                            action_name,
                            control_by_seed_seat[(int(seed), int(seat))],
                        )
                        counter_key = f"{opportunity['current_step']}|{action_name}"
                        eligible_counts[counter_key] = (
                            eligible_counts.get(counter_key, 0) + int(eligible)
                        )
                        if eligible:
                            jobs.append((opponent_meta, opportunity, action_name, int(seed), int(seat)))
                        else:
                            skipped_jobs += 1
    total = len(jobs)
    done_count = 0
    for opponent_meta, opportunity, action_name, seed, seat in jobs:
        done_count += 1
        print(
            f"[{done_count}/{total}] {opponent_meta['name']} "
            f"{action_name} {opportunity['current_step']} "
            f"seed={seed} seat={seat}",
            flush=True,
        )
        control = controls[(opponent_meta["name"], int(seed), int(seat))]
        candidate_opportunity = dict(opportunity)
        future_trace = _trace_at(control, opportunity["future_step"])
        if future_trace:
            candidate_opportunity["control_future_available"] = int(
                future_trace.get("available_after_units", 0)
            )
        candidate = SingleEventBidirectional(candidate_opportunity, action_name)
        opponent, _ = load_spec(opponent_meta)
        result = _run_game(candidate, opponent, seed, seat)
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
            or (control["margin"] >= 0 and result["margin"] < 0)
        )
        rows.append({
                            "item": "MILK",
                            "current_step": opportunity["current_step"],
                            "future_step": opportunity["future_step"],
                            "current_quantity": opportunity["current_quantity"],
                            "future_quantity": opportunity["future_quantity"],
                            "gap": opportunity["gap"],
                            "action": action_name if candidate.event_units > 0 else "CONTROL",
                            "requested_action": action_name,
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
                            "changed_units": int(candidate.event_units),
                            "activation_reason": candidate.activation_reason,
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
        "engine_version": engine_version,
        "required_engine_version": REQUIRED_ENGINE_VERSION,
        "engine_compatible": bool(engine_compatible),
        "training_data_valid": bool(engine_compatible),
        "configuration": GAME_CONFIGURATION,
        "episode_steps": EPISODE_STEPS,
        "seeds": list(seeds),
        "events": opportunities,
        "opponents": [row["name"] for row in opponents],
        "samples": len(rows),
        "controls": len(control_rows),
        "candidate_jobs": len(jobs),
        "skipped_ineligible_jobs": skipped_jobs,
        "eligible_counts": eligible_counts,
        "all_done": int(all(row["candidate_done"] and row["control_done"] for row in rows)),
        "repayment_failures": sum(row["repayment_failure"] for row in rows),
        "actions": treatment_actions,
        "filters": {
            "min_step": min_step,
            "max_step": max_step,
            "max_events": max_events,
        },
    }
    (output / "collection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument(
        "--event", action="append", type=int, default=None,
        help="focus on one or more route steps; omit to collect all MILK events",
    )
    parser.add_argument("--opponent", action="append", default=None)
    parser.add_argument(
        "--action",
        action="append",
        dest="treatment_actions",
        choices=[name for name in RL010_ACTIONS if name != "CONTROL"],
        help="counterfactual action(s); omit for all four actions",
    )
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--allow-engine-mismatch",
        action="store_true",
        help="diagnostic only: collect data on a non-1.32.6 engine; never fit it",
    )
    args = parser.parse_args()
    report = collect(
        args.output,
        tuple(args.seed or DEFAULT_SEEDS),
        tuple(args.event or DEFAULT_EVENTS),
        tuple(args.opponent or []),
        allow_engine_mismatch=args.allow_engine_mismatch,
        treatment_actions=tuple(args.treatment_actions or []),
        min_step=args.min_step,
        max_step=args.max_step,
        max_events=args.max_events,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
