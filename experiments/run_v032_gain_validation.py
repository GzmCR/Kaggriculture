"""Collect V032 timing-event counterfactuals for V032-R1 calibration.

For each local notebook opponent game this runner first records the events the
old V032 timing arm would accept.  It then reruns the same environment with
only one accepted event enabled and compares it with V27 order-only control.
The fixed ``nowinlog`` replays are intentionally not used here.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

from kaggle_environments import make

from rl_010_opponents import load_spec, unique_loadable_specs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMING = ROOT / "baseline/artifacts/v032_route_conditioned_timing/v032_v27_timing/main.py"
DEFAULT_CONTROL = ROOT / "baseline/artifacts/v032_route_conditioned_timing/v032_v27_order_only/main.py"
DEFAULT_R1_TIMING = ROOT / "baseline/artifacts/v032_route_conditioned_timing_r1/v032_r1_v27_timing/main.py"
DEFAULT_R1_CONTROL = ROOT / "baseline/artifacts/v032_route_conditioned_timing_r1/v032_r1_v27_order_only/main.py"


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"v032_r1_validation_{tag}_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CallProbe:
    def __init__(self, module, target_step=None, forced_profile=None):
        self.module = module
        self.target_step = target_step
        self.forced_profile = forced_profile
        self.events = []
        self._original_choose = getattr(module, "_v032_choose_timing", None)
        self._original_gain = getattr(module, "_v032_expected_gain", None)
        self.gains = []
        if self._original_choose is not None:
            module._v032_choose_timing = self._choose
        if self._original_gain is not None:
            module._v032_expected_gain = self._gain

    def _gain(self, obs, item, now_q, future_q, due, transfer, mode, profile):
        value = self._original_gain(obs, item, now_q, future_q, due, transfer, mode, profile)
        self.gains.append({"step": int(obs.get("step", 0) or 0), "item": str(item),
                           "now_q": int(now_q), "future_q": int(future_q),
                           "due": int(due), "transfer": int(transfer),
                           "mode": str(mode), "raw_gain": float(value)})
        return value

    def _choose(self, obs, action, state, step, profile):
        if self.forced_profile is not None:
            state["known"] = True
            profile = self.forced_profile
        if self.target_step is not None and int(step) != int(self.target_step):
            return action
        before = copy.deepcopy(state.get("pending"))
        result = self._original_choose(obs, action, state, step, profile)
        after = state.get("pending")
        if after and not before:
            matching = [row for row in self.gains if row["step"] == int(step)
                        and row["item"] == str(after["item"])
                        and row["transfer"] == int(after["quantity"])
                        and row["due"] == int(after["due"])
                        and row["mode"] == str(after["mode"])]
            event = dict(after)
            event.update({"step": int(step), "raw_gain": matching[-1]["raw_gain"] if matching else None})
            self.events.append(event)
        return result

    def close(self):
        if self._original_choose is not None:
            self.module._v032_choose_timing = self._original_choose
        if self._original_gain is not None:
            self.module._v032_expected_gain = self._original_gain


class _OpponentRecorder:
    def __init__(self, agent):
        self.agent = agent
        self.schedule = {}

    def __call__(self, obs, config=None):
        try:
            action = self.agent(obs, config)
        except TypeError:
            action = self.agent(obs)
        step = int(obs.get("step", 0) or 0)
        self.schedule[step] = [list(order) for order in (action.get("market", []) or [])]
        return action


def _run_game(agent, opponent, seed, seat, capture_opponent=False):
    recorder = _OpponentRecorder(opponent) if capture_opponent else None
    opponent_player = recorder if recorder is not None else opponent
    players = [agent, opponent_player] if int(seat) == 0 else [opponent_player, agent]
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": int(seed)}, debug=False)
    env.run(players)
    final = env.steps[-1]
    me, them = int(seat), 1 - int(seat)
    money = [float(final[i].observation["farms"][i]["money"]) for i in (me, them)]
    result = {"mine": money[0], "opponent": money[1], "margin": money[0] - money[1],
              "done": int(all(str(x.status) == "DONE" for x in final))}
    if recorder is not None:
        result["opponent_schedule"] = recorder.schedule
    return result


def _run_probe(module, opponent, seed, seat, target_step=None, forced_profile=None):
    probe = CallProbe(module, target_step=target_step, forced_profile=forced_profile)
    result = _run_game(probe.module.agent, opponent, seed, seat)
    result["events"] = probe.events
    result["gain_calls"] = probe.gains
    probe.close()
    return result


def collect(opponent_names, seeds, max_events, output, timing_path=DEFAULT_TIMING, control_path=DEFAULT_CONTROL):
    specs, _ = unique_loadable_specs()
    if opponent_names:
        wanted = set(opponent_names)
        specs = [row for row in specs if row["name"] in wanted]
    rows = []
    for spec in specs:
        _, metadata = load_spec(spec)
        for seed in seeds:
            for seat in (0, 1):
                print(f"probe {spec['name']} seed={seed} seat={seat}", flush=True)
                def fresh_opponent():
                    return load_spec(spec)[0]
                timing_module = _load(timing_path, f"timing_{spec['name']}_{seed}_{seat}")
                profiles = getattr(timing_module, "V032_PROFILES", []) or []
                forced_profile = profiles[0] if profiles else {
                    "supply_forecast": {item: {"default": 0.0} for item in ("MILK", "STRAWBERRY", "WOOL", "MELON")},
                }
                full = _run_probe(timing_module, fresh_opponent(), seed, seat, forced_profile=forced_profile)
                control_module = _load(control_path, f"control_{spec['name']}_{seed}_{seat}")
                control = _run_game(control_module.agent, fresh_opponent(), seed, seat)
                for index, event in enumerate(full["events"][:max_events]):
                    one_module = _load(timing_path, f"one_{spec['name']}_{seed}_{seat}_{index}")
                    one = _run_probe(one_module, fresh_opponent(), seed, seat,
                                     target_step=int(event["step"]), forced_profile=forced_profile)
                    rows.append({
                        "source_hash": metadata["source_sha256"],
                        "opponent": spec["name"], "seed": int(seed), "seat": int(seat),
                        "event_index": int(index), "step": int(event["step"]),
                        "item": str(event["item"]), "mode": str(event["mode"]),
                        "transfer": int(event["quantity"]), "due": int(event["due"]),
                        "raw_gain": event.get("raw_gain"),
                        "control_margin": float(control["margin"]),
                        "full_timing_margin": float(full["margin"]),
                        "single_event_margin": float(one["margin"]),
                        "actual_margin_delta": float(one["margin"] - control["margin"]),
                        "own_cash_delta": float(one["mine"] - control["mine"]),
                        "opponent_cash_delta": float(one["opponent"] - control["opponent"]),
                        "done": int(control["done"] and one["done"]),
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output), "sources": len(specs)}, indent=2))
    return rows


class R1Probe:
    """Force one anonymous profile while collecting R1's accepted events."""

    def __init__(self, module, target_step=None, profile=None):
        self.module = module
        self.target_step = target_step
        self.profile = profile or {"supply_forecast": {item: {"default": 0.0}
                                                         for item in ("MILK", "STRAWBERRY", "WOOL", "MELON")}}
        self.events = []
        self.gains = []
        self.original_match = module._v032_match_profile
        self.original_choose = module._v032_r1_choose
        self.original_expected = module._v032_r1_expected_gain
        module._v032_match_profile = self._match
        module._v032_r1_choose = self._choose
        module._v032_r1_expected_gain = self._expected

    def _match(self, obs, state, step):
        state["known"] = True
        state["profile"] = self.profile
        return self.profile

    def _expected(self, obs, control_action, candidate_action, profile, item,
                  first_due, second_due, transfer, mode, config):
        raw_scenarios = self.module._v032_r1_raw_gain(
            obs, control_action, candidate_action, profile, item,
            first_due, second_due, transfer, mode, config,
        )
        raw = None if raw_scenarios is None else float(raw_scenarios[1])
        self.gains.append({"step": int(obs.get("step", 0) or 0), "item": str(item),
                           "due": int(first_due), "second_due": int(second_due),
                           "transfer": int(transfer), "mode": str(mode),
                           "raw_gain": raw})
        # R1 calibration is set to zero correction and support gate for data
        # collection; fitting happens after these real rollout features exist.
        return self.original_expected(obs, control_action, candidate_action, profile,
                                      item, first_due, second_due, transfer, mode, config)

    def _choose(self, obs, action, state, step, profile, config):
        if self.target_step is not None and int(step) != int(self.target_step):
            return action
        before = copy.deepcopy(state.get("pending"))
        result = self.original_choose(obs, action, state, step, profile, config)
        after = state.get("pending")
        if after and not before:
            matching = [row for row in self.gains if row["step"] == int(step)
                        and row["item"] == str(after["item"])
                        and row["transfer"] == int(after["quantity"])
                        and row["due"] == int(after["due"])
                        and row["mode"] == str(after["mode"])]
            event = dict(after)
            event.update({"step": int(step), "raw_gain": matching[-1]["raw_gain"] if matching else None})
            self.events.append(event)
        return result

    def close(self):
        self.module._v032_match_profile = self.original_match
        self.module._v032_r1_choose = self.original_choose
        self.module._v032_r1_expected_gain = self.original_expected


def _run_r1_probe(module, opponent, seed, seat, target_step=None, profile=None):
    probe = R1Probe(module, target_step=target_step, profile=profile)
    result = _run_game(probe.module.agent, opponent, seed, seat)
    result["events"] = probe.events
    result["gain_calls"] = probe.gains
    probe.close()
    return result


def collect_r1(opponent_names, seeds, max_events, output,
               timing_path=DEFAULT_R1_TIMING, control_path=DEFAULT_R1_CONTROL):
    specs, _ = unique_loadable_specs()
    if opponent_names:
        wanted = set(opponent_names)
        specs = [row for row in specs if row["name"] in wanted]
    rows = []
    for spec in specs:
        _, metadata = load_spec(spec)
        for seed in seeds:
            for seat in (0, 1):
                print(f"r1 probe {spec['name']} seed={seed} seat={seat}", flush=True)
                profile_module = _load(timing_path, f"profile_{spec['name']}_{seed}_{seat}")
                profiles = getattr(profile_module, "V032_PROFILES", []) or []
                profile = profiles[0] if profiles else None
                # Zero residual + support gate gives an unbiased raw R1
                # estimator for data collection; no timing model is trained in
                # this step.
                profile_module.V032_R1_CALIBRATION = {
                    "global": {"support_groups": 24, "median_residual": 0.0, "coefficients": []}
                }
                control_module = _load(control_path, f"control_{spec['name']}_{seed}_{seat}")
                control = _run_game(control_module.agent, load_spec(spec)[0], seed, seat, capture_opponent=True)
                profile_module.V032_R1_OFFLINE_OPPONENT_SCHEDULE = control.get("opponent_schedule", {})
                full = _run_r1_probe(profile_module, load_spec(spec)[0], seed, seat, profile=profile)
                for index, event in enumerate(full["events"][:max_events]):
                    one_module = _load(timing_path, f"one_{spec['name']}_{seed}_{seat}_{index}")
                    one_module.V032_R1_CALIBRATION = {
                        "global": {"support_groups": 24, "median_residual": 0.0, "coefficients": []}
                    }
                    one_module.V032_R1_OFFLINE_OPPONENT_SCHEDULE = control.get("opponent_schedule", {})
                    one = _run_r1_probe(one_module, load_spec(spec)[0], seed, seat,
                                        target_step=int(event["step"]), profile=profile)
                    rows.append({
                        "source_hash": metadata["source_sha256"], "opponent": spec["name"],
                        "seed": int(seed), "seat": int(seat), "event_index": int(index),
                        "step": int(event["step"]), "item": str(event["item"]),
                        "mode": str(event["mode"]), "transfer": int(event["quantity"]),
                        "due": int(event["due"]), "raw_gain": event.get("raw_gain"),
                        "control_margin": float(control["margin"]),
                        "single_event_margin": float(one["margin"]),
                        "actual_margin_delta": float(one["margin"] - control["margin"]),
                        "own_cash_delta": float(one["mine"] - control["mine"]),
                        "opponent_cash_delta": float(one["opponent"] - control["opponent"]),
                        "done": int(control["done"] and one["done"]),
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output), "sources": len(specs), "estimator": "r1"}, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026])
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--timing", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--estimator", choices=("r1", "old"), default="r1")
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v032_route_conditioned_timing_r1/gain_diagnostics/events.jsonl")
    args = parser.parse_args()
    collector = collect_r1 if args.estimator == "r1" else collect
    if args.estimator == "r1":
        collector(args.opponents, args.seeds, max(1, args.max_events), args.output,
                  args.timing if args.timing != DEFAULT_TIMING else DEFAULT_R1_TIMING,
                  args.control if args.control != DEFAULT_CONTROL else DEFAULT_R1_CONTROL)
    else:
        collector(args.opponents, args.seeds, max(1, args.max_events), args.output,
                  args.timing, args.control)
