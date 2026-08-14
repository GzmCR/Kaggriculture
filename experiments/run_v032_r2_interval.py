"""Run V032-R2 safe interval timing diagnostics.

The runner first captures a normal V27 order-only game, then uses the same
seed and the same opponent action tape for paired control/candidate games.
Only one current-to-next same-item SELL interval is changed per candidate.
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
from v032_r2_interval import (
    R2_MAX_ORDERS,
    R2_SHED_CAPACITY,
    r2_adjust_delay,
    r2_adjust_future,
    r2_local_prediction,
    r2_mandatory_costs,
    r2_reorder_existing_orders,
    r2_safety_gate,
    r2_sell_quantity,
    r2_simulate_interval,
    r2_storage_usage,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "baseline/history/v031_route_market_combo/v27_order_only/main.py"
DEFAULT_OUTPUT = ROOT / "baseline/artifacts/v032_route_conditioned_timing_r2/intervals.jsonl"
GAME_CONFIGURATION = {
    "episodeSteps": 720,
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


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"v032_r2_{tag}_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(agent, obs, config=None):
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


def _copy_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return copy.deepcopy(action)


class _Recorder:
    def __init__(self, function):
        self.function = function
        self.actions = {}
        self.observations = {}

    def __call__(self, obs, config=None):
        step = int(obs.get("step", 0) or 0)
        self.observations[step] = copy.deepcopy(obs)
        action = _copy_action(_call(self.function, obs, config))
        self.actions[step] = copy.deepcopy(action)
        return action


class _FixedTape:
    def __init__(self, actions):
        self.actions = {int(k): copy.deepcopy(v) for k, v in actions.items()}

    def __call__(self, obs, config=None):
        step = int(obs.get("step", 0) or 0)
        return copy.deepcopy(self.actions.get(step, {"farmer": ["PASS"], "hands": [], "market": []}))


class _DelayedRoute:
    def __init__(self, module, start_step, end_step, item, transfer, recorder=None):
        self.module = module
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.item = str(item).upper()
        self.transfer = int(transfer)
        self.recorder = recorder

    def __call__(self, obs, config=None):
        step = int(obs.get("step", 0) or 0)
        action = _copy_action(self.module.agent(obs, config))
        if step == self.start_step:
            market = r2_adjust_delay(action.get("market", []) or [], self.item, self.transfer)
            if market is not None:
                action["market"] = market
                action = self.module._v031_reorder_existing(obs, action)
        elif step == self.end_step:
            market = r2_adjust_future(action.get("market", []) or [], self.item, self.transfer)
            if market is not None:
                action["market"] = market
                action = self.module._v031_reorder_existing(obs, action)
        if self.recorder is not None:
            self.recorder.observations[step] = copy.deepcopy(obs)
            self.recorder.actions[step] = copy.deepcopy(action)
        return action


class _CommitLogger:
    """Temporarily logs actual target SELL commits in the environment."""

    def __init__(self, item):
        self.item = str(item).upper()
        self.rows = []
        self.step = None
        self.module = None
        self.original_process = None
        self.original_commit = None

    def __enter__(self):
        import kaggle_environments.envs.kaggriculture.kaggriculture as module

        self.module = module
        self.original_process = module._process_market
        self.original_commit = module._commit_unit
        owner = self

        def process(state, env):
            owner.step = int(state[0].observation.get("step", 0) or 0)
            return owner.original_process(state, env)

        def commit(op, item, price, farm, private, market, shed_capacity=100):
            player = None
            if op == "SELL" and str(item).upper() == owner.item:
                # _commit_unit receives the farm object, so identify the
                # player from the farm object in the current state when the
                # wrapper is invoked by the engine.
                current_state = getattr(owner, "state", None)
                if current_state is not None:
                    shared_farms = current_state[0].observation.farms
                    for index, shared_farm in enumerate(shared_farms):
                        if shared_farm is farm:
                            player = index
                            break
                owner.rows.append({"step": owner.step, "player": player,
                                   "item": str(item).upper(), "price": float(price)})
            return owner.original_commit(op, item, price, farm, private, market, shed_capacity)

        def process_with_state(state, env):
            owner.state = state
            try:
                return process(state, env)
            finally:
                owner.state = None

        module._process_market = process_with_state
        module._commit_unit = commit
        return self

    def __exit__(self, exc_type, exc, tb):
        self.module._process_market = self.original_process
        self.module._commit_unit = self.original_commit
        return False


def _run(players, seed, item, tag):
    logger = _CommitLogger(item)
    env = make("kaggriculture", configuration={**GAME_CONFIGURATION, "seed": int(seed)}, debug=False)
    with logger:
        env.run(players)
    final = env.steps[-1]
    money = [float(final[i].observation["farms"][i]["money"]) for i in (0, 1)]
    result = {
        "money": money,
        "margin": money[0] - money[1],
        "done": int(all(str(state.status) == "DONE" for state in final)),
        "commits": logger.rows,
        "tag": tag,
    }
    return result


def _capture(route_module, opponent, seed, seat):
    route_rec = _Recorder(route_module.agent)
    opp_rec = _Recorder(opponent)
    players = [route_rec, opp_rec] if int(seat) == 0 else [opp_rec, route_rec]
    env = make("kaggriculture", configuration={**GAME_CONFIGURATION, "seed": int(seed)}, debug=False)
    env.run(players)
    return route_rec, opp_rec, int(seat), int(all(str(state.status) == "DONE" for state in env.steps[-1]))


def _find_events(actions, item, cutoff=648):
    events = []
    steps = sorted(int(step) for step in actions)
    rows = [(step, r2_sell_quantity(actions[step].get("market", []), item))
            for step in steps if step < int(cutoff)]
    rows = [(step, quantity) for step, quantity in rows if quantity > 0]
    for index, (step, quantity) in enumerate(rows[:-1]):
        future_step, future_quantity = rows[index + 1]
        if future_step > step:
            events.append({"start_step": step, "end_step": future_step,
                           "current_quantity": quantity, "future_quantity": future_quantity})
    return events


def _action_maps(route_rec, opp_rec, seat, item, start, end, transfer, route_module):
    # Maps are always indexed by environment player id.
    route_actions = route_rec.actions
    opp_actions = opp_rec.actions
    own_by_step = {}
    opp_by_step = {}
    for step in range(int(start), int(end) + 1):
        route_action = _copy_action(route_actions.get(step, {}))
        opponent_action = _copy_action(opp_actions.get(step, {}))
        if step == int(start):
            changed = r2_adjust_delay(route_action.get("market", []) or [], item, transfer)
            if changed is None:
                return None
            route_action["market"] = changed
            route_action = route_module._v031_reorder_existing(route_rec.observations[step], route_action)
        elif step == int(end):
            changed = r2_adjust_future(route_action.get("market", []) or [], item, transfer)
            if changed is None:
                return None
            route_action["market"] = changed
            route_action = route_module._v031_reorder_existing(route_rec.observations[step], route_action)
        if int(seat) == 0:
            own_by_step[step], opp_by_step[step] = route_action, opponent_action
        else:
            own_by_step[step], opp_by_step[step] = opponent_action, route_action
    return own_by_step, opp_by_step


def _player_action_maps(route_rec, opp_rec, seat, start, end):
    result = [{}, {}]
    for step in range(int(start), int(end) + 1):
        route_action = _copy_action(route_rec.actions.get(step, {}))
        opp_action = _copy_action(opp_rec.actions.get(step, {}))
        if int(seat) == 0:
            result[0][step], result[1][step] = route_action, opp_action
        else:
            result[0][step], result[1][step] = opp_action, route_action
    return result


def _player_obs_maps(route_rec, opp_rec, seat, start, end):
    # Each recorder's private observation belongs to that player; both have
    # the same public market state.
    maps = [{}, {}]
    for step in range(int(start), int(end) + 1):
        route_obs = route_rec.observations.get(step, {})
        opp_obs = opp_rec.observations.get(step, {})
        if int(seat) == 0:
            maps[0][step], maps[1][step] = route_obs, opp_obs
        else:
            maps[0][step], maps[1][step] = opp_obs, route_obs
    return maps


def _order_context(actions, obs_maps, start, end):
    result = {}
    for step in range(int(start), int(end) + 1):
        per_player = []
        for player in (0, 1):
            obs = obs_maps[player].get(step, {}) or {}
            farms = obs.get("farms", []) or []
            farm = farms[player] if len(farms) > player else {}
            hires = int(farm.get("hires_today", 0) or 0)
            unlocked = len(farm.get("unlocked_quadrants", []) or [])
            hire_costs = []
            land_costs = []
            hire_offset = 0
            land_offset = 0
            action = actions[player].get(step, {}) or {}
            for index, order in enumerate(action.get("market", []) or []):
                op = str(order[0]).upper() if isinstance(order, (list, tuple)) and order else ""
                if op == "HIRE":
                    a, b = 0, 1
                    for _ in range(hires + hire_offset):
                        a, b = b, a + b
                    while len(hire_costs) <= index:
                        hire_costs.append(0.0)
                    hire_costs[index] = float(max(1, a))
                    hire_offset += 1
                elif op == "BUY_LAND":
                    land_index = max(0, min(2, unlocked - 1 + land_offset))
                    while len(land_costs) <= index:
                        land_costs.append(0.0)
                    land_costs[index] = float((1000, 2000, 4000)[land_index])
                    land_offset += 1
            per_player.append({"HIRE": hire_costs, "BUY_LAND": land_costs})
        result[step] = per_player
    return result


def _target_rev(rows, start, end, player):
    return sum(float(row["price"]) for row in rows
               if row.get("player") == int(player) and int(start) <= int(row.get("step", -1)) <= int(end))


def _non_market_action(action):
    action = action or {}
    return {"farmer": action.get("farmer", ["PASS"]),
            "hands": action.get("hands", [])}


def _market_without_item(action, item):
    return [list(order) for order in (action or {}).get("market", []) or []
            if not (isinstance(order, (list, tuple)) and len(order) >= 2
                    and str(order[0]).upper() == "SELL"
                    and str(order[1]).upper() == str(item).upper())]


def _action_diff(control_actions, candidate_actions, start, end, item):
    non_target = 0
    farmer_hands = 0
    market = 0
    for step in range(int(start), int(end) + 1):
        left = control_actions.get(step, {}) or {}
        right = candidate_actions.get(step, {}) or {}
        if _non_market_action(left) != _non_market_action(right):
            farmer_hands += 1
        if _market_without_item(left, item) != _market_without_item(right, item):
            non_target += 1
        if left.get("market", []) != right.get("market", []):
            market += 1
    return farmer_hands, non_target, market


def _run_pair(route_path, opp_spec, seed, seat, item, event, transfer):
    # First pass records the route, opponent tape and safety snapshots.
    capture_route = _load(route_path, f"capture_route_{opp_spec['name']}_{seed}_{seat}")
    opponent, metadata = load_spec(opp_spec)
    route_rec, opp_rec, seat, capture_done = _capture(capture_route, opponent, seed, seat)
    start, end = int(event["start_step"]), int(event["end_step"])
    player_actions = _player_action_maps(route_rec, opp_rec, seat, start, end)
    player_obs = _player_obs_maps(route_rec, opp_rec, seat, start, end)
    obs_start = player_obs[0][start]
    market = obs_start.get("market", {}) or {}
    start_inventory = dict(market.get("inventory", {}) or {})
    start_money = [float((player_obs[p][start].get("farms", [{}, {}])[p] or {}).get("money", 0)) for p in (0, 1)]
    order_context = _order_context(player_actions, player_obs, start, end)
    shops = list((obs_start.get("town", {}) or {}).get("unlocked_shops", []) or [])
    sheds_by_step = [{}, {}]
    for p in (0, 1):
        for step in range(start, end + 1):
            sheds_by_step[p][step] = dict(((player_obs[p][step].get("private", {}) or {}).get("shed", {}) or {}))
    control_orders = player_actions
    candidate_maps = _action_maps(route_rec, opp_rec, seat, item, start, end, transfer, capture_route)
    if candidate_maps is None:
        return None
    candidate_orders = [candidate_maps[0], candidate_maps[1]]
    control_market_orders = {
        player: {step: (action.get("market", []) or []) for step, action in control_orders[player].items()}
        for player in (0, 1)
    }
    candidate_market_orders = {
        player: {step: (action.get("market", []) or []) for step, action in candidate_orders[player].items()}
        for player in (0, 1)
    }
    common_kwargs = {
        "start_inventory": start_inventory,
        "start_money": start_money,
        "sheds_by_step": sheds_by_step,
        "start_step": start,
        "end_step": end,
        "target_item": item,
        "shops": shops,
        "shop_interval": GAME_CONFIGURATION["townShopSellInterval"],
        "center_interval": GAME_CONFIGURATION["townCenterSellInterval"],
        "max_orders": R2_MAX_ORDERS,
        "shed_capacity": R2_SHED_CAPACITY,
        "order_context_by_step": order_context,
    }

    # Run the paired control before the shadow rollout.  A route may execute
    # DROP/HARVEST before the market in the same turn, while the callback's
    # private.shed snapshot is from before those unit actions.  The actual
    # control commit ledger supplies only the inventory that was demonstrably
    # available in that same step; it does not preload any future production.
    opponent_tape = opp_rec.actions
    control_route = _load(route_path, f"control_{opp_spec['name']}_{seed}_{seat}_{item}_{start}_{transfer}")
    control_candidate_rec = _Recorder(control_route.agent)
    fixed_opponent = _FixedTape(opponent_tape)
    control_players = [control_candidate_rec, fixed_opponent] if seat == 0 else [fixed_opponent, control_candidate_rec]
    control_actual = _run(control_players, seed, item, "control")
    control_available = {}
    for row in control_actual["commits"]:
        step = int(row.get("step", -1))
        player = row.get("player")
        if step < start or step > end or player not in (0, 1):
            continue
        control_available[(int(player), step)] = control_available.get((int(player), step), 0) + 1
    availability_reasons = []
    if control_available.get((seat, start), 0) < int(event["current_quantity"]):
        availability_reasons.append("current_target_not_fully_executed")
    if control_available.get((seat, end), 0) < int(event["future_quantity"]):
        availability_reasons.append("future_target_not_fully_executed")
    for p in (0, 1):
        for step in range(start, end + 1):
            available = control_available.get((p, step), 0)
            try:
                current = int(sheds_by_step[p][step].get(str(item).upper(), 0) or 0)
            except (TypeError, ValueError):
                current = 0
            sheds_by_step[p][step][str(item).upper()] = max(current, available)

    control_sim = r2_simulate_interval(
        orders_by_step=control_market_orders[0], opponent_orders_by_step=control_market_orders[1],
        **common_kwargs,
    )
    candidate_sim = r2_simulate_interval(
        orders_by_step=candidate_market_orders[0], opponent_orders_by_step=candidate_market_orders[1],
        extra_player=seat, extra_units=transfer,
        **common_kwargs,
    )
    prediction = r2_local_prediction(control_sim, candidate_sim, seat)
    mandatory = r2_mandatory_costs(control_orders[seat], player_obs[seat], start, end, player=seat)
    safe, safety_reasons = r2_safety_gate(
        control_sim, candidate_sim, player_obs[seat], item, transfer, start, end,
        mandatory_costs=mandatory, player=seat,
        control_obs_by_step=player_obs[seat],
        immediate_cash_delta=(candidate_sim["target_trace"].get(start, [0.0, 0.0])[seat]
                              - control_sim["target_trace"].get(start, [0.0, 0.0])[seat]),
    )

    # The candidate calls the live V27 route, then applies only this event's
    # quantity change.  The opponent remains on the captured action tape.
    timing_route = _load(route_path, f"candidate_{opp_spec['name']}_{seed}_{seat}_{item}_{start}_{transfer}")
    timing_rec = _Recorder(lambda obs, config=None: {"farmer": ["PASS"], "hands": [], "market": []})
    timing_agent = _DelayedRoute(timing_route, start, end, item, transfer, timing_rec)
    candidate_players = [timing_agent, fixed_opponent] if seat == 0 else [fixed_opponent, timing_agent]
    candidate_actual = _run(candidate_players, seed, item, "candidate")
    actual_interval_own = _target_rev(candidate_actual["commits"], start, end, seat) - _target_rev(control_actual["commits"], start, end, seat)
    actual_interval_opp = _target_rev(candidate_actual["commits"], start, end, 1 - seat) - _target_rev(control_actual["commits"], start, end, 1 - seat)
    actual_interval_margin = actual_interval_own - actual_interval_opp
    actual_final_margin = float(candidate_actual["margin"] - control_actual["margin"])
    farmer_hands_diff, non_target_diff, market_diff = _action_diff(
        control_candidate_rec.actions, timing_rec.actions, start, end, item,
    )
    if availability_reasons:
        safe = False
        safety_reasons.extend(reason for reason in availability_reasons if reason not in safety_reasons)
    if farmer_hands_diff or non_target_diff:
        safe = False
        safety_reasons.append("non_target_route_diff")
    return {
        "opponent": opp_spec["name"],
        "source_hash": metadata.get("source_sha256", ""),
        "seed": int(seed), "seat": int(seat), "item": str(item),
        "start_step": start, "end_step": end,
        "current_quantity": int(event["current_quantity"]),
        "future_quantity": int(event["future_quantity"]),
        "transfer": int(transfer),
        "capture_done": int(capture_done),
        "predicted_local_margin_delta": float(prediction["predicted_local_margin_delta"]),
        "predicted_own_delta": float(prediction["own_delta"]),
        "predicted_opponent_delta": float(prediction["opponent_delta"]),
        "predicted_control_target_revenue": list(control_sim["target_revenue"]),
        "predicted_candidate_target_revenue": list(candidate_sim["target_revenue"]),
        "actual_interval_margin_delta": float(actual_interval_margin),
        "actual_interval_own_delta": float(actual_interval_own),
        "actual_interval_opponent_delta": float(actual_interval_opp),
        "actual_final_margin_delta": actual_final_margin,
        "control_target_commits": [row for row in control_actual["commits"] if start <= int(row.get("step", -1)) <= end],
        "candidate_target_commits": [row for row in candidate_actual["commits"] if start <= int(row.get("step", -1)) <= end],
        "control_margin": float(control_actual["margin"]),
        "candidate_margin": float(candidate_actual["margin"]),
        "safe": int(bool(safe)),
        "safety_reasons": list(safety_reasons),
        "control_failed": list(control_sim["failed"]),
        "candidate_failed": list(candidate_sim["failed"]),
        "candidate_actual_done": int(candidate_actual["done"]),
        "control_actual_done": int(control_actual["done"]),
        "farmer_hands_action_diff": int(farmer_hands_diff),
        "non_target_market_action_diff": int(non_target_diff),
        "market_action_diff": int(market_diff),
        "max_control_storage": max(r2_storage_usage(player_obs[seat][step]) for step in range(start, end + 1)),
        "max_candidate_storage_bound": max(r2_storage_usage(player_obs[seat][step]) + (transfer if step > start else 0) for step in range(start, end + 1)),
        "mandatory_cost_total": float(sum(mandatory.values())),
        "candidate_min_money_interval": float(min(trace[seat] for trace in candidate_sim["money_trace"].values())) if candidate_sim["money_trace"] else None,
    }


def collect(opponent_names, seeds, items, max_events, ratios, output,
            route_path=ROUTE_PATH):
    specs, _ = unique_loadable_specs()
    if opponent_names:
        wanted = set(opponent_names)
        specs = [row for row in specs if row["name"] in wanted]
    rows = []
    for spec in specs:
        for seed in seeds:
            for seat in (0, 1):
                route_module = _load(route_path, f"events_{spec['name']}_{seed}_{seat}")
                opponent, _ = load_spec(spec)
                route_rec, _, _, _ = _capture(route_module, opponent, seed, seat)
                for item in items:
                    events = _find_events(route_rec.actions, item)[:max(1, int(max_events))]
                    for event in events:
                        for ratio in ratios:
                            transfer = max(1, int(float(event["current_quantity"]) * float(ratio) + 0.5))
                            if transfer >= int(event["current_quantity"]):
                                transfer = int(event["current_quantity"]) - 1
                            if transfer <= 0:
                                continue
                            print(f"r2 {spec['name']} seed={seed} seat={seat} {item} {event['start_step']}->{event['end_step']} u={transfer}", flush=True)
                            row = _run_pair(route_path, spec, seed, seat, item, event, transfer)
                            if row is not None:
                                row["ratio"] = float(ratio)
                                rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output), "opponents": len(specs)}, indent=2))
    return rows


def summarize(rows):
    return _summarize(rows, include_items=True)


def _summarize(rows, include_items):
    safe_rows = [row for row in rows if row.get("safe")]
    def stats(values):
        if not values:
            return {"n": 0}
        mean = sum(values) / len(values)
        mae = sum(abs(value) for value in values) / len(values)
        return {"n": len(values), "mean": mean, "mae": mae}
    errors = [float(row["predicted_local_margin_delta"]) - float(row["actual_interval_margin_delta"]) for row in safe_rows]
    signs = [int((float(row["predicted_local_margin_delta"]) > 0) == (float(row["actual_interval_margin_delta"]) > 0)) for row in safe_rows if float(row["predicted_local_margin_delta"]) != 0 and float(row["actual_interval_margin_delta"]) != 0]
    result = {
        "rows": len(rows),
        "safe_rows": len(safe_rows),
        "safe_rate": len(safe_rows) / len(rows) if rows else 0.0,
        "prediction_error": stats(errors),
        "sign_accuracy": sum(signs) / len(signs) if signs else None,
        "positive_prediction_negative_actual": sum(1 for row in safe_rows if float(row["predicted_local_margin_delta"]) > 0 and float(row["actual_interval_margin_delta"]) < 0),
        "mean_actual_final_margin_delta_safe": (sum(float(row["actual_final_margin_delta"]) for row in safe_rows) / len(safe_rows)) if safe_rows else None,
    }
    if include_items:
        result["by_item"] = {
            item: _summarize([row for row in rows if row.get("item") == item], include_items=False)
            for item in sorted({str(row.get("item")) for row in rows})
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponents", nargs="*", default=["v27_current", "v13_r3", "adaptive_replay", "frontier_current"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026, 217, 317, 733])
    parser.add_argument("--items", nargs="+", default=["MILK", "STRAWBERRY"])
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.25, 0.50])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--route", type=Path, default=ROUTE_PATH)
    args = parser.parse_args()
    rows = collect(args.opponents, args.seeds, args.items, args.max_events, args.ratios, args.output, args.route)
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summarize(rows), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summarize(rows), indent=2))
