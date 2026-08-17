"""Run V032-R3 paired real-engine timing experiments.

The runner keeps V27 order-only as the route and fixes the opponent action
tape.  Each candidate changes exactly one target-product event:

* ADVANCE: add already-shed inventory at ``t-h`` and subtract it at ``t``;
* DELAY: use the V032-R2 current-to-next-event transfer.

Rows that cannot pass the real shed pre-check are recorded as SKIPPED instead
of inventing a candidate game.  Evaluated rows run a fresh control and
candidate Kaggriculture game under the same seed and fixed opponent tape.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

from rl_010_opponents import load_spec, unique_loadable_specs
from v032_r3_bidirectional import (
    R3_CUTOFF,
    R3_FRACTIONS,
    R3_HORIZONS,
    R3_ITEMS,
    R3_MAX_ORDERS,
    R3_SHED_CAPACITY,
    r2_adjust_delay,
    r2_adjust_future,
    r2_local_prediction,
    r2_mandatory_costs,
    r2_reorder_existing_orders,
    r2_safety_gate,
    r2_simulate_interval,
    r3_adjust_sell,
    r3_available_extra_inventory,
    r3_copy_action,
    r3_find_advance_events,
    r3_find_delay_events,
    r3_non_target_signature,
    r3_quantity_candidates,
    r3_sell_quantity,
    r3_target_inventory_after_actions,
)

import run_v032_r2_interval as r2runner


ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "baseline/history/v031_route_market_combo/v27_order_only/main.py"
DEFAULT_OUTPUT = ROOT / "baseline/artifacts/v032_r3_bidirectional/r3.jsonl"
GAME_CONFIGURATION = dict(r2runner.GAME_CONFIGURATION)


def _load_module(path, tag):
    spec = importlib.util.spec_from_file_location(
        f"v032_r3_{tag}_{time.time_ns()}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy(action):
    return r3_copy_action(action)


def _player_action_maps(route_rec, opp_rec, seat, start, end):
    result = [{}, {}]
    for step in range(int(start), int(end) + 1):
        route_action = _copy(route_rec.actions.get(step, {}))
        opponent_action = _copy(opp_rec.actions.get(step, {}))
        if int(seat) == 0:
            result[0][step], result[1][step] = route_action, opponent_action
        else:
            result[0][step], result[1][step] = opponent_action, route_action
    return result


def _player_obs_maps(route_rec, opp_rec, seat, start, end):
    result = [{}, {}]
    for step in range(int(start), int(end) + 1):
        route_obs = route_rec.observations.get(step, {})
        opponent_obs = opp_rec.observations.get(step, {})
        if int(seat) == 0:
            result[0][step], result[1][step] = route_obs, opponent_obs
        else:
            result[0][step], result[1][step] = opponent_obs, route_obs
    return result


def _market_maps(actions):
    return {
        player: {
            int(step): (action.get("market", []) or [])
            for step, action in actions[player].items()
        }
        for player in (0, 1)
    }


def _prepare_sheds(player_actions, player_obs, start, end, item):
    """Use observed shed plus same-turn DROP/PLACE availability.

    This is still conservative: HARVEST is not treated as shed inventory,
    because harvested units remain in a farmer/hand inventory until a later
    DROP/PLACE action.
    """
    result = [{}, {}]
    item = str(item).upper()
    for player in (0, 1):
        for step in range(int(start), int(end) + 1):
            obs = player_obs[player].get(step, {}) or {}
            private = obs.get("private", {}) or {}
            shed = dict((private.get("shed", {}) or {}))
            action = player_actions[player].get(step, {}) or {}
            state = r3_target_inventory_after_actions(
                obs, action, item, GAME_CONFIGURATION
            )
            shed[item] = max(
                int(shed.get(item, 0) or 0),
                int(state.get("shed_after_actions", 0) or 0),
            )
            result[player][step] = shed
    return result


def _order_context(actions, obs_maps, start, end):
    return r2runner._order_context(actions, obs_maps, start, end)


def _apply_candidate_action(route_module, action, obs, item, kind, transfer,
                            is_start):
    action = _copy(action)
    delta = 0
    if kind == "ADVANCE":
        delta = int(transfer) if is_start else -int(transfer)
    elif kind == "DELAY":
        delta = -int(transfer) if is_start else int(transfer)
    else:
        raise ValueError(f"unknown timing kind: {kind}")
    changed = r3_adjust_sell(action, item, delta, R3_MAX_ORDERS)
    if changed is None:
        return None
    return route_module._v031_reorder_existing(obs, changed)


def _candidate_maps(route_module, route_rec, opp_rec, seat, item, kind,
                    start, end, transfer):
    result = [{}, {}]
    for step in range(int(start), int(end) + 1):
        route_action = _copy(route_rec.actions.get(step, {}))
        opponent_action = _copy(opp_rec.actions.get(step, {}))
        if step == int(start) or step == int(end):
            route_obs = route_rec.observations.get(step, {}) or {}
            route_action = _apply_candidate_action(
                route_module, route_action, route_obs, item, kind,
                transfer, is_start=(step == int(start)),
            )
            if route_action is None:
                return None
        if int(seat) == 0:
            result[0][step], result[1][step] = route_action, opponent_action
        else:
            result[0][step], result[1][step] = opponent_action, route_action
    return result


class _TimingRoute:
    def __init__(self, module, kind, start, end, item, transfer, recorder=None):
        self.module = module
        self.kind = str(kind)
        self.start = int(start)
        self.end = int(end)
        self.item = str(item).upper()
        self.transfer = int(transfer)
        self.recorder = recorder

    def __call__(self, obs, config=None):
        step = int(obs.get("step", 0) or 0)
        action = _copy(self.module.agent(obs, config))
        if step == self.start or step == self.end:
            action = _apply_candidate_action(
                self.module, action, obs, self.item, self.kind,
                self.transfer, is_start=(step == self.start),
            )
            if action is None:
                action = {"farmer": ["PASS"], "hands": [], "market": []}
        if self.recorder is not None:
            self.recorder.observations[step] = copy.deepcopy(obs)
            self.recorder.actions[step] = copy.deepcopy(action)
        return action


def _commit_count(rows, player, step):
    return sum(
        1 for row in rows
        if row.get("player") == int(player) and int(row.get("step", -1)) == int(step)
    )


def _target_revenue(rows, start, end, player):
    return r2runner._target_rev(rows, start, end, player)


def _target_prices(rows, start, end, player):
    return [
        float(row["price"])
        for row in rows
        if row.get("player") == int(player)
        and int(start) <= int(row.get("step", -1)) <= int(end)
    ]


def _action_diffs(control_actions, candidate_actions, start, end, item):
    farmer_hands = 0
    non_target = 0
    market = 0
    for step in range(int(start), int(end) + 1):
        left = control_actions.get(step, {}) or {}
        right = candidate_actions.get(step, {}) or {}
        if {"farmer": left.get("farmer", ["PASS"]),
            "hands": left.get("hands", [])} != {
                "farmer": right.get("farmer", ["PASS"]),
                "hands": right.get("hands", []),
        }:
            farmer_hands += 1
        if r3_non_target_signature(left, item) != r3_non_target_signature(right, item):
            non_target += 1
        if left.get("market", []) != right.get("market", []):
            market += 1
    return farmer_hands, non_target, market


def _overflow(actions, start, end):
    return sum(
        1 for step in range(int(start), int(end) + 1)
        if len((actions.get(step, {}) or {}).get("market", []) or []) > R3_MAX_ORDERS
    )


def _skip_row(opponent_name, source_hash, seed, seat, item, event, transfer,
              reason, available=None):
    if isinstance(reason, (list, tuple, set)):
        reasons = [str(value) for value in reason]
    else:
        reasons = [str(reason)]
    return {
        "status": "SKIPPED",
        "opponent": opponent_name,
        "source_hash": source_hash,
        "seed": int(seed),
        "seat": int(seat),
        "item": str(item).upper(),
        "kind": event["kind"],
        "start_step": int(event["start_step"]),
        "end_step": int(event["end_step"]),
        "horizon": int(event["horizon"]),
        "current_quantity": int(event.get("current_quantity", 0)),
        "future_quantity": int(event.get("future_quantity", 0)),
        "transfer": int(transfer),
        "available_extra_inventory": available,
        "safe": 0,
        "safety_reasons": sorted(set(reasons)),
        "predicted_local_margin_delta": None,
        "actual_interval_margin_delta": None,
        "actual_final_margin_delta": None,
    }


def _expected_counts(kind, control_start, control_end, transfer):
    if kind == "ADVANCE":
        return control_start + int(transfer), control_end - int(transfer)
    return control_start - int(transfer), control_end + int(transfer)


def _new_cache():
    return {
        "captures": {},
        "controls": {},
        "stats": {
            "capture_runs": 0,
            "opponent_loads": 0,
            "control_runs": 0,
            "candidate_runs": 0,
            "capture_cache_hits": 0,
            "control_cache_hits": 0,
        },
    }


def _spec_cache_key(opponent_spec):
    return str(
        opponent_spec.get("source_sha256")
        or opponent_spec.get("name")
        or opponent_spec.get("path")
    )


def _row_key(row):
    """Return the stable identity of one paired counterfactual row.

    A row is uniquely determined by the opponent source, game condition, and
    one route event/transfer.  Results and diagnostics are deliberately not
    part of the key, so a partially written JSONL can be resumed safely.
    """
    def field(name, default):
        value = row.get(name)
        return default if value is None else int(value)

    return (
        str(row.get("source_hash") or row.get("opponent") or ""),
        field("seed", 0),
        field("seat", 0),
        str(row.get("item", "")).upper(),
        str(row.get("kind", "")).upper(),
        field("start_step", -1),
        field("end_step", -1),
        field("horizon", 0),
        field("transfer", 0),
    )


def _event_key(opponent_spec, seed, seat, item, event, transfer):
    return _row_key({
        "source_hash": _spec_cache_key(opponent_spec),
        "seed": seed,
        "seat": seat,
        "item": item,
        "kind": event["kind"],
        "start_step": event["start_step"],
        "end_step": event["end_step"],
        "horizon": event["horizon"],
        "transfer": transfer,
    })


def _load_existing_rows(path):
    """Load completed rows and identify a possibly incomplete final line.

    The runner writes one JSON object per line.  If the process is interrupted
    while writing the final object, that tail is discarded before appending;
    malformed interior lines are treated as corruption instead of silently
    changing the experiment.
    """
    path = Path(path)
    if not path.exists():
        return [], set(), 0, 0
    data = path.read_bytes()
    rows = []
    keys = set()
    malformed = 0
    valid_end = 0
    offset = 0
    for line in data.splitlines(keepends=True):
        line_end = offset + len(line)
        stripped = line.strip()
        if not stripped:
            valid_end = line_end
            offset = line_end
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            # A final line without a complete JSON object is the normal
            # interruption case.  Anything before it needs human attention.
            if line_end == len(data):
                malformed += 1
                break
            raise RuntimeError(
                f"malformed JSONL line {len(rows) + malformed + 1} in {path}"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row is not an object in {path}")
        key = _row_key(row)
        if key not in keys:
            rows.append(row)
            keys.add(key)
        valid_end = line_end
        offset = line_end
    return rows, keys, valid_end, malformed


def _capture_cached(route_path, opponent_spec, seed, seat, cache):
    """Capture one normal game per opponent/seed/seat.

    The captured observations and actions are immutable inputs for all item,
    event, horizon, and transfer candidates in this run.  Reusing them does
    not change the paired-game semantics; it only removes duplicate 720-turn
    capture games.
    """
    key = (str(route_path), _spec_cache_key(opponent_spec), int(seed), int(seat))
    if key in cache["captures"]:
        cache["stats"]["capture_cache_hits"] += 1
        return cache["captures"][key]
    route_module = _load_module(route_path, f"route_{seed}_{seat}")
    opponent, metadata = load_spec(opponent_spec)
    cache["stats"]["opponent_loads"] += 1
    route_rec, opp_rec, actual_seat, capture_done = r2runner._capture(
        route_module, opponent, seed, seat
    )
    entry = {
        "route_module": route_module,
        "route_rec": route_rec,
        "opp_rec": opp_rec,
        "seat": actual_seat,
        "capture_done": capture_done,
        "metadata": metadata,
    }
    cache["captures"][key] = entry
    cache["stats"]["capture_runs"] += 1
    return entry


def _control_cached(route_path, opponent_spec, seed, seat, item,
                    opp_rec, cache):
    """Run one V27 control game per opponent/seed/seat/item."""
    key = (str(route_path), _spec_cache_key(opponent_spec), int(seed), int(seat), str(item).upper())
    if key in cache["controls"]:
        cache["stats"]["control_cache_hits"] += 1
        return cache["controls"][key]
    control_module = _load_module(route_path, f"control_{seed}_{seat}_{item}")
    control_rec = r2runner._Recorder(control_module.agent)
    fixed_opponent = r2runner._FixedTape(opp_rec.actions)
    control_players = (
        [control_rec, fixed_opponent] if int(seat) == 0
        else [fixed_opponent, control_rec]
    )
    control_actual = r2runner._run(control_players, seed, item, "r3_control")
    entry = {"actual": control_actual, "rec": control_rec}
    cache["controls"][key] = entry
    cache["stats"]["control_runs"] += 1
    return entry


def run_pair(route_path, opponent_spec, seed, seat, item, event, transfer,
             cache=None, skip_unsafe_candidates=False):
    cache = cache or _new_cache()
    captured = _capture_cached(route_path, opponent_spec, seed, seat, cache)
    route_module = captured["route_module"]
    route_rec = captured["route_rec"]
    opp_rec = captured["opp_rec"]
    seat = int(captured["seat"])
    capture_done = captured["capture_done"]
    metadata = captured["metadata"]
    item = str(item).upper()
    kind = str(event["kind"]).upper()
    start, end = int(event["start_step"]), int(event["end_step"])
    source_hash = metadata.get("source_sha256", "")
    if not capture_done:
        return _skip_row(opponent_spec["name"], source_hash, seed, seat, item,
                         event, transfer, "capture_not_done")

    own_start_action = route_rec.actions.get(start, {}) or {}
    available = r3_available_extra_inventory(
        route_rec.observations.get(start, {}) or {},
        own_start_action,
        item,
        GAME_CONFIGURATION,
    )
    if kind == "ADVANCE" and int(transfer) > int(available):
        return _skip_row(opponent_spec["name"], source_hash, seed, seat, item,
                         event, transfer, "no_warehouse_inventory", available)

    player_actions = _player_action_maps(route_rec, opp_rec, seat, start, end)
    player_obs = _player_obs_maps(route_rec, opp_rec, seat, start, end)
    candidate_maps = _candidate_maps(
        route_module, route_rec, opp_rec, seat, item, kind,
        start, end, transfer,
    )
    if candidate_maps is None:
        return _skip_row(opponent_spec["name"], source_hash, seed, seat, item,
                         event, transfer, "action_transform_invalid", available)

    control_market = _market_maps(player_actions)
    candidate_market = _market_maps(candidate_maps)
    obs_start = player_obs[0][start]
    market = obs_start.get("market", {}) or {}
    start_inventory = dict(market.get("inventory", {}) or {})
    start_prices = dict(market.get("prices", {}) or {})
    start_money = [
        float((player_obs[p][start].get("farms", [{}, {}])[p] or {}).get("money", 0))
        for p in (0, 1)
    ]
    sheds_by_step = _prepare_sheds(player_actions, player_obs, start, end, item)
    order_context = _order_context(player_actions, player_obs, start, end)
    shops = list((obs_start.get("town", {}) or {}).get("unlocked_shops", []) or [])
    common = {
        "start_inventory": start_inventory,
        "start_money": start_money,
        "sheds_by_step": sheds_by_step,
        "start_step": start,
        "end_step": end,
        "target_item": item,
        "shops": shops,
        "shop_interval": GAME_CONFIGURATION["townShopSellInterval"],
        "center_interval": GAME_CONFIGURATION["townCenterSellInterval"],
        "max_orders": R3_MAX_ORDERS,
        "shed_capacity": R3_SHED_CAPACITY,
        "order_context_by_step": order_context,
    }

    control_sim = r2_simulate_interval(
        orders_by_step=control_market[0],
        opponent_orders_by_step=control_market[1],
        **common,
    )
    candidate_sim = r2_simulate_interval(
        orders_by_step=candidate_market[0],
        opponent_orders_by_step=candidate_market[1],
        extra_player=(int(seat) if kind == "DELAY" else None),
        extra_units=(int(transfer) if kind == "DELAY" else 0),
        **common,
    )
    prediction = r2_local_prediction(control_sim, candidate_sim, seat)

    safety_reasons = []
    mandatory = r2_mandatory_costs(
        player_actions[seat], player_obs[seat], start, end, player=seat
    )
    if kind == "DELAY":
        immediate_delta = (
            candidate_sim["target_trace"].get(start, [0.0, 0.0])[seat]
            - control_sim["target_trace"].get(start, [0.0, 0.0])[seat]
        )
        safe_delay, delay_reasons = r2_safety_gate(
            control_sim, candidate_sim, player_obs[seat], item, transfer,
            start, end, mandatory_costs=mandatory, player=seat,
            control_obs_by_step=player_obs[seat],
            immediate_cash_delta=immediate_delta,
        )
        if not safe_delay:
            safety_reasons.extend(delay_reasons)

    if safety_reasons and skip_unsafe_candidates:
        return _skip_row(
            opponent_spec["name"], source_hash, seed, seat, item,
            event, transfer, safety_reasons, available,
        )

    control_entry = _control_cached(
        route_path, opponent_spec, seed, seat, item, opp_rec, cache
    )
    control_actual = control_entry["actual"]
    control_rec = control_entry["rec"]

    control_start = _commit_count(control_actual["commits"], seat, start)
    control_end = _commit_count(control_actual["commits"], seat, end)
    route_start_quantity = r3_sell_quantity(own_start_action, item)
    route_end_quantity = r3_sell_quantity(route_rec.actions.get(end, {}), item)
    expected_start, expected_end = _expected_counts(
        kind, control_start, control_end, transfer
    )

    timing_module = _load_module(route_path, f"candidate_{seed}_{seat}_{item}")
    timing_rec = r2runner._Recorder(lambda obs, config=None: {
        "farmer": ["PASS"], "hands": [], "market": []
    })
    timing_agent = _TimingRoute(
        timing_module, kind, start, end, item, transfer, timing_rec
    )
    fixed_opponent = r2runner._FixedTape(opp_rec.actions)
    candidate_players = (
        [timing_agent, fixed_opponent] if int(seat) == 0
        else [fixed_opponent, timing_agent]
    )
    cache["stats"]["candidate_runs"] += 1
    candidate_actual = r2runner._run(candidate_players, seed, item, "r3_candidate")

    candidate_start = _commit_count(candidate_actual["commits"], seat, start)
    candidate_end = _commit_count(candidate_actual["commits"], seat, end)
    if control_start < route_start_quantity:
        safety_reasons.append("control_start_target_not_fully_executed")
    if control_end < route_end_quantity:
        safety_reasons.append("control_end_target_not_fully_executed")
    if candidate_start != expected_start:
        safety_reasons.append("candidate_start_quantity_mismatch")
    if candidate_end != expected_end:
        safety_reasons.append("candidate_end_quantity_mismatch")
    if not control_actual["done"] or not candidate_actual["done"]:
        safety_reasons.append("game_not_done")

    farmer_hands_diff, non_target_diff, market_diff = _action_diffs(
        control_rec.actions, timing_rec.actions, start, end, item
    )
    if farmer_hands_diff:
        safety_reasons.append("farmer_hands_action_diff")
    if non_target_diff:
        safety_reasons.append("non_target_market_change")
    if _overflow(timing_rec.actions, start, end):
        safety_reasons.append("market_order_overflow")

    actual_interval_own = (
        _target_revenue(candidate_actual["commits"], start, end, seat)
        - _target_revenue(control_actual["commits"], start, end, seat)
    )
    other = 1 - int(seat)
    actual_interval_opp = (
        _target_revenue(candidate_actual["commits"], start, end, other)
        - _target_revenue(control_actual["commits"], start, end, other)
    )
    actual_interval_margin = actual_interval_own - actual_interval_opp
    candidate_money = candidate_actual["money"]
    control_money = control_actual["money"]
    actual_final_margin = (
        (candidate_money[seat] - candidate_money[other])
        - (control_money[seat] - control_money[other])
    )
    obs_end = player_obs[0].get(end, {}) or {}
    end_market = obs_end.get("market", {}) or {}
    control_prices_own = _target_prices(control_actual["commits"], start, end, seat)
    candidate_prices_own = _target_prices(candidate_actual["commits"], start, end, seat)
    control_prices_opp = _target_prices(control_actual["commits"], start, end, other)
    candidate_prices_opp = _target_prices(candidate_actual["commits"], start, end, other)
    result = {
        "status": "EVALUATED",
        "opponent": opponent_spec["name"],
        "source_hash": source_hash,
        "seed": int(seed), "seat": int(seat), "item": item,
        "kind": kind,
        "start_step": start, "end_step": end,
        "horizon": int(event["horizon"]),
        "current_quantity": int(event.get("current_quantity", 0)),
        "future_quantity": int(event.get("future_quantity", 0)),
        "transfer": int(transfer),
        "available_extra_inventory": int(available),
        "start_market_inventory": int(start_inventory.get(item, 0) or 0),
        "start_market_price": int(start_prices.get(item, 0) or 0),
        "end_market_inventory": int((end_market.get("inventory", {}) or {}).get(item, 0) or 0),
        "end_market_price": int((end_market.get("prices", {}) or {}).get(item, 0) or 0),
        "capture_done": int(capture_done),
        "predicted_local_margin_delta": float(prediction["predicted_local_margin_delta"]),
        "predicted_own_delta": float(prediction["own_delta"]),
        "predicted_opponent_delta": float(prediction["opponent_delta"]),
        "predicted_control_target_revenue": list(control_sim["target_revenue"]),
        "predicted_candidate_target_revenue": list(candidate_sim["target_revenue"]),
        "actual_interval_margin_delta": float(actual_interval_margin),
        "actual_interval_own_delta": float(actual_interval_own),
        "actual_interval_opponent_delta": float(actual_interval_opp),
        "actual_final_margin_delta": float(actual_final_margin),
        "control_own_target_prices": control_prices_own,
        "candidate_own_target_prices": candidate_prices_own,
        "control_opponent_target_prices": control_prices_opp,
        "candidate_opponent_target_prices": candidate_prices_opp,
        "control_start_commits": int(control_start),
        "control_end_commits": int(control_end),
        "candidate_start_commits": int(candidate_start),
        "candidate_end_commits": int(candidate_end),
        "route_start_quantity": int(route_start_quantity),
        "route_end_quantity": int(route_end_quantity),
        "safe": int(not safety_reasons),
        "safety_reasons": sorted(set(safety_reasons)),
        "control_actual_done": int(control_actual["done"]),
        "candidate_actual_done": int(candidate_actual["done"]),
        "farmer_hands_action_diff": int(farmer_hands_diff),
        "non_target_market_action_diff": int(non_target_diff),
        "market_action_diff": int(market_diff),
        "control_failed": list(control_sim["failed"]),
        "candidate_failed": list(candidate_sim["failed"]),
        "mandatory_cost_total": float(sum(mandatory.values())),
    }
    return result


def _specs(names):
    specs, _ = unique_loadable_specs()
    if not names:
        return specs
    wanted = set(names)
    return [spec for spec in specs if spec["name"] in wanted]


def _route_actions():
    module = _load_module(ROUTE_PATH, "static_route")
    actions = getattr(module, "_ACTIONS", None)
    if actions is None:
        raise RuntimeError("V27 route does not expose _ACTIONS")
    return actions


def _events_for_mode(actions, item, mode, min_step, max_step, max_events):
    if mode == "advance":
        events = r3_find_advance_events(
            actions, item, R3_HORIZONS, R3_CUTOFF, min_step, max_step
        )
    elif mode == "combined":
        events = r3_find_advance_events(
            actions, item, R3_HORIZONS, R3_CUTOFF, min_step, max_step
        )
        events.extend(r3_find_delay_events(
            actions, item, R3_CUTOFF, min_step, max_step
        ))
        events.sort(key=lambda row: (row["start_step"], row["end_step"], row["kind"]))
    else:
        raise ValueError(mode)
    if max_events is not None and int(max_events) > 0:
        # Apply the cap per kind so combined mode does not hide all delay
        # events behind the larger advance event set.
        selected = []
        counts = Counter()
        for event in events:
            kind = str(event["kind"])
            if counts[kind] >= int(max_events):
                continue
            selected.append(event)
            counts[kind] += 1
        return selected
    return events


def collect(opponent_names, seeds, seats, items, mode, max_events,
            output, min_step=None, max_step=None, progress_every=25,
            resume=False, flush_every=50, skip_unsafe_candidates=False):
    specs = _specs(opponent_names)
    actions = _route_actions()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    existing_keys = set()
    malformed_tail = 0
    if resume:
        existing_rows, existing_keys, valid_end, malformed_tail = (
            _load_existing_rows(output)
        )
        if malformed_tail:
            with output.open("r+b") as handle:
                handle.truncate(valid_end)
            print(
                f"[V032-R3] truncated {malformed_tail} incomplete final line",
                flush=True,
            )
    rows = list(existing_rows)
    cache = _new_cache()
    completed = 0
    skipped_existing = 0
    file_mode = "a" if resume and output.exists() else "w"
    with output.open(file_mode, encoding="utf-8") as handle:
        for spec in specs:
            for seed in seeds:
                for seat in seats:
                    for item in items:
                        events = _events_for_mode(
                            actions, item, mode, min_step, max_step, max_events
                        )
                        for event in events:
                            source_quantity = int(event["current_quantity"])
                            for transfer in r3_quantity_candidates(
                                source_quantity, source_quantity
                            ):
                                key = _event_key(
                                    spec, seed, seat, item, event, transfer
                                )
                                if key in existing_keys:
                                    skipped_existing += 1
                                    continue
                                row = run_pair(
                                    ROUTE_PATH, spec, seed, seat, item,
                                    event, transfer, cache=cache,
                                    skip_unsafe_candidates=skip_unsafe_candidates,
                                )
                                rows.append(row)
                                existing_keys.add(_row_key(row))
                                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                completed += 1
                                if flush_every and completed % int(flush_every) == 0:
                                    handle.flush()
                                if progress_every and completed % int(progress_every) == 0:
                                    stats = cache["stats"]
                                    print(
                                        "[V032-R3] rows={} captures={} controls={} candidates={} "
                                        "cache_hits={}/{}".format(
                                            completed,
                                            stats["capture_runs"],
                                            stats["control_runs"],
                                            stats["candidate_runs"],
                                            stats["capture_cache_hits"],
                                            stats["control_cache_hits"],
                                        ),
                                        flush=True,
                                    )
        handle.flush()
    run_stats = {
        "rows": len(rows),
        "output": str(output),
        "opponents": len(specs),
        "seeds": list(seeds),
        "seats": list(seats),
        "items": list(items),
        "mode": mode,
        "resumed": bool(resume),
        "existing_rows": len(existing_rows),
        "skipped_existing": skipped_existing,
        "new_rows": completed,
        "malformed_tail_lines": malformed_tail,
        "skip_unsafe_candidates": bool(skip_unsafe_candidates),
        "flush_every": int(flush_every),
        "cache": dict(cache["stats"]),
    }
    output.with_name(output.stem + "_run_stats.json").write_text(
        json.dumps(run_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows), "output": str(output),
        "opponents": len(specs), "seeds": list(seeds),
        "seats": list(seats), "items": list(items), "mode": mode,
        "cache": cache["stats"],
        "resumed": bool(resume),
        "existing_rows": len(existing_rows),
        "skipped_existing": skipped_existing,
        "new_rows": completed,
    }, ensure_ascii=False, indent=2))
    return rows


def summarize(rows):
    evaluated = [row for row in rows if row.get("status") == "EVALUATED"]
    safe = [row for row in evaluated if row.get("safe")]

    def metric(rs, predicted, actual):
        pairs = [
            (float(row[predicted]), float(row[actual]))
            for row in rs
            if row.get(predicted) is not None and row.get(actual) is not None
        ]
        if not pairs:
            return {"n": 0}
        errors = [left - right for left, right in pairs]
        signs = [
            int((left > 0) == (right > 0))
            for left, right in pairs
            if left != 0 and right != 0
        ]
        return {
            "n": len(pairs),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "bias": sum(errors) / len(errors),
            "sign_accuracy": sum(signs) / len(signs) if signs else None,
            "positive_prediction_negative_actual": sum(
                1 for left, right in pairs if left > 0 and right < 0
            ),
            "mean_predicted": sum(left for left, _ in pairs) / len(pairs),
            "mean_actual": sum(right for _, right in pairs) / len(pairs),
        }

    by_item_kind = {}
    for item in R3_ITEMS:
        for kind in ("ADVANCE", "DELAY"):
            group = [
                row for row in evaluated
                if row.get("item") == item and row.get("kind") == kind
            ]
            if group:
                safe_group = [row for row in group if row.get("safe")]
                by_item_kind[f"{item}|{kind}"] = {
                    "rows": len(group),
                    "safe_rows": len(safe_group),
                    "safe_rate": len(safe_group) / len(group),
                    "prediction": metric(safe_group,
                                          "predicted_local_margin_delta",
                                          "actual_interval_margin_delta"),
                    "mean_actual_final_margin_delta": sum(
                        float(row["actual_final_margin_delta"]) for row in safe_group
                    ) / len(safe_group) if safe_group else None,
                }
    return {
        "rows": len(rows),
        "evaluated_rows": len(evaluated),
        "safe_rows": len(safe),
        "safe_rate_evaluated": len(safe) / len(evaluated) if evaluated else 0.0,
        "skipped_rows": len(rows) - len(evaluated),
        "skip_reasons": dict(Counter(
            reason for row in rows if row.get("status") == "SKIPPED"
            for reason in row.get("safety_reasons", [])
        )),
        "safe_prediction": metric(
            safe, "predicted_local_margin_delta", "actual_interval_margin_delta"
        ),
        "safe_final_margin_mean": (
            sum(float(row["actual_final_margin_delta"]) for row in safe) / len(safe)
            if safe else None
        ),
        "by_item_kind": by_item_kind,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("advance", "combined"), default="advance")
    parser.add_argument("--opponents", nargs="*", default=[
        "v27_current", "v13_r3", "adaptive_replay", "frontier_current"
    ])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17, 42, 2026, 217, 317, 733])
    parser.add_argument("--seats", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--items", nargs="+", default=list(R3_ITEMS))
    parser.add_argument("--max-events", type=int, default=0,
                        help="cap events per kind/item; 0 means all")
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print cache progress every N rows; 0 disables")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="flush JSONL every N new rows; 0 flushes only at end")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing JSONL and skip completed rows")
    parser.add_argument(
        "--skip-unsafe-candidates", action="store_true",
        help="skip full candidate games rejected by the delay safety precheck",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = collect(
        args.opponents, args.seeds, args.seats, args.items, args.mode,
        args.max_events, args.output, args.min_step, args.max_step,
        args.progress_every, args.resume, args.flush_every,
        args.skip_unsafe_candidates,
    )
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summarize(rows), indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
