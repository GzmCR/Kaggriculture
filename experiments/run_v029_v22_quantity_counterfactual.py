"""Search small quantity-preserving timing changes around the v22 market route."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from kaggle_environments import make

from run_v026_v22_v022c_recovery import EPISODE_STEPS, ROOT, _opponent, _v22_fresh


SEEDS = (17, 42)
PREMIUM = {"MELON", "STRAWBERRY", "MILK", "WOOL"}
MILK_SAFE_SCHEDULE = (
    ("MILK", 215, 260),
    ("MILK", 288, 308),
    ("MILK", 336, 375),
    ("MILK", 388, 404),
    ("MILK", 504, 522),
    ("MILK", 552, 571),
)


def _call(function, obs, config=None):
    try:
        return function(obs, config)
    except TypeError:
        return function(obs)


def _normalize(value):
    if not isinstance(value, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (value.get("hands") or [])],
        "market": [list(item) for item in (value.get("market") or []) if isinstance(item, list)],
    }


def _sell_quantity(action, item):
    return sum(
        max(0, int(order[2]))
        for order in action.get("market", [])
        if len(order) >= 3 and str(order[0]).upper() == "SELL"
        and str(order[1]).upper() == item
    )


def _route_sell_events():
    """Get nearby same-product windows from raw v22 route quantities."""
    module = _load_v22_module()
    events = {item: [] for item in PREMIUM}
    for step, action in enumerate(module._ACTIONS):
        for order in (action or {}).get("market", []) or []:
            if len(order) < 3 or str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item in events:
                events[item].append((step, max(0, int(order[2]))))
    windows = []
    for item, rows in events.items():
        for index, (current_step, current_quantity) in enumerate(rows[:-1]):
            future_step, future_quantity = rows[index + 1]
            if current_step >= 648 or future_step - current_step > 72:
                continue
            if current_quantity <= 0 or future_quantity <= 0:
                continue
            # Keep the search around the route's collision waves instead of
            # spending hundreds of full games on every tiny production sale.
            collision = current_quantity >= 4 or future_quantity >= 6
            if not collision:
                continue
            windows.append({
                "item": item,
                "current_step": current_step,
                "future_step": future_step,
                "current_quantity": current_quantity,
                "future_quantity": future_quantity,
            })
    return windows


def _load_v22_module():
    path = ROOT / "baseline/history/v027_v22_product_shift/v027a_melon_ratio/main.py"
    spec = importlib.util.spec_from_file_location(f"v029_route_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantityShiftAgent:
    def __init__(self, config):
        self.config = dict(config)
        self.base = _v22_fresh("v22")
        self.pending = 0
        self.pending_schedule = {}
        self.last_step = -1
        self.changed = []
        self.action_diff = 0

    def _reset_if_needed(self, step):
        if step == 0 or step < self.last_step:
            self.pending = 0
            self.pending_schedule = {}
            self.changed = []
            self.action_diff = 0
        self.last_step = step

    def __call__(self, obs, config=None):
        step = max(0, int(obs.get("step", 0) or 0))
        self._reset_if_needed(step)
        base = _normalize(_call(self.base, obs, config))
        action = {
            "farmer": list(base["farmer"]),
            "hands": [list(item) for item in base["hands"]],
            "market": [list(order) for order in base["market"]],
        }
        target = self.config
        item = target["item"]
        direction = target["direction"]
        quantity = int(target["quantity"])
        changed = 0
        seat_filter = target.get("seat_filter")
        if seat_filter is not None and int(obs.get("player", 0) or 0) != int(seat_filter):
            return action

        schedule = target.get("schedule")
        if schedule:
            for shift in schedule:
                if step == int(shift["current_step"]):
                    current = _sell_quantity(action, shift["item"])
                    if shift["direction"] == "advance":
                        moved = min(quantity, max(0, int(shift["future_quantity"])), current)
                        changed += _adjust_sell(action, shift["item"], moved)
                        self.pending_schedule[(int(shift["future_step"]), shift["item"])] = (
                            self.pending_schedule.get((int(shift["future_step"]), shift["item"]), 0)
                            + moved
                        )
                    else:
                        moved = min(quantity, current)
                        changed += _adjust_sell(action, shift["item"], -moved)
                        self.pending_schedule[(int(shift["future_step"]), shift["item"])] = (
                            self.pending_schedule.get((int(shift["future_step"]), shift["item"]), 0)
                            + moved
                        )
                pending_key = (step, shift["item"])
                moved = self.pending_schedule.pop(pending_key, 0)
                if moved:
                    delta = -moved if shift["direction"] == "advance" else moved
                    changed += _adjust_sell(action, shift["item"], delta)
        elif step == int(target["current_step"]):
            current = _sell_quantity(action, item)
            if direction == "advance":
                moved = min(quantity, max(0, int(target["future_quantity"])), current)
                self.pending = moved
                changed = _adjust_sell(action, item, moved)
            else:
                moved = min(quantity, current)
                self.pending = moved
                changed = _adjust_sell(action, item, -moved)
        elif step == int(target["future_step"]) and self.pending:
            if direction == "advance":
                changed = _adjust_sell(action, item, -self.pending)
            else:
                changed = _adjust_sell(action, item, self.pending)
            if changed:
                self.pending = 0

        if changed:
            self.changed.append({
                "step": step,
                "direction": direction,
                "item": item,
                "quantity": int(changed),
            })
            self.action_diff += 1
        return action


def _adjust_sell(action, item, delta):
    """Adjust one existing SELL order, preserving its order slot and product."""
    for index, order in enumerate(action.get("market", [])):
        if (
            len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == item
        ):
            current = max(0, int(order[2]))
            updated = current + int(delta)
            if updated < 0:
                return 0
            action["market"][index] = [order[0], order[1], updated]
            if updated == 0:
                action["market"].pop(index)
            return abs(int(delta))
    return 0


def _candidate_configs(windows, seat_filter=None, schedule_name=None):
    configs = []
    if schedule_name == "milk_safe_6":
        for direction in ("delay", "advance"):
            configs.append({
                "item": "MILK_SCHEDULE",
                "current_step": ",".join(str(row[1]) for row in MILK_SAFE_SCHEDULE),
                "future_step": ",".join(str(row[2]) for row in MILK_SAFE_SCHEDULE),
                "current_quantity": 1,
                "future_quantity": 1,
                "direction": direction,
                "quantity": 1,
                "seat_filter": seat_filter,
                "schedule": [
                    {
                        "item": item,
                        "current_step": current_step,
                        "future_step": future_step,
                        "current_quantity": _route_quantity(item, current_step),
                        "future_quantity": _route_quantity(item, future_step),
                        "direction": direction,
                    }
                    for item, current_step, future_step in MILK_SAFE_SCHEDULE
                ],
            })
        return configs
    for window in windows:
        for direction in ("advance", "delay"):
            configs.append({
                **window,
                "direction": direction,
                "quantity": 1,
                "seat_filter": seat_filter,
            })
    return configs


@lru_cache(maxsize=1)
def _raw_route_quantities():
    module = _load_v22_module()
    quantities = {}
    for step, action in enumerate(module._ACTIONS):
        for order in (action or {}).get("market", []) or []:
            if len(order) < 3 or str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item in PREMIUM:
                quantities[(item, int(step))] = quantities.get((item, int(step)), 0) + max(0, int(order[2]))
    return quantities


def _route_quantity(item, step):
    return int(_raw_route_quantities().get((str(item).upper(), int(step)), 0))


def _run_one(candidate, opponent_name, seed, seat):
    agent = QuantityShiftAgent(candidate)
    opponent = _opponent(opponent_name)
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
        **candidate,
        "opponent": opponent_name,
        "seed": int(seed),
        "seat": int(seat),
        "candidate_money": mine_money,
        "opponent_money": other_money,
        "margin": mine_money - other_money,
        "result": "win" if mine_money > other_money else "loss" if mine_money < other_money else "tie",
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "changed_calls": len(agent.changed),
        "action_diff_calls": agent.action_diff,
        "candidate_status": str(mine.status),
        "opponent_status": str(theirs.status),
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows):
    groups = {}
    for row in rows:
        key = (row["item"], row["current_step"], row["future_step"], row["direction"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        outcomes = Counter(row["result"] for row in group)
        output.append({
            "item": key[0],
            "current_step": key[1],
            "future_step": key[2],
            "direction": key[3],
            "quantity": group[0]["quantity"],
            "games": len(group),
            "mean_money": statistics.mean(row["candidate_money"] for row in group),
            "min_money": min(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "all_done": int(all(row["done"] for row in group)),
            "errors": 0,
            "mean_changed_calls": statistics.mean(row["changed_calls"] for row in group),
        })
    return output


def run(
    output,
    seeds,
    max_windows,
    item_filter=None,
    current_filter=None,
    direction_filter=None,
    seat_filter=None,
    opponent_name="v22",
    schedule_name=None,
):
    windows = _route_sell_events()
    if item_filter:
        windows = [row for row in windows if row["item"] == item_filter]
    if current_filter is not None:
        allowed_steps = {
            int(step) for step in (current_filter if isinstance(current_filter, (list, tuple)) else [current_filter])
        }
        windows = [row for row in windows if row["current_step"] in allowed_steps]
    if max_windows:
        windows = windows[:int(max_windows)]
    candidates = _candidate_configs(windows, seat_filter=seat_filter, schedule_name=schedule_name)
    if direction_filter:
        candidates = [row for row in candidates if row["direction"] == direction_filter]
    rows = []
    total = len(candidates) * len(seeds) * 2
    index = 0
    for candidate in candidates:
        for seed in seeds:
            for seat in (0, 1):
                index += 1
                print(
                    f"[{index}/{total}] {candidate['item']} "
                    f"{candidate['current_step']}->{candidate['future_step']} "
                    f"{candidate['direction']} seed={seed} seat={seat}",
                    flush=True,
                )
                rows.append(_run_one(candidate, opponent_name, seed, seat))
    _write_csv(output / "raw.csv", rows)
    _write_csv(output / "summary.csv", _summary(rows))
    return windows, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baseline/artifacts/v029_v22_quantity_counterfactual",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--item", choices=tuple(sorted(PREMIUM)))
    parser.add_argument("--current-step", action="append", type=int)
    parser.add_argument("--direction", choices=("advance", "delay"))
    parser.add_argument("--seat-filter", type=int, choices=(0, 1))
    parser.add_argument("--opponent", default="v22")
    parser.add_argument("--schedule", choices=("milk_safe_6",))
    args = parser.parse_args()
    run(
        args.output,
        tuple(args.seed or (17,)),
        args.max_windows,
        item_filter=args.item,
        current_filter=args.current_step,
        direction_filter=args.direction,
        seat_filter=args.seat_filter,
        opponent_name=args.opponent,
        schedule_name=args.schedule,
    )
