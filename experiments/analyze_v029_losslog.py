"""Analyze downloaded V029 loss replays as a market-timing holdout set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
SELLABLE = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
MILK_SCHEDULE = ((215, 260), (288, 308), (336, 375), (388, 404), (504, 522), (552, 571))


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sell_quantity(action, item):
    total = 0
    for order in (action or {}).get("market", []) or []:
        if not isinstance(order, list) or len(order) < 3:
            continue
        if str(order[0]).upper() == "SELL" and str(order[1]).upper() == item:
            total += max(0, int(_number(order[2])))
    return total


def _liquid_value(observation):
    prices = observation.get("market", {}).get("prices", {}) or {}
    private = observation.get("private", {}) or {}
    inventory = {}
    for item, quantity in (private.get("shed", {}) or {}).items():
        inventory[str(item).upper()] = inventory.get(str(item).upper(), 0) + max(0, int(_number(quantity)))
    for carried in private.get("inventories", []) or []:
        for item, quantity in (carried or {}).items():
            inventory[str(item).upper()] = inventory.get(str(item).upper(), 0) + max(0, int(_number(quantity)))
    return sum(inventory.get(item, 0) * _number(prices.get(item, 0)) for item in SELLABLE)


def _first_negative(values, start=0):
    return next((index for index in range(start, len(values)) if values[index] < 0), None)


def _persistent_negative(values, start=0, window=24):
    return next(
        (index for index in range(start, len(values) - window) if all(value < 0 for value in values[index : index + window])),
        None,
    )


def _action_for_observation(steps, step, seat):
    # Kaggle replay records the action that advances observation step t in the
    # next step entry. The final entry has no following action to apply.
    index = min(step + 1, len(steps) - 1)
    return steps[index][seat].get("action") or {}


def analyze(path, candidate_name="GzmCR632"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload["steps"]
    names = [agent.get("Name", "") for agent in payload.get("info", {}).get("Agents", [])]
    if candidate_name not in names:
        raise ValueError(f"{candidate_name!r} not found in {names!r}")
    candidate_seat = names.index(candidate_name)
    opponent_seat = 1 - candidate_seat
    cash_gap = []
    mark_gap = []
    timeline = []
    for step, frame in enumerate(steps):
        candidate = frame[candidate_seat]["observation"]
        opponent = frame[opponent_seat]["observation"]
        cash = _number(candidate["farms"][candidate_seat].get("money")) - _number(
            opponent["farms"][opponent_seat].get("money")
        )
        mark = cash + _liquid_value(candidate) - _liquid_value(opponent)
        cash_gap.append(cash)
        mark_gap.append(mark)
        if step % 24 == 0 or step == len(steps) - 1:
            prices = candidate.get("market", {}).get("prices", {}) or {}
            timeline.append(
                {
                    "file": path.name,
                    "step": step,
                    "day": step // 24,
                    "cash_gap": round(cash, 3),
                    "cash_plus_inventory_gap": round(mark, 3),
                    **{f"price_{item.lower()}": _number(prices.get(item)) for item in PREMIUM},
                }
            )

    drops = []
    for item in PREMIUM:
        prices = [_number(frame[candidate_seat]["observation"].get("market", {}).get("prices", {}).get(item)) for frame in steps]
        index, delta = min(((i, prices[i] - prices[i - 1]) for i in range(1, len(prices))), key=lambda pair: pair[1])
        drops.append((delta, item, index, prices[index]))
    drops.sort()
    milk_deltas = []
    for current, future in MILK_SCHEDULE:
        current_price = _number(steps[current][candidate_seat]["observation"]["market"]["prices"].get("MILK"))
        future_price = _number(steps[future][candidate_seat]["observation"]["market"]["prices"].get("MILK"))
        milk_deltas.append(future_price - current_price)

    final = steps[-1][candidate_seat]
    opponent_final = steps[-1][opponent_seat]
    return {
        "file": path.name,
        "episode_id": payload.get("info", {}).get("EpisodeId", payload.get("id")),
        "seed": payload.get("info", {}).get("seed"),
        "candidate": candidate_name,
        "candidate_seat": candidate_seat,
        "opponent": names[opponent_seat],
        "candidate_final_cash": _number(final["observation"]["farms"][candidate_seat].get("money")),
        "opponent_final_cash": _number(opponent_final["observation"]["farms"][opponent_seat].get("money")),
        "final_cash_gap": round(cash_gap[-1], 3),
        "final_cash_plus_inventory_gap": round(mark_gap[-1], 3),
        "first_cash_negative_after_step_240": _first_negative(cash_gap, 240),
        "first_mark_negative_after_step_240": _first_negative(mark_gap, 240),
        "persistent_mark_negative_after_step_240": _persistent_negative(mark_gap, 240),
        "minimum_cash_gap": round(min(cash_gap), 3),
        "minimum_mark_gap": round(min(mark_gap), 3),
        "worst_premium_price_drop": round(drops[0][0], 3),
        "worst_premium": drops[0][1],
        "worst_drop_step": drops[0][2],
        "worst_drop_price": round(drops[0][3], 3),
        "milk_future_minus_current_prices": ",".join(f"{value:+.0f}" for value in milk_deltas),
        "milk_shift_price_sum": round(sum(milk_deltas), 3),
        "status": payload.get("statuses"),
        "timeline": timeline,
    }


def run(input_dir, output_dir, candidate_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [analyze(path, candidate_name) for path in sorted(input_dir.glob("*.json"))]
    timeline = [row for report in reports for row in report.pop("timeline")]
    summary_fields = sorted({key for report in reports for key in report})
    timeline_fields = sorted({key for row in timeline for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(reports)
    with (output_dir / "timeline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timeline_fields)
        writer.writeheader()
        writer.writerows(timeline)
    (output_dir / "summary.json").write_text(json.dumps(reports, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="GzmCR632")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.candidate), indent=2, ensure_ascii=True))
