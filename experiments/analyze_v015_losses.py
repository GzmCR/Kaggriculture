"""Analyze V015a replay losses with mark-to-market inventory value."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
    PRODUCTS,
    market_price,
)

from v019_replay_analysis import reconstruct_turn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "baseline/history/v015a_market_collision/log"
DEFAULT_OUT_DIR = ROOT / "baseline/artifacts/v015a_market_collision/loss_analysis"
EPISODE_STEPS = 720
TURNS_PER_DAY = 24
TEAM_NAME = "GzmCR632"
SELLABLE = tuple(item for item in PRODUCTS if item != "FERTILIZER")


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _obs(steps, step, seat):
    index = min(max(0, _int(step)), len(steps) - 1)
    return steps[index][seat].get("observation", {}) or {}


def _action(steps, step, seat):
    index = min(max(0, _int(step) + 1), len(steps) - 1)
    return steps[index][seat].get("action", {}) or {}


def _market_params(market):
    params = market.get("params") if isinstance(market, dict) else None
    return params if isinstance(params, dict) and params else MARKET_PARAMS


def _liquidation_value(item, quantity, market):
    """Value a stockpile by selling units one by one at current market state."""
    if quantity <= 0:
        return 0.0
    inventory = _int((market.get("inventory", {}) or {}).get(item, 0))
    params = _market_params(market)
    try:
        return sum(
            _float(market_price(item, inventory + offset, params))
            for offset in range(quantity)
        )
    except Exception:
        price = _float((market.get("prices", {}) or {}).get(item, 0))
        return price * quantity


def _inventory_counts(obs):
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    shed = private.get("shed", {}) if isinstance(private, dict) else {}
    inventories = private.get("inventories", []) if isinstance(private, dict) else []
    shed_counts = Counter()
    carried_counts = Counter()
    for item in SELLABLE:
        shed_counts[item] = max(0, _int(shed.get(item, 0)))
    for inventory in inventories or []:
        for item in SELLABLE:
            carried_counts[item] += max(0, _int((inventory or {}).get(item, 0)))
    return shed_counts, carried_counts


def _state(obs):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    farm = farms[player] if 0 <= player < len(farms) else {}
    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    shed, carried = _inventory_counts(obs)
    shed_value = sum(_liquidation_value(item, quantity, market) for item, quantity in shed.items())
    carried_value = sum(_liquidation_value(item, quantity, market) for item, quantity in carried.items())
    return {
        "cash": _float(farm.get("money", 0)),
        "shed_value": shed_value,
        "carried_value": carried_value,
        "equity_shed": _float(farm.get("money", 0)) + shed_value,
        "equity_all": _float(farm.get("money", 0)) + shed_value + carried_value,
        "shed_units": sum(shed.values()),
        "carried_units": sum(carried.values()),
        "shed": dict(shed),
        "carried": dict(carried),
    }


def _first_negative(values):
    for index, value in enumerate(values):
        if value < 0:
            return index
    return None


def _first_persistent_negative(values):
    for index, value in enumerate(values):
        if value < 0 and all(other < 0 for other in values[index:]):
            return index
    return None


def _action_counts(action):
    field = Counter()
    market = Counter()
    quantities = Counter()
    units = [action.get("farmer", [])] + list(action.get("hands", []) or [])
    for unit_action in units:
        if isinstance(unit_action, list) and unit_action:
            field[str(unit_action[0])] += 1
    for order in action.get("market", []) or []:
        if not isinstance(order, list) or not order:
            continue
        operation = str(order[0])
        market[operation] += 1
        if len(order) >= 3:
            quantities[(operation, str(order[1]))] += max(0, _int(order[2]))
    return field, market, quantities


def _compact_counter(counter):
    return ",".join(
        f"{key}={value}"
        for key, value in sorted(counter.items())
        if value
    )


def _sell_text(fills):
    return ",".join(
        f"{item}={_int(values.get('filled', 0))}"
        for item, values in sorted(fills.items())
        if _int(values.get("filled", 0))
    )


def _our_seat(payload):
    names = (payload.get("info", {}) or {}).get("TeamNames", []) or []
    for seat, name in enumerate(names[:2]):
        if str(name).strip() == TEAM_NAME:
            return seat
    return 0


def _load_replays(directory):
    for path in sorted(Path(directory).glob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def analyze_replay(path, payload):
    steps = payload.get("steps", [])
    if len(steps) < EPISODE_STEPS:
        raise ValueError(f"{path} has only {len(steps)} steps")
    our_seat = _our_seat(payload)
    opponent_seat = 1 - our_seat
    timeline = []
    daily = defaultdict(lambda: {
        "our_sell_filled": Counter(),
        "opponent_sell_filled": Counter(),
        "our_sell_revenue": 0.0,
        "opponent_sell_revenue": 0.0,
        "our_field": Counter(),
        "opponent_field": Counter(),
        "our_market": Counter(),
        "opponent_market": Counter(),
        "our_buy_product": 0,
        "opponent_buy_product": 0,
    })

    for step in range(EPISODE_STEPS):
        our_obs = _obs(steps, step, our_seat)
        opponent_obs = _obs(steps, step, opponent_seat)
        our = _state(our_obs)
        opponent = _state(opponent_obs)
        our_action = _action(steps, step, our_seat)
        opponent_action = _action(steps, step, opponent_seat)
        our_field, our_market, our_quantities = _action_counts(our_action)
        opponent_field, opponent_market, opponent_quantities = _action_counts(opponent_action)
        fills = reconstruct_turn(steps, step)["per_seat"]
        our_fills = fills[our_seat]
        opponent_fills = fills[opponent_seat]
        day = step // TURNS_PER_DAY
        row = {
            "episode": str((payload.get("info", {}) or {}).get("EpisodeId", path.stem)),
            "step": step,
            "day": day,
            "hour": step % TURNS_PER_DAY,
            "our_cash": round(our["cash"], 4),
            "opponent_cash": round(opponent["cash"], 4),
            "our_shed_value": round(our["shed_value"], 4),
            "opponent_shed_value": round(opponent["shed_value"], 4),
            "our_carried_value": round(our["carried_value"], 4),
            "opponent_carried_value": round(opponent["carried_value"], 4),
            "our_equity_shed": round(our["equity_shed"], 4),
            "opponent_equity_shed": round(opponent["equity_shed"], 4),
            "our_equity_all": round(our["equity_all"], 4),
            "opponent_equity_all": round(opponent["equity_all"], 4),
            "cash_gap": round(our["cash"] - opponent["cash"], 4),
            "equity_shed_gap": round(our["equity_shed"] - opponent["equity_shed"], 4),
            "equity_all_gap": round(our["equity_all"] - opponent["equity_all"], 4),
            "our_shed_units": our["shed_units"],
            "opponent_shed_units": opponent["shed_units"],
            "our_carried_units": our["carried_units"],
            "opponent_carried_units": opponent["carried_units"],
            "our_field_action": _compact_counter(our_field),
            "opponent_field_action": _compact_counter(opponent_field),
            "our_market_action": _compact_counter(our_market),
            "opponent_market_action": _compact_counter(opponent_market),
            "our_sell_filled": _sell_text(our_fills),
            "opponent_sell_filled": _sell_text(opponent_fills),
            "our_sell_revenue": round(sum(_float(v.get("revenue", 0)) for v in our_fills.values()), 4),
            "opponent_sell_revenue": round(sum(_float(v.get("revenue", 0)) for v in opponent_fills.values()), 4),
        }
        timeline.append(row)

        bucket = daily[day]
        bucket["our_sell_revenue"] += row["our_sell_revenue"]
        bucket["opponent_sell_revenue"] += row["opponent_sell_revenue"]
        bucket["our_field"].update(our_field)
        bucket["opponent_field"].update(opponent_field)
        bucket["our_market"].update(our_market)
        bucket["opponent_market"].update(opponent_market)
        bucket["our_buy_product"] += our_market.get("BUY_PRODUCT", 0)
        bucket["opponent_buy_product"] += opponent_market.get("BUY_PRODUCT", 0)
        for item, values in our_fills.items():
            bucket["our_sell_filled"][item] += _int(values.get("filled", 0))
        for item, values in opponent_fills.items():
            bucket["opponent_sell_filled"][item] += _int(values.get("filled", 0))

    cash_gaps = [row["cash_gap"] for row in timeline]
    shed_gaps = [row["equity_shed_gap"] for row in timeline]
    all_gaps = [row["equity_all_gap"] for row in timeline]
    rewards = payload.get("rewards", []) or []
    winner = opponent_seat if _float(rewards[opponent_seat] if len(rewards) > opponent_seat else 0) > _float(rewards[our_seat] if len(rewards) > our_seat else 0) else our_seat

    def describe(index):
        if index is None:
            return {"step": None, "day": None, "hour": None}
        row = timeline[index]
        return {"step": index, "day": row["day"], "hour": row["hour"]}

    first_cash = _first_negative(cash_gaps)
    first_shed = _first_negative(shed_gaps)
    first_all = _first_negative(all_gaps)
    persistent_cash = _first_persistent_negative(cash_gaps)
    persistent_shed = _first_persistent_negative(shed_gaps)
    persistent_all = _first_persistent_negative(all_gaps)
    minimum_all = min(range(len(all_gaps)), key=lambda index: all_gaps[index])
    minimum_shed = min(range(len(shed_gaps)), key=lambda index: shed_gaps[index])
    final = timeline[-1]
    summary = {
        "file": path.name,
        "episode": timeline[-1]["episode"],
        "seed": (payload.get("info", {}) or {}).get("seed"),
        "teams": (payload.get("info", {}) or {}).get("TeamNames", []),
        "our_seat": our_seat,
        "opponent_seat": opponent_seat,
        "winner_seat": winner,
        "our_final_cash": final["our_cash"],
        "opponent_final_cash": final["opponent_cash"],
        "our_final_shed_value": final["our_shed_value"],
        "opponent_final_shed_value": final["opponent_shed_value"],
        "our_final_equity_shed": final["our_equity_shed"],
        "opponent_final_equity_shed": final["opponent_equity_shed"],
        "our_final_equity_all": final["our_equity_all"],
        "opponent_final_equity_all": final["opponent_equity_all"],
        "final_cash_gap": final["cash_gap"],
        "final_equity_shed_gap": final["equity_shed_gap"],
        "final_equity_all_gap": final["equity_all_gap"],
        "first_cash_behind": describe(first_cash),
        "first_shed_equity_behind": describe(first_shed),
        "first_all_equity_behind": describe(first_all),
        "persistent_cash_deficit": describe(persistent_cash),
        "persistent_shed_equity_deficit": describe(persistent_shed),
        "persistent_all_equity_deficit": describe(persistent_all),
        "worst_shed_equity_gap": {**describe(minimum_shed), "gap": shed_gaps[minimum_shed]},
        "worst_all_equity_gap": {**describe(minimum_all), "gap": all_gaps[minimum_all]},
    }
    daily_rows = []
    for day, bucket in sorted(daily.items()):
        row = timeline[min((day + 1) * TURNS_PER_DAY - 1, len(timeline) - 1)]
        daily_rows.append({
            "episode": summary["episode"],
            "day": day,
            "our_cash": row["our_cash"],
            "opponent_cash": row["opponent_cash"],
            "cash_gap": row["cash_gap"],
            "our_equity_shed": row["our_equity_shed"],
            "opponent_equity_shed": row["opponent_equity_shed"],
            "equity_shed_gap": row["equity_shed_gap"],
            "our_equity_all": row["our_equity_all"],
            "opponent_equity_all": row["opponent_equity_all"],
            "equity_all_gap": row["equity_all_gap"],
            "our_sell_filled": _compact_counter(bucket["our_sell_filled"]),
            "opponent_sell_filled": _compact_counter(bucket["opponent_sell_filled"]),
            "our_sell_revenue": round(bucket["our_sell_revenue"], 4),
            "opponent_sell_revenue": round(bucket["opponent_sell_revenue"], 4),
            "our_buy_product": bucket["our_buy_product"],
            "opponent_buy_product": bucket["opponent_buy_product"],
            "our_field": _compact_counter(bucket["our_field"]),
            "opponent_field": _compact_counter(bucket["opponent_field"]),
            "our_market": _compact_counter(bucket["our_market"]),
            "opponent_market": _compact_counter(bucket["opponent_market"]),
        })
    return summary, timeline, daily_rows


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    summaries = []
    timeline_rows = []
    daily_rows = []
    errors = []
    for path, payload in _load_replays(args.log_dir):
        try:
            summary, timeline, replay_daily = analyze_replay(path, payload)
        except Exception as error:
            errors.append({"file": path.name, "error": repr(error)})
            continue
        summaries.append(summary)
        timeline_rows.extend(timeline)
        daily_rows.extend(replay_daily)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "summary.csv", summaries)
    _write_csv(args.out_dir / "timeline.csv", timeline_rows)
    _write_csv(args.out_dir / "daily.csv", daily_rows)
    report = {
        "log_dir": str(args.log_dir),
        "replays_found": len(list(args.log_dir.glob("*.json"))),
        "replays_analyzed": len(summaries),
        "errors": errors,
        "our_team": TEAM_NAME,
        "value_definition": {
            "shed_value": "cash plus all sellable product units in shed, valued by sequential current-market liquidation prices",
            "all_value": "shed_value plus sellable product units still carried by farmer/hands",
            "fertilizer_and_seeds_excluded": True,
            "animal_assets_excluded": True,
        },
        "summary": summaries,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"replays_analyzed": len(summaries), "errors": errors, "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
