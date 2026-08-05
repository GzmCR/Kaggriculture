"""Write daily cash and mark-to-market diagnostics for V020 replay games."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from kaggle_environments import make

from run_v020_value_aware import OUT_DIR, REPLAY_DIR, V020Agent, load_recorded_opponent
from v020_value_aware_market import PREMIUM_PRODUCTS, liquidation_value
from run_v015a_market_collision import EPISODE_STEPS, ReplayAgent


ROOT = Path(__file__).resolve().parents[1]


def _value(obs, include_carried=True):
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    market = obs.get("market", {}) or {}
    inventory = market.get("inventory", {}) or {}
    shed_value = sum(
        liquidation_value(item, max(0, int(shed.get(item, 0) or 0)), int(inventory.get(item, 10000) or 10000), obs)
        for item in PREMIUM_PRODUCTS
    )
    carried_value = 0.0
    if include_carried:
        for unit in inventories:
            if not isinstance(unit, dict):
                continue
            for item in PREMIUM_PRODUCTS:
                carried_value += liquidation_value(
                    item,
                    max(0, int(unit.get(item, 0) or 0)),
                    int(inventory.get(item, 10000) or 10000),
                    obs,
                )
    cash = float(farm.get("money", 0) or 0)
    return cash, shed_value, carried_value


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(variant):
    daily = []
    summaries = []
    for path in sorted(REPLAY_DIR.glob("*.json")):
        route = load_recorded_opponent(path)
        candidate = V020Agent(variant)
        opponent = ReplayAgent(route["actions"])
        players = [candidate, opponent] if route["seat"] == 0 else [opponent, candidate]
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": EPISODE_STEPS, "seed": route["seed"]},
            debug=False,
        )
        env.run(players)
        candidate_seat = route["seat"]
        opponent_seat = 1 - candidate_seat
        for day in range(30):
            step = min((day + 1) * 24 - 1, len(env.steps) - 1)
            candidate_obs = env.steps[step][candidate_seat].observation
            opponent_obs = env.steps[step][opponent_seat].observation
            c_cash, c_shed, c_carried = _value(candidate_obs)
            o_cash, o_shed, o_carried = _value(opponent_obs)
            daily.append({
                "episode": path.stem,
                "day": day,
                "candidate_cash": round(c_cash, 4),
                "opponent_cash": round(o_cash, 4),
                "cash_gap": round(c_cash - o_cash, 4),
                "candidate_equity_shed": round(c_cash + c_shed, 4),
                "opponent_equity_shed": round(o_cash + o_shed, 4),
                "equity_shed_gap": round(c_cash + c_shed - o_cash - o_shed, 4),
                "candidate_equity_all": round(c_cash + c_shed + c_carried, 4),
                "opponent_equity_all": round(o_cash + o_shed + o_carried, 4),
                "equity_all_gap": round(c_cash + c_shed + c_carried - o_cash - o_shed - o_carried, 4),
                "milk_price": (candidate_obs.get("market", {}).get("prices", {}) or {}).get("MILK"),
                "strawberry_price": (candidate_obs.get("market", {}).get("prices", {}) or {}).get("STRAWBERRY"),
                "melon_price": (candidate_obs.get("market", {}).get("prices", {}) or {}).get("MELON"),
                "wool_price": (candidate_obs.get("market", {}).get("prices", {}) or {}).get("WOOL"),
            })
        final = daily[-1]
        summaries.append({
            "episode": path.stem,
            "variant": variant,
            "final_cash_gap": final["cash_gap"],
            "final_equity_shed_gap": final["equity_shed_gap"],
            "final_equity_all_gap": final["equity_all_gap"],
            "first_cash_behind_day": next((row["day"] for row in daily if row["episode"] == path.stem and row["cash_gap"] < 0), None),
            "first_equity_all_behind_day": next((row["day"] for row in daily if row["episode"] == path.stem and row["equity_all_gap"] < 0), None),
            "candidate_status": env.steps[-1][candidate_seat].status,
            "opponent_status": env.steps[-1][opponent_seat].status,
            "controller_diagnostics": json.dumps(candidate.diagnostics(candidate_seat), ensure_ascii=False, sort_keys=True),
        })
    _write_csv(OUT_DIR / "replay_daily.csv", daily)
    _write_csv(OUT_DIR / "replay_diagnostics.csv", summaries)
    report = {"variant": variant, "episodes": len(summaries), "daily_rows": len(daily), "summaries": summaries}
    (OUT_DIR / "replay_diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"variant": variant, "episodes": len(summaries), "daily_rows": len(daily)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("balanced", "sensitive", "conservative"), default="conservative")
    args = parser.parse_args()
    analyze(args.variant)


if __name__ == "__main__":
    main()
