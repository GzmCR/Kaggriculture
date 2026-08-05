"""Unit and invariant tests for V017."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price
from v017_market_rollout import (
    HORIZON,
    MARKET,
    MarketRolloutController,
    _sell_lockstep,
    _town_demand,
    price_at,
)


PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")


def action(orders):
    return {"farmer": ["EAST"], "hands": [["WEST"]], "market": [list(order) for order in orders]}


def runtime(orders):
    actions = [{"farmer": ["PASS"], "hands": [], "market": [list(order) for order in orders]} for _ in range(720)]
    return {
        "experts": {"expert": {"actions": actions}},
        "board_by_seat": {"0": "expert", "1": "expert"},
    }


def observation(step=0, market_inventory=None, shed=None, town=None):
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "market": {
            "inventory": market_inventory or {item: 10000 for item in MARKET},
            "prices": {item: market_price(item, (market_inventory or {}).get(item, 10000)) for item in MARKET},
        },
        "town": {"unlocked_shops": town or []},
        "farms": [
            {"money": 50000, "hands": [], "tiles": [[None]], "unlocked_quadrants": ["NW"]},
            {"money": 50000, "hands": [], "tiles": [[None]], "unlocked_quadrants": ["NW"]},
        ],
        "private": {"shed": shed or {"MELON": 100, "WHEAT": 20}, "inventories": [{}]},
    }


def test_price_function_matches_environment():
    for item in MARKET:
        for inventory in (0, 1, 9000, 9999, 10000, 10001, 10100, 12000, 15000):
            assert price_at(item, inventory) == market_price(item, inventory), (item, inventory)


def test_lockstep_sell_matches_unit_economics():
    inventory, revenue = _sell_lockstep(10000, 1, 1, "MELON", {})
    assert inventory == 10002
    assert revenue == 250
    inventory, revenue = _sell_lockstep(10000, 2, 0, "MELON", {})
    assert inventory == 10002
    assert revenue == 500


def test_town_consumption_schedule():
    obs = observation(step=0, town=["PIZZA_SHOP"])
    assert _town_demand(obs, "MILK", 0) == 2  # shop + town center
    assert _town_demand(obs, "WOOL", 0) == 1  # town center only
    assert _town_demand(obs, "MILK", 1) == 0
    assert _town_demand(obs, "FERTILIZER", 0) == 0


def test_no_premium_slot_is_exact_noop():
    controller = MarketRolloutController("curve_only", runtime([]), selected_state={0: "expert"})
    obs = observation()
    base = action([["BUY_PRODUCT", "WHEAT", 1]])
    assert controller.apply(obs, base) == base


def test_rollout_only_reduces_existing_premium_order():
    controller = MarketRolloutController("opponent_aware", runtime([["SELL", "MELON", 20], ["BUY_PRODUCT", "WHEAT", 1]]), selected_state={0: "expert"})
    obs = observation(market_inventory={**{item: 10000 for item in MARKET}, "MELON": 10100})
    base = action([["SELL", "MELON", 20], ["BUY_PRODUCT", "WHEAT", 1]])
    output = controller.apply(obs, base)
    premium = [order for order in output["market"] if order[0] == "SELL" and order[1] == "MELON"]
    assert sum(int(order[2]) for order in premium) <= 20
    assert [order for order in output["market"] if order[0] == "BUY_PRODUCT"] == [["BUY_PRODUCT", "WHEAT", 1]]
    assert output["farmer"] == base["farmer"] and output["hands"] == base["hands"]


def test_robust_quota_caps_current_order():
    controller = MarketRolloutController("robust_quota", runtime([["SELL", "WOOL", 20]]), selected_state={0: "expert"})
    obs = observation(shed={"WOOL": 20, "WHEAT": 20})
    base = action([["SELL", "WOOL", 20]])
    output = controller.apply(obs, base)
    premium = [order for order in output["market"] if order[0] == "SELL" and order[1] == "WOOL"]
    assert sum(int(order[2]) for order in premium) <= 10


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"V017 rollout invariants: PASS ({len(tests)} tests, horizon={HORIZON})")


if __name__ == "__main__":
    main()
