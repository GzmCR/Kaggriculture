"""Unit and invariant tests for the V018 wave/MPC controller."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price
from v017_market_rollout import MARKET, price_at
from v018_market_wave import MarketWaveController, WAVE_PARAMS


PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")


def _runtime(orders):
    actions = [
        {"farmer": ["PASS"], "hands": [], "market": [list(order) for order in orders]}
        for _ in range(720)
    ]
    return {
        "experts": {"expert": {"actions": actions}},
        "board_by_seat": {"0": "expert", "1": "expert"},
    }


def _observation(step=0, money=50000, market_inventory=None, shed=None):
    inventory = {item: 10000 for item in MARKET}
    if market_inventory:
        inventory.update(market_inventory)
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "market": {
            "inventory": inventory,
            "prices": {item: market_price(item, inventory[item]) for item in MARKET},
        },
        "town": {"unlocked_shops": []},
        "farms": [
            {"money": money, "hands": [], "tiles": [[None]], "unlocked_quadrants": ["NW"]},
            {"money": 50000, "hands": [], "tiles": [[None]], "unlocked_quadrants": ["NW"]},
        ],
        "private": {
            "shed": shed or {"MELON": 100, "STRAWBERRY": 100, "MILK": 100, "WOOL": 100},
            "inventories": [{}],
        },
    }


def _action(orders, farmer=None, hands=None):
    return {
        "farmer": list(farmer or ["EAST"]),
        "hands": [list(item) for item in (hands or [["WEST"]])],
        "market": [list(order) for order in orders],
    }


def _premium_total(orders, item):
    return sum(int(order[2]) for order in orders if order[0] == "SELL" and order[1] == item)


def test_price_function_matches_environment():
    for item in MARKET:
        for inventory in (0, 1, 9000, 9999, 10000, 10001, 10100, 12000, 15000):
            assert price_at(item, inventory) == market_price(item, inventory), (item, inventory)


def test_no_premium_slot_is_exact_noop():
    controller = MarketWaveController("daily_mpc", runtime=_runtime([]), selected_state={0: "expert"})
    obs = _observation()
    base = _action([["BUY_PRODUCT", "WHEAT", 1]])
    assert controller.apply(obs, base) == base


def test_only_existing_premium_quantity_changes():
    controller = MarketWaveController(
        "fixed_wave",
        runtime=_runtime([["SELL", "MELON", 20], ["BUY_PRODUCT", "WHEAT", 1]]),
        selected_state={0: "expert"},
    )
    obs = _observation(step=0)
    base = _action([["SELL", "MELON", 20], ["BUY_PRODUCT", "WHEAT", 1]])
    output = controller.apply(obs, base)
    assert output["farmer"] == base["farmer"]
    assert output["hands"] == base["hands"]
    assert [order for order in output["market"] if order[0] == "BUY_PRODUCT"] == [["BUY_PRODUCT", "WHEAT", 1]]
    assert _premium_total(output["market"], "MELON") <= 20
    assert output["market"] != []


def test_daily_mpc_cash_guard_is_exact_noop():
    controller = MarketWaveController(
        "daily_mpc",
        runtime=_runtime([["SELL", "MELON", 20], ["BUY_PRODUCT", "WHEAT", 1]]),
        selected_state={0: "expert"},
    )
    obs = _observation(step=0, money=2500)
    base = _action([["SELL", "MELON", 20], ["BUY_PRODUCT", "WHEAT", 1]])
    assert controller.apply(obs, base) == base


def test_robust_mpc_caps_and_preserves_order_limit():
    orders = [
        ["SELL", "MELON", 20],
        ["SELL", "STRAWBERRY", 20],
        ["SELL", "MILK", 20],
        ["SELL", "WOOL", 20],
        ["BUY_PRODUCT", "WHEAT", 1],
    ]
    controller = MarketWaveController("robust_mpc", runtime=_runtime(orders), selected_state={0: "expert"})
    output = controller.apply(_observation(step=360), _action(orders))
    assert len(output["market"]) <= 10
    assert all(len(order) < 3 or int(order[2]) >= 0 for order in output["market"])
    for item in PREMIUM:
        assert _premium_total(output["market"], item) <= 20


def test_fixed_wave_upper_bound_follows_phase():
    for item, params in WAVE_PARAMS.items():
        before = _observation(step=params["start_day"] * 24)
        before["day"] = max(0, params["start_day"] - 1)
        before["step"] = before["day"] * 24
        controller = MarketWaveController(
            "fixed_wave", runtime=_runtime([["SELL", item, 20]]), selected_state={0: "expert"}
        )
        output = controller.apply(before, _action([["SELL", item, 20]]))
        assert _premium_total(output["market"], item) <= int(20 * params["pre_window_ratio"] + 1)


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"V018 market-wave invariants: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
