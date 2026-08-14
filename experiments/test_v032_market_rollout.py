"""Reference checks for the V032-R1 market rollout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price as env_market_price


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v032_market_rollout", ROOT / "experiments/v032_market_rollout.py")
ROLL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROLL)


def test_official_price_points():
    points = (1, 9000, 9950, 10000, 10025, 10200, 11000, 13000)
    for item in ROLL.R1_MARKET_PARAMS:
        for inventory in points:
            assert ROLL.r1_market_price(item, inventory) == env_market_price(item, inventory), (item, inventory)


def test_lockstep_two_sellers_quote_same_precommit_price():
    result = ROLL.r1_process_market(
        [[["SELL", "MILK", 2]], [["SELL", "MILK", 2]]],
        {"MILK": 10000}, [{"MILK": 2}, {"MILK": 2}], [0, 0],
    )
    p0 = ROLL.r1_market_price("MILK", 10000)
    p1 = ROLL.r1_market_price("MILK", 10002)
    assert result["executed"] == [2, 2]
    assert result["money"] == [p0 + p1, p0 + p1]
    assert result["inventory"]["MILK"] == 10004


def test_order_limit_and_town_consumption():
    orders = [["SELL", "WHEAT", 1] for _ in range(11)]
    result = ROLL.r1_process_market([orders, []], {"WHEAT": 20}, [{"WHEAT": 20}, {}], [0, 0], max_orders=10)
    assert result["truncated_orders"] == [True, False]
    inventory = ROLL.r1_apply_town_consumption({"MILK": 10000}, ["SMOOTHIE_SHOP"], 0)
    assert inventory["MILK"] == 9998


def test_zero_sell_is_removed_and_window_is_reanchored():
    cleaned = ROLL.r1_clean_zero_sells([["SELL", "MILK", 0], ["SELL", "WOOL", 2], ["PASS"]])
    assert cleaned == [["SELL", "WOOL", 2], ["PASS"]]
    result = ROLL.r1_simulate_window(
        {"MILK": 10000}, [0, 0], [{"MILK": 2}, {}],
        {0: [["SELL", "MILK", 1]]}, {0: []}, 0, 0, [],
    )
    assert result["money"][0] == ROLL.r1_market_price("MILK", 10000)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
