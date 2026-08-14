"""Unit checks for V032-R2 strict interval estimation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price as env_market_price


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v032_r2_interval_test_module", ROOT / "experiments/v032_r2_interval.py")
R2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R2)


def test_official_price_points_match():
    for item in R2.R2_MARKET_PARAMS:
        for inventory in (1, 9000, 9950, 10000, 10025, 10200, 11000, 13000):
            assert R2.r2_market_price(item, inventory) == env_market_price(item, inventory)


def test_lockstep_quotes_are_shared_before_commit():
    result = R2.r2_process_market(
        [[["SELL", "MILK", 2]], [["SELL", "MILK", 2]]],
        {"MILK": 10000},
        [{"MILK": 2}, {"MILK": 2}],
        [0, 0],
        "MILK",
    )
    p0 = R2.r2_market_price("MILK", 10000)
    p1 = R2.r2_market_price("MILK", 10002)
    assert result["executed"] == [2, 2]
    assert result["money"] == [p0 + p1, p0 + p1]


def test_town_consumption_uses_actual_intervals():
    inventory = R2.r2_apply_town_consumption({"MILK": 10000}, ["SMOOTHIE_SHOP"], 0, 4, 24)
    assert inventory["MILK"] == 9998
    inventory = R2.r2_apply_town_consumption({"MILK": 10000}, ["SMOOTHIE_SHOP"], 1, 4, 24)
    assert inventory["MILK"] == 10000


def test_delay_conserves_quantity_without_preloading_future_stock():
    current = [["SELL", "MILK", 10], ["SELL", "WHEAT", 3]]
    future = [["SELL", "MILK", 5]]
    adjusted_current = R2.r2_adjust_delay(current, "MILK", 3)
    adjusted_future = R2.r2_adjust_future(future, "MILK", 3)
    assert R2.r2_sell_quantity(adjusted_current, "MILK") == 7
    assert R2.r2_sell_quantity(adjusted_future, "MILK") == 8
    assert R2.r2_sell_quantity(current, "MILK") + R2.r2_sell_quantity(future, "MILK") == 15
    assert R2.r2_sell_quantity(adjusted_current, "MILK") + R2.r2_sell_quantity(adjusted_future, "MILK") == 15


def test_future_inventory_is_not_invented():
    result = R2.r2_simulate_interval(
        start_inventory={"MILK": 10000},
        start_money=[0, 0],
        sheds_by_step=[{0: {"MILK": 10}, 1: {"MILK": 0}}, {0: {}, 1: {}}],
        orders_by_step={0: [["SELL", "MILK", 7]], 1: [["SELL", "MILK", 8]]},
        opponent_orders_by_step={0: [], 1: []},
        start_step=0,
        end_step=1,
        target_item="MILK",
        shops=[],
        extra_player=0,
        extra_units=3,
    )
    assert result["executed"][0] == 10
    assert result["failed"][0] >= 1


def test_market_order_limit_and_target_only_accounting():
    orders = [["SELL", "WHEAT", 1] for _ in range(11)]
    result = R2.r2_process_market(
        [orders, []],
        {"WHEAT": 10000},
        [{"WHEAT": 20}, {}],
        [0, 0],
        "MILK",
    )
    assert result["executed"][0] == 10
    assert result["target_revenue"] == [0.0, 0.0]


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
