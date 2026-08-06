"""Fast V021 invariants and synthetic gate tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from v020_value_aware_market import MARKET
from v021_product_gate import V021ProductGateController


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "baseline/artifacts/v021_product_gate"


def _obs(step=300, own_money=10000, other_money=10000, prices=None, inventory=None):
    prices = dict(prices or {item: values[0] for item, values in MARKET.items()})
    inventory = dict(inventory or {item: 10000 for item in MARKET})
    farm = {"money": own_money, "hands": [], "tiles": [], "unlocked_quadrants": ["NW", "NE", "SW"]}
    other = {"money": other_money, "hands": [], "tiles": [], "unlocked_quadrants": ["NW", "NE", "SW"]}
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, other],
        "private": {"shed": {"MILK": 100, "WOOL": 100}, "inventories": []},
        "market": {"prices": prices, "inventory": inventory},
        "town": {"unlocked_shops": []},
    }


def _base_action():
    return {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["SELL", "MILK", 10],
            ["SELL", "WOOL", 10],
            ["SELL", "STRAWBERRY", 10],
            ["SELL", "MELON", 10],
            ["SELL", "WHEAT", 2],
        ],
    }


def _load_agent(path):
    spec = importlib.util.spec_from_file_location(f"v021_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_passthrough():
    controller = V021ProductGateController("win_guard", runtime={})
    action = _base_action()
    assert controller.apply(_obs(step=648), action) == action


def test_disabled_products_passthrough():
    controller = V021ProductGateController("product_gate", runtime={})
    obs = _obs(step=300, prices={"MILK": 100, "WOOL": 100, "STRAWBERRY": 80, "MELON": 150})
    result = controller.apply(obs, _base_action())
    result_by_item = {order[1]: order[2] for order in result["market"] if order[0] == "SELL"}
    assert result_by_item["STRAWBERRY"] == 10
    assert result_by_item["MELON"] == 10


def test_win_guard_blocks_when_behind():
    controller = V021ProductGateController("win_guard", runtime={})
    obs = _obs(step=300, own_money=1000, other_money=3000, prices={"MILK": 100, "WOOL": 100})
    result = controller.apply(obs, _base_action())
    assert result == _base_action()


def test_only_existing_orders_and_quantity_bound():
    controller = V021ProductGateController("win_guard", runtime={})
    action = _base_action()
    result = controller.apply(_obs(step=300), action)
    original = {(order[0], order[1]): order[2] for order in action["market"] if len(order) >= 3}
    for order in result["market"]:
        if len(order) >= 3 and order[0] == "SELL" and order[1] in {"MILK", "WOOL", "STRAWBERRY", "MELON"}:
            assert (order[0], order[1]) in original
            assert 0 <= order[2] <= original[(order[0], order[1])]
    assert len(result["market"]) <= 10


def test_recovery_releases_pending_budget():
    controller = V021ProductGateController("product_gate", runtime={})
    state = controller._state(0)
    state["pending_units"]["MILK"] = 1
    state["defer_budget"]["MILK"] = 0
    state["shock_cooldown"]["MILK"] = True
    signal = {"shock": False, "recovered": True}
    controller._update_recovery(state, "MILK", signal)
    controller._update_recovery(state, "MILK", signal)
    assert state["pending_units"]["MILK"] == 0
    assert state["defer_budget"]["MILK"] == 1
    assert state["shock_cooldown"]["MILK"] is False


def test_self_contained_artifacts_import():
    for name in ("v021a_safety_patch.py", "v021b_product_gate.py", "v021c_win_guard.py", "main.py"):
        module = _load_agent(ARTIFACTS / name)
        assert callable(module.agent)


def main():
    tests = [
        test_terminal_passthrough,
        test_disabled_products_passthrough,
        test_win_guard_blocks_when_behind,
        test_only_existing_orders_and_quantity_bound,
        test_recovery_releases_pending_budget,
        test_self_contained_artifacts_import,
    ]
    for test in tests:
        test()
    print(f"V021 product-gate invariants: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
