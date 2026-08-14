"""Fast unit tests for V032-R3 action and inventory guards."""

from __future__ import annotations

import copy

from v032_r3_bidirectional import (
    r3_adjust_sell,
    r3_available_extra_inventory,
    r3_find_advance_events,
    r3_find_delay_events,
    r3_non_target_signature,
    r3_quantity_candidates,
    r3_sell_quantity,
)


def _action(market=None, farmer=None, hands=None):
    return {
        "farmer": farmer or ["PASS"],
        "hands": hands or [],
        "market": copy.deepcopy(market or []),
    }


def _obs(shed, item="MILK", player=0):
    return {
        "player": player,
        "private": {
            "shed": {item: shed},
            "inventories": [{}],
        },
        "farms": [
            {"farmer": [4, 4], "hands": [], "tiles": []},
            {"farmer": [4, 4], "hands": [], "tiles": []},
        ],
    }


def test_quantity_candidates_half_up_and_cap():
    assert r3_quantity_candidates(10, 10) == (1, 3, 5)
    assert r3_quantity_candidates(1, 1) == (1,)
    assert r3_quantity_candidates(100, 2) == (1,)
    assert r3_quantity_candidates(0, 0) == ()


def test_adjust_sell_preserves_non_target_and_order_limit():
    original = _action([
        ["SELL", "MILK", 10],
        ["BUY_SEED", "WHEAT", 1],
    ])
    added = r3_adjust_sell(original, "MILK", 3)
    assert r3_sell_quantity(added, "MILK") == 13
    assert r3_non_target_signature(added, "MILK") == r3_non_target_signature(original, "MILK")

    reduced = r3_adjust_sell(original, "MILK", -10)
    assert r3_sell_quantity(reduced, "MILK") == 0
    assert reduced["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert r3_adjust_sell(original, "MILK", -11) is None

    ten_orders = _action([["BUY_SEED", "WHEAT", 1]] * 10)
    assert r3_adjust_sell(ten_orders, "MILK", 1) is None


def test_real_shed_extra_inventory_excludes_same_turn_sell():
    obs = _obs(5)
    action = _action([["SELL", "MILK", 3]])
    assert r3_available_extra_inventory(obs, action, "MILK") == 2

    # A current SELL does not make future production sellable.  With no
    # shed item, the extra inventory is zero even if the order asks for one.
    assert r3_available_extra_inventory(_obs(0), _action([["SELL", "MILK", 1]]), "MILK") == 0


def test_event_generation_for_advance_and_delay():
    actions = {
        400: _action([["SELL", "MILK", 10]]),
        410: _action([["SELL", "MILK", 5]]),
        500: _action([["SELL", "STRAWBERRY", 4]]),
    }
    advances = r3_find_advance_events(actions, "MILK", horizons=(1, 2, 3), min_step=400)
    assert {(row["start_step"], row["end_step"], row["horizon"])
            for row in advances} == {
                (397, 400, 3), (398, 400, 2), (399, 400, 1),
                (407, 410, 3), (408, 410, 2), (409, 410, 1),
            }
    delays = r3_find_delay_events(actions, "MILK", min_step=400)
    assert delays == [{
        "kind": "DELAY", "item": "MILK", "start_step": 400,
        "end_step": 410, "horizon": 10, "current_quantity": 10,
        "future_quantity": 5,
    }]


def test_non_target_signature_detects_changes_but_allows_reorder():
    left = _action([
        ["SELL", "MILK", 10],
        ["BUY_SEED", "WHEAT", 1],
        ["SELL", "WOOL", 2],
    ])
    reordered = _action([
        ["SELL", "WOOL", 2],
        ["SELL", "MILK", 10],
        ["BUY_SEED", "WHEAT", 1],
    ])
    changed = _action([
        ["SELL", "MILK", 10],
        ["BUY_SEED", "WHEAT", 2],
        ["SELL", "WOOL", 2],
    ])
    assert r3_non_target_signature(left, "MILK") == r3_non_target_signature(reordered, "MILK")
    assert r3_non_target_signature(left, "MILK") != r3_non_target_signature(changed, "MILK")


if __name__ == "__main__":
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} V032-R3 unit tests")
