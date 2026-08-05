"""Unit tests for the V016 market-only selector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v015a_market_overlay import MarketCollisionOverlay, prepare_observation
from v016_market_selector import MarketValueSelector


EXPERT_NAMES = ("mohit", "automatylicza", "manual_player", "navazsh_fathi", "lucien_de_rubempre")


def action(orders):
    return {"farmer": ["PASS"], "hands": [], "market": [list(order) for order in orders]}


def obs(step=0, prices=None, opponent_melon=0, shed=None):
    prices = prices or {"MELON": 220, "STRAWBERRY": 180, "MILK": 190, "WOOL": 200, "WHEAT": 10}
    opponent_tiles = [[None]]
    for _ in range(opponent_melon):
        opponent_tiles[0].append({"kind": "PLANT", "crop": "MELON", "yield_units": 4})
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "market": {"prices": prices, "inventory": {},},
        "farms": [
            {"money": 50000, "hires_today": 0, "unlocked_quadrants": ["NW", "NE", "SW"], "hands": [], "tiles": [[None]]},
            {"money": 50000, "hands": [], "tiles": opponent_tiles},
        ],
        "private": {"shed": shed or {"WHEAT": 20}, "inventories": [{}]},
    }


def runtime(spec):
    actions = {}
    for name in EXPERT_NAMES:
        orders = spec.get(name, [])
        actions[name] = {"actions": [action(orders) for _ in range(720)]}
    return {"experts": actions, "board_by_seat": {"0": "automatylicza", "1": "automatylicza"}}


def test_value_only_prefers_revenue():
    rt = runtime({
        "mohit": [["SELL", "MELON", 100]],
        "automatylicza": [["SELL", "MILK", 1]],
    })
    selector = MarketValueSelector("value_only", rt)
    assert selector.choose(obs()) == "mohit"


def test_collision_hedged_prefers_less_exposed_supply():
    rt = runtime({
        "mohit": [["SELL", "MELON", 100]],
        "automatylicza": [["SELL", "MILK", 1]],
    })
    selector = MarketValueSelector("collision_hedged", rt)
    assert selector.choose(obs(opponent_melon=50)) == "automatylicza"


def test_aggressive_remains_revenue_seeking():
    rt = runtime({
        "mohit": [["SELL", "MELON", 100]],
        "automatylicza": [["SELL", "MILK", 1]],
    })
    selector = MarketValueSelector("aggressive_value", rt)
    assert selector.choose(obs(opponent_melon=1)) == "mohit"


def test_selection_is_daily_and_replaces_market_only():
    rt = runtime({"mohit": [["SELL", "MELON", 4]], "automatylicza": [["SELL", "MILK", 1]]})
    selector = MarketValueSelector("value_only", rt)
    first = selector.apply(obs(0), {"farmer": ["EAST"], "hands": [["WEST"]], "market": []})
    second = selector.apply(obs(1), {"farmer": ["NORTH"], "hands": [["SOUTH"]], "market": []})
    next_day = selector.apply(obs(24), {"farmer": ["PASS"], "hands": [], "market": []})
    assert first["farmer"] == ["EAST"] and first["hands"] == [["WEST"]]
    assert second["farmer"] == ["NORTH"] and second["hands"] == [["SOUTH"]]
    assert first["market"] == [["SELL", "MELON", 4]]
    assert second["market"] == first["market"]
    assert next_day["market"] == first["market"]
    assert selector.diagnostics()["selection_count"] == 2


def test_overlay_still_limits_delay_and_terminal_flush():
    rt = runtime({"mohit": [["SELL", "MILK", 4]], "automatylicza": [["SELL", "MILK", 4]]})
    selector = MarketValueSelector("collision_hedged", rt)
    overlay = MarketCollisionOverlay()
    first = prepare_observation(obs(0), overlay)
    out0 = overlay.apply(first, selector.apply(first, action([["SELL", "MILK", 4]])))
    assert out0["market"] == [["SELL", "MILK", 4]]
    shock = obs(1, prices={"MELON": 220, "STRAWBERRY": 180, "MILK": 100, "WOOL": 200, "WHEAT": 10})
    shock["market"]["inventory"]["MILK"] = 20
    prepared = prepare_observation(shock, overlay)
    out1 = overlay.apply(prepared, selector.apply(prepared, action([["SELL", "MILK", 4]])))
    assert out1["market"] == [["SELL", "MILK", 3]]
    terminal = obs(696, shed={"MILK": 4})
    prepared = prepare_observation(terminal, overlay)
    out2 = overlay.apply(prepared, selector.apply(prepared, action([])))
    assert len(out2["market"]) <= 10
    assert any(order[0] == "SELL" and order[1] == "MILK" for order in out2["market"])


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"V016 selector invariants: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()

