"""Fast invariants for the V015a market-only overlay."""

from __future__ import annotations

from v015a_market_overlay import MarketCollisionOverlay


def observation(step, price=100, supply=100, shed=None):
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "market": {
            "prices": {item: price for item in ("MELON", "STRAWBERRY", "MILK", "WOOL")},
            "inventory": {item: supply for item in ("MELON", "STRAWBERRY", "MILK", "WOOL")},
        },
        "farms": [{"hands": [], "farmer": [0, 0], "tiles": [[None]]}],
        "private": {"shed": shed or {}},
    }


def base_action(*orders):
    return {"farmer": ["PASS"], "hands": [], "market": [list(order) for order in orders]}


def main():
    # Without a price shock the complete action is unchanged.
    overlay = MarketCollisionOverlay()
    first = overlay.apply(observation(0), base_action(["SELL", "MILK", 4], ["BUY_SEED", "WHEAT", 1]))
    assert first == base_action(["SELL", "MILK", 4], ["BUY_SEED", "WHEAT", 1])

    # A 30% one-turn fall plus a supply jump delays only one unit of the
    # product, while preserving farmer/hands and the non-premium order.
    shock = overlay.apply(
        observation(1, price=70, supply=120),
        base_action(["SELL", "MILK", 5], ["SELL", "WHEAT", 3]),
    )
    assert shock["farmer"] == ["PASS"] and shock["hands"] == []
    assert [order for order in shock["market"] if order[0] == "SELL" and order[1] == "WHEAT"] == [["SELL", "WHEAT", 3]]
    assert [order for order in shock["market"] if order[0] == "SELL" and order[1] == "MILK"] == [["SELL", "MILK", 4]]
    assert overlay.diagnostics()["delayed_units"] == 1

    # A recovery releases the held unit only through an existing premium sale.
    recovered = overlay.apply(observation(2, price=100, supply=90), base_action(["SELL", "MILK", 2]))
    assert recovered["market"] == [["SELL", "MILK", 3]]
    assert overlay.diagnostics()["released_units"] == 1

    # Duplicate same-product slots still defer at most one unit per turn.
    overlay.reset()
    overlay.apply(observation(0), base_action(["SELL", "WOOL", 4]))
    duplicate = overlay.apply(
        observation(1, price=70, supply=120),
        base_action(["SELL", "WOOL", 4], ["SELL", "WOOL", 4]),
    )
    assert sum(int(order[2]) for order in duplicate["market"] if order[1] == "WOOL") == 7
    assert overlay.diagnostics()["delayed_units"] == 1

    # Terminal cleanup can append a premium order but never exceeds ten orders.
    overlay.reset()
    terminal = overlay.apply(
        observation(696, shed={"MILK": 12, "WOOL": 8, "STRAWBERRY": 6}),
        base_action(*[["BUY_PRODUCT", "WHEAT", 1]] * 9),
    )
    assert len(terminal["market"]) <= 10
    assert any(order[0] == "SELL" for order in terminal["market"])

    print("V015a invariants: PASS")


if __name__ == "__main__":
    main()
