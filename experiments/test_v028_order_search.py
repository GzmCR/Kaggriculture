"""Structural tests for V028 quantity-preserving order permutations."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "v028a": ROOT / "baseline/history/v028_order_search/v028a_marginal_order/main.py",
    "v028b": ROOT / "baseline/history/v028_order_search/v028b_safe_order/main.py",
    "v028c": ROOT / "baseline/history/v028_order_search/v028c_robust_order/main.py",
}


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"test_v028_{tag}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _farm(hands):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    return {
        "money": 200000,
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": [[(index + 1) % 10, (index + 1) // 10] for index in range(hands)],
        "unlocked_quadrants": ["NW"],
    }


def _obs(module, step, market=None):
    route_step = min(max(0, step), len(module._ACTIONS) - 1)
    hands = len((module._ACTIONS[route_step] or {}).get("hands", []) or [])
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [_farm(hands), _farm(hands)],
        "private": {
            "shed": {
                "MELON": 1000,
                "STRAWBERRY": 1000,
                "MILK": 1000,
                "WOOL": 1000,
            },
            "inventories": [{} for _ in range(hands + 1)],
        },
        "market": market or {
            "inventory": {
                "MELON": 9970,
                "STRAWBERRY": 9970,
                "MILK": 9970,
                "WOOL": 9970,
                "WHEAT": 10000,
                "FERTILIZER": 10000,
            },
            "prices": {
                "MELON": 250,
                "STRAWBERRY": 120,
                "MILK": 160,
                "WOOL": 200,
            },
        },
        "town": {"unlocked_shops": []},
    }


def _sell_multiset(market, premium_only=False):
    rows = []
    for order in market or []:
        if not module_is_sell(order):
            continue
        if premium_only and str(order[1]).upper() not in {"MELON", "STRAWBERRY", "MILK", "WOOL"}:
            continue
        rows.append((str(order[1]).upper(), int(order[2])))
    return sorted(rows)


def module_is_sell(order):
    return isinstance(order, (list, tuple)) and len(order) >= 3 and str(order[0]).upper() == "SELL"


def _find_rich_window(module):
    for step in range(0, 648):
        base = module._v028_base_action(_obs(module, step), step)
        if len(module._v028_premium_positions(base.get("market", []))) >= 2:
            return step
    raise AssertionError("route has no multi-premium order window")


def main():
    for name, path in CANDIDATES.items():
        module = _load(path, name)
        assert len(module._ACTIONS) in {719, 720}, (name, len(module._ACTIONS))
        callables = [key for key, value in module.__dict__.items() if callable(value)]
        assert callables[-1] == "agent", (name, callables[-5:])
        step = _find_rich_window(module)
        obs = _obs(module, step)
        base = module._v028_base_action(obs, step)
        actual = module.agent(obs)
        assert actual["farmer"] == base["farmer"], name
        assert actual["hands"] == base["hands"], name
        assert len(actual["market"]) <= 10, name
        assert _sell_multiset(actual["market"]) == _sell_multiset(base["market"]), name
        assert _sell_multiset(actual["market"], premium_only=True) == _sell_multiset(
            base["market"], premium_only=True
        ), name
        for index, order in enumerate(base["market"]):
            if not module._v028_is_premium_sell(order):
                assert actual["market"][index] == order, (name, index, order, actual["market"])
        terminal_obs = _obs(module, 672)
        terminal_base = module._v028_base_action(terminal_obs, 672)
        terminal_actual = module.agent(terminal_obs)
        assert terminal_actual == terminal_base, name

    # The simulator must credit a queue mismatch when the shadow sells a
    # different premium product in the same order slot.
    module = _load(CANDIDATES["v028a"], "synthetic")
    obs = _obs(module, 0)
    ours = [["SELL", "MELON", 4]]
    same = [["SELL", "MELON", 4]]
    different = [["SELL", "STRAWBERRY", 4]]
    same_value = module._v028_simulate(ours, same, obs)
    different_value = module._v028_simulate(ours, different, obs)
    assert different_value >= same_value, (same_value, different_value)
    print(f"V028 structural tests passed: {len(CANDIDATES)} candidates")


if __name__ == "__main__":
    main()
