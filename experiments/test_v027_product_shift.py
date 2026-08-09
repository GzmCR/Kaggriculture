"""Structural and synthetic tests for V027 market-only candidates."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "v027a": ROOT / "baseline/history/v027_v22_product_shift/v027a_melon_ratio/main.py",
    "v027b": ROOT / "baseline/history/v027_v22_product_shift/v027b_mirror_gated/main.py",
    "v027c": ROOT / "baseline/history/v027_v22_product_shift/v027c_product_specific/main.py",
}


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"test_v027_{tag}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _farm(hands, unlocked=None, animals=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    for index, animal in enumerate(animals or []):
        x, y = index % 10, index // 10
        tiles[y][x] = {"kind": "PASTURE", "animal": animal, "yield_units": 2}
    return {
        "money": 200000,
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": [[(index + 1) % 10, (index + 1) // 10] for index in range(hands)],
        "unlocked_quadrants": unlocked or ["NW"],
    }


def _obs(module, step, opponent=None):
    route_step = min(max(0, step), len(module._ACTIONS) - 1)
    hands = len((module._ACTIONS[route_step] or {}).get("hands", []) or [])
    mine = _farm(hands)
    other = opponent or _farm(0)
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [mine, other],
        "private": {
            "shed": {"MELON": 1000, "STRAWBERRY": 1000},
            "inventories": [{} for _ in range(hands + 1)],
        },
        "market": {
            "inventory": {"MELON": 9000, "STRAWBERRY": 9000},
            "prices": {"MELON": 300, "STRAWBERRY": 140},
        },
        "town": {"unlocked_shops": []},
    }


def _find_shift_window(module, item="MELON"):
    for step in range(0, 648):
        current = module._v027_route_quantity(step, item)
        if current <= 0:
            continue
        for future in range(step + 1, min(672, len(module._ACTIONS))):
            if module._v027_route_quantity(future, item) >= 4:
                return step, future
    raise AssertionError(f"no {item} shift window in route")


def _sell_qty(action, item):
    return sum(
        int(order[2]) for order in action.get("market", [])
        if len(order) >= 3 and order[0] == "SELL" and order[1] == item
    )


def _expected_v22(module, obs, step):
    route_step = min(max(0, step), len(module._ACTIONS) - 1)
    action = module._align_hands(module._copy_action(module._ACTIONS[route_step]), obs)
    return module._align_hands(module._impact_slots(obs, action), obs)


def main():
    modules = {name: _load(path, name) for name, path in CANDIDATES.items()}
    for name, module in modules.items():
        assert len(module._ACTIONS) in {719, 720}, (name, len(module._ACTIONS))
        callables = [key for key, value in module.__dict__.items() if callable(value)]
        assert callables[-1] == "agent", (name, callables[-5:])
        step, future = _find_shift_window(module)
        module._WEED_STATE[0] = {"last_step": -1, "active": {}}
        base = _expected_v22(module, _obs(module, step), step)
        actual = module.agent(_obs(module, step))
        assert actual["farmer"] == base["farmer"]
        assert actual["hands"] == base["hands"]
        assert len(actual["market"]) <= 10
        if name == "v027a":
            assert module._V027_STATS["shifted_sell_units"]["MELON"] > 0, name
            assert module._V027_STATS["shifted_sell_units"]["MELON"] <= 6, name
            assert _sell_qty(actual, "MELON") > _sell_qty(base, "MELON"), name
        else:
            assert module._V027_STATS["shifted_sell_units"]["MELON"] == 0, name
            assert actual["market"] == base["market"], name
        if name == "v027c":
            assert module._V027_STATS["shifted_sell_units"]["STRAWBERRY"] == 0, name
        before = _expected_v22(module, _obs(module, 672), 672)
        after = module.agent(_obs(module, 672))
        assert after == before, name

    mirror = modules["v027b"]
    animals = ["COW"] * 7 + ["SHEEP"] * 5
    public_match = _farm(13, ["NW", "NE", "SW"], animals)
    mirror.agent(_obs(mirror, 0))
    mirror.agent(_obs(mirror, 8 * 24 + 6, public_match))
    mirror.agent(_obs(mirror, 12 * 24 + 6, public_match))
    assert mirror._V027_STATE[0]["mirror_mode"] is True
    assert mirror._V027_STATS["mirror_latches"] == 1
    mismatch = _farm(0, ["NW"], [])
    mirror.agent(_obs(mirror, 13 * 24 + 6, mismatch))
    mirror.agent(_obs(mirror, 14 * 24 + 6, mismatch))
    assert mirror._V027_STATE[0]["mirror_mode"] is False
    assert mirror._V027_STATS["mirror_releases"] == 1
    print(f"V027 structural tests passed: {len(modules)} candidates")


if __name__ == "__main__":
    main()
