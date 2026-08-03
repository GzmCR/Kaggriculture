"""Small deterministic checks for the V007 local overlays.

These checks do not replace the full environment matrix.  They verify the
key contract of V007: overlay code can replace only PASS and must not invent
movement or displace a baseline maintenance action.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empty_tiles(size=10):
    return [[None for _ in range(size)] for _ in range(size)]


def main():
    module = load(
        ROOT / "baseline/history/v007b_idle_fertilizer/main.py",
        "v007_invariant_module",
    )
    terminal_module = load(
        ROOT / "baseline/history/v007a_terminal_safe/main.py",
        "v007_terminal_invariant_module",
    )
    tiles = empty_tiles()
    tiles[0][0] = {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": 0,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "yield_units": 1,
        "fertilized_until_day": -1,
    }
    farm = {
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": [[1, 0]],
    }
    private = {
        "shed": {},
        "inventories": [{"FERTILIZER": 1}, {"WHEAT": 1}],
    }
    obs = {
        "day": 6,
        "step": 144,
        "market": {"prices": {"MELON": 250, "FERTILIZER": 100}},
    }

    baseline_field = {
        "farmer": ["PASS"],
        "hands": [["FEED"]],
        "liquidation": False,
    }
    overlaid = module._v007_apply_idle_fertilizer(
        obs, farm, private, baseline_field
    )
    assert overlaid["farmer"] == ["FERTILIZE"]
    assert overlaid["hands"] == [["FEED"]]
    assert overlaid["farmer"][0] not in {"NORTH", "SOUTH", "EAST", "WEST"}

    # A non-PASS baseline operation is immutable.
    protected = {
        "farmer": ["WATER"],
        "hands": [["PLANT", "WHEAT"]],
        "liquidation": False,
    }
    assert module._v007_apply_idle_fertilizer(
        obs, farm, private, protected
    ) == protected

    # Fertilizer is disabled during liquidation.
    liquidation = {
        "farmer": ["PASS"],
        "hands": [["PASS"]],
        "liquidation": True,
    }
    assert module._v007_apply_idle_fertilizer(
        obs, farm, private, liquidation
    ) == liquidation

    # The terminal overlay is also local and only substitutes PASS.
    terminal = {
        "farmer": ["PASS"],
        "hands": [["HARVEST"]],
        "liquidation": True,
    }
    terminal_result = terminal_module._v007_apply_terminal_overlay(
        {"step": 704}, farm, private, terminal
    )
    assert terminal_result["farmer"] == ["HARVEST"]
    assert terminal_result["hands"] == [["HARVEST"]]

    print("V007 overlay invariants: PASS")


if __name__ == "__main__":
    main()
