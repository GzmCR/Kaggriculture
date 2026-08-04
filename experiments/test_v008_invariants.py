"""Deterministic contract checks for the V008 hybrid router."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location("v008_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    candidate = load(ROOT / "baseline/history/v008_hybrid_router/main.py")
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[0][0] = {
        "kind": "PLANT",
        "crop": "MELON",
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 1,
    }
    obs = {
        "player": 0,
        "step": 18,
        "day": 0,
        "hour": 18,
        "farms": [{
            "money": 3000,
            "tiles": tiles,
            "farmer": [0, 0],
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }, {
            "money": 3000,
            "tiles": [["LOCKED" for _ in range(10)] for _ in range(10)],
            "farmer": [0, 0],
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }
    action = candidate.agent(obs)
    assert isinstance(action, dict)
    assert action["farmer"]
    assert action["hands"] == []

    route = {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"]], "market": []}
    normalized = candidate._v008_normalize_action(route, 1)
    assert len(normalized["hands"]) == 1
    padded = candidate._v008_normalize_action(route, 3)
    assert len(padded["hands"]) == 3

    current = {"farmer": ["WATER"], "hands": [], "market": []}
    assert candidate._v008_emergency_maintenance(obs, route, current)
    print("V008 invariants: PASS")


if __name__ == "__main__":
    main()
