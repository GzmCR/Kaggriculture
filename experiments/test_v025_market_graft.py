"""Structural tests for the V025 route14/V022c market graft."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "v025a_route14_v022c_market",
    "v025b_route14_v022c_open_market",
    "v025c_route14_v022c_mirror_market",
)


def load(name):
    path = ROOT / "baseline/history" / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def obs(module, step=0):
    action = module._V025_ACTIONS[step]
    hands = len(action.get("hands", []) or [])
    tiles = [[None for _ in range(10)] for _ in range(10)]
    farm = {
        "money": 200000, "tiles": tiles, "farmer": [0, 0],
        "hands": [[i + 1, 0] for i in range(hands)],
        "unlocked_quadrants": ["NW", "NE", "SW"],
    }
    return {
        "player": 0, "step": step, "farms": [farm, farm.copy()],
        "private": {"shed": {item: 1000 for item in ("MILK", "WOOL", "STRAWBERRY", "MELON", "WHEAT")}, "inventories": [{} for _ in range(hands + 1)]},
        "market": {"prices": {item: 100 for item in ("MILK", "WOOL", "STRAWBERRY", "MELON", "WHEAT")}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def main():
    for name in CANDIDATES:
        module = load(name)
        assert len(module._V025_ACTIONS) == 720
        for step in (0, 24, 48, 400, 718, 719):
            action = module.agent(obs(module, step))
            assert len(action.get("market", [])) <= 10
            assert isinstance(action.get("farmer"), list)
        assert module._V025_STATS["errors"] == 0, (name, module._V025_STATS)
    print(f"V025 market graft tests passed: {len(CANDIDATES)} candidates")


if __name__ == "__main__":
    main()
