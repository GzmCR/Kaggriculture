"""Fast structural and synthetic checks for generated V024 agents."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "v024a_route14_control",
    "v024b_route14_weed",
    "v024c_route14_order_memory",
    "v024d_route14_strict_r3",
)


def _load(name):
    path = ROOT / "baseline/history" / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(module, step=0, weed_actor=None):
    route = module._V024_ACTIONS
    action = route[step]
    hands = len(action.get("hands", []) or [])
    tiles = [[None for _ in range(10)] for _ in range(10)]
    positions = [[0, 0]] + [[i + 1, 0] for i in range(hands)]
    if weed_actor is not None:
        x, y = positions[weed_actor]
        tiles[y][x] = {"kind": "WEED"}
    farm = {
        "money": 200000,
        "tiles": tiles,
        "farmer": positions[0],
        "hands": positions[1:],
        "unlocked_quadrants": ["NW", "NE", "SW"],
    }
    return {
        "player": 0, "step": step, "farms": [farm, farm.copy()],
        "private": {"shed": {item: 1000 for item in ("MILK", "WOOL", "STRAWBERRY", "MELON", "WHEAT")}, "inventories": [{} for _ in range(hands + 1)]},
        "market": {"prices": {item: 100 for item in ("MILK", "WOOL", "STRAWBERRY", "MELON", "WHEAT")}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def _actor_action(action, actor):
    return action.get("farmer", ["PASS"]) if actor == 0 else (action.get("hands", []) or [])[actor - 1]


def main():
    for name in CANDIDATES:
        module = _load(name)
        assert len(module._V024_ACTIONS) == 720, name
        for step in (0, 48, 400, 718, 719):
            action = module.agent(_obs(module, step))
            assert isinstance(action, dict)
            assert len(action.get("market", [])) <= 10
            assert isinstance(action.get("farmer"), list)
        # At the first planned PLANT/BUILD_PASTURE, V024b/c/d must intercept a
        # visible weed; V024a is the no-recovery control.
        if name != "v024a_route14_control":
            for step, route_action in enumerate(module._V024_ACTIONS):
                ops = [route_action.get("farmer", ["PASS"]), *route_action.get("hands", [])]
                actor = next((i for i, op in enumerate(ops) if op and op[0] in {"PLANT", "BUILD_PASTURE"}), None)
                if actor is None:
                    continue
                first = module.agent(_obs(module, step, actor))
                assert _actor_action(first, actor)[0] == "DIG", (name, step, actor, first)
                second_obs = _obs(module, step + 1, None)
                second = module.agent(second_obs)
                assert _actor_action(second, actor)[0] in {"PLANT", "BUILD_PASTURE"}, (name, step, actor, second)
                break
            else:
                raise AssertionError(f"no plant action found for {name}")
    print(f"V024 structural tests passed: {len(CANDIDATES)} candidates")


if __name__ == "__main__":
    main()
