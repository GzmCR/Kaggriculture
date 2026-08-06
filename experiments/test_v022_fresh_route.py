"""V022 candidate import, legality and actor-local WEED tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = tuple(
    ROOT / "baseline/artifacts" / name / "main.py"
    for name in (
        "v022a_weed_recovery",
        "v022b_fresh_medoid",
        "v022c_medoid_recovery",
        "v022d_medoid_recovery_tactical",
    )
)


def _load(path):
    spec = importlib.util.spec_from_file_location(f"v022_test_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(step=100, weed=False, hands=1):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    if weed:
        tiles[0][0] = {"kind": "WEED"}
    farm = {
        "money": 10000,
        "farmer": [0, 0],
        "hands": [[1, 0] for _ in range(hands)],
        "tiles": tiles,
        "unlocked_quadrants": ["NW", "NE", "SW"],
    }
    other = {
        "money": 10000,
        "farmer": [0, 0],
        "hands": [[1, 0] for _ in range(hands)],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
        "unlocked_quadrants": ["NW", "NE", "SW"],
    }
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, other],
        "private": {"shed": {}, "seeds": {"WHEAT": 1}, "inventories": [{}, {}]},
        "market": {"prices": {"WHEAT": 10, "MILK": 100, "WOOL": 100}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def test_all_candidates_import():
    for path in CANDIDATES:
        module = _load(path)
        assert callable(module.agent)


def test_action_shape_and_order_cap():
    for path in CANDIDATES:
        module = _load(path)
        action = module.agent(_obs(step=300))
        assert isinstance(action, dict)
        assert isinstance(action.get("farmer"), list)
        assert isinstance(action.get("hands"), list)
        assert isinstance(action.get("market"), list)
        assert len(action["market"]) <= 10
        assert all(isinstance(item, list) and item for item in action["market"])


def test_v022a_weed_transaction_is_actor_local():
    module = _load(CANDIDATES[0])
    obs = _obs(step=100, weed=True, hands=1)
    action = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["WATER"]],
        "market": [["BUY_SEED", "WHEAT", 1]],
    }
    repaired = module._v022_weed_repair_action(obs, action, 100)
    assert repaired["farmer"] == ["DIG"]
    assert repaired["hands"] == [["WATER"]]
    assert repaired["market"] == action["market"]

    obs_retry = _obs(step=101, weed=False, hands=1)
    retry = module._v022_weed_repair_action(obs_retry, action, 101)
    assert retry["farmer"] == ["PLANT", "WHEAT"]
    assert retry["hands"] == [["WATER"]]
    assert retry["market"] == action["market"]


def test_v022a_no_weed_passthrough():
    module = _load(CANDIDATES[0])
    obs = _obs(step=200, weed=False, hands=1)
    action = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["WATER"]],
        "market": [["BUY_SEED", "WHEAT", 1]],
    }
    assert module._v022_weed_repair_action(obs, action, 200) == module._v022_align_hands(action, obs)


def test_weed_recovery_has_bounded_catchup():
    module = _load(CANDIDATES[2])
    assert int(getattr(module, "_WEED_REPLAY_STEPS", 0)) == 8
    assert callable(getattr(module, "_weed_repair_action", None))


def main():
    tests = [
        test_all_candidates_import,
        test_action_shape_and_order_cap,
        test_v022a_weed_transaction_is_actor_local,
        test_v022a_no_weed_passthrough,
        test_weed_recovery_has_bounded_catchup,
    ]
    for test in tests:
        test()
    print(f"V022 fresh-route invariants: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
