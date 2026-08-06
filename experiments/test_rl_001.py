"""Fast invariant tests for the RL-001 market selector."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import rl_001_selector as rl  # noqa: E402


def synthetic_obs(step=0):
    tile = [[None for _ in range(10)] for _ in range(10)]
    tile[0][0] = {"kind": "PLANT", "crop": "MELON", "yield_units": 4, "consecutive_unwatered": 0}
    tile[0][1] = {"kind": "PASTURE", "animal": "COW", "yield_units": 2, "consecutive_unfed": 0}
    other = [[None for _ in range(10)] for _ in range(10)]
    other[0][0] = {"kind": "PLANT", "crop": "STRAWBERRY", "yield_units": 5}
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [
            {"money": 5000, "tiles": tile, "farmer": [0, 0], "hands": [[1, 0]],
             "unlocked_quadrants": ["NW"], "hires_today": 1},
            {"money": 4500, "tiles": other, "farmer": [0, 0], "hands": [],
             "unlocked_quadrants": ["NW"], "hires_today": 0},
        ],
        "private": {
            "shed": {"MELON": 12, "MILK": 5, "WHEAT": 2},
            "seeds": {},
            "inventories": [{"MELON": 3}, {"MILK": 2}],
        },
        "market": {
            "prices": {"MELON": 100, "MILK": 70, "WOOL": 80, "STRAWBERRY": 90, "WHEAT": 10},
            "inventory": {"MELON": 100, "MILK": 100, "WOOL": 100, "STRAWBERRY": 100, "WHEAT": 100},
        },
        "town": {"unlocked_shops": ["MARKET", "BARN"]},
    }


def test_features_are_finite_and_bounded():
    encoder = rl.FeatureEncoder()
    vector = encoder.encode(synthetic_obs())
    assert vector.shape == (rl.FEATURE_DIM,)
    assert np.isfinite(vector).all()
    assert float(np.max(np.abs(vector))) <= 5.0


def test_overlay_preserves_field_actions_and_order_constraints():
    obs = synthetic_obs()
    base = {
        "farmer": ["EAST"],
        "hands": [["WATER"]],
        "market": [["BUY_SEED", "WHEAT", 1], ["SELL", "MILK", 2],
                   ["SELL", "MELON", 4], ["SELL", "WHEAT", 3]],
    }
    for mode in range(4):
        candidate = rl.apply_overlay(base, obs, mode, base_actions=[base, base])
        assert candidate["farmer"] == base["farmer"]
        assert candidate["hands"] == base["hands"]
        assert len(candidate["market"]) <= 10
        assert [order for order in candidate["market"] if order[0] == "BUY_SEED"] == [base["market"][0]]
        assert sum(int(order[2]) for order in candidate["market"] if order[0] == "SELL" and order[1] == "WHEAT") == 3


def test_window_is_locked():
    runtime = rl.SelectorRuntime(training=False, seed=3)
    modes = []
    for step in range(48):
        obs = synthetic_obs(step)
        modes.append(runtime.choose(obs))
    assert len(set(modes)) == 1
    runtime.choose(synthetic_obs(48))
    assert runtime.last_action is not None


def test_stop_boundary_closes_previous_block():
    runtime = rl.SelectorRuntime(training=True, seed=3)
    for step in (0, 48, 96, 624, 672):
        obs = synthetic_obs(step)
        runtime.choose(obs)
    assert runtime.last_action is None


def test_q_action_range():
    q = rl.DoubleLinearQ(seed=1)
    action = q.select(np.zeros(rl.FEATURE_DIM), epsilon=0.0)
    assert 0 <= action < rl.ACTION_COUNT


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
