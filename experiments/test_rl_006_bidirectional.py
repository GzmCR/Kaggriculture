"""Fast RL-006 invariants and source smoke tests."""

from __future__ import annotations

import ast
import sys
import types

import numpy as np

from build_rl_006_bidirectional import build_source
from rl_006_bidirectional_timing import (
    RL006_ACTION_NAMES,
    RL006_FEATURE_DIM,
    RL006History,
    rl006_action_direction,
    rl006_action_quantity,
    rl006_adjust_sell,
    rl006_features,
)


def test_action_quantities_and_directions():
    opportunity = {
        "item": "MILK",
        "current_step": 215,
        "future_step": 260,
        "current_quantity": 6,
        "future_quantity": 3,
    }
    assert rl006_action_direction(0) == "CONTROL"
    assert rl006_action_direction(1) == "PREEMPT"
    assert rl006_action_direction(6) == "DELAY"
    assert [rl006_action_quantity(i, opportunity) for i in range(1, 7)] == [1, 1, 2, 1, 2, 3]


def test_signed_adjustment_preserves_total():
    action = {"market": [["SELL", "MILK", 6], ["SELL", "WHEAT", 1]]}
    before = sum(order[2] for order in action["market"] if order[1] == "MILK")
    assert rl006_adjust_sell(action, "MILK", -2) == 2
    future = {"market": [["SELL", "MILK", 3], ["SELL", "WHEAT", 1]]}
    assert rl006_adjust_sell(future, "MILK", 2) == 2
    after = sum(order[2] for order in action["market"] if order[1] == "MILK") + sum(
        order[2] for order in future["market"] if order[1] == "MILK"
    )
    assert before + 3 == after


def test_feature_vector_is_finite():
    obs = {
        "step": 215,
        "day": 8,
        "hour": 23,
        "player": 0,
        "farms": [
            {"money": 3000, "tiles": [[None]], "hands": [], "unlocked_quadrants": ["NW"]},
            {"money": 3000, "tiles": [[None]], "hands": [], "unlocked_quadrants": ["NW"]},
        ],
        "private": {"shed": {"MILK": 8}, "inventories": [{"MILK": 1}]},
        "market": {
            "prices": {"MILK": 160, "WOOL": 200, "STRAWBERRY": 120, "MELON": 250},
            "inventory": {"MILK": 10000, "WOOL": 10000, "STRAWBERRY": 10000, "MELON": 10000},
        },
        "town": {"unlocked_shops": []},
    }
    history = RL006History()
    history.observe(obs)
    opportunity = {
        "item": "MILK",
        "current_step": 215,
        "future_step": 260,
        "current_quantity": 6,
        "future_quantity": 3,
        "gap": 45,
    }
    features = rl006_features(obs, opportunity, history, {"market": [["SELL", "MILK", 6]]})
    assert features.shape == (RL006_FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_generated_source_has_public_agent_last():
    payload = {
        "feature_dim": RL006_FEATURE_DIM,
        "feature_mean": [0.0] * RL006_FEATURE_DIM,
        "feature_scale": [1.0] * RL006_FEATURE_DIM,
        "min_support": 12,
        "min_expected_delta": 5.0,
        "lcb_z": 1.5,
        "allowed_actions": [1, 2, 3, 4, 5, 6],
        "models": {},
    }
    source, _ = build_source(payload)
    tree = ast.parse(source)
    callables = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert callables[-1] == "agent"
    module = types.ModuleType("rl006_generated_test")
    exec(compile(source, "rl006_generated_test.py", "exec"), module.__dict__)
    assert callable(module.agent)


if __name__ == "__main__":
    for function in (
        test_action_quantities_and_directions,
        test_signed_adjustment_preserves_total,
        test_feature_vector_is_finite,
        test_generated_source_has_public_agent_last,
    ):
        function()
    print("RL-006 invariant tests passed")
