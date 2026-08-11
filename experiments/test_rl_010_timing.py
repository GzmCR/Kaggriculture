"""Fast invariants and optional 720-turn smoke test for RL-010."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

from kaggle_environments import make

from rl_010_milk_bidirectional import (
    RL010_FEATURE_DIM,
    RL010Runtime,
    rl010_mask_features,
    rl010_round_half_up,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "baseline/artifacts/rl_010_milk_bidirectional"


def _constant_model(intercept, support=24, bad_intercept=0.0):
    model = {
        "mean": [0.0] * RL010_FEATURE_DIM,
        "scale": [1.0] * RL010_FEATURE_DIM,
        "beta": [0.0] * RL010_FEATURE_DIM,
        "intercept": float(intercept),
        "uncertainty": 1.0,
    }
    bad = dict(model)
    bad["intercept"] = float(bad_intercept)
    bad["uncertainty"] = 0.01
    return {"support": support, "margin": model, "bad": bad}


def _obs(step):
    return {
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "player": 0,
        "farms": [
            {
                "money": 50000,
                "farmer": [0, 0],
                "hands": [],
                "tiles": [[None]],
                "unlocked_quadrants": ["NW"],
            },
            {
                "money": 50000,
                "farmer": [0, 0],
                "hands": [],
                "tiles": [[None]],
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {
            "shed": {"MILK": 10},
            "inventories": [{}],
            "seeds": {},
        },
        "market": {"prices": {"MILK": 160}, "inventory": {"MILK": 10000}},
        "town": {"unlocked_shops": []},
    }


def test_rounding_and_mask():
    assert rl010_round_half_up(0.5) == 1
    assert rl010_round_half_up(1.49) == 1
    assert rl010_round_half_up(1.5) == 2
    vector = list(range(RL010_FEATURE_DIM))
    masked = rl010_mask_features(vector, include_opponent=False)
    assert masked[19] == 0 and masked[20] == 0 and masked[30] == 0
    assert masked[18] == vector[18] and masked[21] == vector[21]


def test_bidirectional_delay_and_repayment():
    opportunity = {
        "item": "MILK",
        "current_step": 216,
        "future_step": 264,
        "current_quantity": 6,
        "future_quantity": 3,
        "gap": 48,
    }
    key = "MILK|216|264"
    payload = {
        "feature_dim": RL010_FEATURE_DIM,
        "allowed_actions": ["DELAY_50"],
        "include_opponent_features": True,
        "models": {f"{key}|DELAY_50": _constant_model(10.0)},
    }
    runtime = RL010Runtime(payload=payload, opportunities=[opportunity])
    first = runtime.act(_obs(216), {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 6]]})
    assert first["market"] == [["SELL", "MILK", 3]]
    assert runtime.interventions == 1
    second = runtime.act(_obs(264), {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 3]]})
    assert second["market"] == [["SELL", "MILK", 6]]
    assert runtime.repayment_successes == 1
    assert runtime.repayment_failures == 0


def test_cutoff_and_no_new_order():
    opportunity = {
        "item": "MILK",
        "current_step": 648,
        "future_step": 649,
        "current_quantity": 6,
        "future_quantity": 3,
        "gap": 1,
    }
    key = "MILK|647|648"
    payload = {
        "feature_dim": RL010_FEATURE_DIM,
        "allowed_actions": ["DELAY_50"],
        "models": {f"{key}|DELAY_50": _constant_model(10.0)},
    }
    runtime = RL010Runtime(payload=payload, opportunities=[opportunity])
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    result = runtime.act(_obs(648), action)
    assert result == action
    assert runtime.interventions == 0


def smoke(variant="rl010c_bidirectional_opp"):
    path = ARTIFACT_ROOT / variant / "main.py"
    if not path.exists():
        path = ARTIFACT_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("rl010_timing_smoke", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": 17},
        debug=False,
    )
    env.run([module.agent, "starter"])
    final = env.steps[-1]
    assert all(state.status == "DONE" for state in final)
    runtime = getattr(module, "_RL010_RUNTIME", None)
    assert runtime is not None
    assert runtime.errors == 0
    assert runtime.repayment_failures == 0
    return {
        "variant": variant,
        "done": True,
        "errors": runtime.errors,
        "repayment_failures": runtime.repayment_failures,
        "interventions": runtime.interventions,
        "fallbacks": runtime.fallbacks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", default="rl010c_bidirectional_opp")
    args = parser.parse_args()
    test_rounding_and_mask()
    test_bidirectional_delay_and_repayment()
    test_cutoff_and_no_new_order()
    result = {"unit_tests": "passed"}
    if args.smoke:
        result["smoke"] = smoke(args.variant)
    print(json.dumps(result, indent=2, ensure_ascii=True))
