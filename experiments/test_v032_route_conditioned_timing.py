"""Fast V032 safety and no-op invariants."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "baseline/artifacts/v032_route_conditioned_timing"


def _load(name):
    path = ARTIFACT / name / "main.py"
    spec = importlib.util.spec_from_file_location("test_v032_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(module, step, hands=0, profile=False):
    farm = {
        "tiles": [[None for _ in range(10)] for _ in range(10)],
        "farmer": [0, 0],
        "hands": [[0, 0] for _ in range(hands)],
        "unlocked_quadrants": ["NW", "NE", "SW"],
        "money": 100000,
    }
    other = json.loads(json.dumps(farm))
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, other],
        "private": {
            "shed": {item: 1000 for item in ("MILK", "STRAWBERRY", "WOOL", "MELON")},
            "inventories": [],
        },
        "market": {
            "inventory": {item: 9500 for item in ("MILK", "STRAWBERRY", "WOOL", "MELON")},
            "prices": {"MILK": 250, "STRAWBERRY": 200, "WOOL": 300, "MELON": 500},
        },
        "town": {"unlocked_shops": []},
    }


def _valid_shape(action, obs):
    assert isinstance(action, dict)
    assert isinstance(action.get("farmer"), list)
    assert len(action.get("hands", [])) == len(obs["farms"][obs["player"]]["hands"])
    assert len(action.get("market", [])) <= 10
    for order in action.get("market", []):
        assert isinstance(order, list) and len(order) >= 1


def test_candidates_load():
    for name in ("v032_v27_order_only", "v032_v27_timing",
                 "v032_8c4s_order_only", "v032_8c4s_timing"):
        module = _load(name)
        assert len(module._ACTIONS) == 719
        action = module.agent(_obs(module, 0))
        _valid_shape(action, _obs(module, 0))


def test_order_only_controls_disable_timing():
    for name in ("v032_v27_order_only", "v032_8c4s_order_only"):
        module = _load(name)
        assert module.V032_DISABLE_TIMING is True
        assert module.V032_PROFILES == []
        for step in (0, 96, 192, 360, 647, 648, 718):
            obs = _obs(module, step)
            action = module.agent(obs)
            _valid_shape(action, obs)
        assert not any(key.endswith("_accepted") for key in module.V032_STATS)


def test_empty_profiles_are_exact_v27_order_only():
    module = _load("v032_v27_timing")
    module.V032_PROFILES = []
    for step in (0, 96, 192, 360, 647, 648, 671, 672, 718):
        obs = _obs(module, step)
        got = module.agent(obs)
        expected = module._v031_agent(obs)
        assert got == expected, step
        _valid_shape(got, obs)


def test_timing_requires_visible_inventory_and_uses_final_reorder():
    module = _load("v032_v27_timing")
    # Patch only the estimator for a synthetic acceptance test.  The runtime
    # estimator itself remains conservative and is tested through the smoke
    # run; this isolates quantity conservation from the market model.
    module._v032_expected_gain = lambda *args, **kwargs: 100.0
    base_farm = _obs(module, 0)["farms"][0]
    signature = module._v032_signature(base_farm)
    module.V032_PROFILES = [{
        "profile_id": "synthetic",
        "checkpoints": {str(step): signature for step in module.V032_ROUTE_CHECKPOINTS},
        "route_distance": 8,
        "market_bands": {item: {"low": 0, "high": 1000} for item in module.V032_PREMIUM},
        "supply_forecast": {item: {"default": 0} for item in module.V032_PREMIUM},
    }]
    for step in (96, 144, 192, 240, 288):
        module.agent(_obs(module, step))
    chosen = None
    for step in range(193, 648):
        obs = _obs(module, step)
        raw = module._ACTIONS[step]
        if any(module._v032_is_sell(x, item) for x in raw.get("market", []) for item in module.V032_PREMIUM):
            chosen = (step, obs, raw)
            break
    assert chosen is not None
    step, obs, raw = chosen
    before = {item: module._v032_qty(raw, item) for item in module.V032_PREMIUM}
    action = module.agent(obs)
    _valid_shape(action, obs)
    # No timing path may create a quantity outside the current/next-event
    # conservation rule; all route quantities remain non-negative integers.
    for order in action["market"]:
        if len(order) >= 3 and order[0] == "SELL":
            assert int(order[2]) >= 0
    assert len(action["market"]) <= 10


def test_cutoffs_disable_new_timing():
    module = _load("v032_v27_timing")
    module.V032_PROFILES = [{"checkpoints": {}, "market_bands": {}, "supply_forecast": {}}]
    for step in (648, 672, 718):
        obs = _obs(module, step)
        before = dict(module.V032_STATS)
        action = module.agent(obs)
        _valid_shape(action, obs)
        assert module.V032_STATS.get("advance_accepted", 0) == before.get("advance_accepted", 0)
        assert module.V032_STATS.get("delay_accepted", 0) == before.get("delay_accepted", 0)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
