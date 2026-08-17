"""V032-R1 candidate and transfer invariants."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "baseline/artifacts/v032_route_conditioned_timing_r1"


def _load(name):
    path = ARTIFACT / name / "main.py"
    spec = importlib.util.spec_from_file_location("v032_r1_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(step=0, hands=0):
    farm = {"tiles": [[None for _ in range(10)] for _ in range(10)],
            "farmer": [0, 0], "hands": [[0, 0] for _ in range(hands)],
            "unlocked_quadrants": ["NW", "NE", "SW"], "money": 100000}
    return {"player": 0, "step": step, "day": step // 24, "hour": step % 24,
            "farms": [farm, json.loads(json.dumps(farm))],
            "private": {"shed": {item: 1000 for item in ("MILK", "STRAWBERRY", "WOOL", "MELON")},
                        "inventories": []},
            "market": {"inventory": {item: 9500 for item in ("MILK", "STRAWBERRY", "WOOL", "MELON")},
                        "prices": {item: 200 for item in ("MILK", "STRAWBERRY", "WOOL", "MELON")}},
            "town": {"unlocked_shops": []}}


def test_candidates_smoke_load():
    for name in ("v032_r1_v27_order_only", "v032_r1_v27_timing",
                 "v032_r1_8c4s_order_only", "v032_r1_8c4s_timing"):
        module = _load(name)
        assert len(module._ACTIONS) == 719
        action = module.agent(_obs())
        assert len(action["market"]) <= 10
        assert len(action["hands"]) == 0


def test_zero_quantity_is_not_emitted():
    module = _load("v032_r1_v27_timing")
    action = {"farmer": ["PASS"], "hands": [],
              "market": [["SELL", "MILK", 0], ["SELL", "WOOL", 2]]}
    cleaned = module._v032_r1_remove_zero(action)
    assert cleaned["market"] == [["SELL", "WOOL", 2]]


def test_transfer_conservation():
    module = _load("v032_r1_v27_timing")
    action = {"farmer": ["PASS"], "hands": [],
              "market": [["SELL", "MILK", 10], ["SELL", "WHEAT", 3]]}
    before = module._v032_r1_qty(action, "MILK")
    assert module._v032_r1_reduce(action, "MILK", 4)
    assert module._v032_r1_add(action, "MILK", 4)
    assert module._v032_r1_qty(action, "MILK") == before


def test_transfer_cannot_delete_lockstep_slot():
    module = _load("v032_r1_v27_timing")
    action = {"farmer": ["PASS"], "hands": [],
              "market": [["SELL", "MILK", 4], ["SELL", "WOOL", 2]]}
    assert module._v032_r1_reduce(action, "MILK", 4) is False
    assert action["market"] == [["SELL", "MILK", 4], ["SELL", "WOOL", 2]]


def test_cutoff_and_empty_calibration_fallback():
    module = _load("v032_r1_v27_timing")
    # The checked-in timing candidate contains fitted calibration.  This test
    # is specifically for the empty-payload fallback, so isolate it from the
    # embedded production payload rather than assuming the artifact is empty.
    module.V032_R1_CALIBRATION = {}
    assert module._v032_r1_calibration("MILK", "advance", [1, 2, 3, 4, 5]) is None
    state = module._V032_R1_STATE[0]
    state["pending"] = None
    before = dict(module.V032_R1_STATS)
    action = module.agent(_obs(648))
    assert len(action["market"]) <= 10
    assert module.V032_R1_STATS.get("advance_accepted", 0) == before.get("advance_accepted", 0)
    assert module.V032_R1_STATS.get("delay_accepted", 0) == before.get("delay_accepted", 0)


def test_empty_calibration_has_no_timing_events():
    control = _load("v032_r1_v27_order_only")
    timing = _load("v032_r1_v27_timing")
    for step in (0, 120, 192, 360, 647, 648, 671, 672, 718):
        left = control.agent(_obs(step))
        right = timing.agent(_obs(step))
        assert left == right, step


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print("PASS", name)
