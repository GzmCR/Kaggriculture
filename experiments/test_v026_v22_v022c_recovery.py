"""Fast structural and synthetic-state tests for V026 candidates."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V026A = ROOT / "baseline/artifacts/v026_v22_v022c_recovery/v026a_v22_single_retry/main.py"
V026B = ROOT / "baseline/artifacts/v026_v22_v022c_recovery/v026b_v22_single_retry_guard/main.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def obs(step, tile, shed=None, inventories=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[0][0] = tile
    return {
        "player": 0,
        "step": step,
        "farms": [{
            "farmer": [0, 0],
            "hands": [],
            "tiles": tiles,
            "money": 100000,
        }, {
            "farmer": [0, 0],
            "hands": [],
            "tiles": [[None for _ in range(10)] for _ in range(10)],
            "money": 100000,
        }],
        "private": {
            "shed": shed or {},
            "inventories": inventories or [{}],
            "seeds": {"WHEAT": 1},
        },
        "market": {
            "inventory": {"WHEAT": 10000, "MILK": 10000},
            "prices": {"WHEAT": 25, "MILK": 160},
        },
        "town": {"unlocked_shops": []},
    }


def test_single_retry_and_early_release(module):
    module._V022E_STATE[0] = {"last_step": -1, "active": {}, "suppressed": {}}
    plant = ["PLANT", "WHEAT"]
    dig = module._v022e_adaptive_repair(obs(100, {"kind": "WEED"}), {"farmer": plant, "hands": [], "market": []}, 100)
    assert dig["farmer"] == ["DIG"]
    retry = module._v022e_adaptive_repair(obs(101, {"kind": "WEED"}), {"farmer": plant, "hands": [], "market": []}, 101)
    assert retry["farmer"] == plant
    success = module._v022e_adaptive_repair(
        obs(102, {"kind": "PLANT", "crop": "WHEAT"}),
        {"farmer": ["PASS"], "hands": [], "market": []},
        102,
    )
    assert success["farmer"] == ["PASS"]
    assert not module._V022E_STATE[0]["active"]


def test_failed_retry_is_not_repeated(module):
    module._V022E_STATE[0] = {"last_step": -1, "active": {}, "suppressed": {}}
    plant = ["PLANT", "WHEAT"]
    module._v022e_adaptive_repair(obs(200, {"kind": "WEED"}), {"farmer": plant, "hands": [], "market": []}, 200)
    module._v022e_adaptive_repair(obs(201, {"kind": "WEED"}), {"farmer": plant, "hands": [], "market": []}, 201)
    failed = module._v022e_adaptive_repair(obs(202, {"kind": "WEED"}), {"farmer": ["PASS"], "hands": [], "market": []}, 202)
    assert failed["farmer"] != ["DIG"]
    assert not module._V022E_STATE[0]["active"]
    assert "farmer" in module._V022E_STATE[0]["suppressed"]


def test_guard_clips_only_excess_sell(module):
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "WHEAT", 10], ["SELL", "MILK", 2]],
    }
    clipped = module._v026_sell_guard(
        obs(100, None, shed={"WHEAT": 3}, inventories=[{"MILK": 1}]), action
    )
    assert clipped["market"] == [["SELL", "WHEAT", 3], ["SELL", "MILK", 1]]
    assert action["market"] == [["SELL", "WHEAT", 10], ["SELL", "MILK", 2]]


def main():
    modules = [load(V026A, "v026a_test"), load(V026B, "v026b_test")]
    for module in modules:
        assert len(module._ACTIONS) == 719
        assert module._V022E_MAX_CATCHUP == 8
        test_single_retry_and_early_release(module)
        test_failed_retry_is_not_repeated(module)
    test_guard_clips_only_excess_sell(modules[1])
    print("V026 synthetic tests passed")


if __name__ == "__main__":
    main()
