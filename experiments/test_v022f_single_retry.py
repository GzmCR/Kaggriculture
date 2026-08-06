"""V022f single-retry ablation tests."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "baseline/artifacts/v022f_single_retry/main.py"


def load_module():
    spec = importlib.util.spec_from_file_location(f"v022f_test_{time.time_ns()}", MAIN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def obs(step, kind="WEED", crop=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    if kind is not None:
        tile = {"kind": kind}
        if crop is not None:
            tile["crop"] = crop
        tiles[0][0] = tile
    farm = {"money": 10000, "farmer": [0, 0], "hands": [[1, 0]], "tiles": tiles, "unlocked_quadrants": ["NW"]}
    other = {"money": 10000, "farmer": [0, 0], "hands": [[1, 0]], "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, other],
        "private": {"shed": {}, "seeds": {"MELON": 1}, "inventories": [{}, {}]},
        "market": {"prices": {"MELON": 200}, "inventory": {"MELON": 100}},
        "town": {"unlocked_shops": []},
    }


def action(farmer):
    return {"farmer": list(farmer), "hands": [["WATER"]], "market": [["SELL", "MELON", 1]]}


def reset(module):
    module._V022E_STATE[0] = {"last_step": -1, "active": {}, "suppressed": {}}
    module._V022E_STATS.update({key: 0 for key in module._V022E_STATS})


def test_one_retry_then_abandon():
    module = load_module()
    reset(module)
    assert module._v022e_adaptive_repair(obs(100), action(["PLANT", "MELON"]), 100)["farmer"] == ["DIG"]
    assert module._v022e_adaptive_repair(obs(101), action(["PASS"]), 101)["farmer"] == ["PLANT", "MELON"]
    # The first retry fails; V022f must abandon immediately, not emit DIG again.
    result = module._v022e_adaptive_repair(obs(102), action(["PASS"]), 102)
    assert result["farmer"] == ["PASS"]
    assert module._V022E_STATE[0]["active"] == {}
    assert module._V022E_STATS["abandoned"] == 1
    assert module._V022E_STATS["weed_retries"] == 1
    # Suppression prevents an immediate infinite re-trigger on the same tile.
    again = module._v022e_adaptive_repair(obs(103), action(["PLANT", "MELON"]), 103)
    assert again["farmer"] == ["PLANT", "MELON"]
    assert module._V022E_STATS["repeat_suppressed"] >= 1


def test_success_path_and_invariants():
    module = load_module()
    reset(module)
    market = [["SELL", "MELON", 1]]
    first = module._v022e_adaptive_repair(obs(100), {"farmer": ["PLANT", "MELON"], "hands": [["WATER"]], "market": market}, 100)
    assert first["farmer"] == ["DIG"] and first["market"] == market
    module._v022e_adaptive_repair(obs(101), action(["PASS"]), 101)
    result = module._v022e_adaptive_repair(obs(102, "PLANT", "MELON"), action(["PASS"]), 102)
    assert result["farmer"] == ["PASS"]
    assert result["hands"] == [["WATER"]]
    assert result["market"] == market


def main():
    test_one_retry_then_abandon()
    test_success_path_and_invariants()
    print("V022f single-retry invariants: PASS (2 tests)")


if __name__ == "__main__":
    main()
