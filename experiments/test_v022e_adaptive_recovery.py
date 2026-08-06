"""Synthetic recovery tests for V022e, plus optional 720-step smoke games."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "baseline/artifacts/v022e_adaptive_recovery/main.py"


def load_module():
    spec = importlib.util.spec_from_file_location(f"v022e_test_{time.time_ns()}", MAIN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def obs(step, tile_kind="WEED", crop=None, hands=1):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    if tile_kind is not None:
        tile = {"kind": tile_kind}
        if crop is not None:
            tile["crop"] = crop
        tiles[0][0] = tile
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
        "private": {"shed": {"MELON": 3}, "seeds": {"MELON": 1}, "inventories": [{}, {}]},
        "market": {"prices": {"MELON": 200}, "inventory": {"MELON": 100}},
        "town": {"unlocked_shops": []},
    }


def action(farmer, hands=None, market=None):
    return {
        "farmer": list(farmer),
        "hands": [list(item) for item in (hands or [["PASS"]])],
        "market": [list(item) for item in (market or [["SELL", "MELON", 1]])],
    }


def reset(module):
    module._V022E_STATE[0] = {"last_step": -1, "active": {}, "suppressed": {}}
    module._V022E_STATS.update({key: 0 for key in module._V022E_STATS})


def test_dig_retry_and_pass_release():
    module = load_module()
    reset(module)
    market = [["SELL", "MELON", 1]]
    first = module._v022e_adaptive_repair(obs(100), action(["PLANT", "MELON"], [["WATER"]], market), 100)
    assert first["farmer"] == ["DIG"]
    assert first["hands"] == [["WATER"]]
    assert first["market"] == market
    retry = module._v022e_adaptive_repair(obs(101), action(["PASS"], [["WATER"]], market), 101)
    assert retry["farmer"] == ["PLANT", "MELON"]
    released = module._v022e_adaptive_repair(obs(102, "PLANT", "MELON"), action(["PASS"], [["WATER"]], market), 102)
    assert released["farmer"] == ["PASS"]
    assert module._V022E_STATE[0]["active"] == {}
    assert module._V022E_STATS["early_releases"] == 1
    assert released["market"] == market


def test_successful_water_release():
    module = load_module()
    reset(module)
    module._v022e_adaptive_repair(obs(100), action(["PLANT", "MELON"]), 100)
    module._v022e_adaptive_repair(obs(101), action(["PASS"]), 101)
    result = module._v022e_adaptive_repair(obs(102, "PLANT", "MELON"), action(["WATER"]), 102)
    assert result["farmer"] == ["WATER"]
    assert module._V022E_STATE[0]["active"] == {}


def test_successful_catchup_is_bounded():
    module = load_module()
    reset(module)
    original_actions = module._ACTIONS
    module._ACTIONS = [{"farmer": ["NORTH"], "hands": [], "market": []} for _ in range(720)]
    try:
        module._v022e_adaptive_repair(obs(100), action(["PLANT", "MELON"]), 100)
        module._v022e_adaptive_repair(obs(101), action(["PASS"]), 101)
        for step in range(102, 110):
            result = module._v022e_adaptive_repair(obs(step, "PLANT", "MELON"), action(["EAST"]), step)
            assert result["farmer"] == ["NORTH"]
        assert module._V022E_STATS["catchup_actions"] == 8
        released = module._v022e_adaptive_repair(obs(110, "PLANT", "MELON"), action(["EAST"]), 110)
        assert released["farmer"] == ["EAST"]
        assert module._V022E_STATE[0]["active"] == {}
    finally:
        module._ACTIONS = original_actions


def test_failed_twice_suppresses_loop():
    module = load_module()
    reset(module)
    module._v022e_adaptive_repair(obs(100), action(["PLANT", "MELON"]), 100)
    assert module._v022e_adaptive_repair(obs(101), action(["PASS"]), 101)["farmer"] == ["PLANT", "MELON"]
    assert module._v022e_adaptive_repair(obs(102), action(["PASS"]), 102)["farmer"] == ["DIG"]
    assert module._v022e_adaptive_repair(obs(103), action(["PASS"]), 103)["farmer"] == ["PLANT", "MELON"]
    final = module._v022e_adaptive_repair(obs(104), action(["PLANT", "MELON"]), 104)
    assert final["farmer"] == ["PLANT", "MELON"]
    assert module._V022E_STATE[0]["active"] == {}
    assert module._V022E_STATS["abandoned"] == 1
    suppressed = module._v022e_adaptive_repair(obs(105), action(["PLANT", "MELON"]), 105)
    assert suppressed["farmer"] == ["PLANT", "MELON"]
    assert module._V022E_STATS["repeat_suppressed"] >= 1


def test_actor_market_and_hand_invariants():
    module = load_module()
    reset(module)
    market = [["BUY_SEED", "MELON", 1], ["SELL", "MELON", 2]]
    base = action(["PLANT", "MELON"], [["WATER"], ["FEED"]], market)
    result = module._v022e_adaptive_repair(obs(100, hands=2), base, 100)
    assert result["farmer"] == ["DIG"]
    assert result["hands"] == [["WATER"], ["FEED"]]
    assert result["market"] == market
    assert len(result["hands"]) == 2


def test_no_weed_passthrough():
    module = load_module()
    reset(module)
    base = action(["EAST"], [["WATER"], ["FEED"]], [["BUY_SEED", "MELON", 1]])
    result = module._v022e_adaptive_repair(obs(200, None, hands=2), base, 200)
    assert result == module._v022e_align_hands(base, obs(200, None, hands=2))


def run_smoke():
    from kaggle_environments import make

    rows = []
    for seat in (0, 1):
        module = load_module()
        players = [module.agent, "starter"] if seat == 0 else ["starter", module.agent]
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 17}, debug=False)
        env.run(players)
        rows.append({"seat": seat, "status": [state.status for state in env.steps[-1]]})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    assert all(row["status"] == ["DONE", "DONE"] for row in rows)


def main():
    tests = (
        test_dig_retry_and_pass_release,
        test_successful_water_release,
        test_successful_catchup_is_bounded,
        test_failed_twice_suppresses_loop,
        test_actor_market_and_hand_invariants,
        test_no_weed_passthrough,
    )
    for test in tests:
        test()
    print(f"V022e adaptive recovery invariants: PASS ({len(tests)} tests)")
    import sys
    if "--smoke" in sys.argv:
        run_smoke()


if __name__ == "__main__":
    main()
