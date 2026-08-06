"""Fast V023 invariants plus optional starter smoke games."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "v023a_high_output_14hands",
    "v023b_stable_12hands",
    "v023c_high_hands_15hands",
    "v023d_early_portfolio",
)


def load_module(name: str):
    path = ROOT / "baseline/artifacts" / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def synthetic_obs(step: int = 0, player: int = 0, hands: int = 2, weed: bool = False):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    if weed:
        tiles[0][0] = {"kind": "WEED"}
    farm = {
        "money": 10000,
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": [[1 + i, 0] for i in range(hands)],
        "unlocked_quadrants": ["NW", "NE", "SW"],
        "hires_today": hands,
    }
    other = dict(farm)
    other["tiles"] = [[None for _ in range(10)] for _ in range(10)]
    return {
        "player": player,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, other],
        "private": {"shed": {"MELON": 20, "MILK": 20, "WOOL": 20}, "seeds": {}, "inventories": [[], [], []]},
        "market": {"inventory": {"MELON": 10000, "MILK": 10000, "WOOL": 10000}, "prices": {"MELON": 200, "MILK": 150, "WOOL": 180}},
        "town": {"unlocked_shops": []},
    }


def assert_action(action, expected_hands: int):
    assert isinstance(action, dict)
    assert isinstance(action.get("farmer"), list) and action["farmer"]
    assert isinstance(action.get("hands"), list)
    assert len(action["hands"]) == expected_hands
    assert isinstance(action.get("market"), list)
    assert len(action["market"]) <= 10
    for order in action["market"]:
        assert isinstance(order, list) and order
        if order[0] == "SELL":
            assert len(order) >= 3 and int(order[2]) >= 0


def test_candidate_schema():
    for name in CANDIDATES:
        module, path = load_module(name)
        source = path.read_text(encoding="utf-8")
        assert "TeamNames" not in source
        assert callable(module.agent)
        assert len(module._V023_ROUTES) == 3
        for step in (0, 47, 48, 49, 704, 718, 719):
            action = module.agent(synthetic_obs(step=step))
            assert_action(action, 2)
        assert module._V023_STATS["terminal_liquidations"] >= 2


def test_portfolio_locks_once():
    module, _ = load_module("v023d_early_portfolio")
    module.agent(synthetic_obs(step=47))
    assert module._V023_STATE[0]["route"] is None
    module.agent(synthetic_obs(step=48))
    selected = module._V023_STATE[0]["route"]
    assert selected in module._V023_ROUTES
    module.agent(synthetic_obs(step=49))
    assert module._V023_STATE[0]["route"] == selected


def test_weed_is_actor_local_and_bounded():
    module, _ = load_module("v023a_high_output_14hands")
    module._V023_STATE[0] = {"last_step": -1, "route": None, "active": {}}
    obs = synthetic_obs(step=0, weed=True)
    original = {"farmer": ["PLANT", "MELON"], "hands": [["PASS"], ["PASS"]], "market": [["SELL", "MELON", 1]]}
    repaired = module._v023_weed_overlay(obs, module._v023_copy_action(original), "high_output_14hands", 0)
    assert repaired["farmer"] == ["DIG"]
    assert repaired["hands"] == original["hands"]
    assert repaired["market"] == original["market"]
    retry = module._v023_weed_overlay(synthetic_obs(step=1, weed=True), module._v023_copy_action(original), "high_output_14hands", 1)
    assert retry["farmer"] == ["PLANT", "MELON"]
    for step in range(2, 10):
        action = module._v023_weed_overlay(synthetic_obs(step=step, weed=True), module._v023_copy_action(original), "high_output_14hands", step)
        assert action["hands"] == original["hands"]
    assert module._V023_STATS["weed_repairs"] == 1
    assert module._V023_STATS["weed_retries"] == 1


def run_smoke():
    from kaggle_environments import make

    rows = []
    for name in CANDIDATES:
        module, _ = load_module(name)
        for seat in (0, 1):
            players = [module.agent, "starter"] if seat == 0 else ["starter", module.agent]
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 17}, debug=False)
            env.run(players)
            final = env.steps[-1]
            rows.append({"candidate": name, "seat": seat, "status": [s.status for s in final]})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    assert all(row["status"] == ["DONE", "DONE"] for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    test_candidate_schema()
    test_portfolio_locks_once()
    test_weed_is_actor_local_and_bounded()
    if args.smoke:
        run_smoke()
    print("V023 invariants: PASS")


if __name__ == "__main__":
    main()
