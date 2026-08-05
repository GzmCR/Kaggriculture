"""Fast V019 style-router and replay-normalization invariants."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price
from v019_replay_analysis import BANDS, DEFAULT_REPLAY_DIR, collect_replays, score_band
from v019_style_router import PublicStyleExpertRouter, PublicStyleTracker, public_style_features


def _obs(step=0, hands=None, quadrants=None, cows=8, sheep=5, price_ratio=1.0):
    tiles = [[None for _ in range(5)] for _ in range(5)]
    tiles[0][0] = {"kind": "COOP", "animal": "COW", "fed_today": True, "yield_units": 0}
    if sheep > 0:
        tiles[0][1] = {"kind": "PASTURE", "animal": "SHEEP", "fed_today": True, "yield_units": 0}
    # Add enough visible structures for public-style thresholds.
    for index in range(max(0, cows - 1)):
        x = (index + 2) % 5
        y = (index + 2) // 5
        if y < 5:
            tiles[y][x] = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "yield_units": 0}
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [
            {"hands": [], "tiles": [[None]], "unlocked_quadrants": ["NW"]},
            {"hands": hands or [], "tiles": tiles, "unlocked_quadrants": quadrants or ["NW", "NE", "SW"]},
        ],
        "market": {
            "prices": {"MELON": 250, "STRAWBERRY": 120 * price_ratio, "MILK": 160 * price_ratio, "WOOL": 200, "WHEAT": 25},
            "inventory": {"MELON": 10000, "STRAWBERRY": 10000, "MILK": 10000, "WOOL": 10000, "WHEAT": 10000},
        },
    }


def test_bands():
    assert score_band(1639) == "L1"
    assert score_band(2399) == "L2"
    assert score_band(2400) == "L3"
    assert score_band(3081) == "L4"
    assert len(BANDS) == 4


def test_replay_deduplication():
    replays = collect_replays(DEFAULT_REPLAY_DIR)
    assert len(replays) == 29
    duplicates = {item["episode"]: item for item in replays if item["duplicate_count"] > 1}
    assert set(duplicates) == {"90121273", "90130811"}
    assert all(item["duplicate_exact"] for item in duplicates.values())


def test_high_worker_persists_after_hands_disappear():
    tracker = PublicStyleTracker()
    style, _, _ = tracker.observe(_obs(step=0, hands=[["PASS"]] * 14))
    assert tracker.max_hands == 14
    style, _, _ = tracker.observe(_obs(step=240, hands=[]))
    assert style == "high_worker_maintenance"


def test_reduced_route_is_publicly_detectable():
    tracker = PublicStyleTracker()
    tracker.observe(_obs(step=0, quadrants=["NW", "NE"], cows=6, sheep=0))
    style, confidence, features = tracker.observe(_obs(step=240, quadrants=["NW", "NE"], cows=6, sheep=0))
    assert style == "reduced_ne_only"
    assert confidence >= 0.9
    assert features["has_SW"] == 0


def test_router_hold_and_fallback():
    router = PublicStyleExpertRouter({"standard_converged": "automatylicza"}, hold_days=1)
    available = {"automatylicza": {}, "mohit": {}}
    first = router.choose(_obs(step=0), available, "automatylicza")
    second = router.choose(_obs(step=1), available, "automatylicza")
    assert first[0] == second[0] == "automatylicza"


def test_environment_price_reference():
    assert market_price("MILK", 10000) == 160
    assert market_price("STRAWBERRY", 10000) == 120


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"V019 style-router invariants: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
