"""Synthetic priority and storage tests for V011 candidates."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(label):
    path = ROOT / "baseline/history" / label / "main.py"
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(tile, position, day=10, step=None, inventory=None, shed=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    x, y = tile["position"]
    tiles[y][x] = {key: value for key, value in tile.items() if key != "position"}
    farm = {
        "tiles": tiles,
        "farmer": list(position),
        "hands": [],
        "unlocked_quadrants": ["NW", "NE", "SW"],
        "money": 3000,
    }
    other = {
        "tiles": [[None for _ in range(10)] for _ in range(10)],
        "farmer": [0, 0],
        "hands": [],
        "unlocked_quadrants": ["NW"],
    }
    return {
        "step": day * 24 if step is None else step,
        "day": day,
        "hour": 0,
        "player": 0,
        "farms": [farm, other],
        "private": {
            "inventories": [inventory or {}],
            "shed": shed or {},
            "seeds": {},
        },
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def main():
    water = load("v011a_water_guard")
    water_obs = observation({
        "position": (1, 1),
        "kind": "PLANT",
        "crop": "STRAWBERRY",
        "planted_day": 0,
        "yield_units": 1,
        "watered_today": False,
        "consecutive_unwatered": 1,
    }, (1, 1))
    jobs = water._v011_crop_jobs(water_obs, water_obs["farms"][0])
    assert any(job["kind"] == "WATER" and job["urgent"] for job in jobs)
    action = water._v011_overlay(
        {"farmer": ["PASS"], "hands": [], "market": []},
        water_obs,
    )
    assert action["farmer"] == ["WATER"]

    harvest = load("v011b_harvest_storage")
    harvest_obs = observation({
        "position": (1, 1),
        "kind": "PLANT",
        "crop": "STRAWBERRY",
        "planted_day": 0,
        "yield_units": 3,
        "watered_today": True,
        "consecutive_unwatered": 0,
    }, (1, 1), day=10)
    jobs = harvest._v011_crop_jobs(harvest_obs, harvest_obs["farms"][0])
    assert any(job["kind"] == "HARVEST" for job in jobs)
    action = harvest._v011_overlay(
        {"farmer": ["PASS"], "hands": [], "market": []},
        harvest_obs,
    )
    assert action["farmer"] == ["HARVEST"]

    storage_obs = observation(
        {"position": (4, 4), "kind": "PLANT", "crop": "STRAWBERRY", "yield_units": 0},
        (4, 4),
        inventory={"STRAWBERRY": 5},
        shed={"WHEAT": 85},
    )
    action = harvest._v011_overlay(
        {"farmer": ["PASS"], "hands": [], "market": []},
        storage_obs,
    )
    assert action["farmer"] == ["DROP"]

    combined = load("v011c_priority_scheduler")
    feed_obs = observation({
        "position": (1, 1),
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": 0,
        "yield_units": 1,
        "watered_today": False,
        "consecutive_unwatered": 1,
    }, (1, 1))
    action = combined._v011_overlay(
        {"farmer": ["FEED"], "hands": [], "market": []},
        feed_obs,
    )
    assert action["farmer"] == ["FEED"]
    print("V011 scheduler tests: PASS")


if __name__ == "__main__":
    main()
