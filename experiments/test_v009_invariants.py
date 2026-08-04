"""Replay-state and contract checks for the V009 market-only overlays."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "log/2026-08-04"
CANDIDATES = {
    "v009a": ROOT / "baseline/history/v009a_market_memory/main.py",
    "v009b": ROOT / "baseline/history/v009b_public_meta_counter/main.py",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field_signature(action):
    return (
        tuple(action.get("farmer", [])),
        tuple(tuple(item) for item in action.get("hands", []) or []),
    )


def check_action_shape(action, obs):
    assert isinstance(action, dict)
    assert isinstance(action.get("farmer"), list) and action["farmer"]
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    expected = len(farms[player].get("hands", []) or [])
    assert len(action.get("hands", []) or []) == expected
    market = action.get("market", []) or []
    assert len(market) <= 10
    for order in market:
        assert isinstance(order, list) and order


def check_public_meta_hysteresis():
    """Exercise V009B's positive-identification and activation threshold."""
    candidate = load(
        CANDIDATES["v009b"],
        "candidate_v009b_synthetic_meta",
    )
    tiles = [[None for _ in range(10)] for _ in range(10)]
    kinds = (["COW"] * 8) + (["SHEEP"] * 5) + (["MELON"] * 5) + (["STRAWBERRY"] * 15)
    for index, value in enumerate(kinds):
        y, x = divmod(index, 10)
        if value in {"COW", "SHEEP"}:
            tiles[y][x] = {"kind": "PASTURE", "animal": value}
        else:
            tiles[y][x] = {"kind": "PLANT", "crop": value}
    opponent = {
        "tiles": tiles,
        "hands": [[0, 0]] * 11,
        "unlocked_quadrants": ["NW", "NE", "SW"],
    }
    own = {"tiles": [[None] * 10 for _ in range(10)], "hands": []}
    obs = {"step": 144, "day": 6, "player": 0, "farms": [own, opponent]}
    candidate._v009b_reset(0)
    candidate._v009b_update_profile(obs)
    assert candidate.V009B_STATE["active"] is False
    obs = dict(obs, step=240, day=10)
    candidate._v009b_update_profile(obs)
    assert candidate.V009B_STATE["active"] is True
    assert candidate.V009B_STATE["activation_count"] == 1


def main():
    checked = 0
    for label, path in CANDIDATES.items():
        for replay_path in sorted(REPLAY_DIR.glob("*.json")):
            data = json.loads(replay_path.read_text(encoding="utf-8"))
            for player in (0, 1):
                # Each player has an independent agent process in Kaggle. Load
                # fresh modules so V009's step memory is not cross-wired.
                baseline = load(ROOT / "main.py", f"baseline_{label}_{player}")
                candidate = load(path, f"candidate_{label}_{player}")
                baseline_fields = Counter()
                candidate_fields = Counter()
                for step in data["steps"]:
                    obs = step[player]["observation"]
                    base_action = baseline.agent(obs)
                    candidate_action = candidate.agent(obs)
                    check_action_shape(candidate_action, obs)
                    assert field_signature(candidate_action) == field_signature(base_action), (
                        label,
                        replay_path.name,
                        player,
                        obs.get("step"),
                    )
                    for op in [base_action["farmer"], *(base_action.get("hands", []) or [])]:
                        baseline_fields[op[0]] += 1
                    for op in [candidate_action["farmer"], *(candidate_action.get("hands", []) or [])]:
                        candidate_fields[op[0]] += 1
                    checked += 1
                assert baseline_fields == candidate_fields
    check_public_meta_hysteresis()
    print(f"V009 invariants: PASS ({checked} replay states)")


if __name__ == "__main__":
    main()
