"""Stress-test market variants against fixed downloaded replay traces.

This is an offline adversarial trace test. The replay opponent does not adapt
to changed shared state, so results are diagnostic rather than leaderboard
evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

from kaggle_environments import make

from run_v026_v22_v022c_recovery import ROOT, _v22_fresh


CANDIDATES = {
    "v22": None,
    "v029a": ROOT / "baseline/artifacts/v029_milk_schedule/v029a_milk_safe_schedule/main.py",
    "v030a": ROOT / "baseline/artifacts/v030_adaptive_market/v030a_milk_momentum_gate/main.py",
    "v030b": ROOT / "baseline/artifacts/v030_adaptive_market/v030b_cross_product_guard/main.py",
}


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"v030_badcase_{tag}_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayOpponent:
    def __init__(self, payload, seat):
        self.steps = payload["steps"]
        self.seat = int(seat)

    def __call__(self, obs, config=None):
        del config
        step = max(0, int(obs.get("step", 0) or 0))
        index = min(step + 1, len(self.steps) - 1)
        action = self.steps[index][self.seat].get("action") or {}
        return json.loads(json.dumps(action))


def _run_one(candidate_name, replay_path):
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    names = [agent.get("Name", "") for agent in payload.get("info", {}).get("Agents", [])]
    candidate_seat = names.index("GzmCR632")
    opponent_seat = 1 - candidate_seat
    if candidate_name == "v22":
        candidate = _v22_fresh("v22")
        module = None
    else:
        module = _load(CANDIDATES[candidate_name], candidate_name)
        candidate = module.agent
    opponent = ReplayOpponent(payload, opponent_seat)
    players = [candidate, opponent] if candidate_seat == 0 else [opponent, candidate]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": int(payload.get("info", {}).get("seed", 0))},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine, theirs = final[candidate_seat], final[opponent_seat]
    mine_money = float(mine.observation["farms"][candidate_seat]["money"])
    other_money = float(theirs.observation["farms"][opponent_seat]["money"])
    stats = getattr(module, "_V030_STATS", {}) if module is not None else {}
    return {
        "candidate": candidate_name,
        "replay": replay_path.name,
        "episode_id": payload.get("info", {}).get("EpisodeId"),
        "original_opponent": names[opponent_seat],
        "original_seed": payload.get("info", {}).get("seed"),
        "candidate_money": mine_money,
        "fixed_trace_money": other_money,
        "margin": mine_money - other_money,
        "candidate_status": str(mine.status),
        "opponent_status": str(theirs.status),
        "done": int(mine.status == "DONE" and theirs.status == "DONE"),
        "errors": int(stats.get("errors", 0)),
        "changed_calls": int(stats.get("changed_calls", 0)),
        "changed_units": int(stats.get("changed_units", 0)),
        "milk_delay_accepted": int(stats.get("milk_delay_accepted", 0)),
        "milk_delay_blocked": int(stats.get("milk_delay_blocked", 0)),
        "crash_advance_accepted": int(stats.get("crash_advance_accepted", 0)),
        "pending_failures": int(stats.get("pending_failures", 0)),
    }


def run(input_dir, output_dir, candidates):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        for path in sorted(input_dir.glob("*.json")):
            print(f"{candidate} vs fixed replay {path.name}", flush=True)
            rows.append(_run_one(candidate, path))
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "baseline/artifacts/v029_milk_schedule/v029a_milk_safe_schedule/losslog",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baseline/artifacts/v030_adaptive_market/badcase_replay_7",
    )
    parser.add_argument("--candidates", nargs="+", default=["v22", "v029a", "v030a", "v030b"])
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, tuple(args.candidates)), indent=2))
