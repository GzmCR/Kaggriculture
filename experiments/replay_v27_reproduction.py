"""Compare an online V27 replay with local kaggle-environments execution."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import time
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "baseline/artifacts/v27/91636967.json"
DEFAULT_AGENT = ROOT / "baseline/history/pure_routes/v27/main.py"


def _load_module(path, tag):
    spec = importlib.util.spec_from_file_location(f"replay_v27_{tag}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize(value):
    if not isinstance(value, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(value.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in value.get("hands", []) or []],
        "market": [list(item) for item in value.get("market", []) or [] if isinstance(item, list)],
    }


def _fixed_replay_agent(replay, seat):
    actions = [row[seat].get("action", {}) for row in replay["steps"]]

    def agent(obs, config=None):
        # Kaggle replay step 0 is the framework's initial placeholder. The
        # first action that produces the replay's step-1 state is stored in
        # replay step 1, so local observation step k consumes action[k + 1].
        step = int(obs.get("step", 0) or 0) + 1
        step = max(0, min(len(actions) - 1, step))
        return copy.deepcopy(actions[step])

    return agent


def _config(replay, override_seed=None):
    config = dict(replay.get("configuration", {}) or {})
    info = replay.get("info", {}) or {}
    config["episodeSteps"] = int(config.get("episodeSteps", len(replay.get("steps", []))))
    if override_seed is None:
        config["seed"] = int(info.get("seed"))
    else:
        config["seed"] = int(override_seed)
    return config


def _online_final(replay, seat):
    row = replay["steps"][-1][seat]
    observation = row.get("observation", {})
    farms = observation.get("farms", []) or []
    money = farms[seat].get("money") if seat < len(farms) else None
    return {
        "money": float(money) if money is not None else None,
        "reward": replay.get("rewards", [None, None])[seat],
        "status": replay.get("statuses", [None, None])[seat],
    }


def _compare_observations(replay, env, seat):
    rows = []
    count = min(len(replay["steps"]), len(env.steps))
    for step in range(count):
        online = replay["steps"][step][seat].get("observation", {})
        local = env.steps[step][seat].observation
        checks = {
            "money": (
                online.get("farms", [{}])[seat].get("money"),
                local.get("farms", [{}])[seat].get("money"),
            ),
            "market_prices": (
                online.get("market", {}).get("prices", {}),
                local.get("market", {}).get("prices", {}),
            ),
            "market_inventory": (
                online.get("market", {}).get("inventory", {}),
                local.get("market", {}).get("inventory", {}),
            ),
            "town_shops": (
                online.get("town", {}).get("unlocked_shops", []),
                local.get("town", {}).get("unlocked_shops", []),
            ),
            "hands": (
                online.get("farms", [{}])[seat].get("hands", []),
                local.get("farms", [{}])[seat].get("hands", []),
            ),
        }
        mismatch = [key for key, (left, right) in checks.items() if left != right]
        if mismatch:
            rows.append({"step": step, "fields": mismatch, "online": checks, "local": checks})
    return rows


def run(path, agent_path, seat, override_seed=None):
    replay = json.loads(Path(path).read_text(encoding="utf-8"))
    module = _load_module(Path(agent_path), "candidate")
    replay_agent = _fixed_replay_agent(replay, 1 - seat)
    candidate = module.agent
    players = [candidate, replay_agent] if seat == 0 else [replay_agent, candidate]
    config = _config(replay, override_seed)
    env = make("kaggriculture", configuration=config, debug=False)
    env.run(players)
    final = env.steps[-1]
    local_final = {
        "money": float(final[seat].observation["farms"][seat]["money"]),
        "reward": final[seat].reward,
        "status": final[seat].status,
    }
    online = _online_final(replay, seat)
    mismatches = []
    for step, state in enumerate(env.steps):
        expected = replay["steps"][step][seat].get("action", {})
        actual = state[seat].action or {}
        if _normalize(expected) != _normalize(actual):
            mismatches.append({"step": step, "expected": expected, "actual": actual})
    summary = {
        "replay": str(path),
        "episode_id": replay.get("info", {}).get("EpisodeId"),
        "online_seed": replay.get("info", {}).get("seed"),
        "local_config": config,
        "online": online,
        "local": local_final,
        "money_delta": local_final["money"] - online["money"],
        "action_mismatches": len(mismatches),
        "first_action_mismatch": mismatches[0] if mismatches else None,
        "candidate_stats": getattr(module, "_V027_STATS", None),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.replay, args.agent, args.seat, args.seed), indent=2, ensure_ascii=True))
