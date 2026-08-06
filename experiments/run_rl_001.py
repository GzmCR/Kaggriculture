"""Train and evaluate RL-001's market-only selector.

The local game remains the source of rewards. Replay/notebook agents are
loaded only as fixed opponents; no opponent private state is passed to the
candidate. Use ``--episodes`` for a short pilot before the larger split.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import gzip
import importlib.util
import json
import statistics
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from rl_001_selector import FeatureEncoder, SelectorRuntime
from run_v006_benchmark import load_hamburger_agent
from run_v008_benchmark import load_notebook_agent
from run_v012_top5_vs_v18 import load_v18_agent


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720
TRAIN_START = 100000
VALIDATION_START = 200000
HOLDOUT_START = 300000
DEFAULT_OPPONENTS = (
    "starter",
    "random",
    "v022c",
    "v18",
    "frontier",
    "hamburger",
    "v13_r3",
    "conditional",
)
V022C_PATH = ROOT / "baseline" / "artifacts" / "v022c_medoid_recovery" / "main.py"
FRONTIER_PATH = ROOT / "baseline" / "kaggle-frontier-lab-strategy-improvement.ipynb"
HAMBURGER_PATH = ROOT / "baseline" / "kaggriculture-hamburger.ipynb"
V13_PATH = ROOT / "baseline" / "v13-r3-top-meta-order-safe-premium-control.ipynb"
CONDITIONAL_PATH = ROOT / "baseline" / "177-180-fresh-top-30-v21-1-conditional-memory.ipynb"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise AttributeError(f"{path} has no agent")
    return module


def load_v13_agent(path: Path = V13_PATH):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "Full V13-R3 implementation" not in text or "```python" not in text:
            continue
        source = text.split("```python", 1)[1].split("```", 1)[0]
        namespace = {"__name__": f"v13_{time.time_ns()}"}
        exec(compile(source, str(path), "exec"), namespace)
        if callable(namespace.get("agent")):
            return namespace["agent"]
    raise ValueError(f"V13 agent source not found in {path}")


def load_conditional_agent(path: Path = CONDITIONAL_PATH):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "_AGENT_B85_PARTS" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "_AGENT_B85_PARTS" for target in node.targets):
                continue
            parts = ast.literal_eval(node.value)
            raw = zlib.decompress(base64.b85decode("".join(parts).encode("ascii"))).decode("utf-8")
            namespace = {"__name__": f"conditional_{time.time_ns()}"}
            exec(compile(raw, str(path), "exec"), namespace)
            if callable(namespace.get("agent")):
                return namespace["agent"]
    raise ValueError(f"conditional agent source not found in {path}")


def load_latest_hamburger_agent(path: Path = HAMBURGER_PATH):
    """Load the current Hamburger notebook format as an opponent.

    Older revisions exposed ANCHOR_BLOB/WRAPPER_BLOB; the current notebook
    stores an anchor and a parameterized overlay template instead.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(cell_text)
    values = {}
    wanted = {"ANCHOR_BLOB", "OVERLAY_BLOB", "CANDIDATE_SPECS", "DEFAULT_NAME"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)
    if not all(name in values for name in wanted):
        raise ValueError("current Hamburger blobs/specs are incomplete")

    def unpack(payload):
        return gzip.decompress(base64.b64decode(payload.encode("ascii"))).decode("utf-8")

    anchor = unpack(values["ANCHOR_BLOB"])
    overlay_template = unpack(values["OVERLAY_BLOB"])
    spec = values["CANDIDATE_SPECS"].get(values["DEFAULT_NAME"]) or {}
    overlay = (
        overlay_template
        .replace("__MIRROR_HORIZON__", str(int(spec.get("mirror_horizon", 0))))
        .replace("__MIRROR_ITEMS__", repr(tuple(spec.get("mirror_items", ()))))
        .replace("__MIRROR_FRACTION__", repr(float(spec.get("mirror_fraction", 1.0))))
        .replace("__SAFE_TERMINAL__", repr(bool(spec.get("safe_terminal", False))))
        .replace("__CASHFLOW_MODE__", repr(spec.get("cashflow_mode")))
    )
    source = anchor.rstrip() + "\n\n" + overlay
    namespace = {"__name__": f"hamburger_{time.time_ns()}"}
    exec(compile(source, str(path), "exec"), namespace)
    if not callable(namespace.get("agent")):
        raise ValueError("current Hamburger source did not define agent")
    return namespace["agent"]


def opponent_factory(name):
    if name in ("starter", "random"):
        return lambda: name
    if name == "v022c":
        return lambda: load_module(V022C_PATH, f"rl_v022c_{time.time_ns()}").agent
    if name == "v18":
        return lambda: load_v18_agent()
    if name == "frontier":
        return lambda: load_notebook_agent(FRONTIER_PATH, f"rl_frontier_{time.time_ns()}")
    if name == "hamburger":
        return lambda: load_latest_hamburger_agent(HAMBURGER_PATH)
    if name == "v13_r3":
        return lambda: load_v13_agent()
    if name == "conditional":
        return lambda: load_conditional_agent()
    raise ValueError(f"unknown opponent: {name}")


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def scan_replays(folder: Path):
    """Validate and split replay files by complete EpisodeId, never by turn."""
    episodes = {}
    if folder is None or not folder.exists():
        return {"folder": str(folder) if folder else None, "files": 0, "episodes": 0, "valid": 0}
    for path in sorted(folder.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        info = payload.get("info", {}) or {}
        episode_id = str(info.get("EpisodeId", payload.get("id", path.stem)))
        steps = payload.get("steps", []) or []
        row = {
            "episode_id": episode_id,
            "path": str(path),
            "score_band": path.parent.name,
            "seed": info.get("seed"),
            "steps": len(steps),
            "complete": len(steps) >= EPISODE_STEPS,
        }
        episodes.setdefault(episode_id, row)
    ordered = [episodes[key] for key in sorted(episodes)]
    valid = [row for row in ordered if row["complete"]]
    first = int(len(valid) * 0.60)
    second = int(len(valid) * 0.80)
    return {
        "folder": str(folder),
        "files": len(list(folder.rglob("*.json"))),
        "episodes": len(ordered),
        "valid": len(valid),
        "invalid_or_incomplete": len(ordered) - len(valid),
        "train": valid[:first],
        "validation": valid[first:second],
        "holdout": valid[second:],
    }


def calibrate_from_replays(folder: Path, seed=0):
    """Warm-start the linear value scale from complete public replays.

    Replay cash deltas are used only as state-distribution calibration. All
    four actions receive the same initial value estimate, so the local game
    remains responsible for learning relative overlay quality.
    """
    manifest = scan_replays(folder)
    rows = manifest.get("train", [])
    encoder = FeatureEncoder()
    features = []
    rewards = []
    weights = []
    band_counts = Counter()
    for row in rows:
        path = Path(row["path"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        steps = payload.get("steps", []) or []
        for seat in (0, 1):
            encoder.reset()
            for start in range(0, 672, 48):
                try:
                    current = steps[start][seat].get("observation") or {}
                    future = steps[min(start + 48, len(steps) - 1)][seat].get("observation") or {}
                    vector = encoder.encode(current)
                    reward = _money(future, seat) - _money(current, seat)
                except (IndexError, AttributeError, TypeError, KeyError):
                    continue
                if not (vector.shape == (96,) and vector.dtype.kind == "f"):
                    continue
                features.append(vector)
                rewards.append(float(reward))
                band = row.get("score_band", "unknown")
                band_counts[band] += 1
                weights.append({"1500~2400": 1.0, "2400~2900": 1.5, "2900~3100": 2.0}.get(band, 1.0))
    if not features:
        raise ValueError(f"no replay calibration samples found under {folder}")
    import numpy as np
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(rewards, dtype=np.float64)
    sample_weights = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
    weighted_matrix = matrix * sample_weights
    weighted_target = target * sample_weights[:, 0]
    regularized = weighted_matrix.T @ weighted_matrix + np.eye(matrix.shape[1]) * 10.0
    beta = np.linalg.solve(regularized, weighted_matrix.T @ weighted_target)
    payload = {"q_a": [beta.tolist() for _ in range(4)], "q_b": [beta.tolist() for _ in range(4)]}
    report = {
        "folder": str(folder),
        "episodes_used": len(rows),
        "samples": len(features),
        "score_band_samples": dict(sorted(band_counts.items())),
        "reward_mean": float(target.mean()),
        "reward_min": float(target.min()),
        "reward_max": float(target.max()),
        "weight_norm": float(np.linalg.norm(beta)),
        "method": "weighted ridge warm start; copied to all four actions",
    }
    return payload, report, manifest


def _money(observation, seat):
    farms = observation.get("farms", []) if isinstance(observation, dict) else []
    try:
        return float(farms[seat].get("money", 0) or 0)
    except (IndexError, AttributeError, TypeError):
        return 0.0


def _final_observation(env, seat):
    final = env.steps[-1][seat]
    observation = getattr(final, "observation", None)
    return observation if isinstance(observation, dict) else {}


def _valid_action(action, obs):
    if not isinstance(action, dict) or not isinstance(action.get("farmer"), list):
        return False
    if not isinstance(action.get("hands", []), list) or not isinstance(action.get("market", []), list):
        return False
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    expected = len(farms[player].get("hands", []) or []) if 0 <= player < len(farms) else 0
    if len(action.get("hands", [])) != expected or len(action.get("market", [])) > 10:
        return False
    return all(isinstance(item, list) and item for item in [action["farmer"], *action["hands"], *action["market"]])


class RLProbe:
    def __init__(self, base_agent, runtime, base_actions):
        self.base_agent = base_agent
        self.runtime = runtime
        self.base_actions = base_actions
        self.errors = 0
        self.invalid = 0
        self.field_mismatch = 0
        self.times_ms = []
        self.modes = []
        self.market_counts = Counter()

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        try:
            base = self.base_agent(obs)
            action = self.runtime.act(obs, base, base_actions=self.base_actions)
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if not _valid_action(action, obs):
            self.invalid += 1
        if isinstance(base, dict) and (
            action.get("farmer") != base.get("farmer")
            or action.get("hands") != base.get("hands")
        ):
            self.field_mismatch += 1
        self.modes.append((int(obs.get("step", 0) or 0), self.runtime.mode))
        for order in action.get("market", []) if isinstance(action, dict) else []:
            if isinstance(order, list) and order:
                self.market_counts[str(order[0])] += 1
        return action

    def metrics(self):
        return {
            "errors": self.errors,
            "invalid": self.invalid,
            "field_mismatch": self.field_mismatch,
            "p99_ms": percentile(self.times_ms, 0.99),
            "max_ms": max(self.times_ms or [0.0]),
            "market_counts": dict(self.market_counts),
            "modes": self.modes[::48],
        }


def run_episode(base_agent, base_actions, opponent, seed, seat, runtime, opponent_name, split, index):
    probe = RLProbe(base_agent, runtime, base_actions)
    players = [probe, opponent] if seat == 0 else [opponent, probe]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)}, debug=False)
    started = time.perf_counter()
    env.run(players)
    elapsed = time.perf_counter() - started
    final_obs = _final_observation(env, seat)
    runtime.finish(final_obs)
    mine = _money(final_obs, seat)
    opponent_obs = _final_observation(env, 1 - seat)
    other = _money(opponent_obs, 1 - seat)
    return {
        "split": split,
        "index": index,
        "seed": int(seed),
        "seat": int(seat),
        "opponent": opponent_name,
        "candidate_money": mine,
        "opponent_money": other,
        "margin": mine - other,
        "result": "win" if mine > other else "loss" if mine < other else "tie",
        "candidate_status": getattr(env.steps[-1][seat], "status", "UNKNOWN"),
        "opponent_status": getattr(env.steps[-1][1 - seat], "status", "UNKNOWN"),
        "done": int(all(getattr(item, "status", "") == "DONE" for item in env.steps[-1])),
        "elapsed_s": elapsed,
        **probe.metrics(),
        "transitions": list(runtime.transitions),
    }


def make_learner(seed=0):
    runtime = SelectorRuntime(training=True, seed=seed)
    runtime.q.alpha = 0.0001
    return runtime.q


def run_training(episodes, opponents, output, seed=0, replay_dir=None):
    output.mkdir(parents=True, exist_ok=True)
    base_module = load_module(V022C_PATH, f"rl_train_v022c_{time.time_ns()}")
    learner = make_learner(seed)
    if replay_dir is not None:
        calibrated, calibration_report, replay_manifest = calibrate_from_replays(replay_dir, seed)
        learner.q_a[:] = calibrated["q_a"]
        learner.q_b[:] = calibrated["q_b"]
        (output / "replay_calibration.json").write_text(json.dumps(calibration_report, indent=2) + "\n", encoding="utf-8")
        (output / "replay_manifest.json").write_text(json.dumps(replay_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"replay calibration episodes={calibration_report['episodes_used']} samples={calibration_report['samples']}", flush=True)
    rng_seed = int(seed)
    rows = []
    for index in range(int(episodes)):
        opponent_name = opponents[index % len(opponents)]
        opponent = opponent_factory(opponent_name)()
        runtime = SelectorRuntime(training=True, seed=rng_seed + index + 1)
        runtime.q = learner
        runtime.epsilon = max(0.05, 1.0 - 0.95 * (index / max(1, episodes - 1)))
        seat = index % 2
        row = run_episode(base_module.agent, getattr(base_module, "_ACTIONS", None), opponent,
                          TRAIN_START + index, seat, runtime, opponent_name, "train", index)
        rows.append(row)
        if (index + 1) % 10 == 0 or index == 0:
            print(f"train {index + 1}/{episodes} opponent={opponent_name} money={row['candidate_money']:.0f} modes={row['modes']}", flush=True)
    weights = learner.payload()
    (output / "weights.json").write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    write_rows(output / "train_rows.jsonl", rows)
    summary = summarize(rows)
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return weights, rows


def run_evaluation(weights, episodes, opponents, output, split, start_seed, fixed_mode=None):
    output.mkdir(parents=True, exist_ok=True)
    base_module = load_module(V022C_PATH, f"rl_eval_v022c_{split}_{time.time_ns()}")
    rows = []
    for index in range(int(episodes)):
        opponent_name = opponents[index % len(opponents)]
        opponent = opponent_factory(opponent_name)()
        runtime = SelectorRuntime(weights=weights, training=False, seed=start_seed + index + 1)
        if fixed_mode is not None:
            runtime.fixed_mode = int(fixed_mode)
        seat = index % 2
        row = run_episode(base_module.agent, getattr(base_module, "_ACTIONS", None), opponent,
                          start_seed + index, seat, runtime, opponent_name, split, index)
        rows.append(row)
        print(f"{split} {index + 1}/{episodes} opponent={opponent_name} money={row['candidate_money']:.0f} margin={row['margin']:.0f}", flush=True)
    write_rows(output / f"{split}_rows.jsonl", rows)
    summary = summarize(rows)
    (output / f"{split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(rows):
    if not rows:
        return {"games": 0}
    outcomes = Counter(row["result"] for row in rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["opponent"]].append(row)
    by_opponent = {}
    for name, group in sorted(grouped.items()):
        by_opponent[name] = {
            "games": len(group),
            "mean_cash": statistics.mean(item["candidate_money"] for item in group),
            "min_cash": min(item["candidate_money"] for item in group),
            "mean_margin": statistics.mean(item["margin"] for item in group),
            "wins": sum(item["result"] == "win" for item in group),
            "losses": sum(item["result"] == "loss" for item in group),
            "done": sum(item["done"] for item in group),
            "errors": sum(item["errors"] for item in group),
            "invalid": sum(item["invalid"] for item in group),
            "field_mismatch": sum(item["field_mismatch"] for item in group),
            "p99_ms": max(item["p99_ms"] for item in group),
        }
    return {
        "games": len(rows),
        "mean_cash": statistics.mean(row["candidate_money"] for row in rows),
        "min_cash": min(row["candidate_money"] for row in rows),
        "mean_margin": statistics.mean(row["margin"] for row in rows),
        "wins": outcomes["win"],
        "losses": outcomes["loss"],
        "ties": outcomes["tie"],
        "win_rate": outcomes["win"] / len(rows),
        "done": sum(row["done"] for row in rows),
        "errors": sum(row["errors"] for row in rows),
        "invalid": sum(row["invalid"] for row in rows),
        "field_mismatch": sum(row["field_mismatch"] for row in rows),
        "p99_ms": max(row["p99_ms"] for row in rows),
        "by_opponent": by_opponent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "evaluate", "ablation", "pipeline"), default="train")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--validation-episodes", type=int, default=300)
    parser.add_argument("--holdout-episodes", type=int, default=300)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--output", type=Path, default=ROOT / "baseline" / "artifacts" / "rl_001_macro_market")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.replay_dir is not None:
        replay_report = scan_replays(args.replay_dir)
        replay_output = args.output / "replay_manifest.json"
        replay_output.parent.mkdir(parents=True, exist_ok=True)
        replay_output.write_text(json.dumps(replay_report, indent=2) + "\n", encoding="utf-8")
        print(f"replay_manifest={replay_output} valid={replay_report.get('valid', 0)}", flush=True)

    if args.mode == "train":
        weights, _ = run_training(args.episodes, args.opponents, args.output / "training", args.seed, args.replay_dir)
        print(json.dumps({"weights": str(args.output / "training" / "weights.json")}, indent=2))
        return

    if args.mode == "pipeline":
        weights, _ = run_training(args.episodes, args.opponents, args.output / "training", args.seed, args.replay_dir)
        run_evaluation(weights, args.validation_episodes, args.opponents,
                       args.output / "validation", "validation", VALIDATION_START, None)
        run_evaluation(weights, args.holdout_episodes, args.opponents,
                       args.output / "holdout", "holdout", HOLDOUT_START, None)
        return

    weights_path = args.weights or args.output / "training" / "weights.json"
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    if args.mode == "evaluate":
        run_evaluation(weights, args.episodes, args.opponents, args.output / "evaluation", "holdout", HOLDOUT_START, None)
        return

    ablation_dir = args.output / "ablation"
    for mode in range(4):
        run_evaluation(weights if mode == 0 else {}, args.episodes, args.opponents,
                       ablation_dir / f"mode_{mode}", f"mode_{mode}", HOLDOUT_START, mode)


if __name__ == "__main__":
    main()
