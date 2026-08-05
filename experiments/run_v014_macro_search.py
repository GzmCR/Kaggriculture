"""Search low-dimensional macro strategies around the V012 executor.

V012 is a fixed-route submission, so changing fields such as ``hands`` or
``cows`` while keeping ``use_fixed_schedule=True`` does not actually change
the farm route.  V014 therefore evaluates the same V012 planning and market
code in its observation-driven mode (``use_fixed_schedule=False``), and
searches only macro parameters that the dynamic planner consumes.

The search is deliberately staged:

* calibration: many cheap candidates, usually one seed and seat 0;
* final: the best candidates on paired seeds, both seats, and a wider opponent
  set;
* gate: compare the finalists with a V012 fixed-route control on the exact
  same games.

This is a local parameter search, not a replay or a per-turn action brute
force.  Every action is generated from the current observation, so a changed
hire/crop/animal plan can cause the scheduler to adapt legally.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import random
import statistics
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
V012_PATH = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
BASELINE_PATH = ROOT / "main.py"
DEFAULT_OUT = ROOT / "baseline/artifacts/v014_macro_search"
HISTORY_DIR = ROOT / "baseline/history/v014_macro_search"
EPISODE_STEPS = 720

CALIBRATION_SEEDS = (17,)
FINAL_SEEDS = (17, 42, 2026, 217, 317, 733)

# These are the fields that materially affect the dynamic V012 planner.  The
# remaining V012 flags stay at their validated defaults unless a future
# experiment explicitly expands this list.
MACRO_KEYS = (
    "hands",
    "cows",
    "sheep",
    "strawberries",
    "opening_wheat",
    "opening_melons",
    "opening_carrots",
    "opening_animals",
    "land_ne_day",
    "land_sw_day",
    "animal_nw_day",
    "animal_ne_day",
    "animal_sw_day",
    "feed_days_buffer",
    "cash_reserve",
    "ongoing_harvest_threshold",
)

DEFAULT_MACRO = {
    "use_fixed_schedule": False,
    "hands": 13,
    "cows": 8,
    "sheep": 6,
    "strawberries": 34,
    "opening_wheat": 10,
    "opening_melons": 9,
    "opening_carrots": 2,
    "opening_animals": 2,
    "land_ne_day": 5,
    "land_sw_day": 10,
    "animal_nw_day": 4,
    "animal_ne_day": 8,
    "animal_sw_day": 12,
    "feed_days_buffer": 1,
    "cash_reserve": 150,
    "ongoing_harvest_threshold": 3,
}

PARAM_VALUES = {
    "hands": (10, 11, 12, 13, 14),
    "cows": (6, 7, 8, 9, 10, 12),
    "sheep": (3, 4, 5, 6, 7, 8),
    "strawberries": (28, 30, 34, 38, 42),
    "opening_wheat": (6, 8, 10, 12, 14),
    "opening_melons": (7, 8, 9, 10, 11),
    "opening_carrots": (0, 1, 2, 3, 4),
    "opening_animals": (0, 1, 2, 3, 4),
    "land_ne_day": (3, 4, 5, 6, 7),
    "land_sw_day": (8, 9, 10, 11, 12, 13),
    "animal_nw_day": (2, 3, 4, 5, 6),
    "animal_ne_day": (6, 7, 8, 9, 10),
    "animal_sw_day": (10, 11, 12, 13, 14),
    "feed_days_buffer": (1, 2, 3),
    "cash_reserve": (0, 100, 150, 250, 400, 600),
    "ongoing_harvest_threshold": (1, 2, 3, 4),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} does not define agent")
    return module


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _config_signature(config: dict) -> str:
    return json.dumps(
        {key: config.get(key) for key in MACRO_KEYS},
        sort_keys=True,
        separators=(",", ":"),
    )


def _candidate_name(config: dict, index: int) -> str:
    short = "_".join(str(config[key]) for key in MACRO_KEYS[:6])
    return f"macro_{index:03d}_{short}"


def build_candidates(count: int, seed: int) -> list[dict]:
    """Create a deterministic neighborhood plus seeded macro proposals."""
    rng = random.Random(seed)
    candidates: list[dict] = []
    seen: set[str] = set()

    def add(changes: dict):
        config = dict(DEFAULT_MACRO)
        config.update(changes)
        # Basic timing constraints prevent obviously dominated candidates.
        if config["land_sw_day"] <= config["land_ne_day"]:
            config["land_sw_day"] = min(13, config["land_ne_day"] + 4)
        if config["animal_ne_day"] < config["animal_nw_day"]:
            config["animal_ne_day"] = min(10, config["animal_nw_day"] + 3)
        if config["animal_sw_day"] < config["animal_ne_day"]:
            config["animal_sw_day"] = min(14, config["animal_ne_day"] + 3)
        signature = _config_signature(config)
        if signature in seen or len(candidates) >= count:
            return
        seen.add(signature)
        candidates.append({
            "name": _candidate_name(config, len(candidates)),
            "config": config,
        })

    # The V012 point is useful as a dynamic control, even though it is not
    # the fixed-route V012 submission used as the final control opponent.
    add({})

    # One-factor neighborhood: this is the most interpretable part of the
    # search and makes the first calibration useful for diagnosing direction.
    for key in MACRO_KEYS:
        for value in PARAM_VALUES[key]:
            if value != DEFAULT_MACRO[key]:
                add({key: value})

    # Hand-written combinations represent plausible production structures.
    profiles = (
        {"hands": 12, "cows": 8, "sheep": 5, "opening_wheat": 12, "opening_melons": 8, "strawberries": 30, "cash_reserve": 250},
        {"hands": 13, "cows": 8, "sheep": 6, "opening_wheat": 8, "opening_melons": 10, "strawberries": 38, "cash_reserve": 150},
        {"hands": 12, "cows": 10, "sheep": 4, "opening_wheat": 10, "opening_melons": 9, "strawberries": 34, "cash_reserve": 250},
        {"hands": 14, "cows": 7, "sheep": 7, "opening_wheat": 10, "opening_melons": 9, "strawberries": 38, "cash_reserve": 100},
        {"hands": 11, "cows": 12, "sheep": 2, "opening_wheat": 14, "opening_melons": 7, "strawberries": 28, "feed_days_buffer": 2, "cash_reserve": 400},
        {"hands": 13, "cows": 6, "sheep": 8, "opening_wheat": 10, "opening_melons": 9, "strawberries": 34, "feed_days_buffer": 2},
        {"hands": 13, "land_ne_day": 4, "land_sw_day": 9, "animal_nw_day": 3, "animal_ne_day": 7, "animal_sw_day": 11},
        {"hands": 13, "land_ne_day": 6, "land_sw_day": 12, "animal_nw_day": 5, "animal_ne_day": 9, "animal_sw_day": 13, "cash_reserve": 250},
    )
    for profile in profiles:
        add(profile)

    # Fill the remaining budget with 2-4 coordinate mutations.  We sample
    # from a fixed table rather than arbitrary values so every result is
    # reproducible and easy to rerun from its config JSON.
    while len(candidates) < count:
        keys = rng.sample(MACRO_KEYS, rng.choice((2, 3, 4)))
        changes = {key: rng.choice(PARAM_VALUES[key]) for key in keys}
        add(changes)
        if len(seen) > count * 20 and len(candidates) < count:
            # This is only a guard for unusually small parameter tables.
            add({key: rng.choice(PARAM_VALUES[key]) for key in MACRO_KEYS})
    return candidates


class ObservedAgent:
    def __init__(self, agent):
        self.agent = agent
        self.calls = 0
        self.errors = 0
        self.invalid = 0
        self.times_ms: list[float] = []
        self.field_counts = Counter()
        self.market_counts = Counter()

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        try:
            action = self.agent(obs, config)
        except TypeError:
            try:
                action = self.agent(obs)
            except Exception:
                self.errors += 1
                action = {"farmer": ["PASS"], "hands": [], "market": []}
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        self.calls += 1
        self.times_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if not isinstance(action, dict):
            self.invalid += 1
            return action
        farmer = action.get("farmer")
        hands = action.get("hands", [])
        market = action.get("market", [])
        if not isinstance(farmer, list) or not farmer or not isinstance(hands, list) or not isinstance(market, list):
            self.invalid += 1
        for operation in [farmer, *hands] if isinstance(hands, list) else [farmer]:
            if isinstance(operation, list) and operation:
                self.field_counts[str(operation[0])] += 1
        if isinstance(market, list):
            for order in market:
                if isinstance(order, list) and order:
                    self.market_counts[str(order[0])] += 1
        return action

    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        values = sorted(self.times_ms)
        return float(values[min(len(values) - 1, int(round((len(values) - 1) * 0.99)))])


_WORKER = None


def _worker_state():
    global _WORKER
    if _WORKER is not None:
        return _WORKER
    pid = os.getpid()
    v012_candidate = load_module(V012_PATH, f"v014_v012_candidate_{pid}")
    v012_control = load_module(V012_PATH, f"v014_v012_control_{pid}")
    baseline = load_module(BASELINE_PATH, f"v014_baseline_{pid}")
    _WORKER = {
        "candidate": v012_candidate,
        "control": v012_control,
        "baseline": baseline,
        "v18": None,
        "hamburger": None,
        "frontier": None,
    }
    return _WORKER


def _load_optional_opponent(name: str):
    state = _worker_state()
    if name == "v18" and state["v18"] is None:
        from run_v012_top5_vs_v18 import load_v18_agent
        state["v18"] = load_v18_agent()
    elif name == "hamburger" and state["hamburger"] is None:
        from run_v006_benchmark import load_hamburger_agent
        state["hamburger"] = load_hamburger_agent(ROOT / "baseline/kaggriculture-hamburger.ipynb")
    elif name == "frontier" and state["frontier"] is None:
        from run_v008_benchmark import load_notebook_agent
        state["frontier"] = load_notebook_agent(
            ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb",
            f"v014_frontier_{os.getpid()}",
        )
    return state.get(name)


def _opponent(name: str):
    state = _worker_state()
    if name in {"starter", "random"}:
        return name
    if name == "v012":
        return state["control"].agent
    if name == "baseline":
        return state["baseline"].agent
    return _load_optional_opponent(name)


def _run_one(job: dict) -> dict:
    state = _worker_state()
    candidate_module = state["candidate"]
    if job["candidate"] == "v012_control":
        candidate_module = state["control"]
        candidate_module.configure_strategy()
    else:
        candidate_module.configure_strategy(job["config"])

    candidate = ObservedAgent(candidate_module.agent)
    opponent = _opponent(job["opponent"])
    players = [candidate, opponent] if int(job["seat"]) == 0 else [opponent, candidate]
    started = time.perf_counter()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(job["seed"])},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    seat = int(job["seat"])
    own = final[seat]
    other = final[1 - seat]
    own_money = float(own.observation["farms"][seat]["money"])
    other_money = float(other.observation["farms"][1 - seat]["money"])
    margin = own_money - other_money
    result = "win" if margin > 0 else "loss" if margin < 0 else "tie"
    return {
        "phase": job["phase"],
        "candidate": job["candidate"],
        "config": json.dumps(job.get("config", {}), sort_keys=True),
        "opponent": job["opponent"],
        "seed": int(job["seed"]),
        "seat": seat,
        "candidate_money": own_money,
        "opponent_money": other_money,
        "margin": margin,
        "result": result,
        "candidate_status": own.status,
        "opponent_status": other.status,
        "game_done": int(own.status == "DONE" and other.status == "DONE"),
        "agent_errors": candidate.errors,
        "invalid_actions": candidate.invalid,
        "action_calls": candidate.calls,
        "runtime_p99_ms": candidate.p99_ms(),
        "runtime_max_ms": max(candidate.times_ms or [0.0]),
        "wall_seconds": time.perf_counter() - started,
        "field_counts": json.dumps(dict(sorted(candidate.field_counts.items())), sort_keys=True),
        "market_counts": json.dumps(dict(sorted(candidate.market_counts.items())), sort_keys=True),
    }


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [_run_one(job) for job in jobs]
    context = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            # map preserves job order, which makes logs and CSV diffs reproducible.
            return list(pool.map(_run_one, jobs, chunksize=1))
    except (OSError, PermissionError) as exc:
        # Some notebook/sandbox runtimes disallow POSIX semaphores.  A normal
        # local terminal can still use --workers; restricted environments
        # should remain usable, only slower.
        print(f"multiprocessing unavailable ({exc}); falling back to serial", flush=True)
        return [_run_one(job) for job in jobs]


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    out = []
    for key, group in groups.items():
        wins = sum(row["result"] == "win" for row in group)
        ties = sum(row["result"] == "tie" for row in group)
        losses = sum(row["result"] == "loss" for row in group)
        out.append({
            **dict(zip(fields, key)),
            "games": len(group),
            "mean_cash": statistics.mean(row["candidate_money"] for row in group),
            "min_cash": min(row["candidate_money"] for row in group),
            "max_cash": max(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": wins / len(group),
            "done_rate": statistics.mean(row["game_done"] for row in group),
            "agent_errors": sum(row["agent_errors"] for row in group),
            "invalid_actions": sum(row["invalid_actions"] for row in group),
            "max_p99_ms": max(row["runtime_p99_ms"] for row in group),
            "max_runtime_ms": max(row["runtime_max_ms"] for row in group),
            "total_wall_seconds": sum(row["wall_seconds"] for row in group),
        })
    return sorted(out, key=lambda row: tuple(str(row[field]) for field in fields))


def _rank_calibration(rows: list[dict]) -> list[dict]:
    candidates = sorted({row["candidate"] for row in rows if row["candidate"] != "v012_control"})
    ranked = []
    for name in candidates:
        group = [row for row in rows if row["candidate"] == name]
        direct = [row for row in group if row["opponent"] == "v012"]
        if not direct:
            direct = group
        ranked.append({
            "candidate": name,
            "config": group[0]["config"],
            "games": len(group),
            "mean_cash": statistics.mean(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "direct_mean_margin_vs_v012": statistics.mean(row["margin"] for row in direct),
            "min_cash": min(row["candidate_money"] for row in group),
            "done_rate": statistics.mean(row["game_done"] for row in group),
            "agent_errors": sum(row["agent_errors"] for row in group),
            "invalid_actions": sum(row["invalid_actions"] for row in group),
            "max_p99_ms": max(row["runtime_p99_ms"] for row in group),
        })
    return sorted(
        ranked,
        key=lambda row: (
            -row["done_rate"],
            row["agent_errors"] + row["invalid_actions"],
            -row["direct_mean_margin_vs_v012"],
            -row["min_cash"],
            -row["mean_cash"],
            row["candidate"],
        ),
    )


def _make_jobs(candidates, phase, opponents, seeds, seats):
    jobs = []
    specs = [{"name": "v012_control", "config": {"use_fixed_schedule": True}}] + candidates
    for spec in specs:
        for opponent in opponents:
            for seed in seeds:
                for seat in seats:
                    jobs.append({
                        "phase": phase,
                        "candidate": spec["name"],
                        "config": spec["config"],
                        "opponent": opponent,
                        "seed": seed,
                        "seat": seat,
                    })
    return jobs


def _gate(final_rows: list[dict], finalist_names: list[str]) -> dict:
    control = [row for row in final_rows if row["candidate"] == "v012_control"]
    control_mean = statistics.mean(row["candidate_money"] for row in control) if control else 0.0
    control_min = min((row["candidate_money"] for row in control), default=0.0)
    report = {
        "winner": None,
        "control": {
            "games": len(control),
            "mean_cash": control_mean,
            "min_cash": control_min,
        },
        "candidates": {},
    }
    for name in finalist_names:
        group = [row for row in final_rows if row["candidate"] == name]
        direct = [row for row in group if row["opponent"] == "v012"]
        if not group:
            continue
        mean_cash = statistics.mean(row["candidate_money"] for row in group)
        min_cash = min(row["candidate_money"] for row in group)
        direct_margin = statistics.mean(row["margin"] for row in direct) if direct else 0.0
        metrics = {
            "games": len(group),
            "mean_cash": mean_cash,
            "min_cash": min_cash,
            "mean_gain_pct_vs_control": mean_cash / control_mean - 1.0 if control_mean else 0.0,
            "min_cash_ratio_vs_control": min_cash / control_min if control_min else 0.0,
            "mean_margin_vs_v012": direct_margin,
            "win_rate_vs_v012": sum(row["result"] == "win" for row in direct) / len(direct) if direct else 0.0,
            "all_done": all(row["game_done"] for row in group),
            "agent_errors": sum(row["agent_errors"] for row in group),
            "invalid_actions": sum(row["invalid_actions"] for row in group),
            "max_p99_ms": max(row["runtime_p99_ms"] for row in group),
        }
        # A search candidate must beat or tie the fixed V012 control on the
        # paired matrix.  The 0.5% improvement requirement is intentionally
        # not used here: first we need a statistically credible direction;
        # promotion can apply the stricter competition gate later.
        metrics["passes"] = bool(
            metrics["all_done"]
            and metrics["agent_errors"] == 0
            and metrics["invalid_actions"] == 0
            and metrics["mean_gain_pct_vs_control"] >= 0.0
            and metrics["mean_margin_vs_v012"] >= 0.0
            and metrics["max_p99_ms"] < 1000.0
        )
        report["candidates"][name] = metrics
    passing = [(name, metrics) for name, metrics in report["candidates"].items() if metrics["passes"]]
    if passing:
        report["winner"] = max(
            passing,
            key=lambda item: (
                item[1]["mean_gain_pct_vs_control"],
                item[1]["mean_margin_vs_v012"],
                item[1]["min_cash_ratio_vs_control"],
                item[0],
            ),
        )[0]
    return report


def _write_submission(config: dict, candidate_name: str, out_dir: Path):
    source = V012_PATH.read_text(encoding="utf-8")
    injection = (
        "\n\n# V014 macro-search winner.\n"
        f"_V014_SELECTED_CONFIG = {json.dumps(config, sort_keys=True)}\n"
        "configure_strategy(_V014_SELECTED_CONFIG)\n"
    )
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_path = HISTORY_DIR / "main.py"
    archive_path = out_dir / "submission.tar.gz"
    main_path.write_text(source + injection, encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")
    (out_dir / "submission_manifest.json").write_text(
        json.dumps(
            {
                "candidate": candidate_name,
                "config": config,
                "main": str(main_path),
                "archive": str(archive_path),
                "depends_on_local_logs": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibration", "final", "all"), default="all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=1405)
    parser.add_argument("--calibration-seeds", type=_parse_ints, default=CALIBRATION_SEEDS)
    parser.add_argument("--final-seeds", type=_parse_ints, default=FINAL_SEEDS)
    parser.add_argument("--calibration-seats", type=_parse_ints, default=(0,))
    parser.add_argument("--final-seats", type=_parse_ints, default=(0, 1))
    parser.add_argument(
        "--calibration-opponents",
        type=_parse_names,
        default=("v012", "baseline"),
    )
    parser.add_argument(
        "--final-opponents",
        type=_parse_names,
        default=("v012", "v18", "baseline"),
    )
    parser.add_argument("--write-submission", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(max(1, args.candidate_count), args.search_seed)
    (args.out / "candidate_configs.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8"
    )

    calibration_rows = []
    if args.stage in {"calibration", "all"}:
        jobs = _make_jobs(
            candidates,
            "calibration",
            args.calibration_opponents,
            args.calibration_seeds,
            args.calibration_seats,
        )
        print(f"V014 calibration: {len(candidates)} candidates, {len(jobs)} games, workers={args.workers}", flush=True)
        calibration_rows = _run_jobs(jobs, max(1, args.workers))
        _write_csv(args.out / "calibration_raw.csv", calibration_rows)
        _write_csv(args.out / "calibration_summary.csv", _aggregate(calibration_rows, ("phase", "candidate", "opponent")))
        ranking = _rank_calibration(calibration_rows)
        (args.out / "calibration_ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        finalists = [row["candidate"] for row in ranking[: max(1, args.top_k)]]
        (args.out / "finalists.json").write_text(json.dumps(finalists, indent=2), encoding="utf-8")
        print("V014 calibration finalists:", ", ".join(finalists), flush=True)
        if args.stage == "calibration":
            return
    else:
        finalists = json.loads((args.out / "finalists.json").read_text(encoding="utf-8"))

    config_by_name = {candidate["name"]: candidate["config"] for candidate in candidates}
    final_candidates = [
        {"name": name, "config": config_by_name[name]}
        for name in finalists
        if name in config_by_name
    ]
    final_jobs = _make_jobs(
        final_candidates,
        "final",
        args.final_opponents,
        args.final_seeds,
        args.final_seats,
    )
    print(f"V014 final: {len(final_candidates)} candidates + control, {len(final_jobs)} games, workers={args.workers}", flush=True)
    final_rows = _run_jobs(final_jobs, max(1, args.workers))
    _write_csv(args.out / "final_raw.csv", final_rows)
    _write_csv(args.out / "final_summary.csv", _aggregate(final_rows, ("phase", "candidate", "opponent")))
    gate = _gate(final_rows, finalists)
    (args.out / "gate_report.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)

    winner = gate.get("winner")
    if winner and args.write_submission:
        _write_submission(config_by_name[winner], winner, args.out)
        print(f"V014 winner submission written for {winner}", flush=True)
    elif not winner:
        print("No V014 candidate passed the paired gate; keep V012 as control.", flush=True)


if __name__ == "__main__":
    main()
