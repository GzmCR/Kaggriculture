"""Run V019 replay calibration and public-style counterfactuals."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from run_v015a_market_collision import Probe, load_module, percentile, write_csv
from v019_replay_analysis import (
    DEFAULT_REPLAY_DIR,
    EPISODE_STEPS,
    OUT_DIR as ANALYSIS_DIR,
    PREMIUM,
    _obs,
    reconstruct_turn,
    collect_replays,
)
from v019_style_router import PublicStyleTracker


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "baseline/artifacts/v019_public_style_router"
V012_PATH = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
V018_PATH = ROOT / "baseline/artifacts/v018_market_wave/v018c_robust_mpc.py"
V019_ARTIFACTS = {
    "v019a": OUT_DIR / "v019a_price_priority.py",
    "v019b": OUT_DIR / "v019b_public_style.py",
    "v019c": OUT_DIR / "v019c_weak_counter.py",
}
EXPERTS = ("mohit", "automatylicza", "manual_player", "navazsh_fathi", "lucien_de_rubempre")
DEFAULT_MAPPING = {
    "standard_converged": "automatylicza",
    "reduced_ne_only": "mohit",
    "high_worker_maintenance": "manual_player",
    "premium_concentrated": "navazsh_fathi",
}


def _invoke(agent, obs, config=None):
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


class ReplaySideAgent:
    def __init__(self, steps, seat):
        self.actions = [copy.deepcopy(steps[min(i + 1, EPISODE_STEPS - 1)][seat].get("action") or {}) for i in range(EPISODE_STEPS)]

    def __call__(self, obs, config=None):
        step = max(0, min(int(obs.get("step", 0) or 0), EPISODE_STEPS - 1))
        return copy.deepcopy(self.actions[step])


class FixedExpertAgent:
    def __init__(self, expert):
        self.expert = expert
        self.module = load_module(V012_PATH, f"v019_fixed_{expert}_{time.time_ns()}")

    def __call__(self, obs, config=None):
        base = _invoke(self.module.agent, obs, config)
        step = max(0, min(int(obs.get("step", 0) or 0), EPISODE_STEPS - 1))
        expert_action = self.module._V18_RUNTIME["experts"][self.expert]["actions"][step]
        return {
            "farmer": list(base.get("farmer") or ["PASS"]),
            "hands": [list(item) for item in (base.get("hands") or [])],
            "market": [list(order) for order in (expert_action.get("market") or [])],
        }


class V019Agent:
    def __init__(self, variant, mapping):
        self.variant = variant
        self.module = load_module(V019_ARTIFACTS[variant], f"v019_{variant}_{time.time_ns()}")
        self.base = load_module(V012_PATH, f"v019_compare_{variant}_{time.time_ns()}")
        self.module._V019_ROUTER.mapping = dict(mapping)
        self.field_changed = 0
        self.nonpremium_changed = 0

    @staticmethod
    def _nonpremium(orders):
        return [
            order for order in (orders or [])
            if not (isinstance(order, list) and len(order) >= 2 and order[0] == "SELL" and order[1] in PREMIUM)
        ]

    def __call__(self, obs, config=None):
        base_action = _invoke(self.base.agent, obs, config)
        output = _invoke(self.module.agent, obs, config)
        if output.get("farmer") != base_action.get("farmer") or output.get("hands") != base_action.get("hands"):
            self.field_changed += 1
        if self._nonpremium(output.get("market")) != self._nonpremium(base_action.get("market")):
            self.nonpremium_changed += 1
        return output

    def diagnostics(self, player):
        result = self.module._V019_ROUTER.diagnostics(player)
        result.update({"field_changed": self.field_changed, "nonpremium_changed": self.nonpremium_changed})
        return result


def _style_for_target(replay, candidate_seat):
    tracker = PublicStyleTracker()
    styles = []
    steps = replay["payload"]["steps"]
    for step in range(EPISODE_STEPS):
        obs = dict(_obs(steps, step, candidate_seat))
        obs["player"] = candidate_seat
        style, confidence, features = tracker.observe(obs)
        if step % 24 == 0 and step // 24 >= 10:
            styles.append((style, confidence, features))
    if not styles:
        return "standard_converged", 0.0
    counts = Counter(item[0] for item in styles)
    style = counts.most_common(1)[0][0]
    confidence = max(item[1] for item in styles if item[0] == style)
    return style, confidence


def _transactions(env, seat):
    payload = env.toJSON()
    steps = payload.get("steps", [])
    totals = {item: {"requested": 0, "filled": 0, "revenue": 0.0, "floor_units": 0} for item in PREMIUM}
    for step in range(min(EPISODE_STEPS, len(steps))):
        reconstructed = reconstruct_turn(steps, step)["per_seat"][seat]
        for item in PREMIUM:
            for key in totals[item]:
                totals[item][key] += reconstructed[item][key]
    for item in PREMIUM:
        totals[item]["weighted_price"] = (
            totals[item]["revenue"] / totals[item]["filled"] if totals[item]["filled"] else 0.0
        )
    return totals


def run_game(candidate, replay, candidate_seat, label, opponent_seat=None):
    opponent_seat = 1 - candidate_seat if opponent_seat is None else opponent_seat
    opponent = ReplaySideAgent(replay["payload"]["steps"], opponent_seat)
    players = [candidate, opponent] if candidate_seat == 0 else [opponent, candidate]
    probe = Probe(candidate)
    players = [probe, opponent] if candidate_seat == 0 else [opponent, probe]
    started = time.perf_counter()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(replay["seed"])},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    mine, other = final[candidate_seat], final[1 - candidate_seat]
    my_money = float(mine.observation["farms"][candidate_seat]["money"])
    other_money = float(other.observation["farms"][1 - candidate_seat]["money"])
    margin = my_money - other_money
    row = {
        "episode": replay["episode"],
        "seed": replay["seed"],
        "seat": candidate_seat,
        "variant": label,
        "opponent_style": _style_for_target(replay, candidate_seat)[0],
        "candidate_money": my_money,
        "opponent_money": other_money,
        "margin": margin,
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "game_done": int(mine.status == "DONE" and other.status == "DONE"),
        "wall_seconds": time.perf_counter() - started,
    }
    row.update(probe.metrics())
    if callable(getattr(candidate, "diagnostics", None)):
        diagnostics = candidate.diagnostics(candidate_seat)
        for key, value in diagnostics.items():
            if key == "history" or key.endswith("_history"):
                row[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
    for item, values in _transactions(env, candidate_seat).items():
        for key, value in values.items():
            row[f"actual_{item}_{key}"] = value
    return row


def _targets(replays):
    for replay in replays:
        for seat in (0, 1):
            yield replay, seat


def run_calibration(replays, out_dir=OUT_DIR):
    rows = []
    control_rows = {}
    targets = list(_targets(replays))
    total = len(targets) * (1 + len(EXPERTS))
    index = 0
    for replay, seat in targets:
        index += 1
        print(f"[{index}/{total}] control_v012 episode={replay['episode']} seat={seat}", flush=True)
        row = run_game(load_module(V012_PATH, f"v019_cal_control_{time.time_ns()}").agent, replay, seat, "control_v012")
        rows.append(row)
        control_rows[(replay["episode"], seat)] = row
        for expert in EXPERTS:
            index += 1
            print(f"[{index}/{total}] fixed_{expert} episode={replay['episode']} seat={seat}", flush=True)
            row = run_game(FixedExpertAgent(expert), replay, seat, f"fixed_{expert}")
            row["expert"] = expert
            rows.append(row)
    for row in rows:
        control = control_rows.get((row["episode"], row["seat"]))
        if control is not None:
            row["control_margin"] = control["margin"]
            row["delta_vs_control"] = row["margin"] - control["margin"]
    write_csv(out_dir / "calibration_raw.csv", rows)
    write_csv(out_dir / "calibration_summary.csv", summarize(rows, ("variant", "opponent_style")))
    return rows


def loo_mapping(calibration_rows, test_episode):
    groups = defaultdict(list)
    for row in calibration_rows:
        if row.get("variant", "").startswith("fixed_") and row.get("episode") != test_episode:
            groups[(row.get("opponent_style"), row.get("expert"))].append(float(row.get("delta_vs_control", 0.0)))
    mapping = dict(DEFAULT_MAPPING)
    evidence = {}
    for style in DEFAULT_MAPPING:
        candidates = []
        for expert in EXPERTS:
            values = groups.get((style, expert), [])
            if len(values) >= 3:
                candidates.append((statistics.mean(values), expert, len(values)))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[2], item[1]), reverse=True)
            mapping[style] = candidates[0][1]
            evidence[style] = {"expert": candidates[0][1], "mean_delta": candidates[0][0], "games": candidates[0][2]}
        else:
            evidence[style] = {"expert": mapping[style], "mean_delta": None, "games": 0}
    return mapping, evidence


def run_counterfactual(replays, calibration_rows, out_dir=OUT_DIR):
    rows = []
    targets = list(_targets(replays))
    total = len(targets) * 5
    index = 0
    for replay, seat in targets:
        mapping, evidence = loo_mapping(calibration_rows, replay["episode"])
        for label in ("control_v012", "control_v018", "v019a", "v019b", "v019c"):
            index += 1
            print(f"[{index}/{total}] {label} episode={replay['episode']} seat={seat}", flush=True)
            if label == "control_v012":
                candidate = load_module(V012_PATH, f"v019_cf_v012_{time.time_ns()}").agent
            elif label == "control_v018":
                candidate = load_module(V018_PATH, f"v019_cf_v018_{time.time_ns()}").agent
            else:
                candidate = V019Agent(label, mapping)
            row = run_game(candidate, replay, seat, label)
            row["mapping"] = json.dumps(mapping, sort_keys=True)
            row["mapping_evidence"] = json.dumps(evidence, sort_keys=True)
            rows.append(row)
    write_csv(out_dir / "counterfactual_raw.csv", rows)
    write_csv(out_dir / "counterfactual_summary.csv", summarize(rows, ("variant", "opponent_style")))
    return rows


def summarize(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        results = Counter(row.get("result") for row in group)
        item = dict(zip(fields, key))
        item.update({
            "games": len(group),
            "mean_money": statistics.mean(float(row["candidate_money"]) for row in group),
            "mean_margin": statistics.mean(float(row["margin"]) for row in group),
            "min_money": min(float(row["candidate_money"]) for row in group),
            "wins": results["win"],
            "ties": results["tie"],
            "losses": results["loss"],
            "win_rate": results["win"] / len(group),
            "all_done": int(all(int(row.get("game_done", 0)) for row in group)),
            "agent_errors": sum(int(row.get("agent_errors", 0)) for row in group),
            "invalid_action_shapes": sum(int(row.get("invalid_action_shapes", 0)) for row in group),
            "p99_ms": max(float(row.get("runtime_p99_ms", 0.0)) for row in group),
        })
        for metric in ("field_changed", "nonpremium_changed", "actual_MILK_revenue", "actual_STRAWBERRY_revenue", "actual_MILK_filled", "actual_STRAWBERRY_filled"):
            if any(metric in row for row in group):
                item[f"sum_{metric}"] = sum(float(row.get(metric, 0.0)) for row in group)
        output.append(item)
    return output


def build_gate(rows, replays=None):
    """Evaluate the V019 research gates against a matched V012 control.

    ``invalid_action_shapes`` is compared per replay to the same control,
    because V012 itself has a known daily-boundary shape count in this replay
    harness.  A candidate is not allowed to introduce additional shape
    failures, but inheriting the control count is not treated as a new bug.
    Score bands are episode-level labels: the replay set does not identify
    which seat/team owns the filename score.
    """
    controls = [row for row in rows if row.get("variant") == "control_v012"]
    control_mean = statistics.mean(float(row["candidate_money"]) for row in controls) if controls else 0.0
    control_min = min((float(row["candidate_money"]) for row in controls), default=0.0)
    control_wins = sum(row.get("result") == "win" for row in controls)
    control_games = len(controls)
    control_by_target = {(row.get("episode"), row.get("seat")): row for row in controls}
    band_map = {
        str(replay.get("episode")): list(replay.get("score_bands") or [])
        for replay in (replays or [])
    }
    control_by_band = defaultdict(list)
    for row in controls:
        for band in band_map.get(str(row.get("episode")), []):
            control_by_band[band].append(float(row["candidate_money"]))
    result = {
        "control": "control_v012",
        "control_mean_money": control_mean,
        "control_min_money": control_min,
        "control_win_rate": control_wins / control_games if control_games else 0.0,
        "bands_are_episode_level": True,
        "variants": {},
    }
    for variant in ("control_v018", "v019a", "v019b", "v019c"):
        group = [row for row in rows if row.get("variant") == variant]
        wins = sum(row.get("result") == "win" for row in group)
        nonnegative = []
        no_new_invalid = True
        for row in group:
            control = control_by_target.get((row.get("episode"), row.get("seat")))
            if control is None:
                continue
            nonnegative.append(float(row["candidate_money"]) >= float(control["candidate_money"]))
            no_new_invalid = no_new_invalid and int(row.get("invalid_action_shapes", 0) or 0) <= int(control.get("invalid_action_shapes", 0) or 0)

        band_checks = {}
        for band, control_values in sorted(control_by_band.items()):
            values = [
                float(row["candidate_money"])
                for row in group
                if band in band_map.get(str(row.get("episode")), [])
            ]
            if not values:
                band_checks[band] = {"games": 0, "pass": False}
            else:
                control_band_mean = statistics.mean(control_values)
                candidate_band_mean = statistics.mean(values)
                band_checks[band] = {
                    "games": len(values),
                    "control_mean_money": control_band_mean,
                    "candidate_mean_money": candidate_band_mean,
                    "relative_to_control": candidate_band_mean / control_band_mean if control_band_mean else None,
                    "pass": candidate_band_mean >= control_band_mean * 0.95,
                }
        checks = {
            "all_done": bool(group) and all(int(row.get("game_done", 0)) for row in group),
            "no_errors": bool(group) and sum(int(row.get("agent_errors", 0)) for row in group) == 0,
            "no_new_invalid_vs_control": bool(group) and no_new_invalid,
            "mean_not_lower": bool(group) and statistics.mean(float(row["candidate_money"]) for row in group) >= control_mean,
            "min_not_below_95pct": bool(group) and min(float(row["candidate_money"]) for row in group) >= control_min * 0.95,
            "at_least_60pct_targets_not_lower": bool(nonnegative) and sum(nonnegative) / len(nonnegative) >= 0.60,
            "all_score_bands_within_5pct": bool(band_checks) and all(item["pass"] for item in band_checks.values()),
            "field_unchanged": all(float(row.get("field_changed") or 0.0) == 0 for row in group),
            "p99_under_1000ms": bool(group) and max(float(row.get("runtime_p99_ms") or 0.0) for row in group) < 1000.0,
        }
        result["variants"][variant] = {
            "pass": all(checks.values()),
            "checks": checks,
            "win_rate_not_lower_diagnostic": bool(group) and wins / len(group) >= (control_wins / control_games if control_games else 0.0),
            "games": len(group),
            "mean_money": statistics.mean(float(row["candidate_money"]) for row in group) if group else None,
            "mean_margin": statistics.mean(float(row["margin"]) for row in group) if group else None,
            "wins": wins,
            "nonnegative_target_fraction": sum(nonnegative) / len(nonnegative) if nonnegative else None,
            "score_bands": band_checks,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibrate", "counterfactual", "all"), default="all")
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replays = collect_replays(args.replay_dir)
    calibration_rows = []
    if args.stage in {"calibrate", "all"}:
        calibration_rows = run_calibration(replays, args.out_dir)
    elif (args.out_dir / "calibration_raw.csv").exists():
        with (args.out_dir / "calibration_raw.csv").open(newline="", encoding="utf-8") as handle:
            calibration_rows = list(csv.DictReader(handle))
    counterfactual_rows = []
    if args.stage in {"counterfactual", "all"}:
        if not calibration_rows:
            raise RuntimeError("calibration_raw.csv is required before counterfactual stage")
        counterfactual_rows = run_counterfactual(replays, calibration_rows, args.out_dir)
        report = build_gate(counterfactual_rows, replays)
        (args.out_dir / "gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"calibration_games": len(calibration_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
