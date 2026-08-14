"""Run V032 smoke, local notebook matrix and fixed-nowinlog diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make

from rl_010_opponents import load_spec, unique_loadable_specs


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "baseline/artifacts/v032_route_conditioned_timing"
NOWINLOG = ROOT / "baseline/artifacts/25:27 Strict-Future | v27 Midgame Meta Reset/nowinlog"


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"v032_{tag}_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Probe:
    def __init__(self, module):
        self.module = module
        self.times = []
        self.errors = 0
        self.invalid = 0

    def __call__(self, obs, config=None):
        started = time.perf_counter_ns()
        try:
            action = self.module.agent(obs, config)
        except TypeError:
            try:
                action = self.module.agent(obs)
            except Exception:
                self.errors += 1
                action = {"farmer": ["PASS"], "hands": [], "market": []}
        except Exception:
            self.errors += 1
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        self.times.append((time.perf_counter_ns() - started) / 1e6)
        farms = obs.get("farms", []) or []
        seat = int(obs.get("player", 0) or 0)
        expected = len(farms[seat].get("hands", []) or []) if seat < len(farms) else 0
        valid = isinstance(action, dict) and isinstance(action.get("farmer"), list)
        valid = valid and len(action.get("hands", []) or []) == expected
        valid = valid and len(action.get("market", []) or []) <= 10
        self.invalid += int(not valid)
        return action if isinstance(action, dict) else {"farmer": ["PASS"], "hands": [], "market": []}

    def metric(self):
        values = sorted(self.times)
        def pct(q):
            return values[min(len(values) - 1, int(round((len(values) - 1) * q)))] if values else 0.0
        return {"agent_calls": len(values), "agent_errors": self.errors, "invalid_shapes": self.invalid,
                "p50_ms": pct(.50), "p95_ms": pct(.95), "p99_ms": pct(.99), "max_ms": max(values or [0.0])}


class ReplayOpponent:
    def __init__(self, payload, seat):
        self.steps = payload.get("steps", [])
        self.seat = int(seat)

    def __call__(self, obs, config=None):
        del config
        step = min(max(0, int(obs.get("step", 0) or 0) + 1), len(self.steps) - 1)
        return json.loads(json.dumps(self.steps[step][self.seat].get("action") or {}))


def _result(candidate, opponent, seed, seat, module, opponent_name):
    probe = Probe(module)
    players = [probe, opponent] if seat == 0 else [opponent, probe]
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": int(seed)}, debug=False)
    env.run(players)
    final = env.steps[-1]
    mine = final[seat].observation["farms"][seat]["money"]
    theirs = final[1 - seat].observation["farms"][1 - seat]["money"]
    margin = float(mine) - float(theirs)
    return {"candidate": candidate, "opponent": opponent_name, "seed": int(seed), "seat": int(seat),
            "candidate_money": float(mine), "opponent_money": float(theirs), "margin": margin,
            "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
            "done": int(all(str(x.status) == "DONE" for x in final)), **probe.metric(),
            "timing_stats": json.dumps(getattr(module, "V032_STATS", {}), sort_keys=True)}


def run_local(candidates, seeds, output, opponent_names=None):
    specs, _ = unique_loadable_specs()
    builtin = [{"name": "starter", "builtin": "starter", "source_sha256": "builtin:starter"},
               {"name": "random", "builtin": "random", "source_sha256": "builtin:random"}]
    specs = specs + builtin
    if opponent_names:
        wanted = set(opponent_names)
        specs = [spec for spec in specs if spec["name"] in wanted]
    rows = []
    for candidate in candidates:
        module = _load(ARTIFACT_ROOT / candidate / "main.py", candidate)
        for spec in specs:
            if spec.get("builtin"):
                opponent, meta = spec["builtin"], spec
            else:
                opponent, meta = load_spec(spec)
            for seed in seeds:
                for seat in (0, 1):
                    print(f"{candidate} vs {spec['name']} seed={seed} seat={seat}", flush=True)
                    rows.append(_result(candidate, opponent, seed, seat, module, spec["name"]))
    return rows


def run_nowin(candidates, replay_dir, output):
    rows = []
    for path in sorted(Path(replay_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        agents = payload.get("info", {}).get("Agents", []) or []
        names = [str(x.get("Name", "")) for x in agents]
        if "GzmCR632" not in names:
            continue
        seat = names.index("GzmCR632")
        opponent_seat = 1 - seat
        seed = int(payload.get("info", {}).get("seed", 0) or 0)
        opponent = ReplayOpponent(payload, opponent_seat)
        for candidate in candidates:
            module = _load(ARTIFACT_ROOT / candidate / "main.py", candidate)
            row = _result(candidate, opponent, seed, seat, module, names[opponent_seat])
            row["replay"] = path.name
            rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["candidate"], row["opponent"])].append(row)
    result = []
    for (candidate, opponent), group in sorted(groups.items()):
        wins = sum(row["result"] == "win" for row in group)
        ties = sum(row["result"] == "tie" for row in group)
        result.append({"candidate": candidate, "opponent": opponent, "games": len(group),
                       "wins": wins, "ties": ties, "losses": len(group) - wins - ties,
                       "win_rate": wins / len(group),
                       "score_rate": (wins + .5 * ties) / len(group),
                       "mean_margin": statistics.mean(row["margin"] for row in group),
                       "min_margin": min(row["margin"] for row in group),
                       "all_done": int(all(row["done"] for row in group)),
                       "errors": sum(row["agent_errors"] for row in group),
                       "invalid": sum(row["invalid_shapes"] for row in group),
                       "max_p99_ms": max(row["p99_ms"] for row in group)})
    return result


def summarize_route_attribution(rows):
    """Compare each timing arm with its same-route order-only control."""
    by_key = {(row["candidate"], row.get("opponent"), row.get("seed"), row.get("seat")): row
              for row in rows}
    result = []
    for route in ("v27", "8c4s"):
        control = f"v032_{route}_order_only"
        timing = f"v032_{route}_timing"
        pairs = []
        for key, row in by_key.items():
            if key[0] != timing:
                continue
            base = by_key.get((control, key[1], key[2], key[3]))
            if base is not None:
                pairs.append((base, row))
        if not pairs:
            continue
        deltas = [float(t["margin"]) - float(b["margin"]) for b, t in pairs]
        result.append({"route": route, "control": control, "timing": timing,
                       "paired_games": len(pairs),
                       "timing_nonnegative": sum(delta >= 0 for delta in deltas),
                       "timing_negative": sum(delta < 0 for delta in deltas),
                       "mean_timing_margin_delta": statistics.mean(deltas),
                       "min_timing_margin_delta": min(deltas),
                       "max_timing_margin_delta": max(deltas)})
    return result


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["candidate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    candidates = tuple(args.candidates)
    local = run_local(candidates, tuple(args.seeds), out, args.opponents) if args.stage in {"local", "all"} else []
    nowin = run_nowin(candidates, args.nowinlog, out) if args.stage in {"nowin", "all"} else []
    write_rows(out / "local_raw.csv", local)
    write_rows(out / "nowin_raw.csv", nowin)
    summary = summarize(local + nowin)
    write_rows(out / "summary.csv", summary)
    (out / "route_attribution.json").write_text(
        json.dumps(summarize_route_attribution(local + nowin), indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(json.dumps({"candidates": candidates, "seeds": args.seeds,
        "nowinlog": str(args.nowinlog), "engine": "kaggle-environments==1.32.6"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("local", "nowin", "all"), default="local")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "benchmark")
    parser.add_argument("--nowinlog", type=Path, default=NOWINLOG)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026])
    parser.add_argument("--opponents", nargs="*", default=None)
    parser.add_argument("--candidates", nargs="+", default=["v032_v27_timing", "v032_8c4s_timing"])
    main(parser.parse_args())
