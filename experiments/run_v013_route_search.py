"""Search V013 field routes while keeping V012's market policy fixed.

The runner has two phases:

* calibration: candidate field routes use the automatylicza market tape;
* final: the top five routes are evaluated with the full V012 market agent.

The competition environment silently ignores missing or extra hand actions,
so hand-list mismatches are recorded as diagnostics rather than treated as
agent failures.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import importlib.util
import json
import statistics
import sys
import tarfile
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make


_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

try:
    from .run_v006_benchmark import load_hamburger_agent
    from .run_v008_benchmark import load_module, load_notebook_agent
    from .run_v012_top5_vs_v18 import load_v18_agent
except ImportError:
    from run_v006_benchmark import load_hamburger_agent
    from run_v008_benchmark import load_module, load_notebook_agent
    from run_v012_top5_vs_v18 import load_v18_agent


ROOT = Path(__file__).resolve().parents[1]
V012_PATH = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
LOG_DIR = ROOT / "log/2026-08-04"
NOTEBOOK_DIR = ROOT / "baseline"
OUT_DIR = ROOT / "baseline/artifacts/v013_route_mutation"
HISTORY_DIR = ROOT / "baseline/history/v013_route_mutation"
EPISODE_STEPS = 720
TURNS_PER_DAY = 24
CALIBRATION_SEEDS = (17, 42, 2026)
HOLDOUT_SEEDS = (217, 317, 733)
ALL_SEEDS = CALIBRATION_SEEDS + HOLDOUT_SEEDS


def _copy_action(action):
    action = action if isinstance(action, dict) else {}
    farmer = action.get("farmer")
    if not isinstance(farmer, list) or not farmer:
        farmer = ["PASS"]
    hands = action.get("hands")
    if not isinstance(hands, list):
        hands = []
    market = action.get("market")
    if not isinstance(market, list):
        market = []
    return {
        "farmer": list(farmer),
        "hands": [list(item) if isinstance(item, list) else ["PASS"] for item in hands],
        "market": [list(item) if isinstance(item, list) else ["PASS"] for item in market],
    }


def _field_action(action):
    copied = _copy_action(action)
    return {
        "farmer": copied["farmer"],
        "hands": copied["hands"],
    }


def _route_from_actions(name, actions, source_kind, source_file="", source_seat=None):
    normalized = [_copy_action(actions[min(index, len(actions) - 1)]) for index in range(EPISODE_STEPS)]
    return {
        "name": name,
        "source_kind": source_kind,
        "source_file": source_file,
        "source_seat": source_seat,
        "actions": normalized,
        "field_actions": [_field_action(action) for action in normalized],
        "terminal_cash": None,
    }


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _terminal_cash(steps, seat):
    try:
        observation = steps[min(EPISODE_STEPS - 1, len(steps) - 1)][seat]["observation"]
        return float(observation["farms"][seat]["money"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _load_replay_file(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    if len(steps) < EPISODE_STEPS:
        raise ValueError(f"{path} has only {len(steps)} steps")

    # When the original source-seat metadata is unavailable, use the side
    # with higher terminal cash. This is only a fallback for raw replay files.
    cash = [_terminal_cash(steps, seat) for seat in (0, 1)]
    seat = 0 if (cash[0] or 0) >= (cash[1] or 0) else 1
    actions = []
    for index in range(EPISODE_STEPS):
        entry = steps[min(index + 1, EPISODE_STEPS - 1)][seat]
        actions.append(entry.get("action") or {})
    route = _route_from_actions(
        f"replay_{path.stem}_s{seat}",
        actions,
        "raw_replay",
        path.name,
        seat,
    )
    route["terminal_cash"] = cash[seat]
    return route


def _load_trace_notebook(path, name):
    """Extract a self-contained TRACE_ACTIONS list from a notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "AGENT_SOURCE" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(target, "id", None) == "AGENT_SOURCE" for target in node.targets):
                continue
            source = ast.literal_eval(node.value)
            namespace = {"__name__": name}
            exec(compile(source, str(path), "exec"), namespace)
            trace = namespace.get("TRACE_ACTIONS")
            if isinstance(trace, list) and len(trace) >= EPISODE_STEPS:
                return _route_from_actions(
                    name,
                    trace,
                    "embedded_notebook",
                    path.name,
                    None,
                )
    return None


def load_source_routes(v012_module):
    routes = []
    source_manifest = {
        "replay_directory": str(LOG_DIR),
        "raw_replay_files": [],
        "fallback_used": False,
        "fallback_routes": [],
    }

    if LOG_DIR.exists():
        for path in sorted(LOG_DIR.glob("*.json")):
            try:
                route = _load_replay_file(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            routes.append(route)
            source_manifest["raw_replay_files"].append(path.name)

    if not routes:
        source_manifest["fallback_used"] = True
        runtime = getattr(v012_module, "_V012_RUNTIME", {})
        experts = runtime.get("experts", {}) if isinstance(runtime, dict) else {}
        for name, expert in experts.items():
            actions = expert.get("actions", []) if isinstance(expert, dict) else []
            if len(actions) == EPISODE_STEPS:
                route = _route_from_actions(name, actions, "v012_embedded")
                routes.append(route)
                source_manifest["fallback_routes"].append(name)

        # These two public notebooks contain complete fixed traces and are
        # useful recovery sources when the ignored replay directory is absent.
        for path, name in (
            (NOTEBOOK_DIR / "kaggriculture-adaptive-replay-agent.ipynb", "soil_v25"),
            (NOTEBOOK_DIR / "kaggriculture-agent-builder.ipynb", "kaito_builder"),
        ):
            if path.exists():
                try:
                    route = _load_trace_notebook(path, name)
                except Exception:
                    route = None
                if route is not None:
                    routes.append(route)
                    source_manifest["fallback_routes"].append(name)

    if not routes:
        raise FileNotFoundError(
            "No raw replay JSONs or embedded V012 routes were available"
        )
    source_manifest["source_route_count"] = len(routes)
    source_manifest["source_routes"] = [
        {
            "name": route["name"],
            "kind": route["source_kind"],
            "file": route["source_file"],
            "seat": route["source_seat"],
            "terminal_cash": route["terminal_cash"],
        }
        for route in routes
    ]
    return routes, source_manifest


def _clone_route(route, name, kind):
    return {
        **route,
        "name": name,
        "source_kind": kind,
        "actions": copy.deepcopy(route["actions"]),
        "field_actions": copy.deepcopy(route["field_actions"]),
    }


def _blend_field_routes(left, right, split_step, name, kind):
    actions = []
    for step in range(EPISODE_STEPS):
        field = left["field_actions"][step] if step < split_step else right["field_actions"][step]
        actions.append({"farmer": list(field["farmer"]), "hands": [list(item) for item in field["hands"]]})
    return _route_from_actions(name, actions, kind)


def build_candidates(routes):
    by_name = {route["name"]: route for route in routes}
    auto = next(
        (route for route in routes if "automatylicza" in route["name"].lower()),
        routes[0],
    )
    source_routes = list(routes)
    candidates = []

    # The intended path is 15 raw replay routes. If those files are missing,
    # keep the candidate cardinality stable with explicitly marked derived
    # aliases so the benchmark still exercises the complete matrix.
    for route in source_routes[:15]:
        candidates.append({
            "name": f"raw_{route['name']}",
            "route": route,
            "overlay": "none",
            "kind": "raw_replay" if route["source_kind"] == "raw_replay" else "recovered_source",
        })
    while len(candidates) < 15:
        route = source_routes[(len(candidates) - len(source_routes)) % len(source_routes)]
        alias = _clone_route(
            route,
            f"recovered_alias_{len(candidates) + 1:02d}_{route['name']}",
            "recovered_alias",
        )
        candidates.append({
            "name": f"raw_{alias['name']}",
            "route": alias,
            "overlay": "none",
            "kind": "recovered_alias",
        })

    others = [route for route in source_routes if route["name"] != auto["name"]]
    crossover_sources = (others + [auto])[:5]
    while len(crossover_sources) < 5:
        crossover_sources.append(auto)
    for index, route in enumerate(crossover_sources, start=1):
        blended = _blend_field_routes(
            auto,
            route,
            10 * TURNS_PER_DAY,
            f"crossover_day10_{index:02d}_{route['name']}",
            "day10_crossover",
        )
        candidates.append({
            "name": blended["name"],
            "route": blended,
            "overlay": "none",
            "kind": "day10_crossover",
        })

    for overlay in ("none", "urgent_maintenance", "terminal_storage", "combined"):
        candidates.append({
            "name": f"automatylicza_overlay_{overlay}",
            "route": auto,
            "overlay": overlay,
            "kind": "maintenance_variant",
        })
    return candidates[:24]


def _tile_at(farm, pos):
    if not isinstance(farm, dict) or not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = farm.get("tiles", [])
    if not isinstance(tiles, list) or y < 0 or y >= len(tiles):
        return None
    row = tiles[y]
    if not isinstance(row, list) or x < 0 or x >= len(row):
        return None
    return row[x]


def _maintenance_op(tile):
    if not isinstance(tile, dict):
        return None
    kind = tile.get("kind")
    if kind == "PLANT" and not tile.get("watered_today", False):
        return ["WATER"]
    if kind in {"COOP", "PASTURE"} and not tile.get("fed_today", False):
        return ["FEED"]
    return None


def _apply_overlay(obs, field, overlay, reference):
    if overlay == "none":
        return field
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = int(obs.get("player", 0) or 0) if isinstance(obs, dict) else 0
    if not isinstance(farms, list) or not (0 <= player < len(farms)):
        return field
    farm = farms[player]
    positions = [farm.get("farmer")] + list(farm.get("hands", []) or [])
    private = obs.get("private", {}) if isinstance(obs, dict) else {}
    inventories = private.get("inventories", []) if isinstance(private, dict) else []
    result = {
        "farmer": list(field.get("farmer", ["PASS"])),
        "hands": [list(item) for item in field.get("hands", [])],
    }
    ref_field = _field_action(reference)
    ref_units = [ref_field.get("farmer", ["PASS"])] + ref_field.get("hands", [])

    for index, pos in enumerate(positions):
        tile = _tile_at(farm, pos)
        current = result["farmer"] if index == 0 else (
            result["hands"][index - 1] if index - 1 < len(result["hands"]) else ["PASS"]
        )
        reference_unit = ref_units[index] if index < len(ref_units) else ["PASS"]
        replacement = None
        if overlay in {"urgent_maintenance", "combined"}:
            # Do not interrupt a scheduled movement or harvest. The overlay
            # only takes over an idle unit, which keeps the replay route
            # coherent while still repairing an urgent local omission.
            if current and current[0] == "PASS":
                replacement = _maintenance_op(tile)
        if replacement is None and overlay in {"terminal_storage", "combined"}:
            current_op = current[0] if current else "PASS"
            reference_op = reference_unit[0] if reference_unit else "PASS"
            if current_op == "PASS" and reference_op in {
                "HARVEST",
                "DROP",
                "COLLECT_FERTILIZER",
                "CARE",
            }:
                replacement = list(reference_unit)
            elif current_op == "PASS" and isinstance(inventories, list) and index < len(inventories):
                inventory = inventories[index]
                if isinstance(inventory, dict) and any(int(value or 0) > 0 for value in inventory.values()):
                    if reference_op == "DROP":
                        replacement = ["DROP"]
        if replacement is not None:
            if index == 0:
                result["farmer"] = replacement
            elif index - 1 < len(result["hands"]):
                result["hands"][index - 1] = replacement
    return result


def _invoke(agent, obs, config=None):
    if isinstance(agent, str):
        raise TypeError("string agents are only valid when passed directly to env.run")
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


class CandidatePolicy:
    def __init__(self, spec, v012_agent, market_mode):
        self.spec = spec
        self.v012_agent = v012_agent
        self.market_mode = market_mode

    def __call__(self, obs, config=None):
        step = max(0, min(int(obs.get("step", 0) or 0), EPISODE_STEPS - 1))
        route_action = self.spec["route"]["actions"][step]
        field = _field_action(route_action)
        reference = _invoke(self.v012_agent, obs, config)
        field = _apply_overlay(obs, field, self.spec["overlay"], reference)
        market = (
            reference.get("market", [])
            if self.market_mode == "dynamic"
            else self.spec["market_route"]["actions"][step].get("market", [])
        )
        return {
            "farmer": field["farmer"],
            "hands": field["hands"],
            "market": [list(item) for item in market] if isinstance(market, list) else [],
        }


class ReplayPolicy:
    def __init__(self, route):
        self.route = route

    def __call__(self, obs, config=None):
        step = max(0, min(int(obs.get("step", 0) or 0), EPISODE_STEPS - 1))
        return _copy_action(self.route["actions"][step])


class Probe:
    def __init__(self, agent):
        self.agent = agent
        self.calls = 0
        self.errors = 0
        self.hand_shape_mismatches = 0
        self.market_overflows = 0
        self.times_ms = []
        self.daily_cash = {}
        self.field_counts = Counter()
        self.market_counts = Counter()

    def __call__(self, obs, config=None):
        step = int(obs.get("step", 0) or 0)
        if step == 0:
            self.calls = 0
            self.errors = 0
            self.hand_shape_mismatches = 0
            self.market_overflows = 0
            self.times_ms = []
            self.daily_cash = {}
            self.field_counts.clear()
            self.market_counts.clear()
        started = time.perf_counter_ns()
        try:
            action = _invoke(self.agent, obs, config)
        except Exception:
            self.errors += 1
            farms = obs.get("farms", []) or []
            player = int(obs.get("player", 0) or 0)
            count = len(farms[player].get("hands", []) or []) if 0 <= player < len(farms) else 0
            action = {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(count)], "market": []}
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        self.calls += 1
        self.times_ms.append(elapsed)

        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0) or 0)
        expected = len(farms[player].get("hands", []) or []) if 0 <= player < len(farms) else 0
        hands = action.get("hands", []) if isinstance(action, dict) else []
        if not isinstance(hands, list) or len(hands) != expected:
            self.hand_shape_mismatches += 1
        market = action.get("market", []) if isinstance(action, dict) else []
        if isinstance(market, list) and len(market) > 10:
            self.market_overflows += 1

        if step % TURNS_PER_DAY == TURNS_PER_DAY - 1 and 0 <= player < len(farms):
            self.daily_cash[str(step // TURNS_PER_DAY)] = float(farms[player].get("money", 0))
        operations = []
        if isinstance(action, dict):
            operations.append(action.get("farmer", []))
            operations.extend(action.get("hands", []) or [])
            for operation in operations:
                if isinstance(operation, list) and operation:
                    self.field_counts[str(operation[0])] += 1
            for order in action.get("market", []) or []:
                if isinstance(order, list) and order:
                    self.market_counts[str(order[0])] += 1
        return action

    @property
    def p99_ms(self):
        if not self.times_ms:
            return 0.0
        values = sorted(self.times_ms)
        return float(values[min(len(values) - 1, int(round((len(values) - 1) * 0.99)))])


def run_game(candidate, opponent, seed, seat, phase, candidate_name, opponent_name):
    probe = Probe(candidate)
    players = [probe, opponent] if seat == 0 else [opponent, probe]
    started = time.perf_counter()
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    candidate_state = final[seat]
    opponent_state = final[1 - seat]
    candidate_money = float(candidate_state.observation["farms"][seat]["money"])
    opponent_money = float(opponent_state.observation["farms"][1 - seat]["money"])
    margin = candidate_money - opponent_money
    result = "win" if margin > 0 else "loss" if margin < 0 else "tie"
    return {
        "phase": phase,
        "candidate": candidate_name,
        "opponent": opponent_name,
        "seed": int(seed),
        "seat": int(seat),
        "candidate_money": candidate_money,
        "opponent_money": opponent_money,
        "margin": margin,
        "result": result,
        "candidate_status": candidate_state.status,
        "opponent_status": opponent_state.status,
        "game_done": int(candidate_state.status == "DONE" and opponent_state.status == "DONE"),
        "agent_errors": probe.errors,
        "hand_shape_mismatches": probe.hand_shape_mismatches,
        "market_overflows": probe.market_overflows,
        "action_calls": probe.calls,
        "runtime_p99_ms": probe.p99_ms,
        "runtime_max_ms": max(probe.times_ms or [0.0]),
        "wall_seconds": time.perf_counter() - started,
        "daily_cash": json.dumps(probe.daily_cash, sort_keys=True),
        "field_counts": json.dumps(dict(sorted(probe.field_counts.items())), sort_keys=True),
        "market_counts": json.dumps(dict(sorted(probe.market_counts.items())), sort_keys=True),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    summaries = []
    for key, group in grouped.items():
        results = Counter(row["result"] for row in group)
        summaries.append({
            **dict(zip(group_fields, key)),
            "games": len(group),
            "mean_cash": statistics.mean(row["candidate_money"] for row in group),
            "min_cash": min(row["candidate_money"] for row in group),
            "max_cash": max(row["candidate_money"] for row in group),
            "mean_margin": statistics.mean(row["margin"] for row in group),
            "wins": results["win"],
            "ties": results["tie"],
            "losses": results["loss"],
            "win_rate": results["win"] / len(group),
            "done_rate": statistics.mean(row["game_done"] for row in group),
            "agent_errors": sum(row["agent_errors"] for row in group),
            "hand_shape_mismatches": sum(row["hand_shape_mismatches"] for row in group),
            "market_overflows": sum(row["market_overflows"] for row in group),
            "max_p99_ms": max(row["runtime_p99_ms"] for row in group),
            "max_runtime_ms": max(row["runtime_max_ms"] for row in group),
        })
    return sorted(summaries, key=lambda row: tuple(str(row[field]) for field in group_fields))


def _load_agents(v012_module):
    v012_agent = getattr(v012_module, "agent")
    baseline_module = load_module(ROOT / "main.py", "v013_baseline")
    opponents = {
        "baseline": baseline_module.agent,
        "v012": v012_agent,
        "v18": load_v18_agent(),
        "starter": "starter",
        "random": "random",
        "hamburger": load_hamburger_agent(ROOT / "baseline/kaggriculture-hamburger.ipynb"),
        "frontier": load_notebook_agent(
            ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb",
            "v013_frontier",
        ),
    }
    return v012_agent, opponents


def _make_candidate(spec, v012_agent, market_route, market_mode):
    return CandidatePolicy(
        {**spec, "market_route": market_route},
        v012_agent,
        market_mode,
    )


def _run_candidate_matrix(specs, v012_agent, market_route, opponents, seeds, phase, out_rows):
    total = len(specs) * len(opponents) * len(seeds) * 2
    completed = 0
    for spec in specs:
        mode = "dynamic" if phase == "final" else "fixed"
        candidate = _make_candidate(spec, v012_agent, market_route, mode)
        for opponent_name, opponent in opponents.items():
            for seed in seeds:
                for seat in (0, 1):
                    row = run_game(candidate, opponent, seed, seat, phase, spec["name"], opponent_name)
                    out_rows.append(row)
                    completed += 1
                    print(
                        f"[{completed}/{total}] {phase} {spec['name']} vs {opponent_name} "
                        f"seed={seed} seat={seat} cash={row['candidate_money']:.0f} "
                        f"mismatch={row['hand_shape_mismatches']} status={row['candidate_status']}",
                        flush=True,
                    )


def _run_parent_round_robin(routes, v012_agent, seeds, out_rows):
    parents = [("v012", v012_agent)]
    parents.extend((route["name"], ReplayPolicy(route)) for route in routes[:5])
    for left_index in range(len(parents)):
        for right_index in range(left_index + 1, len(parents)):
            left_name, left = parents[left_index]
            right_name, right = parents[right_index]
            for seed in seeds:
                for seat in (0, 1):
                    out_rows.append(
                        run_game(
                            left,
                            right,
                            seed,
                            seat,
                            "parent_round_robin",
                            f"parent_{left_name}",
                            f"parent_{right_name}",
                        )
                    )


def _candidate_ranking(rows):
    summaries = _aggregate(
        [row for row in rows if row["phase"] == "calibration"],
        ["candidate"],
    )
    return sorted(
        summaries,
        key=lambda row: (
            -row["done_rate"],
            row["agent_errors"],
            -row["min_cash"],
            -row["mean_margin"],
            -row["mean_cash"],
            row["hand_shape_mismatches"],
            row["candidate"],
        ),
    )


def _select_finalists(ranking, candidates, limit=5):
    """Prefer distinct field routes before filling the requested top-k."""
    by_name = {candidate["name"]: candidate for candidate in candidates}
    selected = []
    seen_routes = set()
    for row in ranking:
        spec = by_name.get(row["candidate"])
        if spec is None:
            continue
        signature = json.dumps(
            spec["route"]["field_actions"],
            separators=(",", ":"),
            sort_keys=True,
        )
        if signature in seen_routes:
            continue
        seen_routes.add(signature)
        selected.append(row["candidate"])
        if len(selected) >= limit:
            return selected
    for row in ranking:
        if row["candidate"] not in selected:
            selected.append(row["candidate"])
        if len(selected) >= limit:
            break
    return selected


def _evaluate_gate(rows, candidate_names):
    holdout = [row for row in rows if row["phase"] == "final" and row["seed"] in HOLDOUT_SEEDS]
    control = [row for row in holdout if row["candidate"] == "v012_control"]
    if not control:
        return {"winner": None, "control": {}, "candidates": {}}
    control_mean = statistics.mean(row["candidate_money"] for row in control)
    control_min = min(row["candidate_money"] for row in control)
    control_win_vs_v012 = [
        row for row in control if row["opponent"] == "v012"
    ]
    report = {
        "winner": None,
        "control": {
            "mean_cash": control_mean,
            "min_cash": control_min,
            "games": len(control),
        },
        "candidates": {},
    }
    for name in candidate_names:
        group = [row for row in holdout if row["candidate"] == name]
        direct = [row for row in group if row["opponent"] == "v012"]
        if not group:
            continue
        mean_cash = statistics.mean(row["candidate_money"] for row in group)
        min_cash = min(row["candidate_money"] for row in group)
        mean_direct_margin = statistics.mean([row["margin"] for row in direct] or [0.0])
        metrics = {
            "mean_cash": mean_cash,
            "min_cash": min_cash,
            "mean_gain_pct": mean_cash / control_mean - 1.0,
            "min_cash_ratio": min_cash / control_min,
            "mean_direct_margin_vs_v012": mean_direct_margin,
            "all_done": all(row["game_done"] for row in group),
            "agent_errors": sum(row["agent_errors"] for row in group),
            "hand_shape_mismatches": sum(row["hand_shape_mismatches"] for row in group),
            "max_p99_ms": max(row["runtime_p99_ms"] for row in group),
            "win_rate_vs_v012": (
                sum(row["result"] == "win" for row in direct) / len(direct)
                if direct else 0.0
            ),
        }
        metrics["passes"] = bool(
            metrics["all_done"]
            and metrics["agent_errors"] == 0
            and metrics["min_cash_ratio"] >= 0.99
            and metrics["mean_gain_pct"] >= 0.005
            and metrics["mean_direct_margin_vs_v012"] >= 0.0
        )
        report["candidates"][name] = metrics

    passing = [
        (name, metrics)
        for name, metrics in report["candidates"].items()
        if metrics["passes"]
    ]
    if passing:
        report["winner"] = max(
            passing,
            key=lambda item: (
                item[1]["mean_cash"],
                item[1]["min_cash"],
                item[1]["mean_direct_margin_vs_v012"],
                -item[1]["hand_shape_mismatches"],
                item[0],
            ),
        )[0]
    return report


def _pack(value):
    encoded = base64.b85encode(
        zlib.compress(json.dumps(value, separators=(",", ":")).encode("utf-8"), 9)
    ).decode("ascii")
    return "".join(f"    {chunk!r},\n" for chunk in (encoded[index : index + 100] for index in range(0, len(encoded), 100)))


def _build_submission(spec, v012_source):
    route = [
        {
            "farmer": list(action.get("farmer", ["PASS"])),
            "hands": [list(item) for item in action.get("hands", [])],
        }
        for action in spec["route"]["actions"]
    ]
    payload = _pack(route)
    injection = f'''

# V013 selected field route overlay. The V012 source above remains the market
# policy and supplies the reference action for the optional maintenance layer.
import base64 as _v013_base64
import json as _v013_json
import zlib as _v013_zlib

_V013_ROUTE_PAYLOAD = ''.join([
{payload}])
_V013_ROUTE = _v013_json.loads(
    _v013_zlib.decompress(_v013_base64.b85decode(_V013_ROUTE_PAYLOAD)).decode('utf-8')
)
_V013_BASE_AGENT = agent
_V013_OVERLAY = {spec['overlay']!r}

def _v013_tile_at(farm, pos):
    if not isinstance(farm, dict) or not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = farm.get('tiles', [])
    if not isinstance(tiles, list) or y < 0 or y >= len(tiles) or not isinstance(tiles[y], list):
        return None
    return tiles[y][x] if 0 <= x < len(tiles[y]) else None

def _v013_urgent(tile):
    if not isinstance(tile, dict):
        return None
    if tile.get('kind') == 'PLANT' and not tile.get('watered_today', False):
        return ['WATER']
    if tile.get('kind') in {{'COOP', 'PASTURE'}} and not tile.get('fed_today', False):
        return ['FEED']
    return None

def _v013_apply_overlay(obs, field, reference):
    if _V013_OVERLAY == 'none':
        return field
    farms = obs.get('farms', []) if isinstance(obs, dict) else []
    player = int(obs.get('player', 0) or 0) if isinstance(obs, dict) else 0
    if not isinstance(farms, list) or not (0 <= player < len(farms)):
        return field
    farm = farms[player]
    positions = [farm.get('farmer')] + list(farm.get('hands', []) or [])
    reference_units = [reference.get('farmer', ['PASS'])] + list(reference.get('hands', []) or [])
    result = {{'farmer': list(field.get('farmer', ['PASS'])), 'hands': [list(item) for item in field.get('hands', [])]}}
    for index, pos in enumerate(positions):
        current = result['farmer'] if index == 0 else (result['hands'][index - 1] if index - 1 < len(result['hands']) else ['PASS'])
        replacement = _v013_urgent(_v013_tile_at(farm, pos)) if _V013_OVERLAY in ('urgent_maintenance', 'combined') and current and current[0] == 'PASS' else None
        ref = reference_units[index] if index < len(reference_units) else ['PASS']
        if replacement is None and _V013_OVERLAY in ('terminal_storage', 'combined') and current and current[0] == 'PASS' and ref and ref[0] in ('HARVEST', 'DROP', 'CARE', 'COLLECT_FERTILIZER'):
            replacement = list(ref)
        if replacement is not None:
            if index == 0:
                result['farmer'] = replacement
            elif index - 1 < len(result['hands']):
                result['hands'][index - 1] = replacement
    return result

def agent(obs, config=None):
    step = max(0, min(int(obs.get('step', 0) or 0), len(_V013_ROUTE) - 1))
    reference = _V013_BASE_AGENT(obs)
    field = _v013_apply_overlay(obs, _V013_ROUTE[step], reference)
    return {{'farmer': field['farmer'], 'hands': field['hands'], 'market': list(reference.get('market', []) or [])[:10]}}
'''
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_path = HISTORY_DIR / "main.py"
    archive_path = OUT_DIR / "submission.tar.gz"
    main_path.write_text(v012_source + injection, encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")
    (OUT_DIR / "submission_manifest.json").write_text(
        json.dumps(
            {
                "candidate": spec["name"],
                "overlay": spec["overlay"],
                "self_contained": True,
                "depends_on_local_logs_at_runtime": False,
                "main": str(main_path),
                "archive": str(archive_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return main_path, archive_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibration", "final", "all"), default="all")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    v012_module = _load_module(V012_PATH, "v013_v012")
    v012_agent, opponents = _load_agents(v012_module)
    routes, source_manifest = load_source_routes(v012_module)
    candidates = build_candidates(routes)
    auto = next((route for route in routes if "automatylicza" in route["name"].lower()), routes[0])
    source_manifest["candidate_count"] = len(candidates)
    source_manifest["candidate_names"] = [candidate["name"] for candidate in candidates]
    (args.out / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")

    rows = []
    if args.stage in {"calibration", "all"}:
        calibration_opponents = {
            "baseline": opponents["baseline"],
            "v012": opponents["v012"],
            "hamburger": opponents["hamburger"],
        }
        _run_candidate_matrix(
            candidates,
            v012_agent,
            auto,
            calibration_opponents,
            CALIBRATION_SEEDS,
            "calibration",
            rows,
        )
        _run_parent_round_robin(routes, v012_agent, (17,), rows)
        write_csv(args.out / "calibration_raw.csv", rows)
        write_csv(args.out / "calibration_summary.csv", _aggregate(rows, ["phase", "candidate"]))
        (args.out / "calibration_summary.json").write_text(
            json.dumps(_aggregate(rows, ["phase", "candidate"]), indent=2),
            encoding="utf-8",
        )
        ranking = _candidate_ranking(rows)
        (args.out / "calibration_ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        top_names = _select_finalists(ranking, candidates, 5)
        (args.out / "finalists.json").write_text(json.dumps(top_names, indent=2), encoding="utf-8")
        if args.stage == "calibration":
            return
    else:
        top_names = json.loads((args.out / "finalists.json").read_text(encoding="utf-8"))

    if args.stage in {"final", "all"}:
        if not rows:
            calibration_path = args.out / "calibration_raw.csv"
            if not calibration_path.exists():
                raise FileNotFoundError("Run --stage calibration before --stage final")
            with calibration_path.open(encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))
            top_names = json.loads((args.out / "finalists.json").read_text(encoding="utf-8"))
            # CSV values are strings; the final runner only needs candidate names.
        spec_by_name = {candidate["name"]: candidate for candidate in candidates}
        final_specs = [spec_by_name[name] for name in top_names if name in spec_by_name]
        final_rows = []
        final_opponents = {
            "baseline": opponents["baseline"],
            "v012": opponents["v012"],
            "v18": opponents["v18"],
            "starter": opponents["starter"],
            "random": opponents["random"],
            "hamburger": opponents["hamburger"],
            "frontier": opponents["frontier"],
        }
        _run_candidate_matrix(
            final_specs,
            v012_agent,
            auto,
            final_opponents,
            ALL_SEEDS,
            "final",
            final_rows,
        )
        control_spec = {
            "name": "v012_control",
            "route": auto,
            "overlay": "none",
            "kind": "control",
        }
        # A control is run through the same final matrix so the gate is paired
        # with the exact current environment and opponent set.
        _run_candidate_matrix(
            [control_spec],
            v012_agent,
            auto,
            final_opponents,
            ALL_SEEDS,
            "final",
            final_rows,
        )
        all_final_rows = final_rows
        write_csv(args.out / "final_raw.csv", all_final_rows)
        final_summary = _aggregate(all_final_rows, ["phase", "candidate", "opponent"])
        write_csv(args.out / "final_summary.csv", final_summary)
        (args.out / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
        gate = _evaluate_gate(all_final_rows, [spec["name"] for spec in final_specs])
        (args.out / "gate_report.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        winner_name = gate.get("winner")
        if winner_name:
            winner = spec_by_name[winner_name]
            main_path, archive_path = _build_submission(winner, V012_PATH.read_text(encoding="utf-8"))
            gate["submission"] = {"main": str(main_path), "archive": str(archive_path)}
            (args.out / "gate_report.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
            print(f"V013 winner: {winner_name}; wrote {main_path} and {archive_path}")
        else:
            print("No V013 candidate passed the holdout gate; V012 remains the control.")
        print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
