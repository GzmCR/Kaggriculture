"""Compare core action and production metrics for baseline and top-agent opponents."""

from __future__ import annotations

import importlib.util
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from run_v008_benchmark import DEFAULT_SEEDS, load_opponents


ROOT = Path(__file__).resolve().parents[1]
EPISODE_STEPS = 720


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_agent(agent, obs, config):
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


def public_metrics(obs, player):
    farms = obs.get("farms", []) or []
    if not (0 <= player < len(farms)):
        return {}
    farm = farms[player]
    plants = Counter()
    animals = Counter()
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                plants[str(tile.get("crop"))] += 1
            animal = tile.get("animal")
            if animal:
                animals[str(animal)] += 1
    return {
        "visible_hands": len(farm.get("hands", []) or []),
        "unlocked": len(farm.get("unlocked_quadrants", []) or []),
        "plants": plants,
        "animals": animals,
    }


class Tracer:
    def __init__(self, agent, player):
        self.agent = agent
        self.player = player
        self.field = Counter()
        self.market = Counter()
        self.quantities = Counter()
        self.max_visible_hands = 0
        self.max_plants = Counter()
        self.max_animals = Counter()
        self.max_unlocked = 0

    def __call__(self, obs, config=None):
        metrics = public_metrics(obs, self.player)
        self.max_visible_hands = max(self.max_visible_hands, metrics.get("visible_hands", 0))
        self.max_unlocked = max(self.max_unlocked, metrics.get("unlocked", 0))
        for item, count in metrics.get("plants", {}).items():
            self.max_plants[item] = max(self.max_plants[item], count)
        for item, count in metrics.get("animals", {}).items():
            self.max_animals[item] = max(self.max_animals[item], count)

        action = call_agent(self.agent, obs, config)
        if not isinstance(action, dict):
            return action
        for operation in [action.get("farmer", []), *(action.get("hands", []) or [])]:
            if isinstance(operation, list) and operation:
                op = str(operation[0])
                self.field[op] += 1
                if op == "PLANT" and len(operation) >= 2:
                    self.field[f"PLANT:{operation[1]}"] += 1
        for order in action.get("market", []) or []:
            if not isinstance(order, list) or not order:
                continue
            op = str(order[0])
            self.market[op] += 1
            if len(order) >= 3 and op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
                item = str(order[1])
                self.quantities[f"{op}:{item}"] += int(order[2] or 0)
        return action


def counter_json(counter):
    return dict(sorted(counter.items()))


def run_pair(opponent_name, seed, seat):
    baseline = load_module(ROOT / "main.py", f"metric_baseline_{opponent_name}_{seed}_{seat}")
    opponent = load_opponents([opponent_name])[opponent_name]
    baseline_trace = Tracer(baseline.agent, seat)
    opponent_trace = Tracer(opponent, 1 - seat)
    players = [baseline_trace, opponent_trace] if seat == 0 else [opponent_trace, baseline_trace]
    env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}, debug=False)
    env.run(players)
    final = env.steps[-1]
    baseline_state = final[seat]
    opponent_state = final[1 - seat]
    return {
        "opponent": opponent_name,
        "seed": seed,
        "seat": seat,
        "baseline_money": float(baseline_state.observation["farms"][seat]["money"]),
        "opponent_money": float(opponent_state.observation["farms"][1 - seat]["money"]),
        "baseline_field": counter_json(baseline_trace.field),
        "opponent_field": counter_json(opponent_trace.field),
        "baseline_market": counter_json(baseline_trace.market),
        "opponent_market": counter_json(opponent_trace.market),
        "baseline_quantities": counter_json(baseline_trace.quantities),
        "opponent_quantities": counter_json(opponent_trace.quantities),
        "baseline_max_plants": counter_json(baseline_trace.max_plants),
        "opponent_max_plants": counter_json(opponent_trace.max_plants),
        "baseline_max_animals": counter_json(baseline_trace.max_animals),
        "opponent_max_animals": counter_json(opponent_trace.max_animals),
        "baseline_max_visible_hands": baseline_trace.max_visible_hands,
        "opponent_max_visible_hands": opponent_trace.max_visible_hands,
        "baseline_max_unlocked": baseline_trace.max_unlocked,
        "opponent_max_unlocked": opponent_trace.max_unlocked,
    }


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["opponent"]].append(row)
    output = {}
    for opponent, group in groups.items():
        metrics = {
            "games": len(group),
            "baseline_money_mean": statistics.mean(row["baseline_money"] for row in group),
            "opponent_money_mean": statistics.mean(row["opponent_money"] for row in group),
            "baseline_money_min": min(row["baseline_money"] for row in group),
            "opponent_money_min": min(row["opponent_money"] for row in group),
        }
        for prefix in ("baseline", "opponent"):
            for kind in ("field", "market", "quantities", "max_plants", "max_animals"):
                if kind.startswith("max_"):
                    merged = {}
                    for row in group:
                        for item, value in row[f"{prefix}_{kind}"].items():
                            merged[item] = max(merged.get(item, 0), value)
                    metrics[f"{prefix}_{kind}"] = dict(sorted(merged.items()))
                else:
                    merged = Counter()
                    for row in group:
                        merged.update(row[f"{prefix}_{kind}"])
                    metrics[f"{prefix}_{kind}"] = dict(sorted(merged.items()))
            metrics[f"{prefix}_max_visible_hands"] = max(row[f"{prefix}_max_visible_hands"] for row in group)
            metrics[f"{prefix}_max_unlocked"] = max(row[f"{prefix}_max_unlocked"] for row in group)
        output[opponent] = metrics
    return output


def main():
    rows = []
    for opponent_name in ("hamburger", "frontier"):
        for seed in DEFAULT_SEEDS:
            for seat in (0, 1):
                rows.append(run_pair(opponent_name, seed, seat))
                print(f"{opponent_name} seed={seed} seat={seat}", flush=True)
    out = ROOT / "baseline/artifacts/v009_market_counter/core_metrics.json"
    out.write_text(json.dumps({"rows": rows, "summary": aggregate(rows)}, indent=2), encoding="utf-8")
    print(json.dumps(aggregate(rows), indent=2))


if __name__ == "__main__":
    main()
