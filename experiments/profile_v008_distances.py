"""Collect frontier state distances on the current policy's tuning games."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

from kaggle_environments import make

from build_v008_hybrid import FRONTIER_NOTEBOOK, ROOT
from run_v006_benchmark import load_hamburger_agent


SEEDS = (17, 42, 2026)
OPPONENTS = ("starter", "random", "hamburger")
EPISODE_STEPS = 720


def load_current():
    spec = importlib.util.spec_from_file_location("v008_profile_current", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def load_frontier_helpers():
    notebook = json.loads(FRONTIER_NOTEBOOK.read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][17].get("source", []))
    tree = ast.parse(cell)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "AGENT_SOURCE" for target in node.targets)
    )
    source = ast.literal_eval(node.value)
    namespace = {"__name__": "v008_profile_frontier"}
    exec(compile(source, str(FRONTIER_NOTEBOOK), "exec"), namespace)
    return namespace["_features"], namespace["STATE_FEATURES"], namespace["PREFERRED_INDEX"]


def distance(features, references):
    values = [
        sum(abs(left - right) for left, right in zip(features, reference))
        for reference in references
    ]
    index = min(range(len(values)), key=lambda item: (values[item], item))
    return index, float(values[index])


def main():
    current = load_current()
    features, state_features, preferred = load_frontier_helpers()
    hamburger = load_hamburger_agent(ROOT / "baseline/kaggriculture-hamburger.ipynb")
    rows = []
    for opponent_name in OPPONENTS:
        opponent = hamburger if opponent_name == "hamburger" else opponent_name
        for seed in SEEDS:
            for seat in (0, 1):
                distances = []

                def candidate(obs, config=None):
                    step = int(obs.get("step", 0) or 0)
                    if step < len(state_features) and step % 24 == 0:
                        live = features(obs)
                        route, value = distance(live, state_features[step])
                        distances.append({"step": step, "route": route, "distance": value})
                    return current(obs, config)

                players = [candidate, opponent] if seat == 0 else [opponent, candidate]
                env = make(
                    "kaggriculture",
                    configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
                    debug=False,
                )
                env.run(players)
                rows.append({
                    "opponent": opponent_name,
                    "seed": seed,
                    "seat": seat,
                    "game_done": all(state.status == "DONE" for state in env.steps[-1]),
                    "distances": distances,
                })
                print(f"profile {opponent_name} seed={seed} seat={seat} points={len(distances)}", flush=True)

    values = sorted(item["distance"] for row in rows for item in row["distances"])

    def quantile(fraction):
        if not values:
            return 0.0
        position = (len(values) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(values) - 1)
        weight = position - low
        return values[low] * (1.0 - weight) + values[high] * weight

    report = {
        "games": len(rows),
        "all_done": all(row["game_done"] for row in rows),
        "points": len(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "quantiles": {str(fraction): quantile(fraction) for fraction in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)},
        "rows": rows,
    }
    out = ROOT / "baseline/artifacts/v008_hybrid_router/distance_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["quantiles"], indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
