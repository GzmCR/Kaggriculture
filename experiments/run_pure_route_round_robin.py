"""Pure fixed-route round robin for the locally identified route families.

This benchmark intentionally does *not* call any candidate's ``agent``
function.  It extracts the frozen action trace and returns the action at the
current step verbatim.  Consequently there is no market controller, order
sorting, preemption, delay, WEED recovery, hand alignment, or terminal
cleanup added by the benchmark.

The market orders embedded in a frozen route are still part of that route;
they are not generated or modified at runtime.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "baseline" / "artifacts" / "pure_route_family_round_robin"
# Fast diagnostic matrix requested for route-only comparison.
SEEDS = (17, 42, 2026)
EPISODE_STEPS = 720

ROUTE_FILES = {
    "V22": (ROOT / "baseline/artifacts/v031_route_market_combo/v22_raw/main.py", "_ACTIONS"),
    "Adaptive_V14": (
        ROOT / "baseline/artifacts/v031_route_market_combo/adaptive_raw/main.py",
        "_ACTIONS",
    ),
    "V27": (ROOT / "baseline/artifacts/v031_route_market_combo/v27_raw/main.py", "_LEGACY_ACTIONS"),
    "Stable12": (ROOT / "baseline/artifacts/v023b_stable_12hands/main.py", "__V023_STABLE12__"),
    "V022c": (ROOT / "baseline/artifacts/v022c_medoid_recovery/main.py", "_ACTIONS"),
}


def _load_module(path: Path, tag: str):
    module_name = f"pure_route_{tag}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_v13_route():
    path = ROOT / "baseline/v13-r3-top-meta-order-safe-premium-control.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        for block in re.findall(r"```python\n(.*?)\n```", text, re.S):
            if "def agent" not in block or ("_ACTIONS" not in block and "TRACE_ACTIONS" not in block):
                continue
            namespace = {"__name__": "pure_route_v13", "__file__": str(path)}
            exec(compile(block, str(path), "exec"), namespace, namespace)
            route = namespace.get("_ACTIONS") or namespace.get("TRACE_ACTIONS")
            if route:
                return route, "notebook-extracted"
    raise RuntimeError(f"could not extract V13-R3 fixed route from {path}")


def _load_routes():
    routes = {}
    sources = {}
    for name, (path, variable) in ROUTE_FILES.items():
        module = _load_module(path, name)
        if variable == "__V023_STABLE12__":
            route = module._V023_ROUTES["stable_12hands"]["actions"]
        else:
            route = getattr(module, variable)
        routes[name] = copy.deepcopy(route)
        sources[name] = str(path)
    routes["V13_R3"], sources["V13_R3"] = _load_v13_route()
    routes["V13_R3"] = copy.deepcopy(routes["V13_R3"])
    return routes, sources


def _fixed_agent(route):
    def agent(obs, config=None):
        step = int(obs.get("step", 0) or 0)
        step = min(max(step, 0), len(route) - 1)
        return copy.deepcopy(route[step])

    return agent


class Probe:
    def __init__(self, fn):
        self.fn = fn
        self.errors = 0
        self.invalid = 0
        self.calls = 0
        self.times_ms = []

    def __call__(self, obs, config=None):
        self.calls += 1
        started = time.perf_counter()
        try:
            action = self.fn(obs, config)
            if not isinstance(action, dict):
                self.invalid += 1
                return {"farmer": ["PASS"], "hands": [], "market": []}
            return action
        except Exception:
            self.errors += 1
            return {"farmer": ["PASS"], "hands": [], "market": []}
        finally:
            self.times_ms.append((time.perf_counter() - started) * 1000.0)


def _percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _run_game(routes, left_name, right_name, seed, left_seat):
    left = Probe(_fixed_agent(routes[left_name]))
    right = Probe(_fixed_agent(routes[right_name]))
    players = [left, right] if left_seat == 0 else [right, left]
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
        debug=False,
    )
    env.run(players)
    final = env.steps[-1]
    left_state = final[left_seat]
    right_state = final[1 - left_seat]
    left_money = left_state.observation["farms"][left_seat]["money"]
    right_money = right_state.observation["farms"][1 - left_seat]["money"]
    margin = left_money - right_money
    times = left.times_ms + right.times_ms
    statuses = [left_state.status, right_state.status]
    return {
        "left": left_name,
        "right": right_name,
        "seed": seed,
        "left_seat": left_seat,
        "left_money": left_money,
        "right_money": right_money,
        "margin": margin,
        "result": "W" if margin > 0 else "L" if margin < 0 else "T",
        "done": int(all(status == "DONE" for status in statuses)),
        "left_status": left_state.status,
        "right_status": right_state.status,
        "left_calls": left.calls,
        "right_calls": right.calls,
        "errors": left.errors + right.errors,
        "invalid": left.invalid + right.invalid,
        "p50_ms": _percentile(times, 0.50),
        "p95_ms": _percentile(times, 0.95),
        "p99_ms": _percentile(times, 0.99),
        "max_ms": max(times) if times else 0.0,
    }


def _summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["left"], row["right"]), []).append(row)
    result = []
    for (left, right), group in grouped.items():
        margins = [row["margin"] for row in group]
        result.append(
            {
                "left": left,
                "right": right,
                "games": len(group),
                "wins": sum(row["result"] == "W" for row in group),
                "ties": sum(row["result"] == "T" for row in group),
                "losses": sum(row["result"] == "L" for row in group),
                "win_rate": sum(row["result"] == "W" for row in group) / len(group),
                "mean_margin": statistics.mean(margins),
                "min_margin": min(margins),
                "mean_left_money": statistics.mean(row["left_money"] for row in group),
                "mean_right_money": statistics.mean(row["right_money"] for row in group),
                "done": sum(row["done"] for row in group),
                "errors": sum(row["errors"] for row in group),
                "invalid": sum(row["invalid"] for row in group),
                "max_p99_ms": max(row["p99_ms"] for row in group),
            }
        )
    return result


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    routes, sources = _load_routes()
    names = list(routes)
    route_lengths = {name: len(route) for name, route in routes.items()}
    route_hashes = {
        name: hashlib.sha256(json.dumps(route, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, route in routes.items()
    }
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    total = len(pairs) * len(SEEDS) * 2
    rows = []
    index = 0
    for left, right in pairs:
        for seed in SEEDS:
            for left_seat in (0, 1):
                index += 1
                print(f"[{index}/{total}] {left} vs {right} seed={seed} left_seat={left_seat}", flush=True)
                rows.append(_run_game(routes, left, right, seed, left_seat))

    summary = _summarize(rows)
    raw_path = ARTIFACT_DIR / "raw.csv"
    summary_path = ARTIFACT_DIR / "summary.csv"
    manifest_path = ARTIFACT_DIR / "manifest.json"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifest_path.write_text(
        json.dumps(
            {
                "mode": "pure_fixed_route",
                "description": "Return frozen route action verbatim; no runtime strategy layer.",
                "seeds": SEEDS,
                "episode_steps": EPISODE_STEPS,
                "candidates": names,
                "source_files": sources,
                "route_lengths": route_lengths,
                "route_sha256": route_hashes,
                "market_orders_are_embedded_route_data": True,
                "runtime_overlays": [],
            },
            indent=2,
        )
    )

    print("\nSUMMARY")
    for row in summary:
        print(
            f"{row['left']} vs {row['right']}: {row['wins']}-{row['ties']}-{row['losses']}, "
            f"mean_margin={row['mean_margin']:.1f}, min_margin={row['min_margin']:.1f}, "
            f"DONE={row['done']}/{row['games']}, errors={row['errors']}, invalid={row['invalid']}, "
            f"max_p99={row['max_p99_ms']:.3f}ms"
        )
    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
