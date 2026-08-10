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
import argparse
import ast
import base64
import csv
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import time
import os
import zlib
from pathlib import Path

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "baseline" / "artifacts" / "pure_route_family_round_robin"
# Fast diagnostic matrix requested for route-only comparison.
SEEDS = (17, 42, 2026)
EPISODE_STEPS = 720
_WORKER_ROUTES = None

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


NOTEBOOK_ROUTES = {
    "HighScore": ROOT / "baseline/2026-08-09/my-2026-08-04-high-score-pipeline.ipynb",
    "FrontierSoil": ROOT / "baseline/2026-08-09/kaggriculture-frontier-the-soil-remembers-rain.ipynb",
    "FrontierMoon": ROOT / "baseline/kaggriculture-frontier-the-moon-counts-melons.ipynb",
    "AgentBuilder": ROOT / "baseline/kaggriculture-agent-builder.ipynb",
}


ARCHIVE_FOLDERS = {
    "V22": "v22",
    "Adaptive_V14": "adaptive_v14",
    "V27": "v27",
    "Stable12": "stable12",
    "V022c": "v022c",
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


def _eval_route_expression(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        namespace = {
            "base64": base64,
            "gzip": gzip,
            "json": json,
            "zlib": zlib,
        }
        try:
            return eval(compile(ast.Expression(node), "<route-expression>", "eval"), namespace, namespace)
        except Exception:
            return None


def _route_assignments(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    result = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            pairs = [(target, node.value) for target in node.targets]
        elif isinstance(node, ast.AnnAssign):
            pairs = [(node.target, node.value)]
        else:
            pairs = []
        for target, value in pairs:
            if isinstance(target, ast.Name):
                result.append((target.id, value))
    return result


def _extract_route_from_source(source, path):
    for name, node in _route_assignments(source):
        if name == "AGENT_SOURCE":
            nested = _eval_route_expression(node)
            if isinstance(nested, str):
                route = _extract_route_from_source(nested, path)
                if route is not None:
                    return route
        elif name in {"_ACTIONS", "TRACE_ACTIONS"}:
            route = _eval_route_expression(node)
            if isinstance(route, list) and route and isinstance(route[0], dict):
                return route
    return None


def _load_notebook_route(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        sources = [text]
        if cell.get("cell_type") == "markdown":
            sources.extend(re.findall(r"```python\s*(.*?)```", text, re.S))
        for source in sources:
            route = _extract_route_from_source(source, path)
            if route is not None:
                return copy.deepcopy(route)
    raise RuntimeError(f"could not extract fixed route from {path}")


def _load_frontier_lab_routes():
    path = ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "AGENT_SOURCE" not in text:
            continue
        for name, node in _route_assignments(text):
            if name != "AGENT_SOURCE":
                continue
            source = _eval_route_expression(node)
            if not isinstance(source, str):
                continue
            packed = None
            for nested_name, nested_node in _route_assignments(source):
                if nested_name == "_PACKED":
                    packed = _eval_route_expression(nested_node)
                    break
            if not isinstance(packed, str):
                continue
            data = json.loads(zlib.decompress(base64.b85decode(packed)).decode("utf-8"))
            traces = data.get("traces")
            if isinstance(traces, list) and all(isinstance(route, list) for route in traces):
                return [copy.deepcopy(route) for route in traces]
    raise RuntimeError(f"could not extract Frontier Lab traces from {path}")


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
        archived = False
        if not path.exists():
            path = ROOT / "baseline/history/pure_routes" / ARCHIVE_FOLDERS[name] / "main.py"
            archived = True
        module = _load_module(path, name)
        if archived:
            route = module._ACTIONS
        elif variable == "__V023_STABLE12__":
            route = module._V023_ROUTES["stable_12hands"]["actions"]
        else:
            route = getattr(module, variable)
        routes[name] = copy.deepcopy(route)
        sources[name] = str(path)
    routes["V13_R3"], sources["V13_R3"] = _load_v13_route()
    routes["V13_R3"] = copy.deepcopy(routes["V13_R3"])
    for name, path in NOTEBOOK_ROUTES.items():
        routes[name] = _load_notebook_route(path)
        sources[name] = str(path)
    frontier_routes = _load_frontier_lab_routes()
    frontier_path = ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb"
    for index, route in enumerate(frontier_routes):
        name = f"FrontierLab_{index}"
        routes[name] = route
        sources[name] = f"{frontier_path}#trace-{index}"
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


def _init_worker(routes):
    global _WORKER_ROUTES
    _WORKER_ROUTES = routes


def _run_game_task(task):
    left, right, seed, left_seat = task
    return _run_game(_WORKER_ROUTES, left, right, seed, left_seat)


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
    parser = argparse.ArgumentParser(description="Run a pure fixed-route round robin.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of independent Kaggriculture worker processes.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

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
    tasks = [
        (left, right, seed, left_seat)
        for left, right in pairs
        for seed in SEEDS
        for left_seat in (0, 1)
    ]
    rows = []
    if args.workers == 1:
        _init_worker(routes)
        for index, task in enumerate(tasks, 1):
            rows.append(_run_game_task(task))
            if index == 1 or index % 25 == 0 or index == total:
                print(f"[{index}/{total}] completed", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(routes,),
        ) as executor:
            futures = [executor.submit(_run_game_task, task) for task in tasks]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                rows.append(future.result())
                if index == 1 or index % 25 == 0 or index == total:
                    print(f"[{index}/{total}] completed", flush=True)

    rows.sort(key=lambda row: (row["left"], row["right"], row["seed"], row["left_seat"]))

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
                "workers": args.workers,
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
