"""Structural tests for RL-010 mechanism x route cross-grafts.

These tests intentionally inspect only the frozen candidate artifacts.  They
do not use replay files at runtime and do not run a market overlay of their
own.  The game smoke test is kept separate so this file remains fast enough
to use before a large matrix run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from build_cross_graft_validation import (
    ARTIFACT_ROOT,
    MECHANISM_NOTEBOOKS,
    ROOT,
    load_archive_routes,
)


def _load(path, tag):
    spec = importlib.util.spec_from_file_location(f"cross_test_{tag}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_action(action, candidate, step):
    assert isinstance(action, dict), f"{candidate} step {step}: action is not dict"
    assert isinstance(action.get("farmer"), list), f"{candidate} step {step}: farmer"
    assert isinstance(action.get("hands"), list), f"{candidate} step {step}: hands"
    assert isinstance(action.get("market"), list), f"{candidate} step {step}: market"
    assert len(action["market"]) <= 10, f"{candidate} step {step}: >10 market orders"
    for order in action["market"]:
        assert isinstance(order, list), f"{candidate} step {step}: non-list order"
        if len(order) >= 3 and str(order[0]).upper() in {"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
            quantity = order[2]
            assert isinstance(quantity, int) and quantity >= 0, (
                f"{candidate} step {step}: invalid quantity {order!r}"
            )


def check_artifacts(artifact_root=ARTIFACT_ROOT):
    artifact_root = Path(artifact_root)
    routes, route_manifest = load_archive_routes()
    manifest_path = artifact_root / "manifest.json"
    assert manifest_path.exists(), f"missing {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates", {})
    expected = len(MECHANISM_NOTEBOOKS) * len(routes)
    assert len(candidates) == expected, (len(candidates), expected)

    checked = 0
    for candidate_name, metadata in sorted(candidates.items()):
        main_path = artifact_root / candidate_name / "main.py"
        candidate_manifest = json.loads(
            (artifact_root / candidate_name / "manifest.json").read_text(encoding="utf-8")
        )
        assert main_path.exists(), main_path
        assert candidate_manifest["candidate"] == candidate_name
        route_name = candidate_manifest["route"]
        mechanism_name = candidate_manifest["mechanism"]
        assert route_name in route_manifest
        assert mechanism_name in MECHANISM_NOTEBOOKS
        assert candidate_manifest["route_length"] == 719
        assert candidate_manifest["root_main_modified"] is False

        module = _load(main_path, candidate_name)
        route = getattr(module, "_CROSS_GRAFT_ROUTE", None)
        assert isinstance(route, list), f"{candidate_name}: missing route payload"
        assert len(route) == 719, f"{candidate_name}: route length {len(route)}"
        assert candidate_manifest["route_sha256"] == route_manifest[route_name]["route_sha256"]

        aliases = candidate_manifest["route_aliases"]
        assert aliases, f"{candidate_name}: no aliases"
        for alias in aliases:
            value = getattr(module, alias, None)
            assert value == route, f"{candidate_name}: alias {alias} not grafted"
        for step, action in enumerate(route):
            _check_action(action, candidate_name, step)
        checked += 1

    # The mechanism and route identity must remain independently auditable.
    # In particular, changing the route must not change the mechanism hash.
    for mechanism_name in MECHANISM_NOTEBOOKS:
        rows = [
            row for row in candidates.values() if row["mechanism"] == mechanism_name
        ]
        assert len({row["mechanism_sha256"] for row in rows}) == 1
        assert len({row["route_sha256"] for row in rows}) == len(routes)
    return {"candidates": checked, "routes": sorted(routes), "mechanisms": sorted(MECHANISM_NOTEBOOKS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    print(json.dumps(check_artifacts(args.artifact_root), indent=2))
