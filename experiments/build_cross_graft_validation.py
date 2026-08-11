"""Build complete-mechanism x external-route validation agents.

The mechanism source is kept intact.  Only its global frozen action table is
overridden with a serialized route payload before the source agent is called.
No shared market overlay is injected.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import gzip
import hashlib
import json
import tarfile
import time
import zlib
from pathlib import Path

from top10_opponents import decode_notebook_source


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "baseline/artifacts/rl_010_milk_bidirectional/cross_graft"
HISTORY_ROOT = ROOT / "baseline/history/rl_010_cross_graft"

MECHANISM_NOTEBOOKS = {
    "v27": ROOT / "baseline/2026-08-09/25-27-strict-future-v27-midgame-meta-reset.ipynb",
    "adaptive": ROOT / "baseline/2026-08-09/kaggriculture-adaptive-replay-agent.ipynb",
    "v13_r3": ROOT / "baseline/v13-r3-top-meta-order-safe-premium-control.ipynb",
    "v21_1": ROOT / "baseline/177-180-fresh-top-30-v21-1-conditional-memory.ipynb",
    "hamburger": ROOT / "baseline/kaggriculture-hamburger.ipynb",
    "frontier": ROOT / "baseline/2026-08-09/kaggriculture-frontier-the-soil-remembers-rain.ipynb",
}

ROUTE_FILES = {
    "v22": (ROOT / "baseline/history/pure_routes/v22/main.py", "_ACTIONS"),
    "v27": (ROOT / "baseline/history/pure_routes/v27/main.py", "_ACTIONS"),
    "adaptive_v14": (ROOT / "baseline/history/pure_routes/adaptive_v14/main.py", "_ACTIONS"),
    "v13_r3": (ROOT / "baseline/history/pure_routes/v13_r3/main.py", "_ACTIONS"),
    "stable12": (ROOT / "baseline/history/pure_routes/stable12/main.py", "_ACTIONS"),
    "v022c": (ROOT / "baseline/history/pure_routes/v022c/main.py", "_ACTIONS"),
}


def _manifest_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _clean_future_import(source):
    return source.replace("from __future__ import annotations\n", "", 1)


def _decode_hamburger(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "ANCHOR_BLOB" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "ANCHOR_BLOB"
                for target in node.targets
            ):
                continue
            blob = ast.literal_eval(node.value)
            return gzip.decompress(base64.b64decode(blob)).decode("utf-8")
    raise ValueError(f"missing Hamburger ANCHOR_BLOB in {path}")


def decode_mechanism(name):
    path = MECHANISM_NOTEBOOKS[name]
    if name == "hamburger":
        source = _decode_hamburger(path)
        encoding = "anchor_blob"
    else:
        source, encoding = decode_notebook_source(path)
    return _clean_future_import(source), {
        "name": name,
        "source_path": str(path.relative_to(ROOT)),
        "encoding": encoding,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_bytes": len(source.encode("utf-8")),
    }


def _load_route_module(path):
    namespace = {"__name__": f"cross_route_{time.time_ns()}", "__file__": str(path)}
    source = Path(path).read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def load_archive_routes():
    routes = {}
    metadata = {}
    for name, (path, variable) in ROUTE_FILES.items():
        namespace = _load_route_module(path)
        route = namespace.get(variable)
        if not isinstance(route, list):
            raise ValueError(f"{name} route variable {variable} is not a list")
        route = _normalize_route(route)
        routes[name] = route
        metadata[name] = _route_metadata(name, route, str(path.relative_to(ROOT)), "archive")
    return routes, metadata


def _normalize_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(value or ["PASS"]) for value in action.get("hands", []) or []],
        "market": [list(value) for value in action.get("market", []) or [] if isinstance(value, (list, tuple))],
    }


def _normalize_route(route):
    route = [_normalize_action(value) for value in list(route or [])]
    if len(route) < 719:
        route.extend([{ "farmer": ["PASS"], "hands": [], "market": [] } for _ in range(719 - len(route))])
    return route[:719]


def _route_metadata(name, route, source, source_kind, extra=None):
    counts = {}
    max_hands = 0
    for action in route:
        max_hands = max(max_hands, len(action.get("hands", []) or []))
        for unit_action in [action.get("farmer", []), *(action.get("hands", []) or [])]:
            if unit_action:
                op = str(unit_action[0]).upper()
                counts[op] = counts.get(op, 0) + 1
        for order in action.get("market", []) or []:
            if len(order) >= 3 and str(order[0]).upper() == "SELL":
                key = f"SELL_{str(order[1]).upper()}"
                counts[key] = counts.get(key, 0) + max(0, int(order[2]))
    payload = json.dumps(route, sort_keys=True, separators=(",", ":"))
    result = {
        "name": name,
        "source": source,
        "source_kind": source_kind,
        "route_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "length": len(route),
        "max_hands": max_hands,
        "action_counts": counts,
    }
    if extra:
        result.update(extra)
    return result


def load_replay_route(name, replay_path, seat):
    replay_path = Path(replay_path).resolve()
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    raw = []
    for step in range(min(719, max(0, len(steps) - 1))):
        state = steps[step + 1]
        if not isinstance(state, list) or int(seat) >= len(state):
            raw.append({"farmer": ["PASS"], "hands": [], "market": []})
            continue
        raw.append(_normalize_action(state[int(seat)].get("action", {})))
    route = _normalize_route(raw)
    return route, _route_metadata(
        name,
        route,
        str(replay_path),
        "replay",
        {"replay_seat": int(seat), "episode_id": payload.get("id", payload.get("episodeId", ""))},
    )


def _route_payload(route):
    encoded = base64.b85encode(
        zlib.compress(json.dumps(route, separators=(",", ":")).encode("utf-8"), 9)
    ).decode("ascii")
    return encoded


def _route_aliases(mechanism_name, source):
    names = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            upper = node.id.upper()
            if node.id in {"_ACTIONS", "TRACE_ACTIONS", "_LEGACY_ACTIONS", "_REBALANCE_ACTIONS"}:
                names.add(node.id)
            elif "ACTION" in upper and ("TRACE" in upper or upper.endswith("_ACTIONS")):
                names.add(node.id)
    if mechanism_name == "v27":
        names.update({"_LEGACY_ACTIONS", "_REBALANCE_ACTIONS"})
    if "TRACE_ACTIONS" in source:
        names.add("TRACE_ACTIONS")
    if "_ACTIONS" in source:
        names.add("_ACTIONS")
    return sorted(names)


def build_candidate(mechanism_name, mechanism_source, route_name, route, route_meta):
    aliases = _route_aliases(mechanism_name, mechanism_source)
    if not aliases:
        raise ValueError(f"no route aliases found for mechanism {mechanism_name}")
    encoded = _route_payload(route)
    assignments = "\n".join(
        f"{alias} = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)"
        for alias in aliases
    )
    suffix = f'''

# Cross-graft: preserve mechanism {mechanism_name}, replace only frozen route {route_name}.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode({encoded!r})
).decode("utf-8"))
{assignments}
_CROSS_GRAFT_MECHANISM = {mechanism_name!r}
_CROSS_GRAFT_ROUTE_NAME = {route_name!r}
_CROSS_GRAFT_ROUTE_SHA256 = {route_meta['route_sha256']!r}
'''
    return mechanism_source.rstrip() + suffix, aliases


def _archive(main_path, archive_path):
    with tarfile.open(archive_path, "w:gz") as handle:
        handle.add(main_path, arcname="main.py")


def build(artifact_root=ARTIFACT_ROOT, history_root=HISTORY_ROOT, replay_specs=None):
    artifact_root = Path(artifact_root).resolve()
    history_root = Path(history_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)

    routes, route_manifest = load_archive_routes()
    replay_manifest = []
    for spec in replay_specs or []:
        name, path, seat = spec
        route, metadata = load_replay_route(name, path, seat)
        routes[name] = route
        route_manifest[name] = metadata
        replay_manifest.append(metadata)

    mechanisms = {}
    for name in MECHANISM_NOTEBOOKS:
        source, metadata = decode_mechanism(name)
        mechanisms[name] = source
        (artifact_root / "mechanisms").mkdir(parents=True, exist_ok=True)

    candidates = {}
    for mechanism_name, mechanism_source in mechanisms.items():
        source_sha = hashlib.sha256(mechanism_source.encode("utf-8")).hexdigest()
        for route_name, route in routes.items():
            candidate_name = f"{mechanism_name}_x_{route_name}"
            source, aliases = build_candidate(
                mechanism_name,
                mechanism_source,
                route_name,
                route,
                route_manifest[route_name],
            )
            target = artifact_root / candidate_name
            history_target = history_root / candidate_name
            target.mkdir(parents=True, exist_ok=True)
            history_target.mkdir(parents=True, exist_ok=True)
            main_path = target / "main.py"
            main_path.write_text(source, encoding="utf-8")
            (history_target / "main.py").write_text(source, encoding="utf-8")
            _archive(main_path, target / "submission.tar.gz")
            manifest = {
                "candidate": candidate_name,
                "mechanism": mechanism_name,
                "route": route_name,
                "mechanism_sha256": source_sha,
                "route_sha256": route_manifest[route_name]["route_sha256"],
                "route_length": route_manifest[route_name]["length"],
                "route_aliases": aliases,
                "main_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "main_bytes": len(source.encode("utf-8")),
                "archive": _manifest_path(target / "submission.tar.gz"),
                "root_main_modified": False,
            }
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            candidates[candidate_name] = manifest

    manifest = {
        "mode": "full_mechanism_external_route_cross_graft",
        "mechanisms": {
            name: {
                "source": str(MECHANISM_NOTEBOOKS[name].relative_to(ROOT)),
                "source_sha256": hashlib.sha256(decode_mechanism(name)[0].encode("utf-8")).hexdigest(),
            }
            for name in MECHANISM_NOTEBOOKS
        },
        "routes": route_manifest,
        "replay_routes": replay_manifest,
        "candidates": candidates,
        "root_main_modified": False,
    }
    (artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = """# RL-010 cross-graft validation

Each candidate preserves the complete source mechanism from one notebook and
replaces only its frozen route table with an independently archived route.
The builder does not append the V031 shared market overlay.

Core matrix: six mechanisms x six archived routes. Optional replay routes are
added with `--replay name=path:seat` and are kept for validation only.
"""
    (artifact_root / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def _parse_replay(value):
    name, rest = value.split("=", 1)
    path, seat = rest.rsplit(":", 1)
    return name, Path(path), int(seat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--replay", action="append", default=[], help="NAME=REPLAY_JSON:SEAT")
    args = parser.parse_args()
    manifest = build(
        args.artifact_root,
        args.history_root,
        [_parse_replay(value) for value in args.replay],
    )
    print(json.dumps({"candidates": len(manifest["candidates"]), "routes": list(manifest["routes"])}, indent=2))
