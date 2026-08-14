"""Build V032 V27 and optional 8C4S route candidates.

Routes are extracted offline, compressed into each candidate, and combined
with the V032 runtime overlay. The resulting submission does not import this
builder or any notebook/replay at runtime.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import tarfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V27_SOURCE = ROOT / "baseline/history/v031_route_market_combo/v27_order_only/main.py"
EIGHT_COW_SOURCE = ROOT / "baseline/2026-08-09/v16-rc5-high-score-8c-4s-premium-market-lead.ipynb"
OVERLAY = ROOT / "experiments/v032_route_conditioned_timing.py"
ARTIFACT_ROOT = ROOT / "baseline/artifacts/v032_route_conditioned_timing"
HISTORY_ROOT = ROOT / "baseline/history/v032_route_conditioned_timing"


def _normalize_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(x or ["PASS"]) for x in action.get("hands", []) or []],
        "market": [list(x) for x in action.get("market", []) or [] if isinstance(x, (list, tuple))],
    }


def _normalize_route(route):
    route = [_normalize_action(x) for x in list(route or [])]
    if len(route) < 719:
        route.extend({"farmer": ["PASS"], "hands": [], "market": []} for _ in range(719 - len(route)))
    return route[:719]


def _literal_or_eval(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        namespace = {"base64": base64, "json": json, "zlib": zlib}
        try:
            return eval(compile(ast.Expression(node), "<route>", "eval"), namespace, namespace)
        except Exception:
            return None


def _assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            pairs = [(target, node.value) for target in node.targets]
        elif isinstance(node, ast.AnnAssign):
            pairs = [(node.target, node.value)]
        else:
            pairs = []
        for target, value in pairs:
            if isinstance(target, ast.Name) and target.id == name:
                return _literal_or_eval(value)
    return None


def extract_8c4s_route(path=EIGHT_COW_SOURCE):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "_ACTIONS" not in text:
            continue
        text = re.sub(r"^%%writefile[^\n]*\n", "", text)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        route = _assignment(tree, "_ACTIONS") or _assignment(tree, "TRACE_ACTIONS")
        if isinstance(route, list) and route and isinstance(route[0], dict):
            return _normalize_route(route)
    raise RuntimeError(f"unable to extract 8C4S route from {path}")


def load_route(name):
    if name == "v27":
        namespace = {"__name__": "v032_v27_route", "__file__": str(V27_SOURCE)}
        exec(compile(V27_SOURCE.read_text(encoding="utf-8"), str(V27_SOURCE), "exec"), namespace, namespace)
        return _normalize_route(namespace["_ACTIONS"]), str(V27_SOURCE.relative_to(ROOT))
    if name == "8c4s":
        return extract_8c4s_route(), str(EIGHT_COW_SOURCE.relative_to(ROOT))
    raise ValueError(name)


def _payload(route):
    return base64.b85encode(zlib.compress(json.dumps(route, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")


def _route_hash(route):
    return hashlib.sha256(json.dumps(route, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _archive(path, archive):
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(path, arcname="main.py")


def build_candidate(route_name, route, profiles, output_root, history_root, timing_enabled=True):
    source = V27_SOURCE.read_text(encoding="utf-8")
    route_blob = _payload(route)
    profile_blob = base64.b85encode(zlib.compress(json.dumps(profiles, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
    injection = f'''\n# V032 offline route/profile payload.\nimport base64 as _v032_base64\nimport json as _v032_json\nimport zlib as _v032_zlib\n_ACTIONS = _v032_json.loads(_v032_zlib.decompress(_v032_base64.b85decode({route_blob!r})).decode("utf-8"))\n_LEGACY_ACTIONS = _ACTIONS\n_REBALANCE_ACTIONS = _ACTIONS\nV032_PROFILES = _v032_json.loads(_v032_zlib.decompress(_v032_base64.b85decode({profile_blob!r})).decode("utf-8"))\nV032_DISABLE_TIMING = {not timing_enabled!r}\n'''
    source += injection + "\n" + OVERLAY.read_text(encoding="utf-8")
    source += f"\nV032_ROUTE_NAME = {route_name!r}\nV032_ROUTE_SHA256 = {_route_hash(route)!r}\n"
    suffix = "timing" if timing_enabled else "order_only"
    name = f"v032_{route_name}_{suffix}"
    manifest = {
        "candidate": name,
        "route_name": route_name,
        "route_sha256": _route_hash(route),
        "route_length": len(route),
        "route_source": str(load_route(route_name)[1]),
        "profile_count": len(profiles),
        "timing_enabled": bool(timing_enabled),
        "engine": "kaggle-environments==1.32.6",
        "runtime_inputs": ["observation", "embedded_anonymous_profiles"],
        "runtime_forbidden_inputs": ["replay", "notebook", "TeamName", "score", "seed", "network"],
        "root_main_modified": False,
    }
    for root in (Path(output_root) / name, Path(history_root) / name):
        root.mkdir(parents=True, exist_ok=True)
        main = root / "main.py"
        main.write_text(source, encoding="utf-8")
        _archive(main, root / "submission.tar.gz")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build(profiles_path=None, output_root=ARTIFACT_ROOT, history_root=HISTORY_ROOT, include_8c4s=True):
    profiles = []
    if profiles_path and Path(profiles_path).exists():
        profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    manifests = {}
    for route_name in ["v27"] + (["8c4s"] if include_8c4s else []):
        route, _ = load_route(route_name)
        route_manifests = {}
        # The order-only artifact is the necessary route control.  It embeds
        # no profiles and disables the timing overlay, while still using the
        # same V27 price-impact reorder and legality layer as the timing arm.
        route_manifests["order_only"] = build_candidate(
            route_name, route, [], output_root, history_root, timing_enabled=False
        )
        route_manifests["timing"] = build_candidate(
            route_name, route, profiles, output_root, history_root, timing_enabled=True
        )
        manifests[route_name] = route_manifests
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"routes": manifests, "profile_source": str(profiles_path or "none")}, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# V032 route-conditioned timing\n\n"
        "V032 uses V27 order-only as the default route. The optional 8C4S candidate "
        "is extracted offline from the latest high-score notebook and is not selected automatically.\n\n"
        "Runtime order: timing transfer -> V27 price-impact SELL reorder -> legality/hand alignment. "
        "Unknown or weakly matched opponents are a strict V27 order-only fallback.\n\n"
        "Profiles are anonymous offline data. The submission does not read replay files, notebooks, "
        "names, scores, seeds or the network.\n",
        encoding="utf-8",
    )
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument("--no-8c4s", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.profiles, include_8c4s=not args.no_8c4s), indent=2))
