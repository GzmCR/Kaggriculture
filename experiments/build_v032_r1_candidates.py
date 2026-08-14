"""Build self-contained V032-R1 route candidates.

The generated candidates contain only a frozen route, anonymous profiles,
the V032 matching layer, the shared market rollout, and optional calibration
coefficients.  No replay/notebook files are read at runtime.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import tarfile
import zlib
from pathlib import Path

from build_v032_route_candidates import (
    ARTIFACT_ROOT as OLD_ARTIFACT_ROOT,
    EIGHT_COW_SOURCE,
    V27_SOURCE,
    _assignment,
    _normalize_route,
    _payload,
    _route_hash,
    _literal_or_eval,
    extract_8c4s_route,
)


ROOT = Path(__file__).resolve().parents[1]
OVERLAY_V032 = ROOT / "experiments/v032_route_conditioned_timing.py"
ROLLOUT = ROOT / "experiments/v032_market_rollout.py"
OVERLAY_R1 = ROOT / "experiments/v032_r1_overlay.py"
ARTIFACT_ROOT = ROOT / "baseline/artifacts/v032_route_conditioned_timing_r1"
HISTORY_ROOT = ROOT / "baseline/history/v032_route_conditioned_timing_r1"


def load_route(name):
    if name == "v27":
        namespace = {"__name__": "v032_r1_v27_route", "__file__": str(V27_SOURCE)}
        exec(compile(V27_SOURCE.read_text(encoding="utf-8"), str(V27_SOURCE), "exec"), namespace, namespace)
        return _normalize_route(namespace["_ACTIONS"]), str(V27_SOURCE.relative_to(ROOT))
    if name == "8c4s":
        return extract_8c4s_route(), str(EIGHT_COW_SOURCE.relative_to(ROOT))
    raise ValueError(name)


def _archive(main, target):
    with tarfile.open(target, "w:gz") as handle:
        handle.add(main, arcname="main.py")


def _json_blob(value):
    return base64.b85encode(zlib.compress(json.dumps(value, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")


def build_candidate(route_name, route, profiles, calibration, timing_enabled, output_root, history_root):
    source = V27_SOURCE.read_text(encoding="utf-8")
    source += "\n# V032-R1 route/profile payload.\n"
    source += "import base64 as _v032_r1_base64\nimport json as _v032_r1_json\nimport zlib as _v032_r1_zlib\n"
    source += f"_ACTIONS = _v032_r1_json.loads(_v032_r1_zlib.decompress(_v032_r1_base64.b85decode({_json_blob(route)!r})).decode('utf-8'))\n"
    source += "_LEGACY_ACTIONS = _ACTIONS\n_REBALANCE_ACTIONS = _ACTIONS\n"
    source += f"V032_PROFILES = _v032_r1_json.loads(_v032_r1_zlib.decompress(_v032_r1_base64.b85decode({_json_blob(profiles)!r})).decode('utf-8'))\n"
    source += f"V032_R1_CALIBRATION = _v032_r1_json.loads(_v032_r1_zlib.decompress(_v032_r1_base64.b85decode({_json_blob(calibration)!r})).decode('utf-8'))\n"
    source += f"V032_DISABLE_TIMING = {True!r}\nV032_R1_DISABLE_TIMING = {not timing_enabled!r}\n"
    source += "\n" + OVERLAY_V032.read_text(encoding="utf-8")
    source += "\n" + ROLLOUT.read_text(encoding="utf-8")
    source += "\n" + OVERLAY_R1.read_text(encoding="utf-8")
    # The overlay declares safe defaults.  Re-apply the build-time values
    # after the source is concatenated so the defaults cannot shadow payloads.
    source += f"\nV032_R1_CALIBRATION = _v032_r1_json.loads(_v032_r1_zlib.decompress(_v032_r1_base64.b85decode({_json_blob(calibration)!r})).decode('utf-8'))\n"
    source += f"V032_R1_DISABLE_TIMING = {not timing_enabled!r}\n"
    route_sha = _route_hash(route)
    mechanism_sha = hashlib.sha256((OVERLAY_V032.read_text() + ROLLOUT.read_text() + OVERLAY_R1.read_text()).encode()).hexdigest()
    name = f"v032_r1_{route_name}_{'timing' if timing_enabled else 'order_only'}"
    manifest = {
        "candidate": name,
        "route_name": route_name,
        "route_sha256": route_sha,
        "route_length": len(route),
        "route_source": load_route(route_name)[1],
        "mechanism_sha256": mechanism_sha,
        "calibration_embedded": bool(calibration),
        "timing_enabled": bool(timing_enabled),
        "engine": "kaggle-environments==1.32.6",
        "runtime_inputs": ["observation", "embedded_anonymous_profiles", "embedded_calibration"],
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


def build(profiles_path=None, calibration_path=None, include_8c4s=True):
    profiles = json.loads(Path(profiles_path).read_text()) if profiles_path and Path(profiles_path).exists() else []
    calibration = json.loads(Path(calibration_path).read_text()) if calibration_path and Path(calibration_path).exists() else {}
    manifests = {}
    for route_name in ["v27"] + (["8c4s"] if include_8c4s else []):
        route, _ = load_route(route_name)
        manifests[route_name] = {
            "order_only": build_candidate(route_name, route, [], {}, False, ARTIFACT_ROOT, HISTORY_ROOT),
            "timing": build_candidate(route_name, route, profiles, calibration, True, ARTIFACT_ROOT, HISTORY_ROOT),
        }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "manifest.json").write_text(json.dumps({"routes": manifests, "calibration": str(calibration_path or "none")}, indent=2) + "\n")
    (ARTIFACT_ROOT / "README.md").write_text(
        "# V032-R1\n\n"
        "R1 replaces the old two-event heuristic with a shared official-price lockstep rollout, "
        "two future same-item events, simultaneous margin estimation, and optional residual calibration.\n\n"
        "An empty calibration payload intentionally makes the timing candidate fall back to order-only. "
        "Generate calibration from local notebook opponents before treating timing results as a candidate.\n",
        encoding="utf-8",
    )
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, default=ROOT / "baseline/artifacts/v032_route_conditioned_timing/profiles/profiles.json")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--no-8c4s", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.profiles, args.calibration, not args.no_8c4s), indent=2))
