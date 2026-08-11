"""Fit and package RL-010 for the frozen V27 order-only route."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from rl_010_milk_bidirectional import rl010_fit_models, rl010_load_samples


ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "baseline/history/v031_route_market_combo/v27_order_only/main.py"
DEFAULT_SAMPLES = ROOT / "baseline/artifacts/rl_010_milk_bidirectional/data_train/samples.jsonl"
OUT_HISTORY = ROOT / "baseline/history/rl_010_milk_bidirectional"
OUT_ARTIFACT = ROOT / "baseline/artifacts/rl_010_milk_bidirectional"


def build_source(payload):
    source = BASE_SOURCE.read_text(encoding="utf-8")
    # This is embedded as Python source, not JSON.  ``json.dumps`` would emit
    # lowercase ``true``/``false`` and make the generated main.py invalid.
    payload_text = repr(payload)
    overlay = f'''

# RL-010: MILK timing overlay.  The route and WEED layer are prepared first,
# then the RL quantity transfer is applied, and only then V27's existing
# price-impact order ranking is run.
RL010_PAYLOAD = {payload_text}
_RL010_OPPORTUNITIES = rl010_route_opportunities(_ACTIONS)
_RL010_RUNTIME = RL010Runtime(payload=RL010_PAYLOAD, opportunities=_RL010_OPPORTUNITIES)

def _rl010_route_action(obs, config=None):
    step = _v031_step(obs)
    action = _v031_copy_action(_ACTIONS[step])
    action = _v031_weed_action(obs, action, step)
    return _v031_align_hands(action, obs)

def agent(obs, config=None):
    try:
        route_action = _rl010_route_action(obs, config)
        adjusted_action = _RL010_RUNTIME.act(obs, route_action)
        final_action = _v031_reorder_existing(obs, adjusted_action)
        final_action = _v031_align_hands(final_action, obs)
        _RL010_RUNTIME.record_final_action(final_action)
        return final_action
    except Exception:
        _RL010_RUNTIME.errors += 1
        return _v031_agent(obs, config)
'''
    from rl_010_milk_bidirectional import __file__ as core_path
    core = Path(core_path).read_text(encoding="utf-8")
    return core + "\n\n" + source.rstrip() + overlay


VARIANTS = {
    "rl010a_delay_only": {
        "allowed_actions": ["DELAY_25", "DELAY_50"],
        "include_opponent_features": True,
    },
    "rl010b_bidirectional_no_opp": {
        "allowed_actions": ["ADVANCE_25", "ADVANCE_50", "DELAY_25", "DELAY_50"],
        "include_opponent_features": False,
    },
    "rl010c_bidirectional_opp": {
        "allowed_actions": ["ADVANCE_25", "ADVANCE_50", "DELAY_25", "DELAY_50"],
        "include_opponent_features": True,
    },
}


def build(samples_path=DEFAULT_SAMPLES, output=OUT_ARTIFACT, variant="rl010c_bidirectional_opp"):
    samples_path = Path(samples_path).resolve()
    samples = rl010_load_samples(samples_path)
    if variant not in VARIANTS:
        raise ValueError(f"unknown RL-010 variant: {variant}")
    settings = VARIANTS[variant]
    payload, report = rl010_fit_models(samples, **settings)
    payload["variant"] = variant
    source = build_source(payload)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history_dir = OUT_HISTORY / variant
    history_dir.mkdir(parents=True, exist_ok=True)
    data = source.encode("utf-8")
    (history_dir / "main.py").write_bytes(data)
    (output / "main.py").write_bytes(data)
    (output / "weights.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    archive_path = output / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output / "main.py", arcname="main.py")
    manifest = {
        "name": variant,
        "status": (
            "fitted_v27_order_only_bidirectional_milk"
            if payload.get("models")
            else "smoke_control_only_no_supported_models"
        ),
        "control": "v27_order_only",
        "feature_dim": payload["feature_dim"],
        "actions": ["ADVANCE_25", "ADVANCE_50", "CONTROL", "DELAY_25", "DELAY_50"],
        "allowed_actions": settings["allowed_actions"],
        "include_opponent_features": settings["include_opponent_features"],
        "min_support": payload["min_support"],
        "min_expected_delta": payload["min_expected_delta"],
        "training_samples": len(samples),
        "training_data": str(samples_path.relative_to(ROOT)),
        "route_source": str(BASE_SOURCE.relative_to(ROOT)),
        "route_sha256": hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest(),
        "main_sha256": hashlib.sha256(data).hexdigest(),
        "main_bytes": len(data),
        "archive": str(archive_path.relative_to(ROOT)),
        "root_main_modified": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, report


def build_all(samples_path=DEFAULT_SAMPLES, output_root=OUT_ARTIFACT):
    output_root = Path(output_root).resolve()
    results = {}
    for variant in VARIANTS:
        results[variant] = build(samples_path, output_root / variant, variant)[0]
    # Keep the full opponent-feature version at the historical root location
    # so older runners continue to find ``rl010``.
    root_manifest, root_report = build(samples_path, output_root, "rl010c_bidirectional_opp")
    OUT_HISTORY.mkdir(parents=True, exist_ok=True)
    (OUT_HISTORY / "main.py").write_bytes((output_root / "main.py").read_bytes())
    results["primary"] = root_manifest
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=OUT_ARTIFACT)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="rl010c_bidirectional_opp")
    parser.add_argument("--all-variants", action="store_true")
    args = parser.parse_args()
    if args.all_variants:
        print(json.dumps(build_all(args.samples, args.output), indent=2))
    else:
        manifest, report = build(args.samples, args.output, args.variant)
        print(json.dumps({"manifest": manifest, "fit": report}, indent=2))
