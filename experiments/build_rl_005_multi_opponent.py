"""Build RL-005 from paired data collected against the 2026-08-09 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from build_v022e_adaptive_recovery import ROOT, _decode_notebook_agent


SELECTOR = ROOT / "experiments/rl_004_trade_timing.py"
DEFAULT_SAMPLES = ROOT / "baseline/artifacts/rl_005_multi_opponent/data_train/samples.jsonl"
OUT_HISTORY = ROOT / "baseline/history/rl_005_multi_opponent"
OUT_ARTIFACT = ROOT / "baseline/artifacts/rl_005_multi_opponent"


def build_source(payload):
    base_source, source_sha = _decode_notebook_agent(
        ROOT / "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb"
    )
    marker = "\ndef agent(obs):"
    if base_source.count(marker) != 1:
        raise ValueError("expected one v22 agent definition")
    base_source = base_source.replace(marker, "\ndef _rl005_v22_agent(obs):", 1)
    selector = SELECTOR.read_text(encoding="utf-8")
    weights = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    wrapper = f'''\n\n# RL-005: multi-opponent, observation-aware timing selector.\nRL005_PAYLOAD = {weights}\n_RL005_OPPORTUNITIES = rl004_route_opportunities(_ACTIONS)\n_RL005_RUNTIME = RL004Runtime(payload=RL005_PAYLOAD, opportunities=_RL005_OPPORTUNITIES)\n\ndef agent(obs, config=None):\n    """Public entry point; V22 owns every farmer/hand action."""\n    try:\n        base = _rl005_v22_agent(obs)\n        return _RL005_RUNTIME.act(obs, base)\n    except Exception:\n        _RL005_RUNTIME.errors += 1\n        return rl004_align_hands(_rl005_v22_agent(obs), obs)\n'''
    return selector + "\n\n" + base_source.rstrip() + wrapper, source_sha


def build(samples_path, output=OUT_ARTIFACT):
    from rl_004_trade_timing import rl004_fit_models, rl004_load_samples

    samples_path = Path(samples_path).resolve()
    samples = rl004_load_samples(samples_path)
    payload, report = rl004_fit_models(samples)
    source, source_sha = build_source(payload)
    output.mkdir(parents=True, exist_ok=True)
    OUT_HISTORY.mkdir(parents=True, exist_ok=True)
    history_main = OUT_HISTORY / "main.py"
    artifact_main = output / "main.py"
    data = source.encode("utf-8")
    history_main.write_bytes(data)
    artifact_main.write_bytes(data)
    (output / "weights.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    archive_path = output / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(artifact_main, arcname="main.py")
    manifest = {
        "name": "rl_005_multi_opponent",
        "status": "multi_opponent_observation_aware_event_delay",
        "control": "v22_price_impact_route",
        "feature_dim": payload["feature_dim"],
        "min_support": payload["min_support"],
        "min_expected_delta": payload["min_expected_delta"],
        "lcb_z": payload["lcb_z"],
        "models": len(payload["models"]),
        "training_samples": len(samples),
        "training_data": str(samples_path.relative_to(ROOT)),
        "source_notebook": "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb",
        "source_sha256": source_sha,
        "main_sha256": hashlib.sha256(data).hexdigest(),
        "main_bytes": len(data),
        "archive": str(archive_path.relative_to(ROOT)),
        "root_main_modified": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    args = parser.parse_args()
    manifest, report = build(args.samples)
    print(json.dumps({"manifest": manifest, "fit": report}, indent=2))
