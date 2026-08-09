"""Build self-contained RL-008 small-quantity timing candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tarfile
from pathlib import Path

from build_v022e_adaptive_recovery import ROOT, _decode_notebook_agent


SELECTORS = (
    ROOT / "experiments/rl_006_bidirectional_timing.py",
    ROOT / "experiments/rl_008_small_shift_timing.py",
)
DEFAULT_SAMPLES = ROOT / "baseline/artifacts/rl_008_small_shift_timing/data_train/samples.jsonl"
OUT_HISTORY = ROOT / "baseline/history/rl_008_small_shift_timing"
OUT_ARTIFACT = ROOT / "baseline/artifacts/rl_008_small_shift_timing"
V22_SOURCE = ROOT / "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb"


def build_source(payload):
    base_source, source_sha = _decode_notebook_agent(V22_SOURCE)
    marker = "\ndef agent(obs):"
    if base_source.count(marker) != 1:
        raise ValueError("expected one V22 agent definition")
    base_source = base_source.replace(marker, "\ndef _rl008_v22_agent(obs):", 1)
    selector_parts = [SELECTORS[0].read_text(encoding="utf-8")]
    selector_parts.append(
        SELECTORS[1].read_text(encoding="utf-8").replace(
            "from __future__ import annotations\n", "", 1
        )
    )
    selector = "\n\n".join(selector_parts)
    payload_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    wrapper = f'''

# RL-008: small-quantity bidirectional premium timing selector.
RL008_PAYLOAD = {payload_text}
_RL008_OPPORTUNITIES = rl008_preempt_opportunities(_ACTIONS) + rl008_delay_opportunities(_ACTIONS)
_RL008_RUNTIME = RL008Runtime(payload=RL008_PAYLOAD, opportunities=_RL008_OPPORTUNITIES)

def agent(obs, config=None):
    """Public entry point; V022 owns every farmer/hand action."""
    try:
        base = _rl008_v22_agent(obs)
        return _RL008_RUNTIME.act(obs, base)
    except Exception as exc:
        _RL008_RUNTIME.errors += 1
        _RL008_RUNTIME.last_error = f'{{type(exc).__name__}}: {{exc}}'
        return rl006_align_hands(_rl008_v22_agent(obs), obs)
'''
    return selector + "\n\n" + base_source.rstrip() + wrapper, source_sha


def _write_archive(main_path, archive_path):
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")


def build(samples_path=DEFAULT_SAMPLES, output=OUT_ARTIFACT):
    from rl_008_small_shift_timing import rl008_fit_models, rl008_load_samples

    samples_path = Path(samples_path).resolve()
    samples = rl008_load_samples(samples_path)
    payload, report = rl008_fit_models(samples)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    OUT_HISTORY.mkdir(parents=True, exist_ok=True)

    variants = {
        "gated_preempt": (tuple(range(1, 7)), "gated"),
        "ungated_preempt": (tuple(range(1, 7)), "ungated"),
        "gated_bidirectional": (tuple(range(1, 10)), "gated"),
        "ungated_bidirectional": (tuple(range(1, 10)), "ungated"),
    }
    manifests = {}
    for variant, (allowed_actions, gate_mode) in variants.items():
        variant_payload = copy.deepcopy(payload)
        variant_payload["allowed_actions"] = list(allowed_actions)
        variant_payload["gate_mode"] = gate_mode
        source, source_sha = build_source(variant_payload)
        data = source.encode("utf-8")
        variant_dir = output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        main_path = variant_dir / "main.py"
        main_path.write_bytes(data)
        archive_path = variant_dir / "submission.tar.gz"
        _write_archive(main_path, archive_path)
        history_dir = OUT_HISTORY / variant
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / "main.py").write_bytes(data)
        (history_dir / "submission.tar.gz").write_bytes(archive_path.read_bytes())
        manifest = {
            "name": f"rl_008_{variant}",
            "status": "small_quantity_bidirectional_timing_contextual_bandit",
            "variant": variant,
            "allowed_actions": list(allowed_actions),
            "gate_mode": gate_mode,
            "control": "v22_price_impact_route",
            "feature_dim": payload["feature_dim"],
            "min_support": payload["min_support"],
            "min_expected_delta": payload["min_expected_delta"],
            "lcb_z": payload["lcb_z"],
            "models": len(payload["models"]),
            "training_samples": len(samples),
            "training_data": str(samples_path.relative_to(ROOT)),
            "source_notebook": str(V22_SOURCE.relative_to(ROOT)),
            "source_sha256": source_sha,
            "main_sha256": hashlib.sha256(data).hexdigest(),
            "main_bytes": len(data),
            "archive": str(archive_path.relative_to(ROOT)),
            "root_main_modified": False,
        }
        (variant_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (history_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests[variant] = manifest

    (output / "weights.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {"variants": manifests, "fit": report}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=OUT_ARTIFACT)
    args = parser.parse_args()
    manifests, report = build(args.samples, args.output)
    print(json.dumps({"manifest": manifests, "fit": report}, indent=2))
