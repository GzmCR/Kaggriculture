"""Build the self-contained RL-003 event-level timing selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from build_v022e_adaptive_recovery import ROOT, _decode_notebook_agent


SOURCE_NOTEBOOK = ROOT / "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb"
SELECTOR = ROOT / "experiments/rl_003_trade_timing.py"
OUT_HISTORY = ROOT / "baseline/history/rl_003_trade_timing"
OUT_ARTIFACT = ROOT / "baseline/artifacts/rl_003_trade_timing"


def build_source(weights):
    base_source, source_sha = _decode_notebook_agent(SOURCE_NOTEBOOK)
    marker = "\ndef agent(obs):"
    if base_source.count(marker) != 1:
        raise ValueError("expected one v22 agent definition")
    base_source = base_source.replace(marker, "\ndef _rl003_v22_agent(obs):", 1)
    selector = SELECTOR.read_text(encoding="utf-8")
    payload = json.dumps(weights, separators=(",", ":"), ensure_ascii=True)
    wrapper = f'''

# RL-003: event-level one-unit premium sale timing selector.
RL003_WEIGHTS = {payload}
_RL003_OPPORTUNITIES = route_opportunities(_ACTIONS)
_RL003_RUNTIME = TimingRuntime(weights=RL003_WEIGHTS, opportunities=_RL003_OPPORTUNITIES)

def agent(obs, config=None):
    """Public entry point; v22 owns every farmer/hand action."""
    try:
        base = _rl003_v22_agent(obs)
        return _RL003_RUNTIME.act(obs, base)
    except Exception:
        return _align_hands(_rl003_v22_agent(obs), obs)
'''
    return selector + "\n\n" + base_source.rstrip() + wrapper, source_sha


def build(weights_path):
    weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    source, source_sha = build_source(weights)
    OUT_HISTORY.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACT.mkdir(parents=True, exist_ok=True)
    history_main = OUT_HISTORY / "main.py"
    artifact_main = OUT_ARTIFACT / "main.py"
    source_bytes = source.encode("utf-8")
    history_main.write_bytes(source_bytes)
    artifact_main.write_bytes(source_bytes)
    (OUT_ARTIFACT / "weights.json").write_text(
        json.dumps(weights, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    archive_path = OUT_ARTIFACT / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(artifact_main, arcname="main.py")
    manifest = {
        "name": "rl_003_trade_timing",
        "status": "event_level_delay_one",
        "control": "v22_price_impact_route",
        "action_count": 5,
        "actions": {
            "0": "v22 control",
            "1": "delay one MILK unit",
            "2": "delay one WOOL unit",
            "3": "delay one STRAWBERRY unit",
            "4": "delay one MELON unit",
        },
        "feature_dim": 40,
        "max_delayed_orders": 8,
        "source_notebook": str(SOURCE_NOTEBOOK.relative_to(ROOT)),
        "source_sha256": source_sha,
        "main_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "main_bytes": len(source.encode("utf-8")),
        "archive": str(archive_path.relative_to(ROOT)),
        "root_main_modified": False,
    }
    (OUT_ARTIFACT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return history_main, artifact_main, archive_path, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=OUT_ARTIFACT / "fit_v2/weights.json",
    )
    args = parser.parse_args()
    paths = build(args.weights)
    print(json.dumps({"paths": [str(path) for path in paths[:3]], "manifest": paths[3]}, indent=2))
