"""Build the self-contained RL-001 submission from the V022c control.

The generated file keeps V022c's complete field/labor route and embeds only
the small NumPy market selector plus its learned weights. The root baseline
and the V022c source are read-only inputs to this builder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V022C = ROOT / "baseline" / "artifacts" / "v022c_medoid_recovery" / "main.py"
SELECTOR = ROOT / "experiments" / "rl_001_selector.py"
OUT_HISTORY = ROOT / "baseline" / "history" / "rl_001_macro_market"
OUT_ARTIFACT = ROOT / "baseline" / "artifacts" / "rl_001_macro_market"


def zero_weights():
    return {
        "q_a": [[0.0] * 96 for _ in range(4)],
        "q_b": [[0.0] * 96 for _ in range(4)],
    }


def load_weights(path: Path | None):
    if path is None or not path.exists():
        return zero_weights()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    q_a = payload.get("q_a")
    q_b = payload.get("q_b")
    if not (isinstance(q_a, list) and isinstance(q_b, list)):
        raise ValueError("weights JSON must contain q_a and q_b")
    if len(q_a) != 4 or len(q_b) != 4 or any(len(row) != 96 for row in q_a + q_b):
        raise ValueError("weights JSON must have shape (4, 96) for q_a and q_b")
    return {"q_a": q_a, "q_b": q_b}


def rename_final_agent(source: str) -> str:
    marker = "\ndef agent(obs):"
    index = source.rfind(marker)
    if index < 0:
        raise ValueError("could not locate V022c final agent definition")
    return source[:index] + "\ndef _v022c_agent(obs):" + source[index + len(marker):]


def build_source(weights):
    control = rename_final_agent(V022C.read_text(encoding="utf-8"))
    control = control.replace("from __future__ import annotations\n\n", "", 1)
    selector = SELECTOR.read_text(encoding="utf-8")
    payload = json.dumps(weights, separators=(",", ":"), ensure_ascii=True)
    wrapper = f'''

# RL-001: high-level market selector on top of the V022c control.
RL_WEIGHTS = {payload}
_RL_RUNTIME = SelectorRuntime(weights=RL_WEIGHTS, training=False, seed=0)

def agent(obs, config=None):
    """Public Kaggle entry point; V022c still owns farmer and hand actions."""
    try:
        base = _v022c_agent(obs)
        return _RL_RUNTIME.act(obs, base, base_actions=_ACTIONS)
    except Exception:
        return _align_hands(_v022c_agent(obs), obs)
'''
    # Put the selector first so V022c's original helper names are defined
    # last. This prevents the embedded experiment helpers from changing the
    # control route when zero weights are used.
    return selector + "\n\n" + control + wrapper


def write_artifacts(weights):
    OUT_HISTORY.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACT.mkdir(parents=True, exist_ok=True)
    source = build_source(weights)
    history_main = OUT_HISTORY / "main.py"
    artifact_main = OUT_ARTIFACT / "main.py"
    history_main.write_text(source, encoding="utf-8")
    shutil.copyfile(history_main, artifact_main)
    (OUT_ARTIFACT / "weights.json").write_text(
        json.dumps(weights, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "name": "rl_001_macro_market",
        "status": "control_zero_weights",
        "control": "baseline/artifacts/v022c_medoid_recovery/main.py",
        "block_steps": 48,
        "rl_stop_step": 672,
        "action_count": 4,
        "feature_dim": 96,
        "weights": "weights.json",
        "zero_weights_are_v022c_control": True,
        "pilot_weights_are_not_promoted": True,
    }
    (OUT_ARTIFACT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    tar_path = OUT_ARTIFACT / "submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(artifact_main, arcname="main.py")
    return history_main, artifact_main, tar_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    weights = load_weights(args.weights)
    paths = write_artifacts(weights)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
