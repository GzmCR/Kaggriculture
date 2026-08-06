"""Build V022 fresh-route and local-recovery candidates.

The two supplied notebooks contain self-contained, anonymized compressed
agents in their final code cells.  This builder extracts those artifacts
offline and creates four independent candidates.  Runtime candidates never
read notebooks, replay files, identities, scores, seeds, or external APIs.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import shutil
import tarfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V012 = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
TACTICAL_NOTEBOOK = ROOT / "baseline/144-150-fresh-top-30-v21-tactical-memory.ipynb"
RECOVERY_NOTEBOOK = ROOT / "baseline/159-160-vs-frontier-v20-weed-slip-recovery.ipynb"
WEED_OVERLAY = Path(__file__).with_name("v022_weed_overlay.py")

CANDIDATES = (
    "v022a_weed_recovery",
    "v022b_fresh_medoid",
    "v022c_medoid_recovery",
    "v022d_medoid_recovery_tactical",
)

README = """# V022 fresh-route experiment

This artifact contains four independent candidates derived from the two
supplied public notebooks.

- `v022a_weed_recovery`: clean V012 route plus actor-local visible-WEED
  `DIG -> retry -> at most 8 turns of catch-up`.  Market orders are untouched.
- `v022b_fresh_medoid`: anonymous complete route extracted from the
  159/160 notebook, with no runtime tactical overlay.
- `v022c_medoid_recovery`: the same anonymous route plus local WEED recovery.
- `v022d_medoid_recovery_tactical`: the 144/150 notebook artifact, containing
  its fit-only medoid, balanced market hazard, one-turn half-quantity
  preemption, public-similarity gate and bounded WEED recovery.

The notebook payloads are used only at build time.  Runtime code uses no
notebook, replay, team name, score, seed lookup, network or external API.
Each candidate has its own `main.py` and `submission.tar.gz` with `main.py` at
the archive root.  The root repository `main.py` is not modified.

The notebook scores are local/replay research results, not official Kaggle
leaderboard guarantees.  Candidates must pass both fixed-replay leave-one-out
checks and fresh closed-loop games before any promotion is considered.
"""


def _decode_notebook_agent(path: Path) -> tuple[str, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "_AGENT_B85_PARTS" not in source:
            continue
        tree = ast.parse(source)
        encoded = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "_AGENT_B85_PARTS" for target in node.targets):
                encoded = ast.literal_eval(node.value)
                break
        if encoded is None:
            continue
        payload = zlib.decompress(base64.b85decode("".join(encoded))).decode("utf-8")
        return payload, hashlib.sha256(payload.encode()).hexdigest()
    raise RuntimeError(f"no embedded agent payload found in {path}")


def _append(source: str, suffix: str) -> str:
    return source.rstrip() + "\n\n" + suffix.strip() + "\n"


def _plain_medoid_wrapper() -> str:
    return r'''
# V022b: anonymous fresh medoid route only; no runtime tactical overlay.
_V022_MEDOID_ORIGINAL_AGENT = agent


def agent(obs):
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
        return _align_hands(_copy_action(_ACTIONS[step]), obs)
    except Exception:
        return _V022_MEDOID_ORIGINAL_AGENT(obs)
'''


def _recovery_wrapper() -> str:
    return r'''
# V022c: anonymous fresh medoid plus actor-local visible-WEED recovery.
_V022_MEDOID_ORIGINAL_AGENT = agent


def agent(obs):
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
        action = _weed_repair_action(obs, _copy_action(_ACTIONS[step]), step)
        return _align_hands(action, obs)
    except Exception:
        return _V022_MEDOID_ORIGINAL_AGENT(obs)
'''


def _build_sources() -> dict[str, tuple[str, dict]]:
    v012_source = V012.read_text(encoding="utf-8")
    weed_overlay = WEED_OVERLAY.read_text(encoding="utf-8")
    recovery_source, recovery_sha = _decode_notebook_agent(RECOVERY_NOTEBOOK)
    tactical_source, tactical_sha = _decode_notebook_agent(TACTICAL_NOTEBOOK)
    return {
        "v022a_weed_recovery": (
            _append(v012_source, weed_overlay),
            {"source": "v012", "source_sha256": hashlib.sha256(v012_source.encode()).hexdigest()},
        ),
        "v022b_fresh_medoid": (
            _append(recovery_source, _plain_medoid_wrapper()),
            {"source": "159-160 notebook embedded medoid", "source_sha256": recovery_sha},
        ),
        "v022c_medoid_recovery": (
            _append(recovery_source, _recovery_wrapper()),
            {"source": "159-160 notebook embedded medoid", "source_sha256": recovery_sha},
        ),
        "v022d_medoid_recovery_tactical": (
            tactical_source,
            {"source": "144-150 notebook embedded medoid and hazard", "source_sha256": tactical_sha},
        ),
    }


def _write_submission(candidate: str, source: str, metadata: dict) -> dict:
    history_dir = ROOT / "baseline/history" / candidate
    artifact_dir = ROOT / "baseline/artifacts" / candidate
    history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_main = history_dir / "main.py"
    artifact_main = artifact_dir / "main.py"
    history_main.write_text(source, encoding="utf-8")
    artifact_main.write_text(source, encoding="utf-8")
    archive = artifact_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(artifact_main, arcname="main.py")
    digest = hashlib.sha256(source.encode()).hexdigest()
    manifest = {
        "candidate": candidate,
        "main_sha256": digest,
        "main_bytes": len(source.encode()),
        "archive": archive.name,
        **metadata,
    }
    (artifact_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "README.md").write_text(README, encoding="utf-8")
    return manifest


def build() -> list[dict]:
    sources = _build_sources()
    manifests = []
    for candidate in CANDIDATES:
        source, metadata = sources[candidate]
        manifests.append(_write_submission(candidate, source, metadata))
    artifact_root = ROOT / "baseline/artifacts/v022_fresh_route"
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "README.md").write_text(README, encoding="utf-8")
    return manifests


if __name__ == "__main__":
    for row in build():
        print(f"{row['candidate']}: {row['main_bytes']} bytes {row['main_sha256']}")
