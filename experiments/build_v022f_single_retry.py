"""Build V022f: V022e adaptive recovery without a second retry."""

from __future__ import annotations

import hashlib
import json
import tarfile

from build_v022e_adaptive_recovery import OVERLAY, ROOT, SOURCE_NOTEBOOK, _decode_notebook_agent


CANDIDATE = "v022f_single_retry"
README = """# V022f single-retry WEED recovery

V022f is an ablation of V022e. It keeps the same complete route and market
actions, but removes the second `DIG -> retry` attempt. After the first retry
fails, the actor is suppressed briefly and returns to the current route.
V022c and V022e remain unchanged; this candidate is experimental.
"""


def _single_retry_overlay() -> str:
    first_failure = "\n".join([
        '            _V022E_STATS["retry_failures"] += 1',
        '            if int(transaction.get("retry_count", 0)) < 1:',
        '                transaction["retry_count"] = 1',
        '                transaction["phase"] = "second_dig"',
        '                unit_actions[index] = ["DIG"]',
        '            else:',
        '                active.pop(actor, None)',
        '                suppressed[actor] = {',
        '                    "until": step + _V022E_SUPPRESSION,',
        '                    "position": tuple(position) if isinstance(position, (list, tuple)) else None,',
        '                }',
        '                _V022E_STATS["abandoned"] += 1',
        '            continue',
        '',
    ])
    single_failure = "\n".join([
        '            _V022E_STATS["retry_failures"] += 1',
        '            active.pop(actor, None)',
        '            suppressed[actor] = {',
        '                "until": step + _V022E_SUPPRESSION,',
        '                "position": tuple(position) if isinstance(position, (list, tuple)) else None,',
        '            }',
        '            _V022E_STATS["abandoned"] += 1',
        '            continue',
        '',
    ])
    result = OVERLAY.replace(first_failure, single_failure)
    second_phase = "\n".join([
        '        if phase == "second_dig":',
        '            unit_actions[index] = list(transaction["intended"])',
        '            transaction["phase"] = "confirm"',
        '            _V022E_STATS["weed_retries"] += 1',
        '            continue',
        '',
    ])
    result = result.replace(second_phase, "")
    result = result.replace('            "retry_count": 0,\n', '')
    return result


def _write(source: str, source_sha: str) -> dict:
    history_dir = ROOT / "baseline/history" / CANDIDATE
    artifact_dir = ROOT / "baseline/artifacts" / CANDIDATE
    history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "main.py").write_text(source, encoding="utf-8")
    artifact_main = artifact_dir / "main.py"
    artifact_main.write_text(source, encoding="utf-8")
    archive = artifact_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(artifact_main, arcname="main.py")
    manifest = {
        "candidate": CANDIDATE,
        "source": str(SOURCE_NOTEBOOK.relative_to(ROOT)),
        "source_sha256": source_sha,
        "main_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "main_bytes": len(source.encode("utf-8")),
        "archive": str(archive.relative_to(ROOT)),
        "max_catchup": 8,
        "max_retries": 1,
        "market_unchanged": True,
    }
    (artifact_dir / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "README.md").write_text(README, encoding="utf-8")
    return manifest


def build() -> dict:
    source, source_sha = _decode_notebook_agent(SOURCE_NOTEBOOK)
    return _write(source.rstrip() + "\n\n" + _single_retry_overlay().strip() + "\n", source_sha)


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
