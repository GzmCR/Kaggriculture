"""Executable opponent catalog for RL-010 data collection.

Notebook names and scores are offline metadata only.  The runtime policy never
imports this module and never sees an opponent identifier.
"""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import time
import types
from pathlib import Path

from top10_opponents import decode_notebook_source


ROOT = Path(__file__).resolve().parents[1]

OPPONENT_SPECS = [
    {
        "name": "v27_current",
        "path": ROOT / "baseline/2026-08-09/25-27-strict-future-v27-midgame-meta-reset.ipynb",
        "family": "current_high",
    },
    {
        "name": "v14_public",
        "path": ROOT / "baseline/2026-08-09/84-84-base-public-holdout-v14-clone-preemption.ipynb",
        "family": "market_specialist",
    },
    {
        "name": "adaptive_replay",
        "path": ROOT / "baseline/2026-08-09/02-adaptive-replay-agent.ipynb",
        "family": "current_high",
    },
    {
        "name": "frontier_current",
        "path": ROOT / "baseline/2026-08-09/kaggriculture-frontier-the-soil-remembers-rain.ipynb",
        "family": "market_specialist",
    },
    {
        "name": "strong_barnyard",
        "path": ROOT / "baseline/2026-08-09/strong-barnyard-economist.ipynb",
        "family": "market_specialist",
    },
    {
        "name": "high_score_pipeline",
        "path": ROOT / "baseline/2026-08-09/my-2026-08-04-high-score-pipeline.ipynb",
        "family": "current_high",
    },
    {
        "name": "hamburger",
        "path": ROOT / "baseline/kaggriculture-hamburger.ipynb",
        "family": "early_rule",
        "special": "anchor",
    },
    {
        "name": "v13_r3",
        "path": ROOT / "baseline/v13-r3-top-meta-order-safe-premium-control.ipynb",
        "family": "market_specialist",
    },
    {
        "name": "v21_1",
        "path": ROOT / "baseline/177-180-fresh-top-30-v21-1-conditional-memory.ipynb",
        "family": "market_specialist",
    },
    {
        "name": "yummers",
        "path": ROOT / "baseline/kaggriculture-yummers.ipynb",
        "family": "early_rule",
        "special": "anchor",
    },
]


def _load_anchor(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "ANCHOR_BLOB" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "ANCHOR_BLOB"
                for target in node.targets
            ):
                continue
            blob = ast.literal_eval(node.value)
            raw = gzip.decompress(base64.b64decode(blob.encode("ascii"))).decode("utf-8")
            namespace = {
                "__name__": f"rl010_anchor_{time.time_ns()}",
                "__file__": str(path),
            }
            exec(compile(raw, str(path), "exec"), namespace)
            return namespace["agent"], raw
    raise ValueError(f"missing ANCHOR_BLOB in {path}")


def load_spec(spec):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = ROOT / path
    if spec.get("special") == "anchor":
        agent, source = _load_anchor(path)
        encoding = "anchor_blob"
    else:
        source, encoding = decode_notebook_source(path)
        namespace = {
            "__name__": f"rl010_opponent_{spec['name']}_{time.time_ns()}",
            "__file__": str(path),
        }
        exec(compile(source, str(path), "exec"), namespace)
        agent = namespace.get("agent")
        if not callable(agent):
            raise AttributeError(f"{path} did not define agent")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return agent, {
        **spec,
        "path": str(path.relative_to(ROOT)),
        "encoding": encoding,
        "source_sha256": source_sha,
        "source_bytes": len(source.encode("utf-8")),
    }


def inspect_opponents():
    rows = []
    seen = {}
    for spec in OPPONENT_SPECS:
        try:
            _, metadata = load_spec(spec)
            metadata["duplicate_of"] = seen.get(metadata["source_sha256"], "")
            if not metadata["duplicate_of"]:
                seen[metadata["source_sha256"]] = spec["name"]
            metadata["load_error"] = ""
        except Exception as exc:
            metadata = {
                **spec,
                "path": str(Path(spec["path"]).relative_to(ROOT)),
                "source_sha256": "",
                "source_bytes": 0,
                "duplicate_of": "",
                "load_error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(metadata)
    return rows


def unique_loadable_specs():
    rows = inspect_opponents()
    seen = set()
    result = []
    for row in rows:
        if row.get("load_error") or row.get("duplicate_of"):
            continue
        seen.add(row["source_sha256"])
        result.append(row)
    return result, rows
