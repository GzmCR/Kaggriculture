"""Load the embedded agents from the 2026-08-09 baseline notebooks."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import time
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOP10_DIR = ROOT / "baseline/2026-08-09"

TOP10_NOTEBOOKS = {
    "02_adaptive_replay": TOP10_DIR / "02-adaptive-replay-agent.ipynb",
    "84_public_holdout": TOP10_DIR / "84-84-base-public-holdout-v14-clone-preemption.ipynb",
    "adaptive_farming": TOP10_DIR / "adaptive-farming-strategy-for-kaggriculture.ipynb",
    "adaptive_replay": TOP10_DIR / "kaggriculture-adaptive-replay-agent.ipynb",
    "top_meta": TOP10_DIR / "kaggriculture-findings-from-zero-to-top-meta.ipynb",
    "frontier_soil": TOP10_DIR / "kaggriculture-frontier-the-soil-remembers-rain.ipynb",
    "high_score_pipeline": TOP10_DIR / "my-2026-08-04-high-score-pipeline.ipynb",
    "strong_barnyard": TOP10_DIR / "strong-barnyard-economist.ipynb",
    "rank_c45": TOP10_DIR / "kaggriculture-rank-your-agent.ipynb",
    "kaggriculture": TOP10_DIR / "kaggriculture.ipynb",
}


def unique_top10_names():
    """Return one name per embedded executable source hash."""
    seen = set()
    names = []
    for row in inspect_top10():
        if row.get("load_error"):
            continue
        source_sha = row.get("source_sha256") or row["name"]
        if source_sha in seen:
            continue
        seen.add(source_sha)
        names.append(row["name"])
    return names


def _assignment(tree, names):
    names = set(names)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            return ast.literal_eval(node.value)
    return None


def decode_notebook_source(path):
    """Decode the first self-contained agent payload found in a notebook."""
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        source = _assignment(tree, ("AGENT_SOURCE", "_TOP_AGENT_SOURCE"))
        if isinstance(source, str) and "def agent" in source:
            return source, "literal_source"

        parts = _assignment(tree, ("_AGENT_B85_PARTS",))
        if isinstance(parts, list):
            source = zlib.decompress(base64.b85decode("".join(parts))).decode("utf-8")
            return source, "b85_zlib_source"

        encoded = _assignment(tree, ("C64_B85",))
        if isinstance(encoded, str):
            source = zlib.decompress(base64.b85decode(encoded)).decode("utf-8")
            return source, "b85_zlib_source"

        parts = _assignment(tree, ("_AGENT_B64_PARTS",))
        if isinstance(parts, list):
            source = zlib.decompress(base64.b64decode("".join(parts))).decode("utf-8")
            return source, "b64_zlib_source"

        encoded = _assignment(tree, ("_AGENT_B64",))
        if isinstance(encoded, str):
            source = zlib.decompress(base64.b64decode(encoded)).decode("utf-8")
            return source, "b64_zlib_source"

    raise ValueError(f"no embedded agent source found in {path}")


def load_top10_agent(name, path=None):
    """Return a fresh agent callable and source metadata for one notebook."""
    path = Path(path or TOP10_NOTEBOOKS[name])
    source, encoding = decode_notebook_source(path)
    namespace = {
        "__name__": f"top10_{name}_{time.time_ns()}",
        "__file__": str(path),
    }
    exec(compile(source, str(path), "exec"), namespace)
    agent = namespace.get("agent")
    if not callable(agent):
        raise AttributeError(f"{path} source did not define callable agent")
    return agent, {
        "name": str(name),
        "path": str(path.relative_to(ROOT)),
        "encoding": encoding,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_bytes": len(source.encode("utf-8")),
    }


def inspect_top10():
    rows = []
    seen = {}
    for name, path in TOP10_NOTEBOOKS.items():
        try:
            _, metadata = load_top10_agent(name, path)
            duplicate_of = seen.get(metadata["source_sha256"])
            if duplicate_of is None:
                seen[metadata["source_sha256"]] = name
            metadata["duplicate_of"] = duplicate_of
            metadata["load_error"] = ""
        except Exception as exc:
            metadata = {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "encoding": "",
                "source_sha256": "",
                "source_bytes": 0,
                "duplicate_of": "",
                "load_error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(metadata)
    return rows


if __name__ == "__main__":
    print(json.dumps(inspect_top10(), indent=2, ensure_ascii=True))
