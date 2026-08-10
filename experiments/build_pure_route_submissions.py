"""Build self-contained submissions for the fixed route-only comparison.

The route loader is shared with ``run_pure_route_round_robin.py`` so the
archived submissions and the benchmark use the same extracted traces.  The
generated agent only indexes a frozen action list; it has no market, recovery,
opponent, or terminal strategy layer.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "baseline" / "history" / "pure_routes"
LOADER_PATH = ROOT / "experiments" / "run_pure_route_round_robin.py"

FOLDER_NAMES = {
    "V22": "v22",
    "Adaptive_V14": "adaptive_v14",
    "V27": "v27",
    "Stable12": "stable12",
    "V022c": "v022c",
    "V13_R3": "v13_r3",
}


MAIN_TEMPLATE = '''"""Pure fixed route: {name}.

This submission returns the archived action trace at the current step.  It
does not run the source notebook/agent and does not add market sorting,
preemption, delay, WEED recovery, opponent detection, or terminal cleanup.
Market orders embedded in the trace are part of the fixed route.
"""

import base64
import copy
import json
import zlib


_ACTIONS = json.loads(zlib.decompress(base64.b85decode(
    {payload!r}
)).decode("utf-8"))


def agent(obs, config=None):
    step = int(obs.get("step", 0) or 0)
    step = min(max(step, 0), len(_ACTIONS) - 1)
    return copy.deepcopy(_ACTIONS[step])
'''


README_TEMPLATE = '''# {name}: pure fixed route

This folder archives a standalone route-only submission extracted from the
local Kaggriculture research repository.

Runtime behavior:

- return the frozen action trace for the current `step`;
- keep market orders exactly as embedded in the trace;
- do not reorder or add market orders;
- do not use WEED recovery, opponent state, replay files, notebooks, scores,
  seeds, network access, or external APIs;
- do not add terminal liquidation or dynamic hand alignment.

Route length: `{length}` actions  
Route SHA-256: `{route_sha}`

The route was extracted from the source recorded in the repository manifest.
It is archived for pure production-route comparison, not selected as the
current root `main.py`.
'''


def _load_loader():
    spec = importlib.util.spec_from_file_location("pure_route_loader", LOADER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LOADER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    loader = _load_loader()
    routes, sources = loader._load_routes()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "pure_fixed_route",
        "runtime_overlays": [],
        "market_orders_are_embedded_route_data": True,
        "routes": {},
    }
    for name, folder in FOLDER_NAMES.items():
        route = routes[name]
        encoded = base64.b85encode(
            zlib.compress(json.dumps(route, separators=(",", ":")).encode("utf-8"), 9)
        ).decode("ascii")
        route_sha = hashlib.sha256(
            json.dumps(route, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        target = OUT_ROOT / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / "main.py").write_text(
            MAIN_TEMPLATE.format(name=name, payload=encoded), encoding="utf-8"
        )
        (target / "README.md").write_text(
            README_TEMPLATE.format(name=name, length=len(route), route_sha=route_sha),
            encoding="utf-8",
        )
        manifest["routes"][name] = {
            "folder": folder,
            "length": len(route),
            "route_sha256": route_sha,
            "source": sources[name],
        }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
