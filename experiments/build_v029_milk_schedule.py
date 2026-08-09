"""Build V029 from v22 with a conservative MILK sale-wave schedule."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from build_v022e_adaptive_recovery import ROOT, _decode_notebook_agent


ARTIFACT_ROOT = ROOT / "baseline/artifacts/v029_milk_schedule"
HISTORY_ROOT = ROOT / "baseline/history/v029_milk_schedule"
SOURCE_NOTEBOOK = ROOT / "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb"


README = """# V029: conservative MILK sale-wave schedule

V029 keeps the complete v22 farmer/hands route and v22 price-impact ordering.
It changes only six existing MILK SELL quantities by moving one unit from the
first order to the next same-product order:

`215->260, 288->308, 336->375, 388->404, 504->522, 552->571`

The total MILK quantity is unchanged. No SELL is created, no non-SELL order is
moved, and the terminal period from step 672 uses the original v22 action.
This is an experimental candidate; the repository root `main.py` is unchanged.
"""


RUNTIME = r'''

# V029: quantity-preserving MILK sale-wave delays.
_V029_SCHEDULE = (
    (215, 260),
    (288, 308),
    (336, 375),
    (388, 404),
    (504, 522),
    (552, 571),
)
_V029_CUTOFF = 672
_V029_STATS = {
    "calls": 0,
    "errors": 0,
    "changed_calls": 0,
    "changed_units": 0,
    "schedule_hits": 0,
    "terminal_fallbacks": 0,
}


def _v029_fallback(obs):
    farm = _farm(obs, _seat(obs))
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
        "market": [],
    }


def _v029_base_action(obs, step):
    route_step = min(max(0, int(step)), len(_ACTIONS) - 1)
    action = _weed_repair_action(obs, _copy_action(_ACTIONS[route_step]), _ACTIONS, route_step)
    return _align_hands(_impact_slots(obs, action), obs)


def _v029_adjust_sell(action, item, delta):
    for index, order in enumerate(action.get("market", []) or []):
        if (
            isinstance(order, (list, tuple))
            and len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == item
        ):
            try:
                current = max(0, int(order[2]))
            except (TypeError, ValueError):
                return 0
            updated = current + int(delta)
            if updated < 0:
                return 0
            if updated == 0:
                action["market"].pop(index)
            else:
                action["market"][index] = [order[0], order[1], updated]
            return abs(int(delta))
    return 0


def _v029_market_shift(action, step):
    changed = 0
    for current_step, future_step in _V029_SCHEDULE:
        if int(step) == current_step:
            changed += _v029_adjust_sell(action, "MILK", -1)
        elif int(step) == future_step:
            changed += _v029_adjust_sell(action, "MILK", 1)
    if changed:
        _V029_STATS["changed_calls"] += 1
        _V029_STATS["changed_units"] += int(changed)
        _V029_STATS["schedule_hits"] += 1
    return action


def agent(obs):
    try:
        _V029_STATS["calls"] += 1
        step = max(0, int(_get(obs, "step", 0) or 0))
        action = _v029_base_action(obs, step)
        if step < _V029_CUTOFF:
            action = _v029_market_shift(action, step)
        else:
            _V029_STATS["terminal_fallbacks"] += 1
        return _align_hands(action, obs)
    except Exception:
        _V029_STATS["errors"] += 1
        return _v029_fallback(obs)
'''


def build() -> dict:
    base_source, source_sha = _decode_notebook_agent(SOURCE_NOTEBOOK)
    marker = "\ndef agent(obs):"
    if base_source.count(marker) != 1:
        raise RuntimeError("Expected exactly one public v22 agent definition")
    base_source = base_source.replace(marker, "\ndef _v029_v22_original_agent(obs):", 1)
    source = base_source.rstrip() + "\n\n" + RUNTIME.lstrip() + "\n"

    name = "v029a_milk_safe_schedule"
    history_dir = HISTORY_ROOT / name
    artifact_dir = ARTIFACT_ROOT / name
    history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_main = history_dir / "main.py"
    artifact_main = artifact_dir / "main.py"
    history_main.write_text(source, encoding="utf-8")
    artifact_main.write_text(source, encoding="utf-8")
    archive = artifact_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(artifact_main, arcname="main.py")

    manifest = {
        "candidate": name,
        "source_notebook": str(SOURCE_NOTEBOOK.relative_to(ROOT)),
        "source_sha256": source_sha,
        "main_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "main_bytes": len(source.encode("utf-8")),
        "archive": str(archive.relative_to(ROOT)),
        "route": "v22_embedded_complete_route",
        "market_layer": "six_quantity_preserving_milk_delays",
        "schedule": [[215, 260], [288, 308], [336, 375], [388, 404], [504, 522], [552, 571]],
        "cutoff": 672,
        "root_main_modified": False,
    }
    (artifact_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(README, encoding="utf-8")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "README.md").write_text(README, encoding="utf-8")
    (ARTIFACT_ROOT / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
