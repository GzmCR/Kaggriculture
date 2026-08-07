"""Build V026 candidates by grafting bounded V022 recovery onto v22.

The v22 notebook payload is the only route source.  The V022f overlay is used
only for its actor-local single-retry recovery state machine; its route and
market entry point are deliberately removed and rebuilt below so v22's
price-impact ordering remains active.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from build_v022e_adaptive_recovery import OVERLAY as V022E_OVERLAY
from build_v022e_adaptive_recovery import _decode_notebook_agent
from build_v022e_adaptive_recovery import ROOT
from build_v022f_single_retry import _single_retry_overlay


ARTIFACT_ROOT = ROOT / "baseline/artifacts/v026_v22_v022c_recovery"
HISTORY_ROOT = ROOT / "baseline/history/v026_v22_v022c_recovery"
SOURCE_NOTEBOOK = ROOT / "baseline/44-46-strict-future-top-30-v22-price-impact.ipynb"


README = """# V026: v22 route with V022c-family local recovery

V026 keeps the self-contained 14-hands high-output route and official
price-impact SELL ordering from the 44-46 v22 artifact.  It grafts only the
V022f single-retry actor-local visible-WEED recovery:

`DIG -> one retry -> observe -> early release or bounded catch-up`.

The second retry, mirror gate, opponent exposure market ranking, and old V022c
15-hands route are not included.  V026a changes only recovery.  V026b adds a
last-resort legality guard that clips an existing SELL quantity only when it is
greater than the currently visible shed-plus-carried inventory.

Both candidates are experimental and do not replace the repository root
`main.py`.
"""


def _recovery_helpers() -> str:
    """Return V022f helpers without its route-replacing agent function."""
    source = _single_retry_overlay()
    old_catchup = '''        if phase == "catchup":
            if _v022e_safe_current_action(unit_actions[index], transaction, tile):
                active.pop(actor, None)
                _V022E_STATS["catchup_releases"] += 1
            elif int(transaction.get("catchup_count", 0)) < _V022E_MAX_CATCHUP:
                unit_actions[index] = _v022e_actor_action(step - 1, actor)
                transaction["catchup_count"] = int(transaction.get("catchup_count", 0)) + 1
                _V022E_STATS["catchup_actions"] += 1
            else:
                active.pop(actor, None)
'''
    new_catchup = '''        if phase == "catchup":
            # Re-check the route at the current time before replaying an old
            # action.  This prevents recovery from consuming a current WATER,
            # FEED, CARE, or PASS slot merely because the previous timeline
            # action was still being caught up.
            current_route_action = _v022e_actor_action(step, actor)
            if _v022e_safe_current_action(current_route_action, transaction, tile):
                unit_actions[index] = current_route_action
                active.pop(actor, None)
                _V022E_STATS["catchup_releases"] += 1
            elif int(transaction.get("catchup_count", 0)) < _V022E_MAX_CATCHUP:
                unit_actions[index] = _v022e_actor_action(step - 1, actor)
                transaction["catchup_count"] = int(transaction.get("catchup_count", 0)) + 1
                _V022E_STATS["catchup_actions"] += 1
            else:
                active.pop(actor, None)
'''
    if old_catchup not in source:
        raise RuntimeError("V022f catch-up block not found")
    source = source.replace(old_catchup, new_catchup)
    marker = "\ndef agent(obs):"
    if marker not in source:
        raise RuntimeError("V022f overlay agent marker not found")
    return source.rsplit(marker, 1)[0].rstrip()


V026_AGENT = r'''

# V026: v22 route -> V022f single-retry recovery -> v22 price-impact slots.
def _v026_fallback(obs):
    farm = _farm(obs, _seat(obs))
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
        "market": [],
    }


def _v026_base_action(obs):
    step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
    # Do not call the v22 agent here: its original WEED overlay would run a
    # second state machine.  Start from the exact route, apply one recovery
    # layer, then keep the original v22 price-impact layer.
    action = _copy_action(_ACTIONS[step])
    action = _v022e_adaptive_repair(obs, action, step)
    return _align_hands(_impact_slots(obs, action), obs)


def agent(obs):
    try:
        return _v026_base_action(obs)
    except Exception:
        return _v026_fallback(obs)


_V026_STATS = _V022E_STATS
'''


V026_GUARD = r'''

# V026b: conservative legality guard only; valid v22 SELL orders are no-op.
_V026_GUARD_STATS = {
    "sell_orders_checked": 0,
    "sell_orders_clipped": 0,
    "sell_units_clipped": 0,
    "market_orders_truncated": 0,
}


def _v026_visible_inventory(obs):
    private = _get(obs, "private", {}) or {}
    available = {}
    for item, quantity in dict(_get(private, "shed", {}) or {}).items():
        try:
            available[str(item)] = available.get(str(item), 0) + max(0, int(quantity or 0))
        except (TypeError, ValueError):
            continue
    for inventory in list(_get(private, "inventories", []) or []):
        for item, quantity in dict(inventory or {}).items():
            try:
                available[str(item)] = available.get(str(item), 0) + max(0, int(quantity or 0))
            except (TypeError, ValueError):
                continue
    return available


def _v026_sell_guard(obs, action):
    action = _copy_action(action)
    available = _v026_visible_inventory(obs)
    output = []
    for order in list(action.get("market") or []):
        if not _is_sell(order):
            output.append(order)
            continue
        _V026_GUARD_STATS["sell_orders_checked"] += 1
        item = str(order[1])
        try:
            requested = max(0, int(order[2]))
        except (TypeError, ValueError):
            requested = 0
        allowed = min(requested, available.get(item, 0))
        if allowed != requested:
            _V026_GUARD_STATS["sell_orders_clipped"] += 1
            _V026_GUARD_STATS["sell_units_clipped"] += requested - allowed
        output.append(["SELL", item, allowed])
        available[item] = max(0, available.get(item, 0) - allowed)
    if len(output) > 10:
        _V026_GUARD_STATS["market_orders_truncated"] += len(output) - 10
    action["market"] = output[:10]
    return action


def agent(obs):
    try:
        return _v026_sell_guard(obs, _v026_base_action(obs))
    except Exception:
        return _v026_fallback(obs)
'''


def _write_candidate(name: str, source: str, source_sha: str, guard: bool) -> dict:
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
        "route": "v22_embedded_14hands_high_output",
        "market_layer": "v22_price_impact_slots",
        "recovery": "v022f_single_retry",
        "max_retries": 1,
        "max_catchup": 8,
        "market_changed": bool(guard),
        "root_main_modified": False,
    }
    (artifact_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "README.md").write_text(
        README + f"\nCandidate: `{name}`\n",
        encoding="utf-8",
    )
    return manifest


def build() -> dict:
    v22_source, source_sha = _decode_notebook_agent(SOURCE_NOTEBOOK)
    helpers = _recovery_helpers()
    base = v22_source.rstrip() + "\n\n" + helpers + V026_AGENT
    manifests = {
        "v026a_v22_single_retry": _write_candidate(
            "v026a_v22_single_retry", base + "\n", source_sha, guard=False
        ),
        "v026b_v22_single_retry_guard": _write_candidate(
            "v026b_v22_single_retry_guard", base + V026_GUARD + "\n", source_sha, guard=True
        ),
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "README.md").write_text(README, encoding="utf-8")
    (ARTIFACT_ROOT / "build_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifests


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
