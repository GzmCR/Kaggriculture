"""Build the V022e adaptive visible-WEED recovery candidate."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import tarfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "baseline/159-160-vs-frontier-v20-weed-slip-recovery.ipynb"
CANDIDATE = "v022e_adaptive_recovery"


README = """# V022e adaptive WEED recovery

V022e keeps the same complete anonymous route and market actions as V022c.
Only the actor-local visible-WEED transaction is changed:

`DIG -> retry -> observe tile -> release when safe, otherwise bounded catch-up`.

The retry is confirmed from the next observation.  A failed retry gets one
additional `DIG -> retry` attempt; a second failure suppresses the same actor
and tile briefly so the agent cannot loop forever.  Other actors and market
orders are copied unchanged.  The root `main.py` and V022c archive are not
modified by this builder.
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
        return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()
    raise RuntimeError(f"no embedded agent payload found in {path}")


OVERLAY = r'''

# V022e: adaptive actor-local visible-WEED recovery.
import copy as _v022e_copy_module

_V022E_STATE = {
    0: {"last_step": -1, "active": {}, "suppressed": {}},
    1: {"last_step": -1, "active": {}, "suppressed": {}},
}
_V022E_MAX_CATCHUP = 8
_V022E_SUPPRESSION = 8
_V022E_STATS = {
    "weed_repairs": 0,
    "weed_retries": 0,
    "retry_success_first": 0,
    "retry_success_second": 0,
    "retry_failures": 0,
    "early_releases": 0,
    "catchup_releases": 0,
    "catchup_actions": 0,
    "abandoned": 0,
    "repeat_suppressed": 0,
}


def _v022e_copy_action(action):
    action = _v022e_copy_module.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in (action.get("hands") or [])],
        "market": [list(item) for item in (action.get("market") or []) if isinstance(item, list) and item],
    }


def _v022e_seat(obs):
    return 1 if int(obs.get("player", 0) or 0) == 1 else 0


def _v022e_align_hands(action, obs):
    action = _v022e_copy_action(action)
    seat = _v022e_seat(obs)
    farms = obs.get("farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    expected = len(farm.get("hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(item or ["PASS"]) for item in hands[:expected]]
    return action


def _v022e_tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (farm.get("tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError, KeyError):
        return "LOCKED"


def _v022e_route_action(step):
    actions = globals().get("_ACTIONS", []) or []
    if not actions:
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return _v022e_copy_action(actions[min(max(int(step), 0), len(actions) - 1)])


def _v022e_actor_action(step, actor):
    action = _v022e_route_action(step)
    if actor == "farmer":
        return list(action.get("farmer") or ["PASS"])
    index = int(actor)
    hands = action.get("hands", []) or []
    return list(hands[index] if index < len(hands) else ["PASS"])


def _v022e_positions(obs):
    seat = _v022e_seat(obs)
    farms = obs.get("farms", []) or []
    farm = farms[seat] if seat < len(farms) else {}
    return farm, [farm.get("farmer"), *(farm.get("hands", []) or [])]


def _v022e_actor_index(actor):
    return 0 if actor == "farmer" else int(actor) + 1


def _v022e_expected(intended):
    operation = str(intended[0]).upper() if isinstance(intended, list) and intended else ""
    if operation == "PLANT" and len(intended) >= 2:
        return "PLANT", str(intended[1]).upper()
    if operation == "BUILD_PASTURE":
        return "PASTURE", None
    return None, None


def _v022e_success(tile, transaction):
    if not isinstance(tile, dict):
        return False
    if tile.get("kind") != transaction.get("expected_kind"):
        return False
    expected_crop = transaction.get("expected_crop")
    return expected_crop is None or str(tile.get("crop", "")).upper() == expected_crop


def _v022e_safe_current_action(unit_action, transaction, tile):
    operation = str((unit_action or ["PASS"])[0]).upper()
    if operation == "PASS":
        return True
    expected_kind = transaction.get("expected_kind")
    if not isinstance(tile, dict) or tile.get("kind") != expected_kind:
        return False
    if expected_kind == "PLANT":
        return operation == "WATER"
    return operation in ("FEED", "CARE", "COLLECT_FERTILIZER", "PLACE")


def _v022e_reset_if_needed(state, step):
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "active": {}, "suppressed": {}})
    state["last_step"] = step


def _v022e_adaptive_repair(obs, action, step):
    action = _v022e_align_hands(action, obs)
    seat = _v022e_seat(obs)
    state = _V022E_STATE[seat]
    _v022e_reset_if_needed(state, step)
    farm, positions = _v022e_positions(obs)
    unit_actions = [
        list(action.get("farmer") or ["PASS"]),
        *[list(item or ["PASS"]) for item in action.get("hands", []) or []],
    ]
    active = state.setdefault("active", {})
    suppressed = state.setdefault("suppressed", {})

    for actor, transaction in list(active.items()):
        index = _v022e_actor_index(actor)
        if index >= len(unit_actions) or index >= len(positions):
            active.pop(actor, None)
            continue
        age = step - int(transaction.get("start_step", step))
        position = positions[index]
        tile = _v022e_tile_at(farm, position)
        phase = transaction.get("phase", "retry")

        if phase == "retry":
            unit_actions[index] = list(transaction["intended"])
            transaction["phase"] = "confirm"
            _V022E_STATS["weed_retries"] += 1
            continue

        if phase == "confirm":
            if _v022e_success(tile, transaction):
                if int(transaction.get("retry_count", 0)) == 0:
                    _V022E_STATS["retry_success_first"] += 1
                else:
                    _V022E_STATS["retry_success_second"] += 1
                if _v022e_safe_current_action(unit_actions[index], transaction, tile):
                    active.pop(actor, None)
                    _V022E_STATS["early_releases"] += 1
                elif age <= int(transaction.get("catchup_until", step)):
                    unit_actions[index] = _v022e_actor_action(step - 1, actor)
                    transaction["phase"] = "catchup"
                    transaction["catchup_count"] = int(transaction.get("catchup_count", 0)) + 1
                    _V022E_STATS["catchup_actions"] += 1
                else:
                    active.pop(actor, None)
                continue

            _V022E_STATS["retry_failures"] += 1
            if int(transaction.get("retry_count", 0)) < 1:
                transaction["retry_count"] = 1
                transaction["phase"] = "second_dig"
                unit_actions[index] = ["DIG"]
            else:
                active.pop(actor, None)
                suppressed[actor] = {
                    "until": step + _V022E_SUPPRESSION,
                    "position": tuple(position) if isinstance(position, (list, tuple)) else None,
                }
                _V022E_STATS["abandoned"] += 1
            continue

        if phase == "second_dig":
            unit_actions[index] = list(transaction["intended"])
            transaction["phase"] = "confirm"
            _V022E_STATS["weed_retries"] += 1
            continue

        if phase == "catchup":
            if _v022e_safe_current_action(unit_actions[index], transaction, tile):
                active.pop(actor, None)
                _V022E_STATS["catchup_releases"] += 1
            elif int(transaction.get("catchup_count", 0)) < _V022E_MAX_CATCHUP:
                unit_actions[index] = _v022e_actor_action(step - 1, actor)
                transaction["catchup_count"] = int(transaction.get("catchup_count", 0)) + 1
                _V022E_STATS["catchup_actions"] += 1
            else:
                active.pop(actor, None)

    for actor, block in list(suppressed.items()):
        index = _v022e_actor_index(actor)
        if index >= len(positions):
            suppressed.pop(actor, None)
            continue
        current_position = positions[index]
        current_tile = _v022e_tile_at(farm, current_position)
        old_position = block.get("position")
        if step >= int(block.get("until", step)) or current_tile != "LOCKED" and not (
            isinstance(current_tile, dict) and current_tile.get("kind") == "WEED"
        ) or (old_position is not None and tuple(current_position) != tuple(old_position)):
            suppressed.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if actor in suppressed:
            _V022E_STATS["repeat_suppressed"] += 1
            continue
        expected_kind, expected_crop = _v022e_expected(intended)
        if expected_kind is None:
            continue
        tile = _v022e_tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {
            "start_step": step,
            "intended": list(intended),
            "expected_kind": expected_kind,
            "expected_crop": expected_crop,
            "retry_count": 0,
            "catchup_until": step + _V022E_MAX_CATCHUP,
            "catchup_count": 0,
            "suppressed_until": None,
            "phase": "retry",
        }
        unit_actions[index] = ["DIG"]
        _V022E_STATS["weed_repairs"] += 1

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _v022e_align_hands(action, obs)


def agent(obs):
    try:
        step = max(0, int(obs.get("step", 0) or 0))
        return _v022e_adaptive_repair(obs, _v022e_route_action(step), step)
    except Exception:
        step = max(0, int(obs.get("step", 0) or 0)) if isinstance(obs, dict) else 0
        return _v022e_align_hands(_v022e_route_action(step), obs if isinstance(obs, dict) else {})
'''


def _write_submission(source: str, source_sha: str) -> dict:
    history_dir = ROOT / "baseline/history" / CANDIDATE
    artifact_dir = ROOT / "baseline/artifacts" / CANDIDATE
    history_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_main = history_dir / "main.py"
    artifact_main = artifact_dir / "main.py"
    history_main.write_text(source, encoding="utf-8")
    artifact_main.write_text(source, encoding="utf-8")
    archive = artifact_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(artifact_main, arcname="main.py")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    manifest = {
        "candidate": CANDIDATE,
        "source": str(SOURCE_NOTEBOOK.relative_to(ROOT)),
        "source_sha256": source_sha,
        "main_sha256": digest,
        "main_bytes": len(source.encode("utf-8")),
        "archive": str(archive.relative_to(ROOT)),
        "max_catchup": 8,
        "max_retries": 2,
        "market_unchanged": True,
    }
    (artifact_dir / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "README.md").write_text(README, encoding="utf-8")
    return manifest


def build() -> dict:
    source, source_sha = _decode_notebook_agent(SOURCE_NOTEBOOK)
    manifest = _write_submission(source.rstrip() + "\n\n" + OVERLAY.strip() + "\n", source_sha)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
