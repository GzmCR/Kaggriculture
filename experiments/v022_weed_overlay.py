"""Append-only actor-local WEED recovery overlay for V022a.

This file is appended to the clean V012 agent by the V022 builder.  It does
not alter market orders or another actor's route.  The wrapper only repairs a
visible, current-tile PLANT/BUILD_PASTURE slip.
"""

import copy as _v022_copy_module


_V022_ORIGINAL_AGENT = agent
_V022_WEED_STATE = {0: {}, 1: {}}
_V022_WEED_REPLAY_STEPS = 8
_V022_DIAGNOSTICS = {
    "weed_repairs": 0,
    "weed_retries": 0,
    "weed_catchup_actions": 0,
}


def _v022_copy_action(action):
    action = _v022_copy_module.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _v022_align_hands(action, obs):
    action = _v022_copy_action(action)
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[seat] if seat < len(farms) else {}
    expected = len(_get(farm, "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _v022_tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _v022_base_trace_action(obs, step, actor):
    """Return the original route action for one actor at a prior step."""
    try:
        player = int(_get(obs, "player", 0) or 0)
        strategy = globals().get("STRATEGY", {})
        if strategy.get("use_fixed_schedule") and strategy.get("fixed_schedule_version") == "v18":
            board_name = _V18_RUNTIME["board_by_seat"][str(1 if player == 1 else 0)]
            schedule = _V18_RUNTIME["experts"][board_name]["actions"]
            trace = schedule[min(max(int(step), 0), len(schedule) - 1)] or {}
        else:
            probe = dict(obs)
            probe["step"] = int(step)
            trace = _V022_ORIGINAL_AGENT(probe) or {}
        if actor == "farmer":
            return list(trace.get("farmer") or ["PASS"])
        hands = trace.get("hands", []) or []
        index = int(actor)
        return list(hands[index] if index < len(hands) else ["PASS"])
    except Exception:
        return ["PASS"]


def _v022_weed_repair_action(obs, action, step):
    action = _v022_align_hands(action, obs)
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    game = _V022_WEED_STATE[seat]
    if step == 0 or "active" not in game or step < int(game.get("last_step", -1)):
        game = {"last_step": step, "active": {}}
        _V022_WEED_STATE[seat] = game
    game["last_step"] = step

    farms = list(_get(obs, "farms", []) or [])
    farm = farms[seat] if seat < len(farms) else {}
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - int(transaction["start"])
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
            _V022_DIAGNOSTICS["weed_retries"] += 1
        elif 2 <= age <= 1 + _V022_WEED_REPLAY_STEPS:
            unit_actions[index] = _v022_base_trace_action(obs, step - 1, actor)
            _V022_DIAGNOSTICS["weed_catchup_actions"] += 1
        elif age > 1 + _V022_WEED_REPLAY_STEPS:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _v022_tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]
        _V022_DIAGNOSTICS["weed_repairs"] += 1

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _v022_align_hands(action, obs)


def agent(obs):
    try:
        base = _V022_ORIGINAL_AGENT(obs)
        step = max(0, int(_get(obs, "step", 0) or 0))
        return _v022_weed_repair_action(obs, base, step)
    except Exception:
        return _V022_ORIGINAL_AGENT(obs)
