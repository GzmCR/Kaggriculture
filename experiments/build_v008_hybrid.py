"""Build the V008 hybrid current/frontier candidate without touching root main.py."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "main.py"
FRONTIER_NOTEBOOK = ROOT / "baseline/kaggle-frontier-lab-strategy-improvement.ipynb"


def extract_frontier_source():
    notebook = json.loads(FRONTIER_NOTEBOOK.read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][17].get("source", []))
    tree = ast.parse(cell)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", None) == "AGENT_SOURCE" for target in node.targets):
            continue
        source = ast.literal_eval(node.value)
        # The current policy uses CROPS, PRODUCTS, and ANIMALS as pricing
        # dictionaries.  Keep the embedded frontier namespace isolated so its
        # compact tuple constants cannot silently alter the fallback policy.
        for name in (
            "EPISODE_IDS",
            "PRODUCTS",
            "CROPS",
            "ANIMALS",
            "LOCK_TURNS",
            "_ROUTE",
        ):
            source = source.replace(name, f"_V008_FRONTIER_{name}")
        return source.replace(
            "\ndef agent(obs, config=None):",
            "\ndef _v008_frontier_agent(obs, config=None):",
            1,
        )
    raise ValueError("AGENT_SOURCE not found in frontier notebook")


HYBRID_TEMPLATE = r'''

# V008 hybrid router. The current rule policy remains the fallback. Frontier
# replay actions are used only inside a compatible, state-matched 24-turn block.
V008_DISTANCE_THRESHOLD = __THRESHOLD__
V008_LOCK_TURNS = 24
V008_ROUTE = None
V008_MODE = "current"
V008_DISTANCE = None
V008_FALLBACKS = 0
V008_ROUTE_SWITCHES = 0
V008_ROUTE_HISTORY = []


def _v008_reset_state():
    global V008_ROUTE, V008_MODE, V008_DISTANCE
    global V008_FALLBACKS, V008_ROUTE_SWITCHES, V008_ROUTE_HISTORY
    V008_ROUTE = None
    V008_MODE = "current"
    V008_DISTANCE = None
    V008_FALLBACKS = 0
    V008_ROUTE_SWITCHES = 0
    V008_ROUTE_HISTORY = []


def _v008_route_index(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        for key in ("route", "index", "value", "id"):
            if key in value:
                return _v008_route_index(value[key])
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _v008_pick_route(obs, step):
    live = _features(obs)
    distances = [
        sum(abs(left - right) for left, right in zip(live, reference))
        for reference in STATE_FEATURES[step]
    ]
    index = min(
        range(len(distances)),
        key=lambda candidate: (
            distances[candidate],
            candidate != PREFERRED_INDEX,
            candidate,
        ),
    )
    return int(index), float(distances[index])


def _v008_route_compatible(obs, route_action):
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    if not (0 <= player < len(farms)):
        return False
    farm_hands = len(farms[player].get("hands", []) or [])
    route_hands = len(route_action.get("hands", []) or [])
    return abs(farm_hands - route_hands) <= 2


def _v008_normalize_action(action, hand_count, max_orders=10):
    result = copy.deepcopy(action) if isinstance(action, dict) else {}
    farmer = result.get("farmer", ["PASS"])
    if not isinstance(farmer, list) or not farmer:
        farmer = ["PASS"]
    hands = result.get("hands", [])
    if not isinstance(hands, list):
        hands = []
    hands = [item if isinstance(item, list) and item else ["PASS"] for item in hands]
    if len(hands) < hand_count:
        hands.extend([["PASS"] for _ in range(hand_count - len(hands))])
    hands = hands[:hand_count]
    market = result.get("market", [])
    if not isinstance(market, list):
        market = []
    return {
        "farmer": farmer,
        "hands": hands,
        "market": market[:max_orders],
    }


def _v008_operations(action):
    operations = []
    if not isinstance(action, dict):
        return operations
    operations.append(action.get("farmer", []))
    operations.extend(action.get("hands", []) or [])
    return [item[0] for item in operations if isinstance(item, list) and item]


def _v008_emergency_maintenance(obs, route_action, current_action):
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    if not (0 <= player < len(farms)):
        return True
    farm = farms[player]
    step = int(obs.get("step", 0) or 0)
    hour = int(obs.get("hour", step % 24) or 0)
    route_ops = _v008_operations(route_action)
    current_ops = _v008_operations(current_action)

    urgent_water = False
    urgent_feed = False
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                if (
                    not tile.get("watered_today", False)
                    and int(tile.get("consecutive_unwatered", 0) or 0) >= 1
                ):
                    urgent_water = True
            elif tile.get("kind") in {"COOP", "PASTURE"}:
                if (
                    not tile.get("fed_today", False)
                    and int(tile.get("consecutive_unfed", 0) or 0) >= 1
                ):
                    urgent_feed = True

    if hour >= 18:
        if urgent_water and "WATER" not in route_ops and "WATER" in current_ops:
            return True
        if urgent_feed and "FEED" not in route_ops and "FEED" in current_ops:
            return True

    if step >= 704:
        terminal_ops = {"HARVEST", "COLLECT_FERTILIZER", "PICKUP", "DROP"}
        route_terminal = sum(operation in terminal_ops for operation in route_ops)
        current_terminal = sum(operation in terminal_ops for operation in current_ops)
        if route_terminal == 0 and current_terminal > 0:
            return True
    return False


def _v008_hand_count(obs):
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    if 0 <= player < len(farms):
        return len(farms[player].get("hands", []) or [])
    return 0


def agent(obs, config=None):
    global V008_ROUTE, V008_MODE, V008_DISTANCE
    global V008_FALLBACKS, V008_ROUTE_SWITCHES, V008_ROUTE_HISTORY
    step = int(obs.get("step", 0) or 0)
    if step == 0:
        _v008_reset_state()

    current_action = _v008_current_agent(obs, config)
    boundary = step == 0 or step % V008_LOCK_TURNS == 0
    if boundary and step < len(STATE_FEATURES):
        try:
            route, distance = _v008_pick_route(obs, step)
            route = _v008_route_index(route)
        except Exception:
            route, distance = None, None
        if route is None or not (0 <= route < len(TRACES)):
            V008_ROUTE = None
            V008_DISTANCE = distance
            V008_MODE = "current"
            V008_FALLBACKS += 1
            V008_ROUTE_HISTORY.append({
                "step": step,
                "route": None,
                "distance": distance,
                "mode": "current",
            })
            return current_action
        route_action = _normalize_action_for_v008(route, step)
        selected = (
            distance <= V008_DISTANCE_THRESHOLD
            and _v008_route_compatible(obs, route_action)
        )
        next_mode = "frontier" if selected else "current"
        if next_mode != V008_MODE:
            V008_ROUTE_SWITCHES += 1
        V008_ROUTE = route
        V008_DISTANCE = distance
        V008_MODE = next_mode
        V008_ROUTE_HISTORY.append({
            "step": step,
            "route": route,
            "distance": distance,
            "mode": next_mode,
        })

    if V008_MODE != "frontier" or V008_ROUTE is None:
        return current_action

    route_index = _v008_route_index(V008_ROUTE)
    if route_index is None or not (0 <= route_index < len(TRACES)):
        V008_FALLBACKS += 1
        return current_action
    if step >= len(TRACES[route_index]):
        V008_FALLBACKS += 1
        return current_action
    route_action = _normalize_action_for_v008(route_index, step)
    route_action = _v008_normalize_action(
        route_action,
        _v008_hand_count(obs),
        int((config or {}).get("maxMarketOrdersPerTurn", 10) or 10),
    )
    if _v008_emergency_maintenance(obs, route_action, current_action):
        V008_FALLBACKS += 1
        return current_action
    return route_action


def _normalize_action_for_v008(route, step):
    route_index = _v008_route_index(route)
    if route_index is None or not (0 <= route_index < len(TRACES)):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return copy.deepcopy(TRACES[route_index][step])
'''


def build_source(threshold: float):
    current = CURRENT.read_text(encoding="utf-8")
    marker = "\ndef agent(obs, config=None):"
    if current.count(marker) != 1:
        raise ValueError("Expected exactly one root agent function")
    current = current.replace(marker, "\ndef _v008_current_agent(obs, config=None):", 1)
    frontier = extract_frontier_source()
    hybrid = HYBRID_TEMPLATE.replace("__THRESHOLD__", repr(float(threshold)))
    return current.rstrip() + "\n\n" + frontier.rstrip() + "\n" + hybrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = build_source(args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding="utf-8")
    compile(source, str(args.out), "exec")
    print({"path": str(args.out), "threshold": args.threshold, "bytes": len(source.encode("utf-8"))})


if __name__ == "__main__":
    main()
