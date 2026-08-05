"""Build self-contained V019 public-style market-router candidates."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V012 = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
ROUTER = ROOT / "experiments/v019_style_router.py"
OUT = ROOT / "baseline/artifacts/v019_public_style_router"
HISTORY = ROOT / "baseline/history/v019_public_style_router"


DEFAULT_MAPPING = {
    # Selected from the 348-game expert calibration with leave-one-episode-
    # out routing.  This is still an offline research prior; it never reads
    # the replay score or TeamName at runtime.
    "standard_converged": "navazsh_fathi",
    "reduced_ne_only": "mohit",
    "high_worker_maintenance": "manual_player",
    "premium_concentrated": "navazsh_fathi",
}


WRAPPER = r'''

# V019 wrapper: immutable V012 field route plus public-state market routing.
_V019_BASE_AGENT = agent
_V019_ROUTER = PublicStyleExpertRouter(
    mapping=_V019_MAPPING,
    hold_days=_V019_HOLD_DAYS,
)


def _v019_step(obs):
    return max(0, min(int(obs.get("step", 0) or 0), 719))


def _v019_copy_market(orders):
    return [list(order) for order in (orders or []) if isinstance(order, list)]


def _v019_target_market(base_orders, expert_orders, targets):
    """Replace only existing target SELL slots with expert quantities."""
    base_orders = _v019_copy_market(base_orders)
    expert_orders = _v019_copy_market(expert_orders)
    target_orders = {}
    for item in targets:
        quantities = [
            max(0, int(order[2]))
            for order in expert_orders
            if len(order) >= 3 and order[0] == "SELL" and order[1] == item
        ]
        target_orders[item] = sum(quantities)

    output = []
    replaced = set()
    for order in base_orders:
        if len(order) >= 2 and order[0] == "SELL" and order[1] in targets:
            item = order[1]
            if item in replaced:
                continue
            replaced.add(item)
            quantity = target_orders.get(item, 0)
            if quantity > 0:
                output.append(["SELL", item, quantity])
        else:
            output.append(list(order))
    return output[:10]


def agent(obs, config=None):
    prepared = dict(obs) if isinstance(obs, dict) else {}
    try:
        base = _V019_BASE_AGENT(prepared, config)
    except TypeError:
        base = _V019_BASE_AGENT(prepared)
    player = int(prepared.get("player", 0) or 0)
    experts = _V18_RUNTIME.get("experts", {})
    fallback = _V18_RUNTIME.get("board_by_seat", {}).get(str(player), "automatylicza")
    if fallback not in experts:
        fallback = sorted(experts)[0] if experts else None
    selected, style, confidence, features = _V019_ROUTER.choose(prepared, experts, fallback)
    step = _v019_step(prepared)
    expert_action = experts[selected]["actions"][min(step, len(experts[selected]["actions"]) - 1)] if selected else {}
    expert_market = expert_action.get("market", []) if isinstance(expert_action, dict) else []
    base_market = base.get("market", []) if isinstance(base, dict) else []
    if _V019_MODE == "price_priority":
        market = _v019_target_market(base_market, expert_market, ("MILK", "STRAWBERRY"))
    elif _V019_MODE == "weak_counter":
        if style == "reduced_ne_only" and confidence >= 0.90:
            market = _v019_copy_market(expert_market)[:10]
        else:
            market = _v019_copy_market(base_market)[:10]
    else:
        market = _v019_copy_market(expert_market)[:10]
    return {
        "farmer": list(base.get("farmer") or ["PASS"]),
        "hands": [list(item) for item in (base.get("hands") or [])],
        "market": market,
    }
'''


def build(mode, mapping, path, hold_days=1):
    base = V012.read_text(encoding="utf-8")
    router = ROUTER.read_text(encoding="utf-8").replace("from __future__ import annotations\n\n", "", 1)
    payload = (
        base
        + "\n\n"
        + router
        + "\n\n"
        + f"_V019_MODE = {mode!r}\n"
        + f"_V019_MAPPING = {dict(mapping)!r}\n"
        + f"_V019_HOLD_DAYS = {int(hold_days)}\n"
        + WRAPPER
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("price_priority", "public_style", "weak_counter"), default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)
    specs = {
        "price_priority": "v019a_price_priority.py",
        "public_style": "v019b_public_style.py",
        "weak_counter": "v019c_weak_counter.py",
    }
    modes = [args.mode] if args.mode else list(specs)
    for mode in modes:
        build(mode, DEFAULT_MAPPING, OUT / specs[mode])
    build("public_style", DEFAULT_MAPPING, HISTORY / "main.py")
    print("built", ", ".join(modes))


if __name__ == "__main__":
    main()
