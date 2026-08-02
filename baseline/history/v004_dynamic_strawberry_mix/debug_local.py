import argparse
import importlib.util
from collections import Counter
from pathlib import Path

from kaggle_environments import make


PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)


def load_agent(path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Agent file not found: {path}\n"
            "Generate or copy the baseline to the project root as main.py."
        )

    spec = importlib.util.spec_from_file_location("debugged_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} must define agent(obs, config=None)")
    return module


def plan_info(module, obs):
    """Read optional scenario-aware planning diagnostics without changing state."""
    try:
        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        roles = module._role_plan(obs, farm)
        survey = module._survey(farm, private, roles, int(obs.get("day", 0)))
        target_hands = module._target_hands(obs, farm, private, roles)
        phase = module._policy_phase(obs, farm, private, survey)
        due_jobs = len(module._field_jobs(obs, farm, private, roles, liquidation=False))
        return {
            "phase": phase,
            "target_hands": target_hands,
            "due_jobs": due_jobs,
            "risk_animals": survey.get("at_risk_animals", 0),
            "risk_crops": survey.get("at_risk_crops", 0),
        }
    except Exception:
        return {}


def _farm_snapshot(obs, farm):
    plants = Counter()
    animals = Counter()
    yield_units = Counter()
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop:
                    plants[crop] += 1
                    yield_units[crop] += int(tile.get("yield_units", 0) or 0)
            animal = tile.get("animal")
            if animal:
                animals[animal] += 1

    private = obs.get("private", {}) or {}
    inventory = Counter()
    for item, count in (private.get("shed", {}) or {}).items():
        inventory[item] += int(count or 0)
    for carried in private.get("inventories", []) or []:
        for item, count in (carried or {}).items():
            inventory[item] += int(count or 0)

    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    return {
        "plants": dict(sorted(plants.items())),
        "animals": dict(sorted(animals.items())),
        "yield_units": dict(sorted(yield_units.items())),
        "inventory": {
            item: inventory[item] for item in PRODUCTS if inventory[item] > 0
        },
        "seeds": {
            item: int(count or 0)
            for item, count in sorted((private.get("seeds", {}) or {}).items())
            if int(count or 0) > 0
        },
        "prices": {
            item: int(prices[item] or 0)
            for item in PRODUCTS
            if item in prices
        },
    }


def _counter_delta(counter):
    return dict(sorted((key, value) for key, value in counter.items() if value))


def _order_estimate(module, obs, farm, order, hire_index=None):
    if not order:
        return None
    operation = order[0]
    if operation in {"SELL", "BUY_PRODUCT"} and len(order) >= 3:
        item = order[1]
        quantity = int(order[2] or 0)
        prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
        unit_price = int(prices.get(item, 0) or 0)
        return operation, item, quantity * unit_price
    if operation == "BUY_SEED" and len(order) >= 3:
        item = order[1]
        quantity = int(order[2] or 0)
        return operation, item, quantity * int(module.CROPS[item]["seed"])
    if operation == "BUY_ANIMAL" and len(order) >= 3:
        item = order[1]
        quantity = int(order[2] or 0)
        return operation, item, quantity * int(module.ANIMALS[item]["cost"])
    if operation == "BUY_LAND":
        unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
        index = max(0, unlocked - 1)
        prices = getattr(module, "LAND_PRICES", ())
        if index < len(prices):
            return operation, f"LAND_{index + 1}", int(prices[index])
    if operation == "HIRE":
        hires = (
            int(farm.get("hires_today", 0) or 0)
            if hire_index is None
            else int(hire_index)
        )
        return operation, "HIRE", int(module._fib(hires))
    return None


def print_daily_snapshot(obs, farm, stats, completed_day):
    snapshot = _farm_snapshot(obs, farm)
    print(
        f"DAILY day={int(completed_day):02d} "
        f"state_day={int(obs.get('day', 0)):02d} "
        f"money={farm.get('money', 0):.0f} "
        f"hands={len(farm.get('hands', []))} "
        f"plants={snapshot['plants']} animals={snapshot['animals']} "
        f"yield={snapshot['yield_units']} inventory={snapshot['inventory']} "
        f"seeds={snapshot['seeds']} prices={snapshot['prices']} "
        f"prev_ops={_counter_delta(stats['day_field_ops'])} "
        f"prev_market={_counter_delta(stats['day_market_orders'])} "
        f"est_sell={_counter_delta(stats['day_sell_value'])} "
        f"est_buy={_counter_delta(stats['day_buy_value'])}"
    )
    stats["day_field_ops"].clear()
    stats["day_market_orders"].clear()
    stats["day_sell_value"].clear()
    stats["day_buy_value"].clear()


def summarize(obs, action, module, stats, print_all=False):
    player = int(obs["player"])
    farm = obs["farms"][player]
    step = int(obs.get("step", 0))
    hour = int(obs.get("hour", 0))
    if hour == 0:
        previous_day = int(obs.get("day", 0)) - 1
        if previous_day >= 0:
            print_daily_snapshot(obs, farm, stats, previous_day)
        else:
            stats["day_field_ops"].clear()
            stats["day_market_orders"].clear()

    stats["calls"] += 1
    for field_action in [action.get("farmer", [])] + list(action.get("hands", [])):
        if field_action:
            stats["field_ops"][field_action[0]] += 1
            stats["day_field_ops"][field_action[0]] += 1
    hire_index = int(farm.get("hires_today", 0) or 0)
    for order in action.get("market", []):
        if order:
            stats["market_orders"][order[0]] += 1
            stats["day_market_orders"][order[0]] += 1
            if order[0] in {"SELL", "BUY_PRODUCT"} and len(order) >= 3:
                key = (order[0], order[1])
                quantity = int(order[2] or 0)
                stats["market_quantities"][key] += quantity
            estimate = _order_estimate(module, obs, farm, order, hire_index)
            if order[0] == "HIRE":
                hire_index += 1
            if estimate is not None:
                operation, item, value = estimate
                bucket = (
                    stats["sell_value"]
                    if operation == "SELL"
                    else stats["buy_value"]
                )
                day_bucket = (
                    stats["day_sell_value"]
                    if operation == "SELL"
                    else stats["day_buy_value"]
                )
                bucket[item] += value
                day_bucket[item] += value

    market = action.get("market", [])
    movement = {"NORTH", "SOUTH", "EAST", "WEST", "PASS"}
    field_ops = [
        item for item in [action.get("farmer", [])] + list(action.get("hands", []))
        if item and item[0] not in movement
    ]
    important = bool(market or field_ops)
    should_print = print_all or step < 24 or hour == 0 or important
    if not should_print:
        return

    diagnostics = plan_info(module, obs)
    print(
        f"[step={step:03d} day={obs.get('day', 0):02d} hour={hour:02d}] "
        f"money={farm.get('money', 0):.0f} hands={len(farm.get('hands', []))} "
        f"land={farm.get('unlocked_quadrants', [])} "
        f"plan={diagnostics}"
    )
    if important or print_all or step < 24:
        print(
            "  farmer=",
            action.get("farmer"),
            "hands=",
            action.get("hands", []),
            "market=",
            market,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--opponent", choices=("pass", "random", "starter"), default="starter")
    parser.add_argument("--print-all", action="store_true")
    args = parser.parse_args()

    module = load_agent(args.agent)
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": args.steps, "seed": args.seed},
        debug=True,
    )
    stats = {
        "calls": 0,
        "field_ops": Counter(),
        "market_orders": Counter(),
        "day_field_ops": Counter(),
        "day_market_orders": Counter(),
        "market_quantities": Counter(),
        "sell_value": Counter(),
        "buy_value": Counter(),
        "day_sell_value": Counter(),
        "day_buy_value": Counter(),
    }

    def traced_agent(obs, config=None):
        action = module.agent(obs, config)
        summarize(obs, action, module, stats, args.print_all)
        return action

    env.run([traced_agent, args.opponent])
    print("FINAL:", [(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])
    print("FIELD_OPS:", dict(stats["field_ops"]))
    print("MARKET_ORDERS:", dict(stats["market_orders"]))
    print("MARKET_QUANTITIES:", dict(stats["market_quantities"]))
    print("EST_SELL_VALUE:", dict(stats["sell_value"]))
    print("EST_BUY_VALUE:", dict(stats["buy_value"]))


if __name__ == "__main__":
    main()
