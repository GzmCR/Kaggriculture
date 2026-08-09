"""Event-level reinforcement learning for v22 premium sale timing.

The field route and v22 price-impact market policy stay fixed.  At an existing
premium SELL wave the policy chooses either the original order or a bounded
one-unit delay to the next same-product wave.  This is deliberately narrower
than RL-001's 48-turn overlay selector so the reward has a causal meaning.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Kaggle's source loader executes raw code without defining __file__.
    ROOT = Path.cwd()
PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
FEATURE_DIM = 40
ACTION_COUNT = 5
ACTION_CONTROL = 0
ACTION_FOR_ITEM = {item: index + 1 for index, item in enumerate(PREMIUM)}
MAX_DELAYED_ORDERS = 8
MIN_GAP = 4
MAX_GAP = 72


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _load_v22_module():
    path = ROOT / "baseline/history/v027_v22_product_shift/v027a_melon_ratio/main.py"
    spec = importlib.util.spec_from_file_location(f"rl003_v22_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_opportunities(actions=None):
    """Return existing premium SELL waves and their next same-product wave."""
    if actions is None:
        actions = _load_v22_module()._ACTIONS
    events = defaultdict(list)
    for step, action in enumerate(actions):
        for order in (action or {}).get("market", []) or []:
            if len(order) < 3 or str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item not in PREMIUM:
                continue
            quantity = max(0, _as_int(order[2]))
            if quantity:
                events[item].append((int(step), quantity))
    opportunities = []
    for item, rows in events.items():
        for index, (current_step, current_quantity) in enumerate(rows[:-1]):
            future_step, future_quantity = rows[index + 1]
            gap = future_step - current_step
            if not (MIN_GAP <= gap <= MAX_GAP):
                continue
            opportunities.append({
                "item": item,
                "current_step": current_step,
                "future_step": future_step,
                "current_quantity": current_quantity,
                "future_quantity": future_quantity,
                "gap": gap,
            })
    return sorted(opportunities, key=lambda row: (row["current_step"], row["item"]))


def _opportunity_index(opportunities=None):
    opportunities = opportunities or route_opportunities()
    index = defaultdict(list)
    for row in opportunities:
        if int(row["current_step"]) < 648 and int(row["future_step"]) < 672:
            index[int(row["current_step"])].append(row)
    for step in index:
        deduped = {}
        for row in index[step]:
            key = (row["item"], int(row["future_step"]))
            if key not in deduped or row["current_quantity"] > deduped[key]["current_quantity"]:
                deduped[key] = row
        index[step] = sorted(deduped.values(), key=lambda row: (row["item"], row["future_step"]))
    return index


def _opportunity_features(item, current_step, future_step, current_quantity, future_quantity, obs=None):
    """Encode only bounded, observable timing context."""
    step = _as_int(current_step)
    future_step = _as_int(future_step)
    current_quantity = max(0, _as_int(current_quantity))
    future_quantity = max(0, _as_int(future_quantity))
    gap = max(0, future_step - step)
    day = step // 24
    hour = step % 24
    values = [
        1.0,
        step / 720.0,
        day / 30.0,
        hour / 24.0,
        min(1.0, current_quantity / 32.0),
        min(1.0, future_quantity / 32.0),
        max(-1.0, min(1.0, (future_quantity - current_quantity) / 32.0)),
        min(1.0, gap / 72.0),
    ]
    values.extend(1.0 if item == name else 0.0 for name in PREMIUM)
    bucket = min(23, max(0, step // 30))
    values.extend(1.0 if bucket == index else 0.0 for index in range(24))

    price = 0.0
    inventory = 0.0
    opponent_supply = 0.0
    money_gap = 0.0
    if isinstance(obs, dict):
        market = _get(obs, "market", {}) or {}
        prices = _get(market, "prices", {}) or {}
        market_inventory = _get(market, "inventory", {}) or {}
        price = _as_float(prices.get(item, 0)) / 300.0
        inventory = _as_float(market_inventory.get(item, 0)) / 10000.0
        farms = list(_get(obs, "farms", []) or [])
        seat = _as_int(_get(obs, "player", 0), 0)
        other = farms[1 - seat] if len(farms) > 1 else {}
        opponent_supply = _public_supply(other, item) / 100.0
        mine = farms[seat] if 0 <= seat < len(farms) else {}
        money_gap = (_as_float(_get(mine, "money", 0)) - _as_float(_get(other, "money", 0))) / 100000.0
    values.extend([
        min(2.0, price),
        min(2.0, inventory),
        min(2.0, opponent_supply),
        max(-2.0, min(2.0, money_gap)),
    ])
    if len(values) != FEATURE_DIM:
        raise AssertionError(f"timing feature size {len(values)} != {FEATURE_DIM}")
    return np.asarray(values, dtype=np.float64)


def _public_supply(farm, item):
    total = 0
    for row in _get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            if item == "MILK" and str(tile.get("animal", "")).upper() == "COW":
                total += max(0, _as_int(tile.get("yield_units", 0)))
            elif item == "WOOL" and str(tile.get("animal", "")).upper() == "SHEEP":
                total += max(0, _as_int(tile.get("yield_units", 0)))
            elif item == "EGG" and str(tile.get("animal", "")).upper() == "GOOSE":
                total += max(0, _as_int(tile.get("yield_units", 0)))
            elif str(tile.get("crop", "")).upper() == item:
                total += max(0, _as_int(tile.get("yield_units", 0)))
    return total


class TimingQ:
    """Linear item-level Q function for an event-level terminal reward."""

    def __init__(self, weights=None, threshold=0.0):
        payload = weights or {}
        self.weights = np.asarray(
            payload.get("weights", np.zeros((ACTION_COUNT, FEATURE_DIM))), dtype=np.float64
        )
        if self.weights.shape != (ACTION_COUNT, FEATURE_DIM):
            raise ValueError("timing weights must have shape (5, 40)")
        self.threshold = float(payload.get("threshold", threshold))

    def values(self, features):
        return self.weights @ np.asarray(features, dtype=np.float64)

    def select(self, features, item):
        values = self.values(features)
        action = ACTION_FOR_ITEM.get(str(item).upper())
        if action is None or values[action] <= self.threshold:
            return ACTION_CONTROL
        return int(action)

    def payload(self):
        return {"weights": self.weights.tolist(), "threshold": self.threshold}


def _copy_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(item or ["PASS"]) for item in action.get("hands", []) or []],
        "market": [list(item) for item in action.get("market", []) or [] if isinstance(item, list)],
    }


def _align_hands(action, obs):
    action = _copy_action(action)
    farms = list(_get(obs, "farms", []) or [])
    seat = _as_int(_get(obs, "player", 0), 0)
    expected = len(_get(farms[seat], "hands", []) or []) if 0 <= seat < len(farms) else 0
    hands = list(action.get("hands", []) or [])
    hands.extend([["PASS"] for _ in range(max(0, expected - len(hands)))])
    action["hands"] = hands[:expected]
    return action


def _adjust_sell(action, item, delta):
    for index, order in enumerate(action.get("market", []) or []):
        if (
            len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == str(item).upper()
        ):
            current = max(0, _as_int(order[2]))
            updated = current + int(delta)
            if updated < 0:
                return 0
            if updated == 0:
                action["market"].pop(index)
            else:
                action["market"][index] = [order[0], order[1], updated]
            return abs(int(delta))
    return 0


class TimingRuntime:
    """Stateful event-level policy used by the local runner and submission."""

    def __init__(self, weights=None, opportunities=None, seed=0):
        del seed
        self.policy = TimingQ(weights)
        self.opportunities = _opportunity_index(opportunities)
        self.pending = defaultdict(int)
        self.last_step = -1
        self.delayed_orders = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.decisions = []
        self.errors = 0

    def reset(self):
        self.pending.clear()
        self.last_step = -1
        self.delayed_orders = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.decisions = []

    def _apply_pending(self, action, step):
        changed = 0
        for key in [key for key in self.pending if key[0] == step]:
            _, item = key
            quantity = self.pending.pop(key, 0)
            if quantity:
                changed += _adjust_sell(action, item, quantity)
        return changed

    def act(self, obs, base_action):
        step = _as_int(_get(obs, "step", 0), 0)
        if step == 0 or step < self.last_step:
            self.reset()
        action = _align_hands(base_action, obs)
        changed = self._apply_pending(action, step)
        opportunities = self.opportunities.get(step, [])
        if opportunities and self.delayed_orders < MAX_DELAYED_ORDERS:
            scored = []
            for opportunity in opportunities:
                features = _opportunity_features(
                    opportunity["item"],
                    opportunity["current_step"],
                    opportunity["future_step"],
                    opportunity["current_quantity"],
                    opportunity["future_quantity"],
                    obs,
                )
                action_id = ACTION_FOR_ITEM[opportunity["item"]]
                scored.append((
                    float(self.policy.values(features)[action_id]),
                    opportunity,
                    features,
                    action_id,
                ))
            predicted_value, opportunity, features, action_id = max(scored, key=lambda row: row[0])
            decision = self.policy.select(features, opportunity["item"])
            moved = 0
            if decision == action_id:
                moved = _adjust_sell(action, opportunity["item"], -1)
                if moved:
                    self.pending[(opportunity["future_step"], opportunity["item"])] += moved
                    self.delayed_orders += 1
                    changed += moved
            self.decisions.append({
                "step": step,
                "item": opportunity["item"],
                "future_step": opportunity["future_step"],
                "action": int(decision),
                "predicted_delay_value": predicted_value,
                "moved": int(moved),
            })
        if changed:
            self.changed_calls += 1
            self.changed_units += int(changed)
        self.last_step = step
        return action


def fit_timing_weights(samples, ridge=10.0, threshold_grid=(-50, -20, -10, 0, 5, 10, 20, 50, 100)):
    """Fit Q(delay|context) from paired cash-delta interventions."""
    if not samples:
        raise ValueError("no timing samples")
    matrix = np.asarray([row["features"] for row in samples], dtype=np.float64)
    target = np.asarray([row["cash_delta"] for row in samples], dtype=np.float64)
    weights = np.zeros((ACTION_COUNT, FEATURE_DIM), dtype=np.float64)
    predictions = np.zeros(len(samples), dtype=np.float64)
    item_reports = {}
    for item, action in ACTION_FOR_ITEM.items():
        indices = [index for index, row in enumerate(samples) if row["item"] == item]
        if not indices:
            continue
        item_matrix = matrix[indices]
        item_target = target[indices]
        beta = np.linalg.solve(
            item_matrix.T @ item_matrix + np.eye(FEATURE_DIM, dtype=np.float64) * float(ridge),
            item_matrix.T @ item_target,
        )
        weights[action] = beta
        predictions[indices] = item_matrix @ beta
        item_reports[item] = {
            "samples": len(indices),
            "beta_norm": float(np.linalg.norm(beta)),
        }
    threshold_rows = []
    for threshold in threshold_grid:
        selected = target[predictions > float(threshold)]
        threshold_rows.append({
            "threshold": float(threshold),
            "selected": int(len(selected)),
            "mean_delta": float(selected.mean()) if len(selected) else 0.0,
            "min_delta": float(selected.min()) if len(selected) else 0.0,
            "positive_rate": float(np.mean(selected > 0)) if len(selected) else 0.0,
        })
    viable = [row for row in threshold_rows if row["selected"] >= max(3, len(samples) // 8)]
    chosen = max(viable or threshold_rows, key=lambda row: (row["mean_delta"], row["min_delta"]))
    report = {
        "samples": len(samples),
        "mean_cash_delta": float(target.mean()),
        "median_cash_delta": float(statistics.median(target.tolist())),
        "min_cash_delta": float(target.min()),
        "max_cash_delta": float(target.max()),
        "item_models": item_reports,
        "threshold_grid": threshold_rows,
        "chosen_threshold": chosen,
    }
    return TimingQ({"weights": weights.tolist(), "threshold": chosen["threshold"]}), report


def build_samples_from_v029(root=ROOT, train_seeds=(17, 42, 2026)):
    """Turn completed V029 counterfactual games into causal timing samples."""
    control = {}
    control_paths = (
        root / "baseline/artifacts/v028_order_search/v22_3seed/matrix_raw.csv",
        root / "baseline/artifacts/v029_v22_quantity_counterfactual/v22_control_holdout/matrix_raw.csv",
    )
    for path in control_paths:
        if not path.exists():
            continue
        for row in csv.DictReader(path.open("r", encoding="utf-8")):
            seed = _as_int(row.get("seed"))
            seat = _as_int(row.get("seat"))
            if seed in train_seeds:
                control[(seed, seat)] = _as_float(row.get("candidate_money"))

    opportunities = {
        (row["item"], int(row["current_step"]), int(row["future_step"])): row
        for row in route_opportunities()
    }
    rows = []
    seen = set()
    artifact_root = root / "baseline/artifacts/v029_v22_quantity_counterfactual"
    for path in sorted(artifact_root.rglob("raw.csv")):
        if "v22_control" in str(path):
            continue
        for row in csv.DictReader(path.open("r", encoding="utf-8")):
            if str(row.get("direction", "")) != "delay":
                continue
            seed = _as_int(row.get("seed"))
            seat = _as_int(row.get("seat"))
            item = str(row.get("item", "")).upper()
            if seed not in train_seeds or item not in PREMIUM:
                continue
            key = (item, _as_int(row.get("current_step")), _as_int(row.get("future_step")), seed, seat)
            if key in seen or (seed, seat) not in control:
                continue
            opportunity = opportunities.get(key[:3])
            if opportunity is None:
                continue
            seen.add(key)
            candidate_money = _as_float(row.get("candidate_money"))
            samples = {
                "item": item,
                "current_step": opportunity["current_step"],
                "future_step": opportunity["future_step"],
                "current_quantity": opportunity["current_quantity"],
                "future_quantity": opportunity["future_quantity"],
                "seed": seed,
                "seat": seat,
                "cash_delta": candidate_money - control[(seed, seat)],
            }
            samples["features"] = _opportunity_features(
                item,
                samples["current_step"],
                samples["future_step"],
                samples["current_quantity"],
                samples["future_quantity"],
            ).tolist()
            rows.append(samples)
    return rows


def save_fit(output, q, report, samples):
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights.json").write_text(json.dumps(q.payload(), indent=2) + "\n", encoding="utf-8")
    (output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/rl_003_trade_timing")
    parser.add_argument("--train-seed", action="append", type=int, default=None)
    args = parser.parse_args()
    seeds = tuple(args.train_seed or (17, 42, 2026))
    samples = build_samples_from_v029(ROOT, seeds)
    q, report = fit_timing_weights(samples)
    save_fit(args.output, q, report, samples)
    print(json.dumps({"samples": len(samples), "weights": str(args.output / "weights.json"), "report": report}, indent=2))
