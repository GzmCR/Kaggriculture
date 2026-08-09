"""RL-008 small-quantity bidirectional timing overlay.

V022c owns production and the base market route.  This module only moves one
small premium SELL quantity at a time and keeps an exact repayment debt.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

try:
    from rl_006_bidirectional_timing import (
        RL006History,
        rl006_adjust_sell,
        rl006_align_hands,
        rl006_clone_distance,
        rl006_features,
        rl006_get,
        rl006_int,
        rl006_normalize_action,
        rl006_private_inventory,
        rl006_public_signature,
        rl006_step,
    )
except ImportError:
    pass


RL008_ACTION_NAMES = {
    0: "CONTROL",
    1: "PREEMPT_H1_1",
    2: "PREEMPT_H1_25",
    3: "PREEMPT_H2_1",
    4: "PREEMPT_H2_25",
    5: "PREEMPT_H3_1",
    6: "PREEMPT_H3_25",
    7: "DELAY_1",
    8: "DELAY_25",
    9: "DELAY_50",
}
RL008_PREEMPT_ACTIONS = tuple(range(1, 7))
RL008_DELAY_ACTIONS = (7, 8, 9)
RL008_NONCONTROL = tuple(range(1, 10))
RL008_FEATURE_DIM = 78
RL008_MIN_SUPPORT = 12
RL008_MIN_EXPECTED_DELTA = 5.0
RL008_LCB_Z = 1.5
RL008_CUTOFF = 648
RL008_PREMIUM = {"MILK", "WOOL", "STRAWBERRY", "MELON"}


def rl008_event_key(opportunity):
    return "{}|{}|{}|{}|{}".format(
        str(opportunity["kind"]).upper(),
        str(opportunity["item"]).upper(),
        rl006_int(opportunity["current_step"]),
        rl006_int(opportunity["future_step"]),
        rl006_int(opportunity.get("horizon", 0)),
    )


def rl008_preempt_opportunities(actions, target_sales=None):
    sales = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            if str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item not in RL008_PREMIUM:
                continue
            quantity = max(0, rl006_int(order[2]))
            if quantity:
                sales[item][int(step)] = sales[item].get(int(step), 0) + quantity
    allowed = set((str(item).upper(), int(step)) for item, step in (target_sales or ()))
    rows = []
    for item, item_sales in sales.items():
        for future_step, future_quantity in sorted(item_sales.items()):
            if allowed and (item, future_step) not in allowed:
                continue
            for horizon in (1, 2, 3):
                current_step = future_step - horizon
                if 0 <= current_step < RL008_CUTOFF:
                    rows.append({
                        "kind": "PREEMPT",
                        "item": item,
                        "current_step": current_step,
                        "future_step": int(future_step),
                        "current_quantity": 0,
                        "future_quantity": int(future_quantity),
                        "gap": horizon,
                        "horizon": horizon,
                    })
    return rows


def rl008_delay_opportunities(actions, requested_events=None):
    from_rl006 = []
    events = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            if str(order[0]).upper() != "SELL" or str(order[1]).upper() not in RL008_PREMIUM:
                continue
            item = str(order[1]).upper()
            events[item][int(step)] = events[item].get(int(step), 0) + max(0, rl006_int(order[2]))
    allowed = set((str(item).upper(), int(step)) for item, step in (requested_events or ()))
    for item, rows in events.items():
        ordered = sorted(rows.items())
        for index, (current_step, current_quantity) in enumerate(ordered[:-1]):
            future_step, future_quantity = ordered[index + 1]
            if not (4 <= future_step - current_step <= 72):
                continue
            if allowed and (item, current_step) not in allowed:
                continue
            from_rl006.append({
                "kind": "DELAY",
                "item": item,
                "current_step": int(current_step),
                "future_step": int(future_step),
                "current_quantity": int(current_quantity),
                "future_quantity": int(future_quantity),
                "gap": int(future_step - current_step),
                "horizon": int(future_step - current_step),
            })
    return from_rl006


def rl008_action_kind(action_id):
    if int(action_id) == 0:
        return "CONTROL"
    return "DELAY" if int(action_id) in RL008_DELAY_ACTIONS else "PREEMPT"


def rl008_action_quantity(action_id, opportunity):
    action_id = int(action_id)
    current = max(0, rl006_int(opportunity.get("current_quantity", 0)))
    future = max(0, rl006_int(opportunity.get("future_quantity", 0)))
    if action_id in (1, 3, 5, 7):
        return 1
    if action_id in (2, 4, 6):
        return min(future, max(1, int(math.ceil(future * 0.25))))
    if action_id == 8:
        return min(current, max(1, int(math.ceil(current * 0.25))))
    if action_id == 9:
        return min(current, max(1, int(math.ceil(current * 0.50))))
    return 0


def rl008_action_matches(action_id, opportunity):
    kind = str(opportunity.get("kind", "")).upper()
    if kind == "DELAY":
        return int(action_id) in RL008_DELAY_ACTIONS
    horizon = rl006_int(opportunity.get("horizon", 0))
    return {
        1: (1,), 2: (1,),
        3: (2,), 4: (2,),
        5: (3,), 6: (3,),
    }.get(int(action_id), ()) == (horizon,)


def rl008_clone_units(obs):
    farms = list(rl006_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10**9
    left = rl006_public_signature(farms[0])
    right = rl006_public_signature(farms[1])
    weights = {"hands": 2, "quadrants": 3, "COW": 2, "SHEEP": 2, "GOOSE": 1}
    return sum(weights.get(key, 1) * abs(int(value) - int(right.get(key, 0))) for key, value in left.items())


def rl008_gate(obs, action, opportunity, quantity):
    market = rl006_get(obs, "market", {}) or {}
    prices = rl006_get(market, "prices", {}) or {}
    item = str(opportunity["item"]).upper()
    if rl008_clone_units(obs) > 6:
        return False
    if float(rl006_get(prices, item, 0) or 0) <= 1:
        return False
    if len(action.get("market", []) or []) >= 10:
        return False
    if rl006_private_inventory(obs, item) < quantity:
        return False
    return True


def rl008_append_sell(action, item, quantity):
    if quantity <= 0 or len(action.get("market", []) or []) >= 10:
        return 0
    action.setdefault("market", []).append(["SELL", str(item).upper(), int(quantity)])
    return int(quantity)


def rl008_shift_sell(action, item, delta):
    """Apply an exact signed quantity across existing same-item SELL orders."""
    item = str(item).upper()
    delta = int(delta)
    if delta == 0:
        return 0
    market = list(action.get("market", []) or [])
    if delta > 0:
        for order in market:
            if len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == item:
                order[2] = max(0, rl006_int(order[2])) + delta
                action["market"] = market
                return delta
        return rl008_append_sell(action, item, delta)

    need = -delta
    available = sum(
        max(0, rl006_int(order[2]))
        for order in market
        if len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == item
    )
    if available < need:
        return 0
    remaining = need
    updated = []
    for order in market:
        if remaining and len(order) >= 3 and str(order[0]).upper() == "SELL" and str(order[1]).upper() == item:
            quantity = max(0, rl006_int(order[2]))
            take = min(quantity, remaining)
            quantity -= take
            remaining -= take
            if quantity <= 0:
                continue
            updated.append([order[0], order[1], quantity])
        else:
            updated.append(order)
    if remaining:
        return 0
    action["market"] = updated
    return need


class RL008Policy:
    def __init__(self, payload=None):
        payload = payload or {}
        self.feature_dim = int(payload.get("feature_dim", RL008_FEATURE_DIM))
        self.min_support = int(payload.get("min_support", RL008_MIN_SUPPORT))
        self.min_expected_delta = float(payload.get("min_expected_delta", RL008_MIN_EXPECTED_DELTA))
        self.lcb_z = float(payload.get("lcb_z", RL008_LCB_Z))
        self.feature_mean = np.asarray(payload.get("feature_mean", [0.0] * self.feature_dim), dtype=np.float64)
        self.feature_scale = np.asarray(payload.get("feature_scale", [1.0] * self.feature_dim), dtype=np.float64)
        self.models = dict(payload.get("models", {}))
        self.allowed_actions = tuple(int(value) for value in payload.get("allowed_actions", RL008_NONCONTROL))
        self.supported_events = set(str(value) for value in payload.get("supported_events", []))
        self.gate_mode = str(payload.get("gate_mode", "ungated"))
        if self.feature_dim != RL008_FEATURE_DIM:
            raise ValueError("RL-008 feature dimension mismatch")

    def predict(self, action_id, features):
        model = self.models.get(str(int(action_id)))
        if not model or int(model.get("support", 0)) < self.min_support:
            return None
        x = (np.asarray(features, dtype=np.float64) - self.feature_mean) / np.maximum(self.feature_scale, 1e-9)
        beta = np.asarray(model.get("beta", [0.0] * self.feature_dim), dtype=np.float64)
        prediction = float(model.get("intercept", 0.0) + x @ beta)
        uncertainty = max(1.0, float(model.get("uncertainty", 1.0)))
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "lcb": prediction - self.lcb_z * uncertainty,
            "support": int(model.get("support", 0)),
        }


class RL008Runtime:
    def __init__(self, payload=None, opportunities=None):
        self.policy = RL008Policy(payload)
        self.index = defaultdict(list)
        for opportunity in opportunities or []:
            self.index[rl006_int(opportunity["current_step"])].append(opportunity)
        self.history = RL006History()
        self.pending = {}
        self.last_step = -1
        self.changed_calls = 0
        self.changed_units = 0
        self.preempt_units = 0
        self.delay_units = 0
        self.fallbacks = 0
        self.errors = 0
        self.last_error = ""
        self.decisions = []

    def reset(self):
        self.history.reset()
        self.pending.clear()
        self.last_step = -1
        self.changed_calls = 0
        self.changed_units = 0
        self.preempt_units = 0
        self.delay_units = 0
        self.fallbacks = 0
        self.errors = 0
        self.last_error = ""
        self.decisions = []

    def _repay(self, action, step):
        changed = 0
        for key in [key for key in self.pending if key[0] == int(step)]:
            _, item = key
            debt = self.pending.pop(key)
            kind = str(debt.get("kind", "PREEMPT")).upper()
            quantity = max(0, rl006_int(debt.get("quantity", 0)))
            delta = -quantity if kind == "PREEMPT" else quantity
            moved = rl008_shift_sell(action, item, delta)
            if moved != abs(delta):
                self.errors += 1
                self.fallbacks += 1
                continue
            self.decisions.append({
                "step": int(step), "future_step": int(step), "item": item,
                "kind": kind, "action": "REPAY", "moved": int(moved),
            })
            changed += moved
        return changed

    def act(self, obs, base_action):
        step = rl006_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl006_align_hands(base_action, obs)
        changed = self._repay(action, step)
        if step < RL008_CUTOFF:
            candidates = self.index.get(step, [])
            scored = []
            for opportunity in candidates:
                event_key = rl008_event_key(opportunity)
                if self.policy.supported_events and event_key not in self.policy.supported_events:
                    continue
                item = str(opportunity["item"]).upper()
                quantity = rl008_action_quantity(7 if opportunity["kind"] == "DELAY" else 1, opportunity)
                if quantity <= 0 or any(value.get("item") == item for value in self.pending.values()):
                    continue
                for action_id in self.policy.allowed_actions:
                    action_id = int(action_id)
                    if not rl008_action_matches(action_id, opportunity):
                        continue
                    quantity = rl008_action_quantity(action_id, opportunity)
                    if quantity <= 0 or len(action.get("market", []) or []) >= 10:
                        continue
                    if opportunity["kind"] == "PREEMPT":
                        if self.policy.gate_mode == "gated" and not rl008_gate(obs, action, opportunity, quantity):
                            continue
                        if rl006_private_inventory(obs, item) < quantity:
                            continue
                    else:
                        if rl006_int(opportunity.get("current_quantity", 0)) < quantity:
                            continue
                    features = rl006_features(obs, opportunity, self.history, action, self.pending)
                    result = self.policy.predict(action_id, features)
                    if result is None or result["prediction"] < self.policy.min_expected_delta or result["lcb"] <= 0:
                        continue
                    scored.append((float(result["lcb"]), opportunity, action_id, quantity, result))
            if scored:
                _, opportunity, action_id, quantity, result = max(scored, key=lambda row: row[0])
                item = str(opportunity["item"]).upper()
                kind = str(opportunity["kind"]).upper()
                if kind == "PREEMPT":
                    moved = rl008_append_sell(action, item, quantity)
                else:
                    moved = rl008_shift_sell(action, item, -quantity)
                if abs(moved) == quantity:
                    key = (int(opportunity["future_step"]), item)
                    self.pending[key] = {
                        "item": item, "quantity": quantity, "kind": kind,
                        "due_step": int(opportunity["future_step"]),
                        "current_step": int(step),
                    }
                    self.decisions.append({
                        "step": int(step), "future_step": int(opportunity["future_step"]),
                        "item": item, "kind": kind, "action": RL008_ACTION_NAMES[action_id],
                        "moved": int(quantity), "prediction": float(result["prediction"]),
                        "uncertainty": float(result["uncertainty"]), "lcb": float(result["lcb"]),
                    })
                    changed += quantity
                    if kind == "PREEMPT":
                        self.preempt_units += quantity
                    else:
                        self.delay_units += quantity
                else:
                    self.fallbacks += 1
            elif candidates:
                self.fallbacks += 1
        if changed:
            self.changed_calls += 1
            self.changed_units += int(changed)
        self.last_step = step
        return action


def rl008_fit_models(samples, ridge=12.0, min_support=RL008_MIN_SUPPORT):
    samples = [row for row in samples if int(row.get("shift_applied", 1)) and int(row.get("future_repaid", 1))]
    if not samples:
        raise ValueError("RL-008 has no valid applied samples")
    matrix = np.asarray([row["features"] for row in samples], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != RL008_FEATURE_DIM:
        raise ValueError(f"invalid RL-008 feature matrix: {matrix.shape}")
    feature_mean = matrix.mean(axis=0)
    feature_scale = np.where(matrix.std(axis=0) < 1e-9, 1.0, matrix.std(axis=0))
    feature_mean[0] = 0.0
    feature_scale[0] = 1.0
    normalized = (matrix - feature_mean) / feature_scale
    grouped = defaultdict(list)
    for index, row in enumerate(samples):
        grouped[str(rl006_int(row["action_id"]))].append((index, row))
    models = {}
    report = {"actions": {}, "skipped_actions": {}}
    rng = np.random.default_rng(8008)
    for key, indexed in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        action_id = int(key)
        if action_id not in RL008_NONCONTROL:
            continue
        rows = [row for _, row in indexed]
        support_keys = {(row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", ""))) for row in rows}
        support = len(support_keys)
        if support < int(min_support):
            report["skipped_actions"][key] = {"rows": len(rows), "support": support}
            continue
        indices = [index for index, _ in indexed]
        x = normalized[indices]
        target = np.asarray([float(row.get("cash_delta", 0.0)) for row in rows], dtype=np.float64)
        design = np.column_stack((np.ones(len(rows)), x))
        penalty = np.eye(RL008_FEATURE_DIM + 1) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        residual = target - design @ coefficients
        groups = defaultdict(list)
        for row, value in zip(rows, residual):
            groups[(row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", "")))].append(float(value))
        group_means = np.asarray([np.mean(values) for values in groups.values()])
        boot = [float(np.mean(rng.choice(group_means, size=len(group_means), replace=True))) for _ in range(128)] if len(group_means) > 1 else []
        residual_std = float(np.std(residual))
        bootstrap_std = float(np.std(boot)) if boot else 0.0
        uncertainty = max(1.0, residual_std * (1.0 + 1.0 / math.sqrt(max(1, support))), bootstrap_std)
        models[key] = {
            "support": support, "rows": len(rows), "intercept": float(coefficients[0]),
            "beta": coefficients[1:].tolist(), "uncertainty": uncertainty,
            "mean_delta": float(target.mean()), "min_delta": float(target.min()),
            "positive_rate": float(np.mean(target > 0)), "residual_std": residual_std,
            "bootstrap_std": bootstrap_std,
        }
        report["actions"][key] = {**models[key], "name": RL008_ACTION_NAMES[action_id]}
    report.update({
        "samples": len(samples), "models": len(models), "feature_dim": RL008_FEATURE_DIM,
        "mean_cash_delta": float(np.mean([float(row.get("cash_delta", 0.0)) for row in samples])),
    })
    payload = {
        "version": "rl008_small_shift_timing", "feature_dim": RL008_FEATURE_DIM,
        "feature_mean": feature_mean.tolist(), "feature_scale": feature_scale.tolist(),
        "min_support": int(min_support), "min_expected_delta": RL008_MIN_EXPECTED_DELTA,
        "lcb_z": RL008_LCB_Z, "allowed_actions": list(RL008_NONCONTROL),
        "supported_events": sorted({rl008_event_key(row) for row in samples}), "models": models,
        "gate_mode": "ungated",
    }
    return payload, report


def rl008_load_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
