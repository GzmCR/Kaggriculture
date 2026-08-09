"""RL-007: learn premium SELL preemption by turn distance.

The V022c production route stays fixed.  A candidate adds a bounded premium
SELL one, two, or three turns before an existing V022c premium sale, then
removes the same quantity from that original sale.  This is deliberately
different from RL-006, which moves quantity between two existing sale waves.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

try:
    from rl_006_bidirectional_timing import (
        RL006_FEATURE_DIM,
        RL006History,
        rl006_align_hands,
        rl006_event_key,
        rl006_features,
        rl006_int,
        rl006_normalize_action,
        rl006_private_inventory,
        rl006_step,
    )
except ImportError:
    # The builder concatenates RL-006 before this module for Kaggle's
    # self-contained loader; local runners import the module normally.
    pass


RL007_ACTION_NAMES = {
    0: "CONTROL",
    1: "PREEMPT_H1",
    2: "PREEMPT_H2",
    3: "PREEMPT_H3",
}
RL007_ACTIONS = (1, 2, 3)
RL007_HORIZONS = {1: 1, 2: 2, 3: 3}
RL007_MIN_SUPPORT = 12
RL007_MIN_EXPECTED_DELTA = 5.0
RL007_LCB_Z = 1.5
RL007_CUTOFF = 648
RL007_MAX_BATCH = 30
RL007_PREMIUM = {"MILK", "WOOL", "STRAWBERRY", "MELON"}
RL007_FEATURE_DIM = 78


def rl007_event_key(opportunity):
    return "{}|{}|{}|{}".format(
        str(opportunity["item"]).upper(),
        rl006_int(opportunity["current_step"]),
        rl006_int(opportunity["future_step"]),
        rl006_int(opportunity["horizon"]),
    )


def rl007_route_opportunities(actions):
    """Create one opportunity for each premium sale and h=1,2,3."""
    route_sales = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            if str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item not in RL007_PREMIUM:
                continue
            quantity = max(0, rl006_int(order[2]))
            if quantity:
                route_sales[item][int(step)] = quantity
    opportunities = []
    for item, rows in route_sales.items():
        for future_step, future_quantity in sorted(rows.items()):
            for horizon in (1, 2, 3):
                current_step = future_step - horizon
                if current_step < 0 or current_step >= RL007_CUTOFF:
                    continue
                opportunities.append({
                    "item": item,
                    "current_step": int(current_step),
                    "future_step": int(future_step),
                    "current_quantity": 0,
                    "future_quantity": int(future_quantity),
                    "gap": int(horizon),
                    "horizon": int(horizon),
                })
    return sorted(opportunities, key=lambda row: (row["current_step"], row["item"], row["horizon"]))


def rl007_action_horizon(action_id):
    return RL007_HORIZONS.get(int(action_id), 0)


def rl007_shift_quantity(opportunity):
    return min(
        RL007_MAX_BATCH,
        max(0, rl006_int(opportunity.get("future_quantity", 0))),
    )


def rl007_append_sell(action, item, quantity):
    item = str(item).upper()
    quantity = max(0, rl006_int(quantity))
    if quantity <= 0 or len(action.get("market", []) or []) >= 10:
        return 0
    action.setdefault("market", []).append(["SELL", item, quantity])
    return quantity


def rl007_reduce_sell(action, item, quantity):
    item = str(item).upper()
    quantity = max(0, rl006_int(quantity))
    for index, order in enumerate(action.get("market", []) or []):
        if len(order) < 3 or str(order[0]).upper() != "SELL" or str(order[1]).upper() != item:
            continue
        available = max(0, rl006_int(order[2]))
        if available < quantity:
            return 0
        remaining = available - quantity
        if remaining:
            action["market"][index] = [order[0], order[1], remaining]
        else:
            action["market"].pop(index)
        return quantity
    return 0


class RL007Policy:
    def __init__(self, payload=None):
        payload = payload or {}
        self.feature_dim = int(payload.get("feature_dim", RL007_FEATURE_DIM))
        self.min_support = int(payload.get("min_support", RL007_MIN_SUPPORT))
        self.min_expected_delta = float(payload.get("min_expected_delta", RL007_MIN_EXPECTED_DELTA))
        self.lcb_z = float(payload.get("lcb_z", RL007_LCB_Z))
        self.feature_mean = np.asarray(payload.get("feature_mean", [0.0] * self.feature_dim), dtype=np.float64)
        self.feature_scale = np.asarray(payload.get("feature_scale", [1.0] * self.feature_dim), dtype=np.float64)
        self.models = dict(payload.get("models", {}))
        self.allowed_actions = tuple(int(value) for value in payload.get("allowed_actions", RL007_ACTIONS))
        self.supported_events = set(str(value) for value in payload.get("supported_events", []))
        if self.feature_dim != RL007_FEATURE_DIM:
            raise ValueError("RL-007 feature dimension mismatch")

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


class RL007Runtime:
    def __init__(self, payload=None, opportunities=None):
        self.policy = RL007Policy(payload)
        self.index = defaultdict(list)
        for opportunity in opportunities or []:
            self.index[rl006_int(opportunity["current_step"])].append(opportunity)
        self.history = RL006History()
        self.pending = {}
        self.last_step = -1
        self.changed_calls = 0
        self.changed_units = 0
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
        self.fallbacks = 0
        self.errors = 0
        self.last_error = ""
        self.decisions = []

    def _repay(self, action, step):
        changed = 0
        for key in [key for key in self.pending if key[0] == int(step)]:
            _, item = key
            debt = self.pending.pop(key)
            quantity = max(0, rl006_int(debt.get("quantity", 0)))
            moved = rl007_reduce_sell(action, item, quantity)
            if moved != quantity:
                self.errors += 1
            changed += moved
        return changed

    def act(self, obs, base_action):
        step = rl006_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl006_align_hands(base_action, obs)
        changed = self._repay(action, step)
        if step < RL007_CUTOFF:
            candidates = self.index.get(step, [])
            scored = []
            for opportunity in candidates:
                event_key = rl007_event_key(opportunity)
                if self.policy.supported_events and event_key not in self.policy.supported_events:
                    continue
                item = opportunity["item"]
                quantity = rl007_shift_quantity(opportunity)
                if quantity <= 0 or len(action.get("market", []) or []) >= 10:
                    continue
                if any(value.get("item") == item for value in self.pending.values()):
                    continue
                if rl006_private_inventory(obs, item) < quantity:
                    continue
                for action_id in self.policy.allowed_actions:
                    if int(action_id) not in RL007_ACTIONS:
                        continue
                    features = rl006_features(obs, opportunity, self.history, action, self.pending)
                    result = self.policy.predict(action_id, features)
                    if result is None:
                        continue
                    if result["prediction"] < self.policy.min_expected_delta or result["lcb"] <= 0.0:
                        continue
                    scored.append((float(result["lcb"]), opportunity, int(action_id), result, features))
            if scored:
                _, opportunity, action_id, result, _ = max(scored, key=lambda row: row[0])
                item = str(opportunity["item"]).upper()
                quantity = rl007_shift_quantity(opportunity)
                moved = rl007_append_sell(action, item, quantity)
                if moved == quantity:
                    key = (int(opportunity["future_step"]), item)
                    self.pending[key] = {
                        "item": item,
                        "quantity": int(quantity),
                        "future_step": int(opportunity["future_step"]),
                        "current_step": int(step),
                        "horizon": int(opportunity["horizon"]),
                    }
                    self.decisions.append({
                        "step": int(step),
                        "future_step": int(opportunity["future_step"]),
                        "horizon": int(opportunity["horizon"]),
                        "item": item,
                        "action": RL007_ACTION_NAMES.get(action_id, "UNKNOWN"),
                        "moved": int(moved),
                        "prediction": float(result["prediction"]),
                        "uncertainty": float(result["uncertainty"]),
                        "lcb": float(result["lcb"]),
                        "support": int(result["support"]),
                    })
                    changed += moved
                else:
                    self.fallbacks += 1
            elif candidates:
                self.fallbacks += 1
        if changed:
            self.changed_calls += 1
            self.changed_units += int(changed)
        self.last_step = step
        return action


def rl007_fit_models(samples, ridge=12.0, min_support=RL007_MIN_SUPPORT):
    if not samples:
        raise ValueError("RL-007 requires counterfactual samples")
    # A candidate that could not append its SELL is not a counterfactual
    # intervention.  Keep it in the collection log, but never fit it as a
    # zero-reward example.
    samples = [row for row in samples if int(row.get("shift_applied", 1))]
    if not samples:
        raise ValueError("RL-007 has no successfully applied counterfactual samples")
    matrix = np.asarray([row["features"] for row in samples], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != RL007_FEATURE_DIM:
        raise ValueError(f"invalid RL-007 feature matrix: {matrix.shape}")
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
    rng = np.random.default_rng(7007)
    for action_key, indexed_rows in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        if int(action_key) not in RL007_ACTIONS:
            continue
        rows = [row for _, row in indexed_rows]
        support_keys = {
            (row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", "")))
            for row in rows
        }
        support = len(support_keys)
        if support < int(min_support):
            report["skipped_actions"][action_key] = {"rows": len(rows), "support": support}
            continue
        indices = [index for index, _ in indexed_rows]
        x = normalized[indices]
        target = np.asarray([float(row.get("cash_delta", 0.0)) for row in rows], dtype=np.float64)
        design = np.column_stack((np.ones(len(rows), dtype=np.float64), x))
        penalty = np.eye(RL007_FEATURE_DIM + 1, dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        residual = target - design @ coefficients
        group_values = defaultdict(list)
        for row, value in zip(rows, residual):
            key = (row.get("seed"), row.get("seat"), row.get("opponent_source_sha256", row.get("opponent", "")))
            group_values[key].append(float(value))
        group_means = np.asarray([np.mean(values) for values in group_values.values()], dtype=np.float64)
        bootstrap = []
        if len(group_means) > 1:
            for _ in range(128):
                bootstrap.append(float(np.mean(rng.choice(group_means, size=len(group_means), replace=True))))
        residual_std = float(np.std(residual))
        bootstrap_std = float(np.std(bootstrap)) if bootstrap else 0.0
        uncertainty = max(1.0, residual_std * (1.0 + 1.0 / math.sqrt(max(1, support))), bootstrap_std)
        models[action_key] = {
            "support": support,
            "rows": len(rows),
            "intercept": float(coefficients[0]),
            "beta": coefficients[1:].tolist(),
            "uncertainty": uncertainty,
            "mean_delta": float(target.mean()),
            "min_delta": float(target.min()),
            "positive_rate": float(np.mean(target > 0)),
            "residual_std": residual_std,
            "bootstrap_std": bootstrap_std,
        }
        report["actions"][action_key] = {
            **models[action_key],
            "name": RL007_ACTION_NAMES.get(int(action_key), action_key),
        }
    supported_events = sorted({
        rl007_event_key(row) for row in samples
    })
    report.update({
        "samples": len(samples),
        "models": len(models),
        "feature_dim": RL007_FEATURE_DIM,
        "mean_cash_delta": float(np.mean([float(row.get("cash_delta", 0.0)) for row in samples])),
    })
    payload = {
        "version": "rl007_next_turn_preemption",
        "feature_dim": RL007_FEATURE_DIM,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "min_support": int(min_support),
        "min_expected_delta": RL007_MIN_EXPECTED_DELTA,
        "lcb_z": RL007_LCB_Z,
        "allowed_actions": list(RL007_ACTIONS),
        "supported_events": supported_events,
        "models": models,
    }
    return payload, report


def rl007_load_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
