"""Observation-aware, conservative sale-timing policy for RL-004.

The v22 farmer and hands route stays fixed. RL-004 only moves one unit from
an existing premium SELL event to the next same-product event when that exact
event has enough paired counterfactual support and a positive lower confidence
bound.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np


RL004_PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
RL004_SUPPORTED_ITEMS = ("MILK", "STRAWBERRY")
RL004_MIN_SUPPORT = 12
RL004_MIN_EXPECTED_DELTA = 5.0
RL004_LCB_Z = 1.5
RL004_MAX_DELAYED_ORDERS = 8
RL004_MIN_GAP = 4
RL004_MAX_GAP = 72
RL004_CUTOFF = 648
RL004_FEATURE_DIM = 29


def rl004_as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def rl004_as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def rl004_get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def rl004_step(obs):
    raw_step = rl004_get(obs, "step", None)
    if raw_step is not None:
        return max(0, rl004_as_int(raw_step))
    day = rl004_as_int(rl004_get(obs, "day", 0))
    hour = rl004_as_int(rl004_get(obs, "hour", 0))
    return max(0, day * 24 + hour)


def rl004_item_key(item, current_step, future_step):
    return f"{str(item).upper()}|{int(current_step)}|{int(future_step)}"


def rl004_route_opportunities(actions):
    events = defaultdict(dict)
    for step, action in enumerate(actions or []):
        for order in (action or {}).get("market", []) or []:
            if (
                not isinstance(order, (list, tuple))
                or len(order) < 3
                or str(order[0]).upper() != "SELL"
            ):
                continue
            item = str(order[1]).upper()
            if item not in RL004_PREMIUM:
                continue
            quantity = max(0, rl004_as_int(order[2]))
            if quantity:
                events[item][int(step)] = events[item].get(int(step), 0) + quantity
    opportunities = []
    for item, rows in events.items():
        ordered = sorted(rows.items())
        for index, (current_step, current_quantity) in enumerate(ordered[:-1]):
            future_step, future_quantity = ordered[index + 1]
            gap = future_step - current_step
            if RL004_MIN_GAP <= gap <= RL004_MAX_GAP:
                opportunities.append({
                    "item": item,
                    "current_step": int(current_step),
                    "future_step": int(future_step),
                    "current_quantity": int(current_quantity),
                    "future_quantity": int(future_quantity),
                    "gap": int(gap),
                })
    return sorted(opportunities, key=lambda row: (row["current_step"], row["item"]))


def rl004_opportunity_index(opportunities):
    index = defaultdict(list)
    for row in opportunities or []:
        if int(row["current_step"]) >= RL004_CUTOFF:
            continue
        if int(row["future_step"]) >= 672:
            continue
        index[int(row["current_step"])].append(row)
    return index


def rl004_public_supply(farm, item):
    total = 0
    for row in rl004_get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            animal = str(tile.get("animal", "")).upper()
            crop = str(tile.get("crop", "")).upper()
            if item == "MILK" and animal == "COW":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
            elif item == "WOOL" and animal == "SHEEP":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
            elif item == "MELON" and crop == "MELON":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
            elif item == "STRAWBERRY" and crop == "STRAWBERRY":
                total += max(0, rl004_as_int(tile.get("yield_units", 0)))
    return total


def rl004_private_inventory(obs, item):
    private = rl004_get(obs, "private", {}) or {}
    shed = rl004_get(private, "shed", {}) or {}
    carried = 0
    for inventory in rl004_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            carried += max(0, rl004_as_int(inventory.get(item, 0)))
    return max(0, rl004_as_int(shed.get(item, 0))) + carried


class RL004FeatureState:
    """Observable price and market-inventory history."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_step = -1
        self.prices = {item: [] for item in RL004_PREMIUM}
        self.inventories = {item: [] for item in RL004_PREMIUM}

    def observe(self, obs):
        step = rl004_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        market = rl004_get(obs, "market", {}) or {}
        prices = rl004_get(market, "prices", {}) or {}
        inventory = rl004_get(market, "inventory", {}) or {}
        for item in RL004_PREMIUM:
            self.prices[item].append((step, rl004_as_float(prices.get(item, 0))))
            self.inventories[item].append((step, rl004_as_float(inventory.get(item, 0))))
            self.prices[item] = self.prices[item][-96:]
            self.inventories[item] = self.inventories[item][-96:]
        self.last_step = step

    @staticmethod
    def _lagged(rows, step, lag):
        target = int(step) - int(lag)
        values = [value for seen_step, value in rows if seen_step <= target]
        return values[-1] if values else None

    def momentum(self, item, step):
        current_price = self.prices[item][-1][1] if self.prices[item] else 0.0
        current_inventory = self.inventories[item][-1][1] if self.inventories[item] else 0.0
        price12 = self._lagged(self.prices[item], step, 12)
        price24 = self._lagged(self.prices[item], step, 24)
        inv12 = self._lagged(self.inventories[item], step, 12)
        inv24 = self._lagged(self.inventories[item], step, 24)
        return (
            current_price,
            current_inventory,
            0.0 if price12 is None else current_price - price12,
            0.0 if price24 is None else current_price - price24,
            0.0 if inv12 is None else current_inventory - inv12,
            0.0 if inv24 is None else current_inventory - inv24,
        )


def rl004_features(obs, opportunity, history, base_action=None):
    item = str(opportunity["item"]).upper()
    step = rl004_as_int(opportunity["current_step"])
    future_step = rl004_as_int(opportunity["future_step"])
    current_quantity = max(0, rl004_as_int(opportunity["current_quantity"]))
    future_quantity = max(0, rl004_as_int(opportunity["future_quantity"]))
    gap = max(0, future_step - step)
    values = [
        1.0,
        step / 720.0,
        future_step / 720.0,
        (step // 24) / 30.0,
        (step % 24) / 24.0,
        min(1.0, current_quantity / 32.0),
        min(1.0, future_quantity / 32.0),
        max(-1.0, min(1.0, (future_quantity - current_quantity) / 32.0)),
        min(1.0, gap / 72.0),
    ]
    values.extend(1.0 if item == name else 0.0 for name in RL004_PREMIUM)

    price, market_inventory, price12, price24, inv12, inv24 = history.momentum(item, step)
    values.extend([
        min(2.0, price / 300.0),
        min(2.0, market_inventory / 10000.0),
        max(-2.0, min(2.0, price12 / 300.0)),
        max(-2.0, min(2.0, price24 / 300.0)),
        max(-2.0, min(2.0, inv12 / 10000.0)),
        max(-2.0, min(2.0, inv24 / 10000.0)),
    ])

    farms = list(rl004_get(obs, "farms", []) or [])
    seat = rl004_as_int(rl004_get(obs, "player", 0))
    mine = farms[seat] if 0 <= seat < len(farms) else {}
    other = farms[1 - seat] if len(farms) > 1 and seat in (0, 1) else {}
    mine_money = rl004_as_float(rl004_get(mine, "money", 0))
    other_money = rl004_as_float(rl004_get(other, "money", 0))
    market_orders = len((base_action or {}).get("market", []) or [])
    values.extend([
        min(2.0, rl004_private_inventory(obs, item) / 100.0),
        min(2.0, rl004_public_supply(mine, item) / 100.0),
        min(2.0, rl004_public_supply(other, item) / 100.0),
        max(-2.0, min(2.0, (mine_money - other_money) / 100000.0)),
        min(1.0, market_orders / 10.0),
        min(1.0, len(rl004_get(rl004_get(obs, "town", {}) or {}, "unlocked_shops", []) or []) / 8.0),
        min(1.0, len(rl004_get(mine, "hands", []) or []) / 20.0),
        min(1.0, len(rl004_get(other, "hands", []) or []) / 20.0),
        min(1.0, len(rl004_get(mine, "unlocked_quadrants", []) or []) / 4.0),
        min(1.0, len(rl004_get(other, "unlocked_quadrants", []) or []) / 4.0),
    ])
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (RL004_FEATURE_DIM,):
        raise AssertionError(f"RL004 feature size {array.size} != {RL004_FEATURE_DIM}")
    return array


def rl004_normalize_action(action):
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(value or ["PASS"]) for value in action.get("hands", []) or []],
        "market": [list(value) for value in action.get("market", []) or [] if isinstance(value, list)],
    }


def rl004_align_hands(action, obs):
    action = rl004_normalize_action(action)
    farms = list(rl004_get(obs, "farms", []) or [])
    seat = rl004_as_int(rl004_get(obs, "player", 0))
    expected = len(rl004_get(farms[seat], "hands", []) or []) if 0 <= seat < len(farms) else 0
    action["hands"].extend([["PASS"] for _ in range(max(0, expected - len(action["hands"])) )])
    action["hands"] = action["hands"][:expected]
    return action


def rl004_adjust_sell(action, item, delta):
    for index, order in enumerate(action.get("market", []) or []):
        if (
            len(order) >= 3
            and str(order[0]).upper() == "SELL"
            and str(order[1]).upper() == str(item).upper()
        ):
            current = max(0, rl004_as_int(order[2]))
            updated = current + int(delta)
            if updated < 0:
                return 0
            if updated == 0:
                action["market"].pop(index)
            else:
                action["market"][index] = [order[0], order[1], updated]
            return abs(int(delta))
    return 0


class RL004Policy:
    def __init__(self, payload=None):
        payload = payload or {}
        self.feature_dim = int(payload.get("feature_dim", RL004_FEATURE_DIM))
        self.min_support = int(payload.get("min_support", RL004_MIN_SUPPORT))
        self.min_expected_delta = float(payload.get("min_expected_delta", RL004_MIN_EXPECTED_DELTA))
        self.lcb_z = float(payload.get("lcb_z", RL004_LCB_Z))
        self.models = dict(payload.get("models", {}))
        if self.feature_dim != RL004_FEATURE_DIM:
            raise ValueError("RL004 feature dimension mismatch")

    def predict(self, key, features):
        model = self.models.get(key)
        if not model or int(model.get("support", 0)) < self.min_support:
            return None
        mean = np.asarray(model.get("mean", [0.0] * self.feature_dim), dtype=np.float64)
        scale = np.asarray(model.get("scale", [1.0] * self.feature_dim), dtype=np.float64)
        beta = np.asarray(model.get("beta", [0.0] * self.feature_dim), dtype=np.float64)
        x = (np.asarray(features, dtype=np.float64) - mean) / np.maximum(scale, 1e-9)
        prediction = float(model.get("intercept", 0.0) + x @ beta)
        uncertainty = max(1.0, float(model.get("uncertainty", 1.0)))
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "lcb": prediction - self.lcb_z * uncertainty,
            "support": int(model.get("support", 0)),
        }

    def select(self, key, item, features):
        if str(item).upper() not in RL004_SUPPORTED_ITEMS:
            return {"selected": False, "reason": "item_not_enabled"}
        result = self.predict(key, features)
        if result is None:
            return {"selected": False, "reason": "unsupported_event"}
        model = self.models.get(key, {})
        # Require a positive training-set margin before trusting a contextual
        # prediction. This keeps sparse event models from extrapolating into a
        # profitable-looking but unsupported market state.
        if float(model.get("train_mean_delta", -math.inf)) < self.min_expected_delta:
            result["selected"] = False
            result["reason"] = "training_mean_below_gate"
            return result
        if float(model.get("train_min_delta", -math.inf)) < 0.0:
            result["selected"] = False
            result["reason"] = "training_min_negative"
            return result
        if result["prediction"] < self.min_expected_delta:
            result["selected"] = False
            result["reason"] = "below_expected_delta"
            return result
        if result["lcb"] <= 0.0:
            result["selected"] = False
            result["reason"] = "non_positive_lcb"
            return result
        result["selected"] = True
        result["reason"] = "positive_lcb"
        return result


class RL004Runtime:
    def __init__(self, payload=None, opportunities=None):
        self.policy = RL004Policy(payload)
        self.index = rl004_opportunity_index(opportunities or [])
        self.history = RL004FeatureState()
        self.pending = defaultdict(int)
        self.last_step = -1
        self.delayed_orders = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.decisions = []
        self.errors = 0

    def reset(self):
        self.history.reset()
        self.pending.clear()
        self.last_step = -1
        self.delayed_orders = 0
        self.changed_calls = 0
        self.changed_units = 0
        self.decisions = []

    def _apply_pending(self, action, step):
        changed = 0
        for key in [key for key in self.pending if key[0] == int(step)]:
            _, item = key
            quantity = self.pending.pop(key, 0)
            if quantity:
                changed += rl004_adjust_sell(action, item, quantity)
        return changed

    def act(self, obs, base_action):
        step = rl004_step(obs)
        if step == 0 or step < self.last_step:
            self.reset()
        self.history.observe(obs)
        action = rl004_align_hands(base_action, obs)
        changed = self._apply_pending(action, step)
        candidates = self.index.get(step, [])
        if candidates and self.delayed_orders < RL004_MAX_DELAYED_ORDERS:
            scored = []
            for opportunity in candidates:
                key = rl004_item_key(
                    opportunity["item"], opportunity["current_step"], opportunity["future_step"]
                )
                features = rl004_features(obs, opportunity, self.history, action)
                result = self.policy.select(key, opportunity["item"], features)
                scored.append((float(result.get("lcb", -math.inf)), opportunity, result))
            _, opportunity, result = max(scored, key=lambda row: row[0])
            moved = 0
            if result.get("selected"):
                moved = rl004_adjust_sell(action, opportunity["item"], -1)
                if moved:
                    self.pending[(int(opportunity["future_step"]), opportunity["item"])] += moved
                    self.delayed_orders += moved
                    changed += moved
            self.decisions.append({
                "step": int(step),
                "item": opportunity["item"],
                "future_step": int(opportunity["future_step"]),
                "selected": bool(result.get("selected", False) and moved),
                "moved": int(moved),
                "reason": result.get("reason", "unsupported_event"),
                "prediction": float(result.get("prediction", 0.0)),
                "lcb": float(result.get("lcb", 0.0)),
                "support": int(result.get("support", 0)),
            })
        if changed:
            self.changed_calls += 1
            self.changed_units += int(changed)
        self.last_step = step
        return action


def rl004_fit_models(samples, ridge=8.0, min_support=RL004_MIN_SUPPORT):
    grouped = defaultdict(list)
    for row in samples:
        key = rl004_item_key(row["item"], row["current_step"], row["future_step"])
        grouped[key].append(row)
    models = {}
    report = {"groups": {}, "skipped_groups": {}}
    for key, rows in sorted(grouped.items()):
        unique_episodes = {(row.get("seed"), row.get("seat"), row.get("opponent", "v22")) for row in rows}
        support = len(unique_episodes)
        if support < int(min_support):
            report["skipped_groups"][key] = {"rows": len(rows), "support": support}
            continue
        matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
        target = np.asarray([rl004_as_float(row["cash_delta"]) for row in rows], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != RL004_FEATURE_DIM:
            raise ValueError(f"invalid feature matrix for {key}: {matrix.shape}")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        mean[0] = 0.0
        scale[0] = 1.0
        scale = np.where(scale < 1e-9, 1.0, scale)
        normalized = (matrix - mean) / scale
        design = np.column_stack((np.ones(len(rows), dtype=np.float64), normalized))
        penalty = np.eye(RL004_FEATURE_DIM + 1, dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        intercept = float(coefficients[0])
        beta = coefficients[1:]
        residual = target - design @ coefficients
        residual_std = float(np.std(residual))
        uncertainty = max(1.0, residual_std * (1.0 + 1.0 / math.sqrt(max(1, support))))
        prediction = intercept + normalized @ beta
        train_mean_delta = float(target.mean())
        train_min_delta = float(target.min())
        train_positive_rate = float(np.mean(target > 0))
        models[key] = {
            "support": support,
            "rows": len(rows),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "beta": beta.tolist(),
            "intercept": intercept,
            "uncertainty": uncertainty,
            "train_mean_delta": train_mean_delta,
            "train_min_delta": train_min_delta,
            "train_positive_rate": train_positive_rate,
        }
        report["groups"][key] = {
            "rows": len(rows),
            "support": support,
            "mean_delta": train_mean_delta,
            "median_delta": float(np.median(target)),
            "min_delta": train_min_delta,
            "max_delta": float(target.max()),
            "positive_rate": train_positive_rate,
            "residual_std": residual_std,
            "mean_prediction": float(prediction.mean()),
        }
    target_values = [rl004_as_float(row["cash_delta"]) for row in samples]
    report["samples"] = len(samples)
    report["models"] = len(models)
    report["mean_cash_delta"] = float(np.mean(target_values)) if target_values else 0.0
    report["min_support"] = int(min_support)
    return {
        "version": "rl004",
        "feature_dim": RL004_FEATURE_DIM,
        "min_support": int(min_support),
        "min_expected_delta": RL004_MIN_EXPECTED_DELTA,
        "lcb_z": RL004_LCB_Z,
        "models": models,
    }, report


def rl004_load_samples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


"""v22 price-impact route agent for Kaggriculture.

The production route is a real fit-only trajectory.  Runtime adaptation is
limited to identity-free weed recovery and in-place ranking of existing SELL
slots by the official 1.32.4 price-impact curve.  Ordinary turns never create,
delete, or resize a market order.
"""
import base64
import copy
import json
import math
import zlib


_ACTIONS = json.loads(zlib.decompress(base64.b85decode(
    (
    'c-rk<U2hyoa{MoR<^$)06zMmvG-nCN6$MJV!FfR}7VsGcjPt|VZ^r$1Yei0XPiJIgWLEVowYs+fIn!O0Rb8DK85#N0|DFBEFTei%'
    'Z@-@X%TH$?Za;oJdptk;&tLxg-~Z>!AHID2`!B!#*Wdp4%jciY-oAU-efh8U;fK$E{rUF&yB}}w&d$$1zTNFUoSm=DKVIMOCx8C9'
    '+r9bn$Nk&g?WeQzSF=C=xVyW5e|ElFKR*8B{AkqgUjO;?hso83@&9zT-+lb_bv*CyA3nYP`ssO+liy8;_w<9~iT^f;4-'
    'fZmKEM7n4$lnZhtKcs-'
    'u(RK>YqM;wZUW*<IUMHh6|71n~q~X>$}_6yXQ&kH#7evcX+nl<kIsg!dtjsBDW&88&>ekgx`<$KQ`gBEuI##(SC>fJngryd*b%f?'
    '&0{3fBJ1tPDl0p?UXsk>yDE=+~D=?v+<t4)X8|`q=q{U-'
    '?3XioPk{vU@N;JW<TSrbaVsJdUiv^W;|RkX})16G?<TWwP7ddTWx+_(Q4yP=wdAVpw1^8S#ADK60J7*lWsP1t4`JebMP%<{xx~H3'
    'dRBk@ogkLkYp<6Lnjm25025gjeBORZsR`waQe$WpCyii2mPFl>uwESNL|nPP0t5t(>3Nt>+dy=f_=?34ldOnVluni_J!#&j@S2hc'
    'e~f0e*V+$;nTajcmH<$@~T|%<NfFMW$HhyHxKt8mVKH&?(TjI-'
    '6lgGBe+F2M0f(N8n5?ao;YUs^3KWZ+g>*TF>P`;sTf0HbvY^!M~?HAo?d2j*7fVn&$pxNp%pM57BuPja5$D)J%$0wI1u3fTA!}r-'
    'qxt26K0Ltb=pn#kByKx9CHvsY=q3MNkCUx`(D!qVas<qZ*Y<<G;uc~>U8h969A_>e0ciya<~2tX3byZN-VsX4alwkm?kKM_Am9$e'
    'Xsv5U2W#yZZrPvR`qYWqr1h~G{v)0l48#nQ&30dK!ID#Z!bhjDOWXl%QkZzWsz#${v0LkZ7UQ2F}HH|zsfDG5$%jXP7)rpRVSX@u'
    '`tEVn~c5I>o+7e4I%g*u$y?lmZ)&k@ui)3ktGIX(39U`3@^?oAhG$~+XNiC|1?Uktk#Ph!Z)25T<WrNuF&(7owt7%5Bkz0p7rsdr'
    'vX}TADZ&G7ILBorc09(O{h){QvqIe3?VLOJv+&fCZsq>32>ZI#~M&_!5x(17Gb)Z#Q^f*e{XLe{-Qd<8v!YOcK-'
    'R)byCeRJa`Wj&(1e@m3L@|KaNGAv{}<ke~$xZ%orqhLB5o^%wR4l$`inFW~BN0XY#G%pQfL{rK3R;C>zlTaVAS(W)TYJ;eOLaw=;'
    'oHUuFdeqtHvx$<Lm(;?Wae4LPn^Y`f3{HxmOp;+fJh{4cj9h6#Wx&*v1L(lA?}oHO11&T_^xPb0F2SIng|uZ>^MfN@rXoSFxv;4l'
    '?{#MG<c=_09tp(RQUGmIMC*f}-MG#OI(i3j?n24>S*;lPqMuNcs+hX{x3Vqy&kV{*bC0@ZRfS3~CS<tHtn{bhS-'
    '4}X2tH+0|hkJwKZr*}(n*;mA(L3tkou?bb|!^Vva)RBxvkdw}gpj@EXm7S5JVc8Wt-L&J?-'
    'Xt<7cG?U?($A1hE(xqWj2To!A<iX27e5Ls4p8A5HL<s{{)-'
    ')JGF;1`4)%gdgs#VR&E~MSCM&mZ9XPqRVi!Buh%{U^%iKM=3}f8;`@63d^Ekx4n*6=ue%;-'
    '^f0Hlvj{~=PJwNt$kh>tdi8r&v&&N*>xBDM<4-bDmJHL+K$g~XJ&;Bfzwwbe##`D*gI7o^CcwsEQ-gq=-'
    '?x9$EyfXOTVIV^WWcHn<t?gBv`@mNC+F^DdE<Vr9p)AB?kGHM?D0e~c>b|sD;J^S&fTYP_iG_LO$k0avVVikAgHEFqBN<OV<M?C?'
    'Hc|^W+ibdF9fuF|1Zt3lX>iKrr3DpA=m-?R^(|rfCYID%4IRYUg-'
    'i^NVIrfOg{=w3EUfD#VRsahG43j(Povi>1mlUp0tcSFRHx<}M-'
    '(3=9j%9UerVBo%+P79gpg>mcSXKe&wXdd)<i|CnWIToLY*P0wbJj#sFiZ)bs`SiRf(y_h0vza$~F;BO^<W<f)|Cz8u0nmMf}DXrL'
    '}CCKOT7UlMn@5KZ==C=T5^MfIr$cvYSUeKtOjIhE2a!;m2i`*2t&L-'
    'w?aGBLl|PWsqcHL2zT&)s7}`U&L`ykpUJ<BE>Mzx_~^l(1LUCGo&X)hL$r&7ulCHk8m0C;SRXdK856~iF>o-'
    '?lxvQb+IWxv&{pK7*7uI%8@|}tB$mj9da+hs!u#6pN*z5D}`YiAC9o_oJR+pn69V*&XkstEMN<=Y%FCF7s#y|Ay7C)!J^v#wWXoB'
    'CLdztRz@HvDC~0ud7YBzXJuG0g!Jz!)mwJW;4q506jC+k!IC#-V+o<>xs({cuDdhg&7ZCf!>1qL-TfIbdEhzSZ$Kohrfk=ok)rUF'
    'q$0@x9gF4Dh0ibaO(|&QqTbV>^_7#4T;+q=wHnN>nu6J<4>baq6;x=6Vds}iy&Gp*#W=pog{;No;jBg}0Q&olgk&0QKID+#(_&^P'
    'XEI)?wp75K(qkn$-'
    '~x5oE*n`$KT+dHB?Z#94aH;ofTA%uRIcEY^wvCzB#T<{vA~Y7OBkjy#XNJ~<Ixh9nFZF95iAEA?m|@6;^S$m_uA|)qGxcIdl(zxb'
    'AnH543b5JuZ7(dZnDr>ON|ivw}_+es-'
    'Z8jdJ#M(EmP;UY|>!qH3;JOFE@r6v2VMa;72brwMfI9Y7c*(R%w>;&&G&KG|jReN|d)jf4tNKJ`8ZLGm)R=UgGp6NR2GS4rPw!Fb'
    'bN^Y#5HB1j4eDeW|mOh=q-k?O!+zxx7-'
    '=*&i%oS4!qwR=I+l&uiDpN=Nuvk0lDEAW#nWx|l)iNmWG#1fC>XTVIT$f#V2#C^|5DTBQhkn|}IbeBo_a3%@Vz&jbaM#b6X<V~9^'
    '*3aMNWgH9N503c3XSMH(sK(WjtJfyk~QWX+GpDeEgWW^U7CqKZ)kpLfoVNvrjUg>4qnvMw?Vvy9Gec;1QJ~jal10h7?VTH%d(quz'
    'B)bl7T6eWlv{FH06#CtY=sKLIH7j&+YBUx`+0r!+eQ{I&J=23z#_vc|MEawL%aQIWAawe>Z?FP|ExLK0E+p|{k*?OA`Csw^FB}KI'
    'o$*99I0Gkl0m)9P`|Dt2CN>`qyc-LXmu5Cd2o!SUN>OzDV@*3+rW<hAgD?;=UTjm9@iL;1$p|E*c%;VQQvPe~$5So8wMuO;r<t77'
    'b(uHBcTY4h;_KMiPAfF!BvjzMoFfYFtu8;UUr8Cm{J0zsZS_lW3V8=F+o8BZ#GAtykeAa2yG$0`v`_X2mq7SrK?pnM$-'
    'RiSbCt9_TfNotPdP3FeoipHvk8_bHdfr%D&Wevel&c4bQfUJ?0lZZJea6gnmC8$PxKJf3Ae^ck41GgWvUhmE`|wJq5ML9PJTI2zv'
    'S<Ux={f<ZuN`z*9Y!6J==9)O*NHS4t=L!Y^j|$z3H;H6;q`>i5F{w6TC0y}FvWZZj=QZs3=;i9Cvp+{##mZrL)dzWaF+^NS(G_5&'
    'MAUy6$1No&H$~^%)k>g(OT6xLX0rx0XLmi9#>)&UkW)E(rDTdd&?g)nlWZ@qii0%$X<Fb;<l%9VOwtElmJ31UZ`$(expkjBMXp>0'
    'QjHVtDOg)>d2b<kkFP`M2F!}7IV%(^P~Eci!N#n=^~cPB6w>VMkO`8)wWW(FCkjJsZ1918i#>u&E+DXDk5QDk|15`lPCIsv21ZHk'
    '*GjjT31#kf+Jl;ht4Y#dXG%!p@nplSvk9p0!~W{)^u4GLCASyvE>_})GHlQ()81{@+s0-'
    'MhCr8^4CoHq^6m+)=dX}a;Ab}!w1k1s*}~2CW22S&8bZACoHjyxkrb|WMrCuS?_Mlv@+iw;eW+a`r&=xGYkNpr9Wn?rZ`d=RKngZ'
    'BH4Va#yP?UA!<ncQXYwrsE{XnsF)=2`8R#H1Qk}`FRP?>0I}AL6vPy1NXgK3On9oa=1E6{Yz?W+(!3rU;LVPYZ7%3bM|YcibhnC|'
    '(e#ij$Od&;dP$Si8YM>eB2lpL&1M~2@2{)?2urA#?_PgDNcJ~x0c!ost5LB&EcUj0!-J-'
    '4w%1DXiT_n?VoKB5<WDX`p;?BC4kXEa$n<w1x23w@J6Q=q($QzkqQ>d70(7*s089)Fz_gxAa$9~WN2qdGH~=UegFkS@W`&-'
    'a=ki!61F$_lU|R0|<w0m+fYwyO?X%FMGmoSxgBwm4Mx#>^NJD5{jBx=o0g8eq^UEQ*9yXJe&x%zj3E@E)43Z@Gbc5xCpT-_fTP@3'
    'V<E6B0c~DJ~T*w-'
    'ARBKzub^L8&Y|}t%?(#3zfzmCz?^9S$YjkJPM4!Y{oH@xDbzQ^ly$bn0X(Uz+KyBKwz@64n@ZDRcAET06Km9OIdM;N>O`?%hE0#_'
    'jcbF-<(xPcW1an>EUD4E71pp`3bB#Aef-uWuGlC$!A*p!vm{Go_<+bsOB!(fCr@Fc#%W*bQ{Hkh$Xg8J0TJ1giwIbCy^rc8FEVl-'
    '4B*sep>in(iZJl!dIR70Tp(qZF0JX?ufbdxWPC~f>vEZU-'
    '7+^}3tx^AoIgaI$9b(!e&%uM^ZO)A#pjuTaPUH|;Qz$1IW^1NJ>SP;`*%nX8sJRISVfR^adROL!WRS%rlRK&hNuje-'
    '<ba4{LziVIgP`qsq7JpjCrBr;SpgqQitoDToU*2*<x3s2Q=Xnu@RU<6q|ls!LNd+P@Ekfzr}&AuzG<FxE{wJ`9oRyWK$k9k#EvBJ'
    'h_GT8TBejpVqrL+3@=emNUe<<m(3>Hg)2xiznaQCu(4&7g*|3A2|x&JV@NZUnu8@3qO#jXPxQ&kDAH_}P-'
    'K@jK`xjF4{wjDaK*%ErGelZDTrI3F|f?HZ4f+bAX_%Kwkj6ft=2RbdHV9YwV=i={bt)LB))y@lC@lig>OqMiPKp4sA=F~=+ljG=}'
    'd3>oy~GdQHx4gZz(XKtuSrQ?Gce1h_*>2KsDP(mKs1ecBl#=v%?xXry&{7@XdJ?IQ)6}tSwbcra(np%GFWzIrM=L-'
    'XxHemrp(7s29W@$Gc(rGjT&?#!G6!Y-'
    'J7b>xs&j>AcFvoC}!vC#tl!ZxH(5B&z!`<&QBYk5?Z4y|vPh^r@~PXjSx!fQ7jh2nhzU13ZB)7Ty!6rwKD<e?~Y77F1?SLj|I-'
    '3;<E^DQ1(}*t#5WK~N6zjHYX7ewAyQr7TM<TL(rNb{O6y&qX>g)@(SL`n5)4T#C>(zLHtSHgd~|2p>TrRuYN2r`GpT=^uzxmo5q1'
    'c1oA8Ut+qPe}XXdGAVH<u8i}{y|bg{?VD{VJVCh7oK_2@!kJq|6R3UC-Y#v<?KefKu_s7#2i-'
    'rk_^6WC*4|6Br%x0YcS>WE9z2fRaw3HZJaVO84UZt~+_P(F!AvVEcQk22%ZO60AY-'
    '~~y~+yKO1{?#rouhcDf=1XKeW2CvZfW?bu?a_A*<{=sqZov3wq6~)((B{tJv`fB1wK|h2zh)DAF^uT8%zlXCkzk4`Os8F=w`s<|6'
    'FHn0><Y$xOXRlrr*`>QMPDX2xOj4M~~tX{#M8>2lQx`h^ojpb|V|!IRq3AvwZCns^=y(+li+3l`V2_O;yaQM;yC7ATz!o?hQmTzv'
    ')h?g^Oc<r35d`RQ_d>s;bD&SjR9*KIYgaSC`XrHNa6M3-'
    'tnlaV|{zoM!4F4f(+{c+<q*J52n=M*&SM@*q0%Bftay}3j<2vjM_E0l_jEL3%Db__~VDpDQ3b9|y4H?QjeX>B<&^U@=I8watPNAO'
    '^>!~yeI-F#)IX@DSwoUyzl!;AhH19<U80sDu^n4xWHvmy?SL7-'
    'VAN_lDD5xFa`Smq(nJp<WKZBljZD|9#<n;+DAdeW}#nl$N><5)7uZ9dT#7m=ELg5NQ#78tsE>uqY&kXuw3EX(^+w4c=|p*DA!AaT'
    ';A=D47EEoIN)MoTmu6aA#rl-zj6E<hA-Q<+Fv(JH@lTC?@wp(>T}2en<fJ(m!+U=5sza`Nz@wsEF_#&$8)<oLjwerP^*(%gw-'
    'D`W}%O5MRf`4e%8{9+Ie%x8<r1KHxV2DE@=x9n3+;&1xOY#DK;`3u~uSxyFdXD-'
    'w+ov}8Kxz(UB1k)u>2h(JeSiw5b3jIuPmL|wXY6OMz_7VQ%B2k^1eX}5%>ffbUNoC%8lW3ikRmD+pep&h!jR5UCO^JZc8GNe*IW_'
    '6$NoY3=nc<>nyoD1q5LXL1bZU8A$FNvWa9%OaBH!Sbr4_m|aqw!J==5zNc0Ce~j|aMPei!kuF8JE^OnFsrVoqO3k+uT<D<jA$&ca'
    'r}Ktqa~TY<p&Cr!<cybL>{xA$%K1K_!6?XU%jaz=pSe0}`3ebUt&?*%JL`vFwiWV@vZtM)at>=ReaD8c$GsfUDf(M25#3NN}mq|j'
    '@+0Up;<Z@XA?Bg+h8IMCBKDP2^C0LhTOZKFePS6!J&t1H}-y;-'
    'Kht1LQ&w&wc8V3*Hf;l(&tD#M48qs$d0>~B>tNBK=d#2rKuS5N}9X{Lcy$x%e9QfPqPr>?9r6T6EoxyWI*0w5@^G_SYOt0Q!bCB@'
    '6BP%r`N#zHZ#K42Xy3$vRnmObYJtNSk9inoak=Wz=nAQF1d$CuFA(2`ncLvSB?MV#-'
    'mQv@d|4wcnz?b@P%?O=+iv62P0bQb<tPU|0bod6^_E*$FpBn~1)Srjd^q{>N(qN~xe+OZQIkz9!%5m2EOU@R)piemYLNV*jZ8ajI'
    'YSF}RMzl-17wYI0|tsK<L*d5?bvD~jyFC=6_MigI!NWw07<x#=qM2RK<5@Ibdx41moJbSGS9blPN!?QDmnmcCBLhz%YffK-'
    '}8fQw7j1C7sH&KtuUeDo|I9?$TmHaw|BW(?-UZEb)WQpDZXH7!KwXB#^STu`Ac@H!J-'
    'DF@aqme{bgYe&MX=zbeR!05w)l%~n+s8h=FsGOHiX0Cy^hFN_+vQ0ZZ0p~h!B(!)TZb8jWEEV;t~#`wQe%Y)f9lp!$HE<4N`R}wV'
    '>O`#TI>J~R8ba&Zkx_McEAn@c^}$9TY7~cN5*ag)tpK%_1dxrf{GBe;wFs{NhJzq#4>1Exq){ifP?>^*c0ipC7HEEn%9SrOllL0('
    'I}ftOdJTlLHPsv_C}AjBOxqV-UA#o4RWcgMXc2uv9{$WFe*(h%hb@a)+nBg9zh}&l#)~?T|yhgmN*$j9D7*gq#wtW*3XDASu>kv0'
    'e~aTpJVtugjIV|HGnqJ1i)sO#71+Vvy-A~Bq1xLd$|p$e%#&Nzeh~SbVU?A;8``ht?&r-'
    'oT4~a5LXKQNn2h9U%&Qi^R~VCPwxN1v2Rv(-PY-45^A&~r~tq&YWm;xu2U{QDha}w{QxwP*nXk}a<epbj^&o{aEF;Fjtjw};-'
    'zY$Zf!Ah%Sq8L$LZRfgwwTonefjl!E*7LORPInFPhe6Cv-^Am-hejVrg8o9*&qL&*MsLqdvSYsVYwKRnsi4R2jl!7vu`e(Q$TJ{S'
    '#a_KW*Q2+xb*2k0GLU&h=}}YDHO_$EGgS$s$t9UZP=cjlP!QG|@iQOv2ZLKvdb}242(UV_V~EDDwtmNviimIzjc>rLTkv3#}Qotn'
    '!`aK*j5Lo!F<+I;tQ`yt+PI!WR(E=K3+w=!tV~qQX@A$z{}q%))BV?uKr`@hK8Ncr_ZGcHT*p5GgMsfUu=T2|*Eh;tXe{O#$g6ju'
    'ufra;XhD+)1hg9D5e*L)Jw#&D+~yy%$q=qz;FlywD#Mg}K5{U_YG_&6nXZG`%@PF|X6IW-{Uke}oel-'
    'm0^?3B?MlUaD1klP{eaMc$}TMK1N|NfbBjeA1dpy5N}|e<eYXTbFP-xHqIXmP<~PB^5*!`cSUy`O=#UtTaD+_#NWdMF;fo@+2r>X'
    'PGk*@NAbq)5u|Ebg&8e1Da<P5b^=8Y~^1QQ_L3s1)Zeo6^c#Ufrz4qJcPD!<~j_9l4cAg$SS#)*kOSKV(}VmUs4*GVH#*mF~ksz{'
    'ind8XC+i}=*oy=bPKYFQs(4&Qz&=$mD^DTZV(W4)x?G3D&KhB!DIC1-'
    'M@OCBo~(fbJJrTQp~ICN!v9}C?rPd<_xJ*?F+57QLNj)@(MhxKYRWXd`JyYsPJT2t7~su0W~JrB4Y5>rHwf=|6z&3nc)M;=UElyq'
    'a9#~&4rf*=E5;f(8z+R_>2$c8n1*<b5u~YXZVz;Ma76=r`1je%&ZDC)sp?XDD(@eGyw_)76b^O;nS$SCEe<#g{j7r+9D!+-'
    '+p_sS9v%d!x#BTfHXc2-(oETp1;UqT9$3<cf-'
    'V!+M$)zT2<R(e6!`0x4Lx=B;Cd!=&*JFbJ<GUTJuhYCso(7#ZYP5*7`+gQjsv9b(HEt%W9j-'
    'rlPmJ&y^a`?m)_3?!&@@I1p9I(mE4Tku<K_+pMcadi^n$53NiT6u_5P)=ITflvz7u2M1MWX{B#jHB0N2zR$H*r_G!hA5fj|s*PsG'
    'fYvCsleGASjk<2)CMB9wBbuY6z>!o)s}bt7ng#Y?rn&**)n%x-'
    '((_#<(dDGIoMICc0zNzTVyg#fzs7rzg%gp*bL^N3{HbP)ov_zpn{3Dqn~CD!Kr4pMj9DOYsID$DWI<<Q43y+&ke0L{zpS$cJ;nvE'
    'u=NU<0wl6lW~-'
    'w}(T*!HXy~8XG`)DHAS$aRWtutFQm*n<VbmPXKtT;Id^bsg`^qAfO5l`diO|lI(qsDwQck3_=6~YhK@2vBbq3KkJN}(rR+4*^gxC'
    'xtslCr`F6RL1%LACy8bR!;5bou5UtRw-'
    'CwvX>P}9+;10Z?z`&EQ)7`rt;Hw9ExIvW})RC6(DLvzeBIZVPA)5U_zvBCf_XmOo9D@-ZbuMvY-JljAI6S9XjnwYd9g25#kBV~N('
    'RxFv5CrqHq+{!+CgXV_=7B`8^<T|N@Y_%l+PKrH)q$LFfT!bYxU<P{f9Zp?cyiCJp(sg>NGB_Bs-'
    'QOW6h_z%QTU_5Js1kE0lE`9#!h=zn5S3UONIZ%wp}aIW$`aKHI$_S%AGpw9ZkOf!YY(50o2;1t11)i4MyP~tC-Tb7j@qk~(Op#S('
    '?dm^9z1ars?I?lxhzY)*JEYoAR(F^Itm%7g}0{GtE$3Eq`U?mRY=_L;+~7mZ4=S>C<%;l)nax|*d~0&v8h3Il+Pxso`T;NSZr72)'
    '|TOZlVgRXf?Oxop>4m|PoG`nLtA2Yfh}g|Aq&p$eWg7160Z##V@6wp25q{6Qrk6#n;dq~g3P$fmxEB9UbCz>`R$t&DqDqot8Yq&E'
    'Ikt+L|967OrpEh=dldija5W4sv8&lNzieY+Exq_mEN6HnoDoJw<RxtlE4^meMlF`(obT@iOp@J^W0dqWgI_-CE0R8Xei#NHig^@<'
    'S)K1QomJ^<dsHu()w_b(vzU?N7jV#xtu-'
    '>T}8zRkqyZZhEn!r^{v$RbI?}Ujy9B(q&3DEY8ipLR96p1bxJ^sa=wKSrX&=srZ2_{5!vg9@vyYb#lBVQT!<S=3se)E97uN%VXIu'
    'kO6__`>I0fc0Gf=)*l7W{jAdg<x+pE_+sA#VQcpR{ywE<I{28W2CbaIPhwxk2d55ybG;LW%Q7WuMpshy-JRJGq+l9_Hq*HBTidb0'
    'cDbNAo+e}ww0Z?+LU8LIR#5TOFwE>VjBmh&$jUo@>>9vDit-'
    '7LeHwKh@g*yv?x{pSD?)v4l4(Lx5Tc1QGtSl$rkrxKJR2du&Dx7u3<O&$cP?_GL(p0J0Y}6DZj#paVU&}W3))-'
    '^YgQ%X>HeALU3zFAtlTc(W6=XGasDz;!hslCEvxK;INh%wjES89o3#ou`O32wuNv9<=RGz0|2QVxela%HH+->VYFdr5j2g-'
    'p`Wpcz|sfm2kU^x;+3CaRceejTd%}dz<G&t%mTLXBQ#rFiGYFq(IrRCJfu#M|3r4gNDTltSGIkKs3HN&@w*=ewOfIrc_&c?2X*<5'
    'QP>R`H0ky@x|GwWPYCAr01JiP%dVV@`C)3E*M`XY`40=FU|y1lRR_F#k`vUT=JWqaF!(^7OY{UX~{!nQUp-'
    '_}(ZvNep50cTGDBJ7qYV;UX`$k-'
    'JsLRq8~6%X3^rKI5MLU(q`u60y#fMG82(F@*A<B}g<l#5MX6v$5@VVjjwH0h#pTdb(xQSR&y!AZ;GI?c6?oe5>hF)oTxDJ4EMy>k'
    'Mwd!<QOeT;=j*4k>+KbQ}mm9%H%vQC9MvBEm4dX*96m+BO)G_^xjq1t3(ooE9blOtLYDgq__0p~ckXL)?)Z(yyhrlJN{<kMKw@yv'
    'l<phq>%vRI>$wXpj)4AEp@LlXTcwXWnWQ;{7R)xwoF1TqdP+6vI+=n$cdi0r1=OS;ezqDZ#_Z|qS+dXQNo3{7nbrWNUSFf^J(0E4'
    'Q9rJ0=!FhRLENyPrR&OV!SV?RO}R1-C|0GopmQJ<sIb4Z+4h!1cpDMCe5B|}r^*ZUag;6-zVDS<pyO?Zi6oKxnLJNU7-'
    '072m*m>$QcXh%MSHylefSvJg7)0T|@F@vW>XsAxR0)!|2o3f`}u=EySSlG@t;BmTaf?w>WO|;-'
    '*TC#@zB7zhaaFjrYwze64TcpNWkV<Go;P{56SvM>C?kllLrp-sVA#;{RMM%X?t5Tc$@NK8cB_{&ajCt|Rh=bZAQOGrTGTd05h7t('
    'e>aBX2rccNorthpCSP92%%Wpl_QQoEuiQCL8?IgZbkg5_{$(&x!U&M-'
    'E3qmT8M53(<quegI1<6C&WeSr~H?Fh*pi0Mu9e!HO8+I;Bcp1Al%O&efQ=Y&+L`I`Fwr~CVHV66mkH`N9*sTax'
    )
)).decode("utf-8"))
_PRICE_FLOOR = 1
_MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align_hands(action, obs):
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(actions, step, actor):
    trace = actions[min(max(int(step), 0), len(actions) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair_action(obs, action, actions, step):
    action = _align_hands(action, obs)
    seat = _seat(obs)
    game = _WEED_STATE[seat]
    if step == 0 or step < game.get("last_step", -1):
        game = {"last_step": step, "active": {}}
        _WEED_STATE[seat] = game
    game["last_step"] = step
    farm = _farm(obs, seat)
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - transaction["start"]
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(actions, step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    raise ValueError(name)


def _market_price(item, inventory):
    base, equilibrium, scale, below_func, below_target, above_func, above_target = (
        _MARKET_PARAMS[item]
    )
    if inventory < equilibrium:
        amplitude = below_target * base / _shape(below_func, scale)
        price = base + amplitude * _shape(below_func, equilibrium - inventory)
    else:
        amplitude = above_target * base / _shape(above_func, scale)
        price = base - amplitude * _shape(above_func, inventory - equilibrium)
    return max(_PRICE_FLOOR, int(round(price)))


def _is_sell(order):
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in _MARKET_PARAMS
    )


def _impact_score(obs, order):
    if not _is_sell(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_quote = float(
        _get(prices, item, _market_price(item, current_inventory)) or 0
    )
    later_quote = float(_market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def _impact_slots(obs, action):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_impact_score(obs, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def _rl004_v22_agent(obs):
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
        action = _weed_repair_action(obs, _copy_action(_ACTIONS[step]), _ACTIONS, step)
        return _align_hands(_impact_slots(obs, action), obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs):
    return agent(obs)

# RL-004: observation-aware, event-supported timing selector.
RL004_PAYLOAD = {"version":"rl004","feature_dim":29,"min_support":12,"min_expected_delta":5.0,"lcb_z":1.5,"models":{"MILK|215|260":{"support":12,"rows":12,"mean":[0.0,0.2986111111111111,0.3611111111111111,0.26666666666666666,0.9583333333333335,0.1875,0.09375,-0.09375,0.625,1.0,0.0,0.0,0.0,0.6605555555555557,0.9975999999999998,-0.032777777777777774,-0.02611111111111111,0.0009000000000000002,0.0006000000000000001,0.06000000000000002,0.0,0.0,0.0,0.09999999999999999,0.25,0.4500000000000001,0.4500000000000001,0.5,0.5],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.06171669642231305,0.002078460969082632,0.017471316881684875,0.021030548034164116,0.00022360679774997898,0.00044721359549995795,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.4291966002185543,1.2613868763731046,-1.554680110147687,-1.4587257373346465,1.1230302390689584,1.1230302390689613,5.995047802914113e-34,0.0,0.0,0.0,-2.062833017580776e-33,0.0,1.650266414064621e-32,1.650266414064621e-32,0.0,0.0],"intercept":24.4999999999999,"uncertainty":1.7138545815992294,"train_mean_delta":24.5,"train_min_delta":12.0,"train_positive_rate":1.0},"MILK|260|283":{"support":12,"rows":12,"mean":[0.0,0.3611111111111111,0.39305555555555566,0.3333333333333333,0.8333333333333334,0.09375,0.09375,0.0,0.3194444444444445,1.0,0.0,0.0,0.0,0.6911111111111111,0.99655,0.021666666666666667,0.050555555555555555,-0.0006500000000000003,-0.0013000000000000006,0.03000000000000001,0.03000000000000001,0.03000000000000001,0.0,0.19999999999999998,0.375,0.39999999999999997,0.39999999999999997,0.5,0.5],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.06436777047982327,0.002569533031505942,0.0053575837561071985,0.016377114414426304,0.00015000000000000001,0.00030000000000000003,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.34905746997630616,0.4776978071715997,0.41007881974910304,0.16454464643821895,-0.11732347989487343,-0.11732347989487345,-1.8477904585192472e-34,-1.8477904585192472e-34,-1.8477904585192472e-34,0.0,2.535211694111811e-34,0.0,5.070423388223622e-34,5.070423388223622e-34,0.0,0.0],"intercept":7.000000000000011,"uncertainty":1.3983409576360988,"train_mean_delta":7.0,"train_min_delta":4.0,"train_positive_rate":1.0},"MILK|288|308":{"support":12,"rows":12,"mean":[0.0,0.39999999999999997,0.42777777777777776,0.39999999999999997,0.0,0.1875,0.09375,-0.09375,0.27777777777777773,1.0,0.0,0.0,0.0,0.6938888888888889,0.9962999999999999,-0.0011111111111111113,0.019999999999999997,-4.9999999999999996e-05,-0.0007000000000000001,0.06000000000000002,0.06000000000000002,0.06000000000000002,0.0,0.9000000000000002,0.5,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.07314158432810089,0.002893095228297879,0.005665577237325316,0.0019245008972987529,0.00015,0.00030000000000000003,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.2013033625005582,1.1930882041293842,-1.092343171585124,0.24602624415774924,1.1837184508812129,1.1837184508812089,6.154951347230194e-33,6.154951347230194e-33,6.154951347230194e-33,0.0,2.575004944537441e-32,0.0,0.0,0.0,0.0,0.0],"intercept":11.499999999999938,"uncertainty":1.0810266912928568,"train_mean_delta":11.5,"train_min_delta":3.0,"train_positive_rate":1.0},"MILK|336|375":{"support":12,"rows":12,"mean":[0.0,0.46666666666666673,0.5208333333333333,0.46666666666666673,0.0,0.28125,0.09375,-0.1875,0.5416666666666667,1.0,0.0,0.0,0.0,0.6905555555555556,0.9960999999999998,0.02666666666666667,0.06277777777777778,-0.0006500000000000003,-0.0013000000000000006,0.08999999999999998,0.06000000000000002,0.06000000000000002,0.0,0.9000000000000002,0.5,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.08893228107545621,0.0034597687784012135,0.013471506281091268,0.034016154477425856,0.00015000000000000001,0.00030000000000000003,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.158499640022232,2.0350721818622044,2.2089895545855613,2.257243551211529,2.3150425126951317,2.3150425126951313,0.0,8.817251752269824e-33,8.817251752269824e-33,0.0,2.508420130584232e-32,0.0,0.0,0.0,0.0,0.0],"intercept":27.666666666666526,"uncertainty":4.001271195873245,"train_mean_delta":27.666666666666668,"train_min_delta":11.0,"train_positive_rate":1.0},"MILK|388|404":{"support":12,"rows":12,"mean":[0.0,0.5388888888888889,0.561111111111111,0.5333333333333333,0.16666666666666666,0.1875,0.09375,-0.09375,0.2222222222222223,1.0,0.0,0.0,0.0,0.5716666666666667,0.9980833333333335,-0.10833333333333334,-0.10444444444444445,0.0021499999999999996,0.0018999999999999996,0.06000000000000002,0.06000000000000002,0.06000000000000002,0.0,0.19999999999999998,0.625,0.39999999999999997,0.39999999999999997,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.1828554377400659,0.0041486610959307006,0.07192460043787624,0.0757350808353421,0.00020615528128088305,0.00041231056256176604,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.236684454539539e-33,0.0,0.0,0.0,0.0,-0.6847255143097514,0.6582019832780153,-1.2158997283561597,-0.45370892075911534,-1.0354163948763855,-1.0354163948763853,-5.591711136348901e-34,-5.591711136348901e-34,-5.591711136348901e-34,0.0,3.942497581566977e-33,0.0,7.884995163133954e-33,7.884995163133954e-33,0.0,0.0],"intercept":7.500000000000039,"uncertainty":3.4025622086621734,"train_mean_delta":7.5,"train_min_delta":3.0,"train_positive_rate":1.0},"MILK|432|450":{"support":12,"rows":12,"mean":[0.0,0.5999999999999999,0.625,0.5999999999999999,0.0,0.1875,0.09375,-0.09375,0.25,1.0,0.0,0.0,0.0,0.41388888888888875,1.0010999999999999,-0.13500000000000004,-0.10111111111111111,0.0027499999999999994,0.0018999999999999996,0.06000000000000002,0.06000000000000002,0.06000000000000002,0.0,0.7999999999999999,0.75,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.28042571252813747,0.004809365862564379,0.0715373510314199,0.05858411700650697,0.00020615528128088297,0.00041231056256176604,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,6.419353454452484,2.5882123232099037,-4.035688318973619,8.067838297746373,-23.512505799252665,-23.512505799252637,1.802648476410616e-32,1.802648476410616e-32,1.802648476410616e-32,0.0,-2.6110662202464996e-32,0.0,0.0,0.0,0.0,0.0],"intercept":-25.666666666666618,"uncertainty":90.88210805274261,"train_mean_delta":-25.666666666666668,"train_min_delta":-244.0,"train_positive_rate":0.8333333333333334},"MILK|452|473":{"support":12,"rows":12,"mean":[0.0,0.6277777777777779,0.6569444444444443,0.5999999999999999,0.8333333333333334,0.09375,0.09375,0.0,0.29166666666666663,1.0,0.0,0.0,0.0,0.39666666666666667,1.0012666666666667,0.01388888888888889,-0.09666666666666668,-0.0003000000000000001,0.0017499999999999998,0.03000000000000001,0.0,0.0,0.0,0.09999999999999999,0.75,0.5999999999999999,0.5999999999999999,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2839079141265768,0.004942221722621884,0.0022906142364542553,0.051135259995572675,0.0001414213562373095,0.0002692582403567253,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.07649304780763903,0.446482396674189,-0.15624490606091437,-1.7722930497490255,1.9638353141260285,1.9597111093533617,-1.9906604593559803e-33,0.0,0.0,0.0,-1.7161934166966233e-33,0.0,-1.3729547333572986e-32,-1.3729547333572986e-32,0.0,0.0],"intercept":22.16666666666667,"uncertainty":3.356346939822527,"train_mean_delta":22.166666666666668,"train_min_delta":12.0,"train_positive_rate":1.0},"MILK|480|504":{"support":12,"rows":12,"mean":[0.0,0.6666666666666666,0.7000000000000001,0.6666666666666666,0.0,0.09375,0.1875,0.09375,0.3333333333333333,1.0,0.0,0.0,0.0,0.3444444444444444,1.0021333333333333,-0.08055555555555556,-0.031111111111111114,0.0014,0.0004999999999999999,0.03000000000000001,0.08999999999999998,0.08999999999999998,0.0,0.9000000000000002,0.75,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.30511483367479053,0.005133766215514247,0.04322279348379984,0.023544022333796766,0.000223606797749979,0.00030000000000000003,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,18.362550999474653,-18.209446571162385,3.155500188949739,-10.344388460943046,33.42674656394025,16.06915477647793,-3.1798366044156495e-32,0.0,0.0,0.0,5.186661090667255e-31,0.0,0.0,0.0,0.0,0.0],"intercept":-15.166666666666616,"uncertainty":41.895852747669,"train_mean_delta":-15.166666666666666,"train_min_delta":-204.0,"train_positive_rate":0.8333333333333334},"MILK|504|522":{"support":12,"rows":12,"mean":[0.0,0.7000000000000001,0.7249999999999998,0.7000000000000001,0.0,0.1875,0.1875,0.0,0.25,1.0,0.0,0.0,0.0,0.43111111111111106,1.0004666666666664,0.05888888888888888,0.08666666666666667,-0.0011,-0.001666666666666667,0.06000000000000002,0.21,0.21,0.0,0.9000000000000002,0.875,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2790105976127913,0.005327496806401914,0.025939150066508338,0.034801021696368506,0.00014142135623730948,0.0002748737083745107,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.07996823701816493,0.5659328121860845,2.7215311668526776,1.3776707154924357,1.823670700941371,3.4048489820984527,-7.1094425347977e-33,0.0,0.0,0.0,1.4055934526269073e-32,0.0,0.0,0.0,0.0,0.0],"intercept":24.66666666666663,"uncertainty":3.806005030403733,"train_mean_delta":24.666666666666668,"train_min_delta":9.0,"train_positive_rate":1.0},"STRAWBERRY|432|454":{"support":12,"rows":12,"mean":[0.0,0.5999999999999999,0.6305555555555555,0.5999999999999999,0.0,0.1875,0.125,-0.0625,0.3055555555555555,0.0,0.0,1.0,0.0,0.7655555555555558,0.9826999999999999,-0.012222222222222225,-0.0022222222222222222,0.00105,9.999999999999998e-05,0.06000000000000002,0.07999999999999999,0.07999999999999999,0.0,0.7999999999999999,0.75,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.03989182904670278,0.0038065732621348583,0.0015713484026367726,0.0024845199749997664,0.00015,0.0003,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.2840262877064067,0.25087437994053785,0.08920473448348827,0.36032286007101827,-0.15125666275906016,-0.15125666275905977,5.9498738334704355e-34,0.0,0.0,0.0,-2.4660954058598682e-33,0.0,0.0,0.0,0.0,0.0],"intercept":4.833333333333326,"uncertainty":1.0,"train_mean_delta":4.833333333333333,"train_min_delta":3.0,"train_positive_rate":1.0},"STRAWBERRY|456|473":{"support":12,"rows":12,"mean":[0.0,0.6333333333333332,0.6569444444444443,0.6333333333333332,0.0,0.125,0.1875,0.0625,0.23611111111111113,0.0,0.0,1.0,0.0,0.7672222222222224,0.9826000000000001,0.0033333333333333335,0.001666666666666667,-0.00024999999999999995,-0.0001,0.039999999999999994,0.29,0.29,0.0,0.7000000000000001,0.75,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.04218199838384049,0.004004996878900175,0.002721655269759087,0.0053575837561071985,0.000206155281280883,0.000412310562561766,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.08096086465150354,0.07168037758728624,0.3161213310488718,0.1489644996216616,-0.2174388072366135,-0.21743880723661355,0.0,0.0,0.0,0.0,-1.4396841367422848e-33,0.0,0.0,0.0,0.0,0.0],"intercept":3.8333333333333357,"uncertainty":1.0,"train_mean_delta":3.8333333333333335,"train_min_delta":2.0,"train_positive_rate":1.0},"STRAWBERRY|480|503":{"support":12,"rows":12,"mean":[0.0,0.6666666666666666,0.6986111111111111,0.6666666666666666,0.0,0.5,0.1875,-0.3125,0.3194444444444445,0.0,0.0,1.0,0.0,0.755,0.9836999999999998,-0.014444444444444446,-0.01222222222222222,0.0013499999999999999,0.0011000000000000003,0.15999999999999998,0.08999999999999998,0.08999999999999998,0.0,0.9000000000000002,0.75,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.04442013223920151,0.004234383071948021,0.002484519974999766,0.0036851386559504443,0.00020615528128088307,0.0004123105625617661,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.4388890081131828,0.385163336080265,0.05590719914195558,0.1687349595793384,-0.4468722739307353,-0.4468722739307349,0.0,0.0,0.0,0.0,-1.1744741562836598e-32,0.0,0.0,0.0,0.0,0.0],"intercept":6.8333333333333135,"uncertainty":1.0282177247681903,"train_mean_delta":6.833333333333333,"train_min_delta":5.0,"train_positive_rate":1.0},"STRAWBERRY|504|523":{"support":12,"rows":12,"mean":[0.0,0.7000000000000001,0.7263888888888889,0.7000000000000001,0.0,0.0625,0.25,0.1875,0.2638888888888889,0.0,0.0,1.0,0.0,0.7327777777777778,0.9856000000000001,0.0016666666666666668,-0.022222222222222223,-5e-05,0.0018999999999999998,0.019999999999999997,0.63,0.63,0.0,0.9000000000000002,0.875,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.05060437203578167,0.00448998886412872,0.0016666666666666668,0.0068493488921877515,0.000206155281280883,0.000412310562561766,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.4751490233840607,0.42649844018277394,-0.2509810532832251,-0.19097674603456574,-0.23572929378874447,-0.2357292937887443,0.0,0.0,0.0,0.0,-1.5269985569894316e-32,0.0,0.0,0.0,0.0,0.0],"intercept":5.8333333333333455,"uncertainty":1.0588868301457188,"train_mean_delta":5.833333333333333,"train_min_delta":4.0,"train_positive_rate":1.0},"STRAWBERRY|528|552":{"support":12,"rows":12,"mean":[0.0,0.7333333333333333,0.7666666666666667,0.7333333333333333,0.0,0.125,0.625,0.5,0.3333333333333333,0.0,0.0,1.0,0.0,0.6577777777777777,0.991,-0.08555555555555557,-0.07500000000000001,0.0063999999999999994,0.005399999999999999,0.039999999999999994,0.32833333333333337,0.32833333333333337,0.0,1.0,0.875,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.06387681045246585,0.004565084884205314,0.015112745009706047,0.013709958532503405,0.00014142135623730948,0.00028284271247461896,1.0,0.005527707983925671,0.005527707983925671,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.43550478439718193,0.4091397419277569,-0.4810003139831974,-0.3665519555076395,-0.36227121927994577,-0.36227121927994593,0.0,-0.14690508231919863,-0.1469050823191987,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"intercept":4.16666666666667,"uncertainty":1.5418930592580409,"train_mean_delta":4.166666666666667,"train_min_delta":0.0,"train_positive_rate":0.8333333333333334},"STRAWBERRY|552|573":{"support":12,"rows":12,"mean":[0.0,0.7666666666666667,0.7958333333333333,0.7666666666666667,0.0,0.625,0.59375,-0.03125,0.29166666666666663,0.0,0.0,1.0,0.0,0.6872222222222223,0.989,0.01888888888888889,0.02944444444444445,-0.0014,-0.0019999999999999996,0.19833333333333333,0.8200000000000002,0.8200000000000002,0.0,1.0,0.875,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.05929888721292634,0.004656178690729105,0.004157397096415491,0.007049209744694179,0.00014142135623730956,0.0002828427124746189,0.005527707983925671,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.3296232066072216,0.3643217384924543,0.11456441646852081,0.35978581725130976,-0.16337464085354705,-0.1633746408535472,0.5269824080086801,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"intercept":3.9166666666666647,"uncertainty":2.0887571749870006,"train_mean_delta":3.9166666666666665,"train_min_delta":0.0,"train_positive_rate":0.9166666666666666}}}
_RL004_OPPORTUNITIES = rl004_route_opportunities(_ACTIONS)
_RL004_RUNTIME = RL004Runtime(payload=RL004_PAYLOAD, opportunities=_RL004_OPPORTUNITIES)

def agent(obs, config=None):
    """Public entry point; v22 owns every farmer/hand action."""
    try:
        base = _rl004_v22_agent(obs)
        return _RL004_RUNTIME.act(obs, base)
    except Exception:
        _RL004_RUNTIME.errors += 1
        return rl004_align_hands(_rl004_v22_agent(obs), obs)
