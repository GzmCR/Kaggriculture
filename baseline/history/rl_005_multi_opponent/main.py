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


def _rl005_v22_agent(obs):
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

# RL-005: multi-opponent, observation-aware timing selector.
RL005_PAYLOAD = {"version":"rl004","feature_dim":29,"min_support":12,"min_expected_delta":5.0,"lcb_z":1.5,"models":{"MILK|215|260":{"support":54,"rows":54,"mean":[0.0,0.2986111111111108,0.3611111111111108,0.266666666666667,0.958333333333334,0.1875,0.09375,-0.09375,0.625,1.0,0.0,0.0,0.0,0.6856790123456792,0.9967999999999996,-0.008518518518518524,-0.015555555555555555,0.0002666666666666669,0.00026666666666666684,0.060000000000000046,0.0,0.0,0.009943703703703703,0.09999999999999994,0.25,0.44999999999999957,0.3277777777777778,0.5,0.5],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.06495845341931622,0.0016492422502470596,0.012182877187702617,0.04473975961641927,0.00023570226039551574,0.0007888106377466156,1.0,1.0,1.0,0.005174128767705833,1.0,1.0,1.0,0.11083298524033336,1.0,1.0],"beta":[0.0,-7.975564973665866e-31,-7.975564973665866e-31,2.1091770610447795e-30,4.218354122089559e-30,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.1760467824225787,0.974444374611181,-0.2354535122663636,-3.3405297364994606,-0.3532303123633266,2.5206986571283028,-1.3753583101905554e-31,0.0,0.0,-1.2483720346465816,-1.5400461107448995e-31,0.0,-1.518870192510054e-30,2.4426778522807475,0.0,0.0],"intercept":11.1481481481479,"uncertainty":1.4056742825224657,"train_mean_delta":11.148148148148149,"train_min_delta":3.0,"train_positive_rate":1.0},"MILK|260|283":{"support":54,"rows":54,"mean":[0.0,0.3611111111111108,0.39305555555555566,0.3333333333333334,0.8333333333333337,0.09375,0.09375,0.0,0.31944444444444453,1.0,0.0,0.0,0.0,0.7038271604938269,0.9963222222222227,0.01864197530864197,0.028024691358024687,-0.0005666666666666668,-0.0008666666666666674,0.030000000000000023,0.030000000000000023,0.003333333333333333,0.01732111111111111,0.19999999999999987,0.375,0.39999999999999974,0.4444444444444443,0.5,0.5],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.04548507378475594,0.0017478734875256625,0.013123636805845242,0.023169784127386215,0.00012472191289246464,0.00041096093353126524,1.0,1.0,0.009428090415820631,0.004417993790828876,1.0,1.0,1.0,0.08314794192830982,1.0,1.0],"beta":[0.0,1.8482134481511723e-31,0.0,0.0,1.725148960287484e-31,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.6546225984181099,0.5544166977459903,0.6962813124397444,0.47734677896632577,-0.2964560547263068,-0.2803480530073062,2.4526888767487888e-33,2.4526888767487888e-33,-0.25212472969388144,0.09668001713688135,1.3077678229642746e-31,0.0,2.6155356459285493e-31,0.43871818543629376,0.0,0.0],"intercept":8.296296296296441,"uncertainty":1.0,"train_mean_delta":8.296296296296296,"train_min_delta":6.0,"train_positive_rate":1.0},"MILK|288|308":{"support":54,"rows":54,"mean":[0.0,0.39999999999999974,0.4277777777777774,0.39999999999999974,0.0,0.1875,0.09375,-0.09375,0.2777777777777781,1.0,0.0,0.0,0.0,0.7180246913580248,0.9957333333333334,0.0065432098765432056,0.021604938271604947,-0.00023333333333333347,-0.0007999999999999996,0.060000000000000046,0.060000000000000046,0.04666666666666665,0.09437703703703705,0.8999999999999991,0.5,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0449440370741897,0.0018997075798366698,0.0029371301860173617,0.006177775804047804,0.00012472191289246467,0.00023094010767585026,1.0,1.0,0.02494438257849295,0.044210755253781174,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,-1.7780058782379335e-31,6.109521813501202e-32,-1.7780058782379335e-31,0.0,0.0,0.0,0.0,-6.109521813501202e-32,0.0,0.0,0.0,0.0,-0.7489203530856765,0.8985133634465365,-2.033577217615198,-1.5823638186089084,0.47687964753707796,0.1356171114229849,-7.403892038165058e-32,-7.403892038165058e-32,-0.5186536082084746,-0.0699663734837708,-6.025507459417894e-31,0.0,0.0,0.0,0.0,0.0],"intercept":8.074074074074074,"uncertainty":1.1847371316951176,"train_mean_delta":8.074074074074074,"train_min_delta":1.0,"train_positive_rate":1.0},"MILK|336|375":{"support":54,"rows":54,"mean":[0.0,0.4666666666666661,0.5208333333333329,0.4666666666666661,0.0,0.28125,0.09375,-0.1875,0.5416666666666671,1.0,0.0,0.0,0.0,0.7461728395061732,0.9943333333333331,0.01592592592592593,0.03543209876543212,-0.0007222222222222221,-0.001444444444444445,0.08999999999999994,0.060000000000000046,0.04666666666666665,0.14538629629629624,0.8999999999999991,0.5,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.04935554808230894,0.0020330600909302474,0.00823189287810847,0.01807957145027603,0.0001930905244109196,0.0002629368792488719,1.0,1.0,0.02494438257849295,0.06791388911283705,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,7.012226731146227e-31,0.0,7.012226731146227e-31,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.284489739073714,0.36548150237458854,2.433905676218753,3.6770730748145946,0.3296143408926035,0.08524367851107548,0.0,-1.5691625187436057e-31,-1.304673063968278,-2.480331108289855,2.152653295857652e-31,0.0,0.0,0.0,0.0,0.0],"intercept":14.888888888888838,"uncertainty":1.823308000156875,"train_mean_delta":14.88888888888889,"train_min_delta":7.0,"train_positive_rate":1.0},"MILK|452|473":{"support":54,"rows":54,"mean":[0.0,0.6277777777777774,0.6569444444444444,0.6000000000000006,0.8333333333333337,0.09375,0.09375,0.0,0.29166666666666646,1.0,0.0,0.0,0.0,0.7127160493827163,0.9951814814814812,0.00666666666666667,-0.024074074074074078,-0.0001888888888888891,0.0008555555555555555,0.030000000000000023,0.0,0.0,0.2587129629629629,0.09999999999999994,0.75,0.6000000000000006,0.5333333333333337,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.1229775885000251,0.0028944438520413737,0.016430424453939874,0.026047334774099616,0.00023778816176702998,0.00042975732457363827,1.0,1.0,1.0,0.1457606362757694,1.0,1.0,1.0,0.04714045207910316,1.0,1.0],"beta":[0.0,-1.336325816789675e-31,0.0,2.67265163357935e-31,1.336325816789675e-31,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.2291919592911267,2.0071503969813826,1.6690346034845973,-1.382256592584588,0.1041483781764238,-3.525166081146842,5.879040067968039e-32,0.0,0.0,-0.5425328458231952,9.664772863474363e-32,0.0,-1.2279424322631803e-30,0.2632805269698914,0.0,0.0],"intercept":12.037037037036841,"uncertainty":3.3217875370595453,"train_mean_delta":12.037037037037036,"train_min_delta":4.0,"train_positive_rate":1.0},"STRAWBERRY|480|503":{"support":54,"rows":54,"mean":[0.0,0.6666666666666669,0.6986111111111121,0.6666666666666669,0.0,0.5,0.1875,-0.3125,0.31944444444444453,0.0,0.0,1.0,0.0,0.7337654320987658,0.985277777777778,-0.003950617283950617,-0.0029012345679012364,0.00038888888888888876,0.0003333333333333335,0.1600000000000001,0.08999999999999994,0.06333333333333335,0.31584666666666683,0.8999999999999991,0.75,0.0,0.0,0.75,0.75],"scale":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.06655869743494955,0.005887987352375029,0.006088310126389112,0.010384506825792778,0.00045201879126239967,0.0007149203529842403,1.0,1.0,0.09843215373488931,0.17238728957613847,1.0,1.0,1.0,1.0,1.0,1.0],"beta":[0.0,0.0,1.2770352713663154e-30,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.32399909890807643,0.6794720194041389,0.8976382836948372,1.1864269946972643,0.5411543159028185,-0.4606486255535138,2.8907754815748927e-31,0.0,-2.584559049554167,-0.4985795011014651,-2.1310958405080863e-30,0.0,0.0,0.0,0.0,0.0],"intercept":7.111111111111138,"uncertainty":1.742847500387789,"train_mean_delta":7.111111111111111,"train_min_delta":-3.0,"train_positive_rate":0.8333333333333334}}}
_RL005_OPPORTUNITIES = rl004_route_opportunities(_ACTIONS)
_RL005_RUNTIME = RL004Runtime(payload=RL005_PAYLOAD, opportunities=_RL005_OPPORTUNITIES)

def agent(obs, config=None):
    """Public entry point; V22 owns every farmer/hand action."""
    try:
        base = _rl005_v22_agent(obs)
        return _RL005_RUNTIME.act(obs, base)
    except Exception:
        _RL005_RUNTIME.errors += 1
        return rl004_align_hands(_rl005_v22_agent(obs), obs)
