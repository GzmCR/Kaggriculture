"""Public-state style features and conservative V019 expert routing.

The router never reads replay filenames, TeamNames, ratings, or private
opponent inventory.  It only consumes the same public farm and shared-market
fields that a submitted Kaggriculture agent receives.
"""

from __future__ import annotations

import copy
from collections import Counter


STYLE_NAMES = (
    "standard_converged",
    "reduced_ne_only",
    "high_worker_maintenance",
    "premium_concentrated",
)
PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")
BASE_PRICES = {"MELON": 250, "STRAWBERRY": 120, "MILK": 160, "WOOL": 200}


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _opponent_farm(obs):
    farms = obs.get("farms", []) if isinstance(obs, dict) else []
    player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
    other = 1 - player
    if isinstance(farms, list) and 0 <= other < len(farms) and isinstance(farms[other], dict):
        return farms[other]
    return {}


def public_style_features(obs):
    """Return compact opponent features available to a live agent."""
    farm = _opponent_farm(obs)
    tiles = farm.get("tiles", []) if isinstance(farm, dict) else []
    crops = Counter()
    animals = Counter()
    ready = Counter()
    weeds = 0
    for row in tiles if isinstance(tiles, list) else []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = str(tile.get("crop", "")).upper()
                crops[crop] += 1
                ready[crop] += max(0, _int(tile.get("yield_units", 0)))
            elif kind in {"COOP", "PASTURE"}:
                animal = str(tile.get("animal", "")).upper()
                animals[animal] += 1
                ready[animal] += max(0, _int(tile.get("yield_units", 0)))
            elif kind == "WEED":
                weeds += 1

    market = obs.get("market", {}) if isinstance(obs, dict) else {}
    prices = market.get("prices", {}) if isinstance(market, dict) else {}
    premium_pipeline = sum(crops[item] + 2.0 * ready[item] for item in ("MELON", "STRAWBERRY"))
    premium_pipeline += animals["COW"] + animals["SHEEP"]
    premium_price_ratio = sum(
        _float(prices.get(item, BASE_PRICES[item]), BASE_PRICES[item]) / BASE_PRICES[item]
        for item in PREMIUM
    ) / len(PREMIUM)
    return {
        "day": _int(obs.get("day", 0)),
        "hour": _int(obs.get("hour", 0)),
        "hands": len(farm.get("hands", []) or []),
        "unlocked_count": len(farm.get("unlocked_quadrants", []) or []),
        "has_NE": int("NE" in (farm.get("unlocked_quadrants", []) or [])),
        "has_SW": int("SW" in (farm.get("unlocked_quadrants", []) or [])),
        "has_SE": int("SE" in (farm.get("unlocked_quadrants", []) or [])),
        "crops_total": sum(crops.values()),
        "animals_total": sum(animals.values()),
        "cows": animals["COW"],
        "sheep": animals["SHEEP"],
        "geese": animals["GOOSE"],
        "premium_pipeline": premium_pipeline,
        "premium_price_ratio": premium_price_ratio,
        "weeds": weeds,
        **{f"crop_{key}": value for key, value in crops.items()},
        **{f"ready_{key}": value for key, value in ready.items()},
    }


class PublicStyleTracker:
    """Stateful daily classifier with persistence and confidence."""

    def __init__(self):
        self.last_step = -1
        self.max_hands = 0
        self.max_pipeline = 0.0
        self.last_style = "standard_converged"
        self.last_confidence = 0.0
        self.history = []

    def reset(self):
        self.__init__()

    def observe(self, obs):
        step = _int(obs.get("step", self.last_step + 1)) if isinstance(obs, dict) else self.last_step + 1
        if step == 0 or step <= self.last_step:
            self.reset()
        self.last_step = step
        features = public_style_features(obs)
        self.max_hands = max(self.max_hands, features["hands"])
        self.max_pipeline = max(self.max_pipeline, features["premium_pipeline"])
        day = features["day"]

        # The reduced route is visible before its sheep necessarily disappear.
        # In the replay set it typically has only NE unlocked, about six or
        # fewer cows, and at most a couple of sheep by day 10.  Waiting for
        # sheep == 0 would classify it only after the production failure had
        # already happened (often at the final step).  The total-animal/crop
        # fallback also covers a player whose livestock count is temporarily
        # hidden by a production transition, while remaining well below the
        # converged 8-cow/5-sheep structure.
        reduced_route = (
            day >= 10
            and not features["has_SW"]
            and (
                (features["cows"] <= 6 and features["sheep"] <= 2)
                or (
                    features["animals_total"] <= 8
                    and features["crops_total"] <= 22
                )
            )
        )
        if reduced_route:
            style, confidence = "reduced_ne_only", 1.0
        elif self.max_hands >= 14:
            style, confidence = "high_worker_maintenance", 0.95
        elif (
            day >= 10
            and self.max_pipeline >= 18
            and features["premium_price_ratio"] <= 0.80
        ):
            style, confidence = "premium_concentrated", 0.75
        else:
            style, confidence = "standard_converged", 0.60

        # Do not let a one-turn ambiguous signal churn the market lane.
        if self.last_style != style and confidence < 0.90:
            style = self.last_style
            confidence = min(confidence, self.last_confidence)
        self.last_style = style
        self.last_confidence = confidence
        if not self.history or self.history[-1]["step"] != step:
            self.history.append({"step": step, "style": style, "confidence": confidence, "features": dict(features)})
        return style, confidence, features


class PublicStyleExpertRouter:
    """Select a complete embedded market expert at daily boundaries."""

    def __init__(self, mapping=None, hold_days=1):
        self.mapping = dict(mapping or {})
        self.hold_days = max(1, _int(hold_days, 1))
        self._states = {}

    def _state(self, player):
        player = _int(player)
        if player not in self._states:
            self._states[player] = {
                "last_step": -1,
                "selected_day": None,
                "selected_expert": None,
                "selected_style": None,
                "selected_confidence": 0.0,
                "tracker": PublicStyleTracker(),
                "history": [],
                "stats": Counter(),
            }
        return self._states[player]

    def reset(self, player=None):
        if player is None:
            self._states.clear()
        else:
            self._states.pop(_int(player), None)

    def choose(self, obs, available_experts, fallback):
        player = _int(obs.get("player", 0)) if isinstance(obs, dict) else 0
        state = self._state(player)
        step = _int(obs.get("step", state["last_step"] + 1)) if isinstance(obs, dict) else state["last_step"] + 1
        if step == 0 or step <= state["last_step"]:
            self.reset(player)
            state = self._state(player)
        state["last_step"] = step
        style, confidence, features = state["tracker"].observe(obs)
        day = _int(obs.get("day", step // 24)) if isinstance(obs, dict) else step // 24
        if (
            state["selected_expert"] is None
            or state["selected_day"] is None
            or day - state["selected_day"] >= self.hold_days
        ):
            candidate = self.mapping.get(style, fallback)
            if candidate not in available_experts:
                candidate = fallback if fallback in available_experts else sorted(available_experts)[0]
            state["selected_expert"] = candidate
            state["selected_style"] = style
            state["selected_confidence"] = confidence
            state["selected_day"] = day
            state["stats"]["selections"] += 1
            if candidate != fallback:
                state["stats"]["non_default_selections"] += 1
            state["history"].append({
                "step": step,
                "day": day,
                "style": style,
                "confidence": confidence,
                "expert": candidate,
                "features": dict(features),
            })
        return state["selected_expert"], style, confidence, features

    def diagnostics(self, player=0):
        state = self._state(player)
        result = dict(state["stats"])
        result.update({
            "selected_expert": state["selected_expert"],
            "selected_style": state["selected_style"],
            "selected_confidence": state["selected_confidence"],
            "history": copy.deepcopy(state["history"]),
        })
        return result
