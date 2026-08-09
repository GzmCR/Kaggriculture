"""Invariant checks for RL-007 turn-level preemption."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from rl_007_next_turn_preemption import (
    RL007_ACTIONS,
    RL007_FEATURE_DIM,
    rl007_append_sell,
    rl007_fit_models,
    rl007_reduce_sell,
    rl007_route_opportunities,
    rl007_shift_quantity,
)


def main():
    route = [{"market": []} for _ in range(6)]
    route[5] = {"market": [["SELL", "MELON", 12]]}
    rows = rl007_route_opportunities(route)
    assert {row["horizon"] for row in rows} == {1, 2, 3}
    assert {row["current_step"] for row in rows} == {2, 3, 4}
    for row in rows:
        action = {"market": []}
        quantity = rl007_shift_quantity(row)
        assert rl007_append_sell(action, "MELON", quantity) == quantity
        assert rl007_reduce_sell(action, "MELON", quantity) == quantity
        assert action["market"] == []

    samples = []
    for action_id in RL007_ACTIONS:
        for group in range(12):
            samples.append({
                "action_id": action_id,
                "seed": group,
                "seat": 0,
                "opponent_source_sha256": str(group),
                "features": np.zeros(RL007_FEATURE_DIM).tolist(),
                "cash_delta": 10.0 + action_id,
                "shift_applied": 1,
                "item": "MELON",
                "current_step": 100 + action_id,
                "future_step": 101 + action_id,
                "horizon": action_id,
            })
    payload, report = rl007_fit_models(samples)
    assert len(payload["models"]) == 3
    assert report["feature_dim"] == RL007_FEATURE_DIM

    built = Path("baseline/artifacts/rl_007_next_turn_preemption/next_turn/main.py")
    if built.exists():
        tree = ast.parse(built.read_text(encoding="utf-8"))
        callables = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert callables[-1] == "agent"
    print("RL-007 invariant tests passed")


if __name__ == "__main__":
    main()
