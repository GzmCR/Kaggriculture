"""Invariant checks for RL-008 small-quantity bidirectional timing."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from rl_008_small_shift_timing import (
    RL008_ACTION_NAMES,
    RL008_FEATURE_DIM,
    rl008_action_matches,
    rl008_action_quantity,
    rl008_delay_opportunities,
    rl008_fit_models,
    rl008_preempt_opportunities,
    rl008_shift_sell,
)


def main():
    route = [{"market": []} for _ in range(8)]
    route[2] = {"market": [["SELL", "MELON", 12]]}
    route[7] = {"market": [["SELL", "MELON", 8]]}
    preempt = rl008_preempt_opportunities(route, [("MELON", 7)])
    assert {row["horizon"] for row in preempt} == {1, 2, 3}
    delays = rl008_delay_opportunities(route)
    assert len(delays) == 1 and delays[0]["current_step"] == 2 and delays[0]["future_step"] == 7

    by_horizon = {row["horizon"]: row for row in preempt}
    assert rl008_action_quantity(1, by_horizon[1]) == 1
    assert rl008_action_quantity(2, by_horizon[1]) == 2
    assert rl008_action_quantity(3, by_horizon[2]) == 1
    assert rl008_action_quantity(4, by_horizon[2]) == 2
    assert rl008_action_quantity(5, by_horizon[3]) == 1
    assert rl008_action_quantity(6, by_horizon[3]) == 2
    assert rl008_action_matches(1, by_horizon[1])
    assert not rl008_action_matches(3, by_horizon[1])

    delay = delays[0]
    assert rl008_action_quantity(7, delay) == 1
    assert rl008_action_quantity(8, delay) == 3
    assert rl008_action_quantity(9, delay) == 6

    action = {"market": [["SELL", "MELON", 2], ["SELL", "WHEAT", 1], ["SELL", "MELON", 5]]}
    assert rl008_shift_sell(action, "MELON", -6) == 6
    assert action["market"] == [["SELL", "WHEAT", 1], ["SELL", "MELON", 1]]
    assert rl008_shift_sell(action, "MELON", 4) == 4
    assert action["market"][-1] == ["SELL", "MELON", 5]

    samples = []
    for action_id in range(1, 10):
        for group in range(12):
            samples.append({
                "action_id": action_id,
                "seed": group,
                "seat": 0,
                "opponent_source_sha256": str(group),
                "features": np.zeros(RL008_FEATURE_DIM).tolist(),
                "cash_delta": 10.0 + action_id,
                "shift_applied": 1,
                "future_repaid": 1,
                "item": "MELON",
                "kind": "PREEMPT",
                "current_step": 100,
                "future_step": 101,
                "horizon": 1,
            })
    payload, report = rl008_fit_models(samples)
    assert set(payload["models"]) == {str(index) for index in range(1, 10)}
    assert report["feature_dim"] == RL008_FEATURE_DIM
    assert set(RL008_ACTION_NAMES) == set(range(10))

    for path in (
        Path("baseline/artifacts/rl_008_small_shift_timing/gated_preempt/main.py"),
        Path("baseline/artifacts/rl_008_small_shift_timing/ungated_bidirectional/main.py"),
    ):
        if path.exists():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            callables = [
                node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            assert callables[-1] == "agent"

    print("RL-008 invariant tests passed")


if __name__ == "__main__":
    main()
