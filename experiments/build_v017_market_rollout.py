"""Build self-contained V017 candidates from the immutable V012 source."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V012 = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
ROLLOUT = ROOT / "experiments/v017_market_rollout.py"
OVERLAY = ROOT / "experiments/v015a_market_overlay.py"
OUT = ROOT / "baseline/artifacts/v017_market_rollout"
HISTORY = ROOT / "baseline/history/v017_market_rollout"


WRAPPER = r'''

# V017 wrapper: fixed V012 field route, product-level rollout, then V015a.
_V017_BASE_AGENT = agent
_V017_CONTROLLER = MarketRolloutController(
    variant=_V017_VARIANT,
    runtime=_V18_RUNTIME,
    selected_state=_V18_SELECTED_MARKET,
    pipeline_fn=_farm_pipeline,
)
_V017_OVERLAY = MarketCollisionOverlay()


def agent(obs, config=None):
    prepared = prepare_observation(obs, _V017_OVERLAY)
    try:
        base_action = _V017_BASE_AGENT(prepared, config)
    except TypeError:
        base_action = _V017_BASE_AGENT(prepared)
    rollout_action = _V017_CONTROLLER.apply(prepared, base_action)
    return _V017_OVERLAY.apply(prepared, rollout_action)
'''


def build(variant, path):
    base = V012.read_text(encoding="utf-8")
    rollout = ROLLOUT.read_text(encoding="utf-8")
    rollout = rollout.replace("from __future__ import annotations\n\n", "", 1)
    overlay = OVERLAY.read_text(encoding="utf-8")
    payload = (
        base
        + "\n\n"
        + rollout
        + "\n\n"
        + overlay
        + "\n\n"
        + f"_V017_VARIANT = {variant!r}\n"
        + WRAPPER
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("curve_only", "opponent_aware", "robust_quota"), default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)
    variants = [args.variant] if args.variant else ["curve_only", "opponent_aware", "robust_quota"]
    filenames = {
        "curve_only": "v017a_curve_only.py",
        "opponent_aware": "v017b_opponent_aware.py",
        "robust_quota": "v017c_robust_quota.py",
    }
    for variant in variants:
        build(variant, OUT / filenames[variant])
        if variant == "robust_quota":
            build(variant, HISTORY / "main.py")
    print("built", ", ".join(variants))


if __name__ == "__main__":
    main()

