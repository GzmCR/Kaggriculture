"""Build self-contained V018 market-wave candidates.

The candidate is deliberately assembled from the immutable V012 route, the
exact V017 market helpers, the V018 controller, and the V015a protection
layer.  This keeps the file suitable for Kaggle's single-file loader while
leaving the active root ``main.py`` untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V012 = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
ROLLOUT = ROOT / "experiments/v017_market_rollout.py"
WAVE = ROOT / "experiments/v018_market_wave.py"
OVERLAY = ROOT / "experiments/v015a_market_overlay.py"
OUT = ROOT / "baseline/artifacts/v018_market_wave"
HISTORY = ROOT / "baseline/history/v018_market_wave"


V018_IMPORT_BLOCK = '''from v017_market_rollout import (
    MARKET,
    PREMIUM_PRODUCTS,
    MAX_MARKET_ORDERS,
    _copy_action,
    _farm_for,
    _inventory_total,
    _opponent_supply_total,
    _sell_lockstep,
    _town_demand,
    _int,
    _num,
    price_at,
)
'''


WRAPPER = r'''

# V018 wrapper: V012 route -> wave/MPC quantity controller -> V015a guard.
_V018_BASE_AGENT = agent
_V018_CONTROLLER = MarketWaveController(
    variant=_V018_VARIANT,
    runtime=_V18_RUNTIME,
    selected_state=_V18_SELECTED_MARKET,
    pipeline_fn=_farm_pipeline,
)
_V018_OVERLAY = MarketCollisionOverlay()


def agent(obs, config=None):
    prepared = prepare_observation(obs, _V018_OVERLAY)
    try:
        base_action = _V018_BASE_AGENT(prepared, config)
    except TypeError:
        base_action = _V018_BASE_AGENT(prepared)
    wave_action = _V018_CONTROLLER.apply(prepared, base_action)
    return _V018_OVERLAY.apply(prepared, wave_action)
'''


def _strip_future(source):
    return source.replace("from __future__ import annotations\n\n", "", 1)


def build(variant, path):
    base = V012.read_text(encoding="utf-8")
    rollout = _strip_future(ROLLOUT.read_text(encoding="utf-8"))
    wave = _strip_future(WAVE.read_text(encoding="utf-8"))
    wave = wave.replace(V018_IMPORT_BLOCK, "", 1)
    overlay = _strip_future(OVERLAY.read_text(encoding="utf-8"))
    payload = (
        base
        + "\n\n"
        + rollout
        + "\n\n"
        + wave
        + "\n\n"
        + overlay
        + "\n\n"
        + f"_V018_VARIANT = {variant!r}\n"
        + WRAPPER
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("fixed_wave", "daily_mpc", "robust_mpc"), default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)
    variants = [args.variant] if args.variant else ["fixed_wave", "daily_mpc", "robust_mpc"]
    filenames = {
        "fixed_wave": "v018a_fixed_wave.py",
        "daily_mpc": "v018b_daily_mpc.py",
        "robust_mpc": "v018c_robust_mpc.py",
    }
    for variant in variants:
        build(variant, OUT / filenames[variant])
    # Keep the most conservative adaptive candidate as the history entry;
    # this is not a promotion to the root submission.
    build("robust_mpc", HISTORY / "main.py")
    print("built", ", ".join(variants))


if __name__ == "__main__":
    main()
