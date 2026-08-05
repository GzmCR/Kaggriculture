"""Build self-contained V016 candidate files from the immutable V012 source."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V012 = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
SELECTOR = ROOT / "experiments/v016_market_selector.py"
OVERLAY = ROOT / "experiments/v015a_market_overlay.py"
OUT = ROOT / "baseline/artifacts/v016_market_value_selector"
HISTORY = ROOT / "baseline/history/v016_market_value_selector"


WRAPPER = r'''

# V016 wrapper: V012 owns the complete farmer/hands route; this layer owns
# only daily market expert selection and the V015a premium collision guard.
_V016_BASE_AGENT = agent
_V016_SELECTOR = MarketValueSelector(
    variant=_V016_VARIANT,
    runtime=_V18_RUNTIME,
    pipeline_fn=_farm_pipeline,
)
_V016_OVERLAY = MarketCollisionOverlay()


def agent(obs, config=None):
    prepared = prepare_observation(obs, _V016_OVERLAY)
    try:
        base_action = _V016_BASE_AGENT(prepared, config)
    except TypeError:
        base_action = _V016_BASE_AGENT(prepared)
    selected_action = _V016_SELECTOR.apply(prepared, base_action)
    return _V016_OVERLAY.apply(prepared, selected_action)
'''


def build(variant, path):
    base = V012.read_text(encoding="utf-8")
    selector = SELECTOR.read_text(encoding="utf-8")
    # Future imports are legal only at the beginning of a Python file.  The
    # selector is also imported as a normal module by tests, but is appended
    # after V012 in a Kaggle self-contained artifact.
    selector = selector.replace("from __future__ import annotations\n\n", "", 1)
    overlay = OVERLAY.read_text(encoding="utf-8")
    payload = base + "\n\n" + selector + "\n\n" + overlay + "\n\n" + f"_V016_VARIANT = {variant!r}\n" + WRAPPER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("value_only", "collision_hedged", "aggressive_value"), default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)
    variants = [args.variant] if args.variant else ["value_only", "collision_hedged", "aggressive_value"]
    for variant in variants:
        filename = {
            "value_only": "v016a_value_only.py",
            "collision_hedged": "v016b_collision_hedged.py",
            "aggressive_value": "v016c_aggressive_value.py",
        }[variant]
        build(variant, OUT / filename)
        if variant == "collision_hedged":
            build(variant, HISTORY / "main.py")
    print("built", ", ".join(variants))


if __name__ == "__main__":
    main()
