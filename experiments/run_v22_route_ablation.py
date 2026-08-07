"""Compare the 44-46 route with and without its price-impact layer."""

from pathlib import Path

from run_v22_validation import OPPONENTS, ROOT, SEEDS, run


if __name__ == "__main__":
    run(
        ("v22", "v22_route_only"),
        OPPONENTS,
        SEEDS,
        ROOT / "baseline/artifacts/v22_route_ablation",
    )
