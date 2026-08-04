"""Static contract checks for the first crop-structure optimization."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    baseline = load(ROOT / "main.py", "v010_baseline_contract")
    assert any("CARROT" in mix for mix in baseline.CROP_MIX.values())
    for label in ("v010a_crop_mix", "v010b_carrot_half"):
        candidate = load(
            ROOT / "baseline/history" / label / "main.py",
            f"{label}_contract",
        )
        for name in ("ANIMALS", "MARKET", "MAX_HANDS", "LIQUIDATION_TURNS"):
            assert getattr(candidate, name) == getattr(baseline, name), name
        assert candidate.CROP_MIX != baseline.CROP_MIX
        assert callable(candidate.agent)
    aggressive = load(
        ROOT / "baseline/history/v010a_crop_mix/main.py",
        "v010a_crop_contract",
    )
    assert all("CARROT" not in mix for mix in aggressive.CROP_MIX.values())
    print("V010 crop invariants: PASS")


if __name__ == "__main__":
    main()
