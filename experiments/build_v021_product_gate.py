"""Build self-contained V021 per-product market-gate candidates."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V012 = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
V020 = ROOT / "experiments/v020_value_aware_market.py"
V021 = ROOT / "experiments/v021_product_gate.py"
HISTORY = ROOT / "baseline/history/v021_product_gate"
ARTIFACT = ROOT / "baseline/artifacts/v021_product_gate"

VARIANTS = {
    "safety_patch": "v021a_safety_patch.py",
    "product_gate": "v021b_product_gate.py",
    "win_guard": "v021c_win_guard.py",
}

WRAPPER = r'''

# V021 wrapper: V012 field route plus bounded per-product market gates.
_V021_RUNTIME = globals().get("_V012_RUNTIME", {})
_V021_CONTROLLER = V021ProductGateController(
    variant=_V021_VARIANT,
    runtime=_V021_RUNTIME,
)


def agent(obs, config=None):
    base = None
    try:
        try:
            base = _v021_base_agent(obs, config)
        except TypeError:
            base = _v021_base_agent(obs)
        return _V021_CONTROLLER.apply(obs, base)
    except Exception:
        if isinstance(base, dict):
            return {
                "farmer": list(base.get("farmer") or ["PASS"]),
                "hands": [list(item) for item in (base.get("hands") or [])],
                "market": [list(order) for order in (base.get("market") or [])[:10]],
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}
'''


def _strip_future_import(source):
    return source.replace("from __future__ import annotations\n\n", "", 1)


def build(variant, target):
    base = V012.read_text(encoding="utf-8")
    marker = "def agent(obs):"
    if marker not in base:
        raise RuntimeError("V012 source agent entrypoint was not found")
    base = base.replace(marker, "def _v021_base_agent(obs):", 1)
    v020 = _strip_future_import(V020.read_text(encoding="utf-8"))
    v021 = _strip_future_import(V021.read_text(encoding="utf-8"))
    # The experiment module's optional import block is unnecessary after the
    # V020 source has been concatenated into the same namespace.
    v021 = v021.replace(
        "try:\n    from v020_value_aware_market import (\n        BALANCED_START,\n        HORIZON,\n        MARKET,\n        MAX_MARKET_ORDERS,\n        PREMIUM_PRODUCTS,\n        ValueAwareMarketController,\n        _copy_action,\n        _expected_future_orders,\n        _int,\n        _num,\n        _market_for,\n        _opponent_supply_profile,\n    )\nexcept ImportError:\n    # The self-contained build has the V020 definitions immediately before\n    # this code, so the names above already exist in that namespace.\n    pass\n\n\n",
        "",
        1,
    )
    payload = (
        base
        + "\n\n"
        + v020
        + "\n\n"
        + v021
        + "\n\n"
        + f"_V021_VARIANT = {variant!r}\n"
        + WRAPPER
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target


def build_all():
    HISTORY.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    targets = {}
    archives = {}
    for variant, filename in VARIANTS.items():
        target = ARTIFACT / filename
        build(variant, target)
        targets[variant] = target
        archive_path = ARTIFACT / f"submission_v021_{variant}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(target, arcname="main.py")
        archives[variant] = archive_path

    history_main = build("win_guard", HISTORY / "main.py")
    artifact_main = build("win_guard", ARTIFACT / "main.py")
    default_archive = ARTIFACT / "submission.tar.gz"
    with tarfile.open(default_archive, "w:gz") as archive:
        archive.add(artifact_main, arcname="main.py")
    (ARTIFACT / "submission_manifest.json").write_text(
        json.dumps({
            "main": str(artifact_main),
            "history_main": str(history_main),
            "archive": str(default_archive),
            "variant_archives": {name: str(path) for name, path in archives.items()},
            "base": "baseline/history/v012_top5_replaced_v18/main.py",
            "field_route_unchanged": True,
            "self_contained": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"built {len(targets)} V021 candidates in {ARTIFACT}")


if __name__ == "__main__":
    build_all()
