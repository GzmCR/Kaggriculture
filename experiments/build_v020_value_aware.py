"""Build self-contained V020 value-aware market submissions."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V012 = ROOT / "baseline/history/v012_top5_replaced_v18/main.py"
CONTROLLER = ROOT / "experiments/v020_value_aware_market.py"
HISTORY = ROOT / "baseline/history/v020_value_aware_market"
ARTIFACT = ROOT / "baseline/artifacts/v020_value_aware_market"
ARCHIVE = ARTIFACT / "submission.tar.gz"


WRAPPER = r'''

# V020 value-aware market overlay.  V012 remains the immutable field route.
_V020_RUNTIME = globals().get("_V012_RUNTIME", {})
_V020_CONTROLLER = ValueAwareMarketController(
    variant=_V020_VARIANT,
    runtime=_V020_RUNTIME,
)


def agent(obs, config=None):
    base = None
    try:
        try:
            base = _v020_base_agent(obs, config)
        except TypeError:
            base = _v020_base_agent(obs)
        return _V020_CONTROLLER.apply(obs, base)
    except Exception:
        if isinstance(base, dict):
            return {
                "farmer": list(base.get("farmer") or ["PASS"]),
                "hands": [list(item) for item in (base.get("hands") or [])],
                "market": [list(order) for order in (base.get("market") or [])[:10]],
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}
'''


def build(variant, target):
    base = V012.read_text(encoding="utf-8")
    marker = "def agent(obs):"
    if marker not in base:
        raise RuntimeError("V012 source agent entrypoint was not found")
    base = base.replace(marker, "def _v020_base_agent(obs):", 1)
    controller = CONTROLLER.read_text(encoding="utf-8")
    controller = controller.replace("from __future__ import annotations\n\n", "", 1)
    payload = (
        base
        + "\n\n"
        + controller
        + "\n\n"
        + f"_V020_VARIANT = {variant!r}\n"
        + WRAPPER
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target


def build_all():
    HISTORY.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    targets = {}
    for variant in ("balanced", "sensitive", "conservative"):
        target = ARTIFACT / f"v020_{variant}.py"
        build(variant, target)
        targets[variant] = target
    history_main = build("balanced", HISTORY / "main.py")
    artifact_main = build("balanced", ARTIFACT / "main.py")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(history_main, arcname="main.py")
    (ARTIFACT / "submission_manifest.json").write_text(
        json.dumps({
            "main": str(artifact_main),
            "history_main": str(history_main),
            "archive": str(ARCHIVE),
            "base": "baseline/history/v012_top5_replaced_v18/main.py",
            "variants": {name: str(path) for name, path in targets.items()},
            "field_route_unchanged": True,
            "self_contained": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {history_main} ({history_main.stat().st_size} bytes)")
    print(f"wrote {artifact_main} ({artifact_main.stat().st_size} bytes)")
    print(f"wrote {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)")


if __name__ == "__main__":
    build_all()
