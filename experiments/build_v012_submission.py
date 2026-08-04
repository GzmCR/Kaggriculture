"""Build a self-contained Kaggle submission for the V012 replaced-v18 agent."""

from __future__ import annotations

import ast
import base64
import json
import tarfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "baseline/40-53-top-10-future-holdout-v18-closed-loop.ipynb"
LOG_DIR = ROOT / "log/2026-08-04"
HISTORY_DIR = ROOT / "baseline/history/v012_top5_replaced_v18"
ARTIFACT_DIR = ROOT / "baseline/artifacts/v012_top5_replaced_v18"
MAIN_PATH = HISTORY_DIR / "main.py"
ARCHIVE_PATH = ARTIFACT_DIR / "submission.tar.gz"

SPECS = (
    {"name": "mohit", "file": "89817349", "seat": 0},
    {"name": "automatylicza", "file": "89830916", "seat": 0},
    {"name": "manual_player", "file": "89820316", "seat": 0},
    {"name": "navazsh_fathi", "file": "89830910", "seat": 0},
    {"name": "lucien_de_rubempre", "file": "89822684", "seat": 1},
)
BOARD_ROUTE = "automatylicza"


def decode_v18_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        if "payload =" not in text or "b85decode" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id != "payload":
                continue
            packed = "".join(ast.literal_eval(node.value.args[0]))
            return zlib.decompress(base64.b85decode(packed)).decode("utf-8")
    raise RuntimeError("Could not decode v18 source")


def load_route(spec):
    payload = json.loads((LOG_DIR / f"{spec['file']}.json").read_text(encoding="utf-8"))
    steps = payload["steps"]
    seat = int(spec["seat"])
    actions = [
        steps[min(index + 1, 719)][seat].get("action") or {}
        for index in range(720)
    ]
    observations = [
        steps[index][seat].get("observation") or {}
        for index in range(720)
    ]
    return {**spec, "actions": actions, "observations": observations}


def build_runtime_payload(source, routes):
    namespace = {"__name__": "v012_builder_v18"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    state_features = namespace["_v18_state_features"]
    experts = {}
    for name, route in routes.items():
        features = [state_features(obs) for obs in route["observations"]]
        experts[name] = {
            "actions": route["actions"],
            "prototypes_by_day": [features[min(day * 24, 719)] for day in range(30)],
            "board_prototype_at_fork": features[632],
        }
    return {
        "experts": experts,
        "board_route": BOARD_ROUTE,
        "board_by_seat": {"0": BOARD_ROUTE, "1": BOARD_ROUTE},
        "market_bias_by_seat": {
            "0": {name: (0.75 if name == BOARD_ROUTE else 0.0) for name in experts},
            "1": {name: (0.75 if name == BOARD_ROUTE else 0.0) for name in experts},
        },
        "board_bias_by_seat": {
            "0": {name: (0.75 if name == BOARD_ROUTE else 0.0) for name in experts},
            "1": {name: (0.75 if name == BOARD_ROUTE else 0.0) for name in experts},
        },
        "distance_strength": 0.5,
        "stay_bonus": 0.5,
        "board_distance_strength": 0.0,
    }


def chunked(value, size=100):
    return [value[index : index + size] for index in range(0, len(value), size)]


def build_main():
    source = decode_v18_source()
    routes = {spec["name"]: load_route(spec) for spec in SPECS}
    runtime = build_runtime_payload(source, routes)
    packed = base64.b85encode(
        zlib.compress(json.dumps(runtime, separators=(",", ":")).encode("utf-8"), 9)
    ).decode("ascii")
    chunks = chunked(packed)
    injection = [
        "",
        "# V012 top-five replay expert replacement.",
        "_V012_EXPERT_PAYLOAD = ''.join([",
    ]
    injection.extend(f"    {chunk!r}," for chunk in chunks)
    injection.extend([
        "])\n",
        "_V012_RUNTIME = json.loads(zlib.decompress(base64.b85decode(_V012_EXPERT_PAYLOAD)).decode('utf-8'))",
        "_V18_RUNTIME['experts'] = _V012_RUNTIME['experts']",
        "_V18_RUNTIME['board_by_seat'] = _V012_RUNTIME['board_by_seat']",
        "_V18_RUNTIME['market_bias_by_seat'] = _V012_RUNTIME['market_bias_by_seat']",
        "_V18_RUNTIME['board_bias_by_seat'] = _V012_RUNTIME['board_bias_by_seat']",
        "_V18_RUNTIME['distance_strength'] = _V012_RUNTIME['distance_strength']",
        "_V18_RUNTIME['stay_bonus'] = _V012_RUNTIME['stay_bonus']",
        "_V18_RUNTIME['board_distance_strength'] = _V012_RUNTIME['board_distance_strength']",
        "STRATEGY['v18_closed_loop_board'] = True",
        "STRATEGY['v18_closed_loop_market'] = True",
    ])
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_PATH.write_text(source + "\n".join(injection) + "\n", encoding="utf-8")
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        archive.add(MAIN_PATH, arcname="main.py")
    (ARTIFACT_DIR / "submission_manifest.json").write_text(
        json.dumps({
            "main": str(MAIN_PATH),
            "archive": str(ARCHIVE_PATH),
            "board_route": BOARD_ROUTE,
            "market_experts": list(routes),
            "self_contained": True,
            "depends_on_local_logs_at_runtime": False,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {MAIN_PATH} ({MAIN_PATH.stat().st_size} bytes)")
    print(f"wrote {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    build_main()
