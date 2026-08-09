"""Run RL-004 decisions over the online RL-003 losslog as holdout."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    spec = importlib.util.spec_from_file_location(f"rl004_holdout_{time.time_ns()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(input_dir, output_dir, candidate_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = [agent.get("Name", "") for agent in payload.get("info", {}).get("Agents", [])]
        if "GzmCR632" not in names:
            continue
        seat = names.index("GzmCR632")
        module = _load(candidate_path)
        for index in range(min(720, len(payload.get("steps", [])))):
            obs = dict(payload["steps"][index][seat].get("observation") or {})
            if "step" not in obs:
                obs["step"] = index
            module.agent(obs)
        runtime = getattr(module, "_RL004_RUNTIME", None)
        decisions = list(getattr(runtime, "decisions", []) or [])
        selected = [row for row in decisions if row.get("selected")]
        rows.append({
            "replay": path.name,
            "episode_id": payload.get("info", {}).get("EpisodeId"),
            "seed": payload.get("info", {}).get("seed"),
            "decision_points": len(decisions),
            "selected_events": len(selected),
            "selected_keys": ";".join(
                f"{row.get('item')}:{row.get('step')}->{row.get('future_step')}"
                for row in selected
            ),
            "unsupported_events": sum(row.get("reason") == "unsupported_event" for row in decisions),
            "errors": int(getattr(runtime, "errors", 0) or 0),
        })
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "losslog_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "losslog_decisions.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "baseline/artifacts/rl_003_trade_timing/losslog",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "baseline/artifacts/rl_004_trade_timing/main.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baseline/artifacts/rl_004_trade_timing/losslog_holdout",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.candidate), indent=2))
