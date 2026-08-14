"""Generate the offline V032 opponent replay pool from executable notebooks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kaggle_environments import make

from rl_010_opponents import unique_loadable_specs, load_spec


ROOT = Path(__file__).resolve().parents[1]


def run(output: Path, seeds=(17, 42, 2026), opponents=None):
    output.mkdir(parents=True, exist_ok=True)
    specs, rows = unique_loadable_specs()
    wanted = set(opponents or [])
    if wanted:
        specs = [row for row in specs if row["name"] in wanted]
    manifest = {"engine": "kaggle-environments==1.32.6", "seeds": list(seeds), "agents": rows, "games": []}
    for spec in specs:
        agent, metadata = load_spec(spec)
        for seed in seeds:
            for candidate_seat in (0, 1):
                path = output / f"{metadata['source_sha256'][:12]}_{int(seed)}_s{candidate_seat}_{time.time_ns()}.json"
                env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": int(seed)}, debug=False)
                players = [agent, "starter"] if candidate_seat == 0 else ["starter", agent]
                env.run(players)
                path.write_text(json.dumps(env.toJSON(), separators=(",", ":")) + "\n", encoding="utf-8")
                manifest["games"].append({"file": path.name, "source_sha256": metadata["source_sha256"], "seed": int(seed), "seat": candidate_seat})
                print(f"wrote {path.name}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/artifacts/v032_route_conditioned_timing/opponent_replays")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026])
    parser.add_argument("--opponents", nargs="*")
    args = parser.parse_args()
    print(json.dumps(run(args.output, tuple(args.seeds), args.opponents), indent=2))
