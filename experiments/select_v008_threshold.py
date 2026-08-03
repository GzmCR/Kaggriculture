"""Select a V008 threshold from tuning rows and verify it on holdout rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUNING = (17, 42, 2026)
DEFAULT_HOLDOUT = (217, 317, 733)


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def metric(rows, candidate, opponents=None):
    selected = [row for row in rows if row["candidate"] == candidate]
    if opponents:
        selected = [row for row in selected if row["opponent"] in opponents]
    if not selected:
        return None
    money = [float(row["candidate_money"]) for row in selected]
    margins = [float(row["margin"]) for row in selected]
    return {
        "games": len(selected),
        "mean_money": sum(money) / len(money),
        "min_money": min(money),
        "max_money": max(money),
        "mean_margin": sum(margins) / len(margins),
        "wins": sum(row["result"] == "win" for row in selected),
        "ties": sum(row["result"] == "tie" for row in selected),
        "losses": sum(row["result"] == "loss" for row in selected),
        "win_rate": sum(row["result"] == "win" for row in selected) / len(selected),
        "all_done": all(row["game_done"] == "1" for row in selected),
        "agent_errors": sum(int(row["agent_errors"]) for row in selected),
        "invalid_action_shapes": sum(
            int(row["invalid_action_shapes"]) for row in selected
        ),
        "p99_ms": max(float(row["runtime_p99_ms"]) for row in selected),
    }


def slice_rows(rows, seeds):
    wanted = {str(seed) for seed in seeds}
    return [row for row in rows if row["seed"] in wanted]


def evaluate(rows, candidates, seeds):
    rows = slice_rows(rows, seeds)
    current = metric(rows, "v008_current")
    current_hamburger = metric(rows, "v008_current", {"hamburger"})
    report = {
        "seeds": list(seeds),
        "current": current,
        "current_hamburger": current_hamburger,
        "candidates": {},
    }
    for candidate in candidates:
        value = metric(rows, candidate)
        hamburger = metric(rows, candidate, {"hamburger"})
        if value is None or hamburger is None:
            continue
        gain = value["mean_money"] / current["mean_money"] - 1.0
        min_ratio = value["min_money"] / current["min_money"]
        hamburger_ratio = (
            hamburger["mean_money"] / current_hamburger["mean_money"]
        )
        passes = (
            value["all_done"]
            and value["agent_errors"] == 0
            and value["invalid_action_shapes"] == 0
            and value["p99_ms"] < 1000.0
            and gain >= 0.005
            and min_ratio >= 0.97
            and value["win_rate"] >= current["win_rate"]
            and hamburger_ratio >= 0.95
        )
        report["candidates"][candidate] = {
            **value,
            "hamburger_mean_money": hamburger["mean_money"],
            "gain_pct": gain,
            "min_ratio": min_ratio,
            "hamburger_ratio": hamburger_ratio,
            "passes_gate": passes,
        }
    return report


def select(report):
    eligible = [
        (name, value)
        for name, value in report["candidates"].items()
        if value["passes_gate"]
    ]
    if not eligible:
        return None
    best_mean = max(value["mean_money"] for _, value in eligible)
    close = [
        item
        for item in eligible
        if (best_mean - item[1]["mean_money"]) / best_mean <= 0.0025
    ]
    close.sort(
        key=lambda item: (
            item[1]["min_money"],
            item[1]["win_rate"],
            item[1]["hamburger_mean_money"],
            item[1]["mean_money"],
        ),
        reverse=True,
    )
    return close[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        nargs="+",
        default=[ROOT / "baseline/artifacts/v008_hybrid_router/v008_raw.csv"],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "baseline/artifacts/v008_hybrid_router/v008_selection.json")
    parser.add_argument("--tuning", type=int, nargs="+", default=list(DEFAULT_TUNING))
    parser.add_argument("--holdout", type=int, nargs="+", default=list(DEFAULT_HOLDOUT))
    args = parser.parse_args()

    rows = load_rows(args.raw)
    candidates = sorted({row["candidate"] for row in rows if row["candidate"] != "v008_current"})
    tuning = evaluate(rows, candidates, args.tuning)
    selected = select(tuning)
    holdout = evaluate(rows, candidates, args.holdout)
    report = {
        "selected": selected,
        "tuning": tuning,
        "holdout": holdout,
        "holdout_passes_selected": bool(
            selected and holdout["candidates"].get(selected, {}).get("passes_gate")
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
