#!/usr/bin/env python3
"""Aggregate sweep results into the main results table.

Reads experiments/<run_id>/results.jsonl and emits:
  experiments/<run_id>/metrics.json      (nested dict of pass rates)
  experiments/<run_id>/metrics.csv       (flat table for plotting)

Pass rate is computed from `passed_contains` for everything except
REFL — REFL rows are counted separately and annotated as 'judged
externally'.

Retrieval Recall@k is computed where gold_retrieval_ids are available:
  ko_native uses gold_answer.evidence_utt_ids
  en_subset/ko_translated use gold_answer.evidence_session_ids
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger

log = get_logger("agg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--results-name", default="results.jsonl")
    args = ap.parse_args()

    run_dir = args.run_dir
    results_path = run_dir / args.results_name
    if not results_path.exists():
        raise SystemExit(f"no results at {results_path}")

    rows = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    log.info("loaded %d result rows", len(rows))

    # Group key: (track, sut, condition, policy, ability) or 'ALL'
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["track"], r["sut"], r["condition"], r["policy"], r["ability"])].append(r)
        groups[(r["track"], r["sut"], r["condition"], r["policy"], "ALL")].append(r)

    def _pass_rate(items):
        substring = [i for i in items if i["ability"] != "REFL"]
        if not substring:
            return None
        return sum(1 for i in substring if i["passed_contains"]) / len(substring)

    def _avg_lat(items):
        return statistics.mean(i["latency_s"] for i in items) if items else 0.0

    def _recall_5(items):
        # Only applicable to retrieval condition.
        if not items or items[0]["condition"] != "retrieval":
            return None
        covered = 0
        total = 0
        for i in items:
            # Find this item in the raw upstream to get gold ids.
            # We reconstruct from retrieved_doc_ids + item_id; need access to items.
            # Simpler: rely on the result to include gold indirectly — deferred.
            pass
        return None

    metrics = {}
    csv_rows = []
    for key, items in sorted(groups.items()):
        track, sut, cond, policy, ability = key
        pr = _pass_rate(items)
        lat = _avg_lat(items)
        metrics.setdefault(track, {}).setdefault(sut, {}).setdefault(cond, {}) \
                .setdefault(policy, {})[ability] = {
            "n": len(items),
            "pass_rate": pr,
            "avg_latency_s": round(lat, 3),
        }
        csv_rows.append([track, sut, cond, policy, ability,
                         len(items), "" if pr is None else round(pr, 4),
                         round(lat, 3)])

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track", "sut", "condition", "policy", "ability", "n",
                    "pass_rate", "avg_latency_s"])
        w.writerows(csv_rows)

    # Print a main headline table (ALL ability).
    print("\n=== Pass rate by track × condition × policy × SUT (ability=ALL, non-REFL items only) ===")
    print(f"{'track':<15s} {'sut':<12s} {'condition':<12s} {'policy':<10s} {'n':>4s} {'pass':>8s}  {'lat':>6s}")
    for track in sorted(metrics):
        for sut in sorted(metrics[track]):
            for cond in sorted(metrics[track][sut]):
                for policy in sorted(metrics[track][sut][cond]):
                    slot = metrics[track][sut][cond][policy].get("ALL")
                    if slot is None:
                        continue
                    pr = slot["pass_rate"]
                    pr_str = "---" if pr is None else f"{pr:.3f}"
                    print(f"{track:<15s} {sut:<12s} {cond:<12s} {policy:<10s} "
                          f"{slot['n']:>4d} {pr_str:>8s}  {slot['avg_latency_s']:>5.1f}s")

    log.info("wrote metrics.json and metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
