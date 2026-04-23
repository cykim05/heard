#!/usr/bin/env python3
"""Run pairwise judge on REFL items in a sweep results.jsonl.

Usage:
    python scripts/08_run_judge.py --run-dir experiments/20260425_.../
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.judge import aggregate_wins, judge_pair
from src.utils.config import REPO_ROOT, load_settings
from src.utils.logging import get_logger
from src.utils.openrouter import OpenRouterClient

log = get_logger("judge")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--results-name", default="results.jsonl")
    ap.add_argument("--out-name", default="judge_verdicts.jsonl")
    ap.add_argument("--aggregate-name", default="judge_aggregate.json")
    ap.add_argument("--budget-cap", type=float, default=0.5)
    args = ap.parse_args()

    results_path = args.run_dir / args.results_name
    if not results_path.exists():
        raise SystemExit(f"no results at {results_path}")

    # Build index: (item_id, sut, condition) -> {policy: response}.
    # Only REFL items with both advisory and reflective policies run get judged.
    by_triple: dict[tuple[str, str, str], dict[str, dict]] = collections.defaultdict(dict)
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("ability") != "REFL":
                continue
            if r["track"] != "ko_native":
                continue
            key = (r["item_id"], r["sut"], r["condition"])
            by_triple[key][r["policy"]] = r

    log.info("found %d (item, sut, condition) REFL triples", len(by_triple))

    settings = load_settings()
    verdicts_path = args.run_dir / args.out_name
    verdicts_path.unlink(missing_ok=True)
    all_verdicts = []
    with OpenRouterClient.from_settings(settings) as client:
        start_budget = client._budget.current()  # noqa: SLF001
        for idx, ((item_id, sut, cond), by_policy) in enumerate(by_triple.items()):
            if "advisory" not in by_policy or "reflective" not in by_policy:
                continue
            spent = client._budget.current() - start_budget  # noqa: SLF001
            if spent >= args.budget_cap:
                log.warning("budget cap %.2f hit at triple %d", args.budget_cap, idx)
                break
            verdicts = judge_pair(
                client,
                item_id=item_id, sut=sut, condition=cond,
                question=by_policy["advisory"]["question"],
                advisory_response=by_policy["advisory"]["response"],
                reflective_response=by_policy["reflective"]["response"],
                seed_base=12000 + idx * 10,
            )
            all_verdicts.extend(verdicts)
            with verdicts_path.open("a", encoding="utf-8") as f:
                for v in verdicts:
                    f.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")
            log.info("judged %s %s %s  spend=%.4f",
                     item_id, sut, cond, client._budget.current() - start_budget)  # noqa: SLF001

    agg = aggregate_wins(all_verdicts)
    (args.run_dir / args.aggregate_name).write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("DONE: %d verdicts -> %s", len(all_verdicts), verdicts_path)
    log.info("aggregate -> %s", args.run_dir / args.aggregate_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
