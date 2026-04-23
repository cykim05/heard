#!/usr/bin/env python3
"""Translate Track A (EN) items to Korean — Track B.

Usage:
    python scripts/05_translate_track_b.py \\
        --in data/final/en_subset/test.jsonl \\
        --out data/final/ko_translated/test.jsonl \\
        --budget-cap 2.0
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.translate import build_track_b_item, translate_item
from src.datagen.utterance_gen import GeneratorRotation
from src.utils.config import REPO_ROOT, load_settings
from src.utils.logging import get_logger
from src.utils.openrouter import BudgetExceeded, OpenRouterClient

log = get_logger("translate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=REPO_ROOT / "data/final/en_subset/test.jsonl")
    ap.add_argument("--out", dest="out_path", type=Path,
                    default=REPO_ROOT / "data/final/ko_translated/test.jsonl")
    ap.add_argument("--metadata-out", type=Path,
                    default=REPO_ROOT / "data/final/ko_translated/metadata.json")
    ap.add_argument("--budget-cap", type=float, default=2.0)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"Track A test.jsonl not found at {args.in_path}. "
                         "Run scripts/02_subsample_longmemeval.py first.")

    settings = load_settings()
    # Track B MVP only translates question + gold_answer — all three
    # generators can handle the tiny payload. Rotate normally.
    rotation = GeneratorRotation.from_models_yaml(settings.models, seed=args.seed * 13)

    with args.in_path.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[: args.limit]
    log.info("loaded %d source items", len(items))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out_path.exists():
        args.out_path.unlink()

    translated = 0
    failed = 0
    gen_counter: collections.Counter[str] = collections.Counter()
    run_start = None

    with OpenRouterClient.from_settings(settings) as client:
        run_start = client._budget.current()  # noqa: SLF001

        for i, item in enumerate(items):
            spent_this_run = client._budget.current() - run_start  # noqa: SLF001
            if spent_this_run >= args.budget_cap:
                log.warning("budget cap %.2f USD hit at item %d; stopping",
                            args.budget_cap, i)
                break

            try:
                tr = translate_item(
                    client, item, rotation=rotation,
                    seed=args.seed + i, sleep_between_calls=args.sleep,
                )
            except BudgetExceeded as e:
                log.error("budget exceeded: %s", e)
                break

            if tr.parse_failed:
                failed += 1
                continue

            out_item = build_track_b_item(item, tr)
            with args.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            translated += 1
            gen_counter[tr.generator_name] += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                log.info(
                    "progress %d/%d  ok=%d fail=%d  run_spend=%.4f USD",
                    i + 1, len(items), translated, failed,
                    client._budget.current() - run_start,  # noqa: SLF001
                )

    metadata = {
        "track": "ko_translated",
        "source": "xiaowu0162/longmemeval:longmemeval_s",
        "derived_from": "data/final/en_subset/test.jsonl",
        "total_in": len(items),
        "translated": translated,
        "parse_failures": failed,
        "generator_distribution": dict(gen_counter),
    }
    args.metadata_out.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("DONE: %d/%d translated -> %s", translated, len(items), args.out_path)
    log.info("metadata -> %s", args.metadata_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
