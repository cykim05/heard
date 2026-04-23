#!/usr/bin/env python3
"""Build Track A — 100 stratified items from LongMemEval_S.

Usage:
    python scripts/02_subsample_longmemeval.py \\
        --out data/final/en_subset/test.jsonl \\
        --seed 42
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.longmemeval import (
    load_longmemeval_s,
    normalize_item,
    stratified_sample,
)
from src.utils.logging import get_logger

log = get_logger("longmemeval")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/final/en_subset/test.jsonl"))
    ap.add_argument("--metadata-out", type=Path,
                    default=Path("data/final/en_subset/metadata.json"))
    ap.add_argument("--per-ability", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("downloading longmemeval_s…")
    data = load_longmemeval_s()
    log.info("loaded %d upstream items", len(data))

    sampled = stratified_sample(data, per_ability=args.per_ability, seed=args.seed)
    ability_counts = collections.Counter(it["_ability"] for it in sampled)
    log.info("stratified sample: %s", dict(ability_counts))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, it in enumerate(sampled):
            normalized = normalize_item(it, ordinal=i)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    metadata = {
        "track": "en_subset",
        "source": "xiaowu0162/longmemeval:longmemeval_s",
        "source_license": "MIT",
        "source_total_items": len(data),
        "sampled_count": len(sampled),
        "per_ability_target": args.per_ability,
        "per_ability_actual": dict(ability_counts),
        "seed": args.seed,
    }
    args.metadata_out.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("wrote %d items to %s", len(sampled), args.out)
    log.info("wrote metadata to %s", args.metadata_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
