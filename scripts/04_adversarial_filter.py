#!/usr/bin/env python3
"""Adversarial filter — apply no-NODE baselines to scenario candidates.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/04_adversarial_filter.py \\
        --in data/scenarios/candidates.jsonl \\
        --out data/scenarios/survivors.jsonl \\
        --verdicts data/scenarios/filter_verdicts.jsonl \\
        --report data/scenarios/filter_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.adversarial_filter import run_filter, save_verdicts_and_survivors
from src.utils.config import REPO_ROOT, load_settings
from src.utils.llm_backend import append_to_registry, load_sut, unload
from src.utils.logging import get_logger

log = get_logger("filter")


SUT_CHOICES: list[tuple[str, str]] = [
    # (logical name, hf_id)
    ("kanana_nano",      "kakaocorp/kanana-1.5-2.1b-instruct-2505"),
    ("qwen25_3b",        "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen25_15b",       "Qwen/Qwen2.5-1.5B-Instruct"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  type=Path,
                    default=REPO_ROOT / "data/scenarios/candidates.jsonl")
    ap.add_argument("--out", dest="out_path", type=Path,
                    default=REPO_ROOT / "data/scenarios/survivors.jsonl")
    ap.add_argument("--verdicts", type=Path,
                    default=REPO_ROOT / "data/scenarios/filter_verdicts.jsonl")
    ap.add_argument("--report", type=Path,
                    default=REPO_ROOT / "data/scenarios/filter_report.json")
    ap.add_argument("--registry", type=Path,
                    default=REPO_ROOT / "experiments/_api_log/filter_model_registry.json")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    with args.in_path.open(encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    log.info("loaded %d candidates from %s", len(candidates), args.in_path)

    log.info("loading %d SUTs…", len(SUT_CHOICES))
    suts = []
    for name, hf_id in SUT_CHOICES:
        log.info("  -> %s (%s)", name, hf_id)
        m = load_sut(hf_id, name=name, quantization="fp16")
        suts.append(m)
        append_to_registry(args.registry, m)
        log.info(
            "     loaded (params=%.2fB  device=%s)",
            m.params / 1e9, m.model.device,
        )

    verdicts = run_filter(
        candidates,
        suts,
        trials=args.trials,
        max_new_tokens=args.max_new_tokens,
    )
    kept = sum(1 for v in verdicts if v.kept)
    log.info(
        "filter DONE — kept %d/%d (%.1f%%)  discarded %d",
        kept, len(verdicts), 100.0 * kept / len(verdicts),
        len(verdicts) - kept,
    )

    save_verdicts_and_survivors(
        verdicts,
        candidates,
        verdicts_path=args.verdicts,
        survivors_path=args.out_path,
        report_path=args.report,
    )
    log.info("report -> %s", args.report)
    log.info("survivors -> %s", args.out_path)

    for m in suts:
        unload(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
