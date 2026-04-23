#!/usr/bin/env python3
"""Run 4-gate auto-validation on Track C survivors (DATASET §5.4.1)."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.utterance_gen import GeneratorRotation
from src.datagen.validation import (
    _load_utterances_by_id,
    aggregate,
    compute_gate3_overlap,
    run_gate1_consistency,
    run_gate4_ambiguity,
)
from src.utils.config import REPO_ROOT, load_settings
from src.utils.logging import get_logger
from src.utils.openrouter import OpenRouterClient

log = get_logger("validate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors", type=Path,
                    default=REPO_ROOT / "data/scenarios/survivors.jsonl")
    ap.add_argument("--corpus", type=Path,
                    default=REPO_ROOT / "data/raw/utterances.jsonl")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data/final/ko_native/test.jsonl")
    ap.add_argument("--metadata-out", type=Path,
                    default=REPO_ROOT / "data/final/ko_native/metadata.json")
    ap.add_argument("--verdicts-out", type=Path,
                    default=REPO_ROOT / "data/scenarios/validation_verdicts.jsonl")
    ap.add_argument("--report-out", type=Path,
                    default=REPO_ROOT / "data/final/validation_report.json")
    ap.add_argument("--budget-cap", type=float, default=1.5)
    ap.add_argument("--gate1-trials", type=int, default=3)
    ap.add_argument("--gate1-min-positive", type=int, default=2)
    ap.add_argument("--gate4-min-clarity", type=int, default=4)
    ap.add_argument("--gate3-threshold", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with args.survivors.open(encoding="utf-8") as f:
        survivors = [json.loads(line) for line in f if line.strip()]
    log.info("loaded %d survivors", len(survivors))

    log.info("gate 3 (n-gram overlap < %.2f)…", args.gate3_threshold)
    gate3 = compute_gate3_overlap(
        survivors, threshold=args.gate3_threshold,
    )
    g3_fail = sum(1 for v in gate3.values() if not v[1])
    log.info("  gate3: pass=%d fail=%d", len(gate3) - g3_fail, g3_fail)

    utterances = _load_utterances_by_id(args.corpus)
    settings = load_settings()

    with OpenRouterClient.from_settings(settings) as client:
        rotation = GeneratorRotation.from_models_yaml(settings.models, seed=args.seed * 17)

        log.info("gate 1 (evidence-answer consistency, %d trials, min %d)…",
                 args.gate1_trials, args.gate1_min_positive)
        gate1 = run_gate1_consistency(
            client, survivors, utterances,
            rotation=rotation,
            trials=args.gate1_trials,
            min_positive=args.gate1_min_positive,
        )
        g1_fail = sum(1 for v in gate1.values() if v[0] is False)
        g1_skip = sum(1 for v in gate1.values() if v[0] is None)
        log.info("  gate1: pass=%d fail=%d skip(ABS)=%d",
                 len(gate1) - g1_fail - g1_skip, g1_fail, g1_skip)

        log.info("gate 4 (question clarity ≥ %d)…", args.gate4_min_clarity)
        gate4 = run_gate4_ambiguity(
            client, survivors,
            rotation=rotation, min_clarity=args.gate4_min_clarity,
        )
        g4_fail = sum(1 for v in gate4.values() if v[1] is False)
        log.info("  gate4: pass=%d fail=%d",
                 len(gate4) - g4_fail, g4_fail)

    verdicts = aggregate(survivors, gate1, gate3, gate4)
    kept = sum(1 for v in verdicts if v.final_pass)
    log.info("ALL GATES: pass=%d/%d (%.1f%%)",
             kept, len(verdicts), 100.0 * kept / len(verdicts))

    # Write verdicts
    args.verdicts_out.parent.mkdir(parents=True, exist_ok=True)
    with args.verdicts_out.open("w", encoding="utf-8") as f:
        from dataclasses import asdict
        for v in verdicts:
            f.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")

    # Final items
    kept_ids = {v.candidate_id for v in verdicts if v.final_pass}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final_items = []
    ability_counts: collections.Counter[str] = collections.Counter()
    persona_counts: collections.Counter[str] = collections.Counter()
    with args.out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(survivors):
            if c["candidate_id"] not in kept_ids:
                continue
            # Shape to heard-bench Track C schema (DATASET §8.2).
            out_item = {
                "item_id": f"ko_native_{i:03d}",
                "track": "ko_native",
                "persona_id": c["persona_id"],
                "ability": c["ability"],
                "question": {
                    "text": c["question_text"],
                    "timestamp": c["question_timestamp"],
                },
                "gold_answer": {
                    "text": c["gold_answer_text"],
                    "contains_tokens": c.get("gold_contains_tokens", []),
                    "excludes_tokens": c.get("gold_excludes_tokens", []),
                    "evidence_utt_ids": c.get("evidence_utt_ids", []),
                },
                "reflective_rubric": (
                    {"criteria": ["specificity", "non_directive", "emotional_attunement", "open_question"]}
                    if c["ability"] == "REFL" else None
                ),
                "metadata": {
                    "generator_model_id": c.get("generator_model_id"),
                    "generator_model_family": c.get("generator_model_family"),
                    "candidate_id": c["candidate_id"],
                    "passed_adversarial_filter": True,
                    "passed_auto_validation_gates": ["gate1", "gate2", "gate3", "gate4"],
                },
            }
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            final_items.append(out_item)
            ability_counts[c["ability"]] += 1
            persona_counts[c["persona_id"]] += 1

    report = {
        "track": "ko_native",
        "candidates_in_survivors": len(survivors),
        "final_kept": len(final_items),
        "ability_counts": dict(ability_counts),
        "persona_counts": dict(persona_counts),
        "gate_fail_counts": {
            "gate1": g1_fail,
            "gate3": g3_fail,
            "gate4": g4_fail,
        },
        "thresholds": {
            "gate1_min_positive": args.gate1_min_positive,
            "gate3_max_overlap": args.gate3_threshold,
            "gate4_min_clarity": args.gate4_min_clarity,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.metadata_out.write_text(
        json.dumps({
            "track": "ko_native",
            "item_count": len(final_items),
            "source": "Generated from corpus at data/raw/utterances.jsonl via "
                      "scripts/03_generate_scenarios.py → 04_adversarial_filter.py → "
                      "06_auto_validation.py",
            "ability_counts": dict(ability_counts),
            "persona_counts": dict(persona_counts),
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("wrote %d final items -> %s", len(final_items), args.out)
    log.info("validation_report -> %s", args.report_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
