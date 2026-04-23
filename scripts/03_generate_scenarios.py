#!/usr/bin/env python3
"""Generate 150 Track C scenario candidates from the utterance corpus.

Cost estimate (150 calls × ~0.005 USD at haiku/mini/flash tier): ~0.8 USD.
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.scenario_gen import (
    CANDIDATE_DISTRIBUTION,
    PERSONA_DISTRIBUTION,
    _utterances_by_persona,
    generate_candidate,
    save_jsonl,
)
from src.datagen.utterance_gen import GeneratorRotation
from src.utils.config import REPO_ROOT, load_settings
from src.utils.logging import get_logger
from src.utils.openrouter import BudgetExceeded, OpenRouterClient

log = get_logger("scenariogen")


def _allocate(total: int, distribution: dict[str, float]) -> dict[str, int]:
    raw = {k: int(round(v * total)) for k, v in distribution.items()}
    # Fix rounding drift so sum == total
    diff = total - sum(raw.values())
    if diff != 0:
        keys = list(distribution.keys())
        for i in range(abs(diff)):
            k = keys[i % len(keys)]
            raw[k] += 1 if diff > 0 else -1
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=REPO_ROOT / "data/raw/utterances.jsonl")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data/scenarios/candidates.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-cap", type=float, default=2.0)
    ap.add_argument("--base-date", type=str, default="2026-02-23")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    base_date = datetime.fromisoformat(args.base_date).replace(
        tzinfo=timezone(timedelta(hours=9))
    )
    settings = load_settings()
    rng = random.Random(args.seed)

    if args.out.exists():
        args.out.unlink()

    # Load corpus and split by persona.
    by_persona = _utterances_by_persona(args.corpus)
    log.info("corpus split: %s",
             {k: len(v) for k, v in by_persona.items()})

    # Work out how many (ability, persona) slots we need.
    ability_counts = dict(CANDIDATE_DISTRIBUTION)  # ability -> count
    ordered_slots: list[tuple[str, str]] = []
    for ability, count in ability_counts.items():
        persona_alloc = _allocate(count, PERSONA_DISTRIBUTION)
        for persona_id, n in persona_alloc.items():
            ordered_slots.extend([(persona_id, ability)] * n)

    assert len(ordered_slots) == sum(ability_counts.values())
    rng.shuffle(ordered_slots)
    log.info("planned slots: %d  ability_persona breakdown=%s",
             len(ordered_slots),
             dict(collections.Counter(ordered_slots)))

    # One rotation instance per persona so each persona's generator mix
    # still gets the anti-fingerprint guarantee of consecutive-different.
    rotations = {
        pid: GeneratorRotation.from_models_yaml(
            settings.models, seed=args.seed * 7 + hash(pid) % 1000,
        )
        for pid in by_persona
    }

    # Load persona cards for prompt context.
    persona_cards = {
        pid: yaml.safe_load((REPO_ROOT / f"configs/personas/{pid}.yaml").read_text(encoding="utf-8"))
        for pid in by_persona
    }

    candidates = []
    run_start_budget = None
    failures = 0

    with OpenRouterClient.from_settings(settings) as client:
        run_start_budget = client._budget.current()  # noqa: SLF001

        for i, (persona_id, ability) in enumerate(ordered_slots):
            spent_this_run = client._budget.current() - run_start_budget  # noqa: SLF001
            if spent_this_run >= args.budget_cap:
                log.warning("budget cap %.2f USD hit at slot=%d; stopping", args.budget_cap, i)
                break

            try:
                cand = generate_candidate(
                    client,
                    persona=persona_cards[persona_id],
                    corpus=by_persona[persona_id],
                    ability=ability,
                    rotation=rotations[persona_id],
                    base_date=base_date,
                    rng=rng,
                    candidate_idx=i,
                    sleep_between_calls=args.sleep,
                )
            except BudgetExceeded as e:
                log.error("budget exceeded: %s", e)
                break

            if cand is None:
                failures += 1
                continue

            candidates.append(cand)
            if (i + 1) % 10 == 0 or (i + 1) == len(ordered_slots):
                log.info(
                    "progress %d/%d  ok=%d fail=%d  spend=%.4f USD",
                    i + 1, len(ordered_slots), len(candidates), failures,
                    client._budget.current() - run_start_budget,  # noqa: SLF001
                )

    save_jsonl(candidates, args.out)
    log.info("DONE: %d candidates saved (failures=%d) -> %s",
             len(candidates), failures, args.out)
    # Ability breakdown
    by_ab = collections.Counter(c.ability for c in candidates)
    log.info("ability counts: %s", dict(by_ab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
