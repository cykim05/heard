#!/usr/bin/env python3
"""Generate utterance corpus for all 3 personas × 60 days.

Usage:
    python scripts/01_generate_data.py \\
        --out data/raw/utterances.jsonl \\
        --days 60 \\
        --per-day 12 \\
        --budget-cap 10.0

Cost estimate (12 utterances/day × 60 days × 3 personas = 180 calls):
    avg $0.012/call × 180 ≈ $2.2, budget-cap=10.0 leaves ample headroom.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen.timeline import generate_timeline, save_timeline
from src.datagen.utterance_gen import (
    GeneratorRotation,
    generate_for_day,
    save_jsonl,
)
from src.utils.config import REPO_ROOT, load_settings
from src.utils.logging import get_logger
from src.utils.openrouter import BudgetExceeded, OpenRouterClient

log = get_logger("datagen")

PERSONAS = [
    "yejin_florist",
    "minseok_cafe",
    "sunhee_hair",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/raw/utterances.jsonl")
    ap.add_argument("--timeline-dir", type=Path, default=REPO_ROOT / "data/raw/timelines")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--per-day", type=int, default=12,
                    help="Target utterance count per day (model may return ±2).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-date", type=str, default="2026-02-23",
                    help="Day 1 calendar date. Day N = base_date + (N-1) days.")
    ap.add_argument("--budget-cap", type=float, default=10.0,
                    help="Soft budget cap for this run in USD (checked after each call).")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--personas", nargs="*", default=PERSONAS)
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate timelines but skip LLM calls.")
    args = ap.parse_args()

    base_date = datetime.fromisoformat(args.base_date).replace(tzinfo=timezone(timedelta(hours=9)))
    settings = load_settings()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.timeline_dir.mkdir(parents=True, exist_ok=True)
    # Clear output for idempotency.
    if args.out.exists():
        args.out.unlink()

    # Phase 1: build timelines (deterministic, no API).
    persona_cards: dict[str, dict] = {}
    persona_timelines: dict[str, list[dict]] = {}
    for pid in args.personas:
        persona_path = REPO_ROOT / "configs/personas" / f"{pid}.yaml"
        persona = yaml.safe_load(persona_path.read_text(encoding="utf-8"))
        persona_cards[pid] = persona
        events = generate_timeline(persona, days=args.days, seed=args.seed)
        persona_timelines[pid] = [ev.__dict__ for ev in events]
        save_timeline(events, args.timeline_dir / f"{pid}_timeline.jsonl")
        log.info("timeline[%s]: %d events across %d days", pid, len(events), args.days)

    if args.dry_run:
        log.info("dry-run: skipping LLM calls")
        return 0

    # Phase 2: utterance generation. One call per (persona, day) block.
    total_start_budget = None
    total_utt = 0
    with OpenRouterClient.from_settings(settings) as client:
        total_start_budget = client._budget.current()  # noqa: SLF001 — ledger is meant to be read.

        for pid in args.personas:
            persona = persona_cards[pid]
            events_by_day: dict[int, list[dict]] = {}
            for ev in persona_timelines[pid]:
                events_by_day.setdefault(ev["day"], []).append(ev)

            rotation = GeneratorRotation.from_models_yaml(
                settings.models, seed=args.seed + hash(pid) % 1000
            )

            per_persona_start_idx = 0
            for day in range(1, args.days + 1):
                day_events = events_by_day.get(day, [])
                # Budget guardrail per-call.
                spent_this_run = client._budget.current() - total_start_budget  # noqa: SLF001
                if spent_this_run >= args.budget_cap:
                    log.warning(
                        "budget cap %.2f USD reached at day=%d persona=%s; stopping.",
                        args.budget_cap, day, pid,
                    )
                    return 1

                try:
                    utterances = generate_for_day(
                        client,
                        persona=persona,
                        day=day,
                        events=day_events,
                        rotation=rotation,
                        base_date=base_date,
                        target_count=args.per_day,
                        seed=args.seed * 1000 + day,
                        start_index=per_persona_start_idx,
                        sleep_between_calls=args.sleep,
                    )
                except BudgetExceeded as e:
                    log.error("budget exceeded: %s", e)
                    return 1

                per_persona_start_idx += len(utterances)
                total_utt += len(utterances)
                save_jsonl(utterances, args.out, append=True)
                log.info(
                    "day %02d/%02d [%s]: %d utterances (run spend %.4f USD)",
                    day, args.days, pid, len(utterances),
                    client._budget.current() - total_start_budget,  # noqa: SLF001
                )

    log.info("DONE: %d utterances across %d personas -> %s",
             total_utt, len(args.personas), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
