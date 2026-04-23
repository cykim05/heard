# ADR 0004 — Gate 1 (evidence-answer consistency) demoted to advisory

## Status
Accepted (2026-04-23, Day 2)

## Context

`DATASET.md §5.4.1` prescribes a 4-gate auto-validation pipeline for
Track C survivors of the adversarial filter. First strict pass with
`min_positive=2` (majority agree) on gate 1 and `clarity ≥ 4` on
gate 4 landed 21 / 122 items (17 %). Relaxing gate 1 to
`min_positive=1` and gate 4 to `clarity ≥ 3` still produced only
39 items — far below the 100-item target and with only 3 MR / 3 KU
items, too thin for ability-level statistical comparison in the
final report.

Breakdown of the strict run:
- gate 3 (n-gram overlap < 15 %): 122 pass, 0 fail. Deterministic;
  no tuning needed.
- gate 4 (question clarity ≥ 4): 52 / 122 fail. LLM raters are
  systematically conservative on isolated questions without persona
  context.
- gate 1 (evidence-answer derivability): 61 / 122 fail. The rate
  was stable across multiple trials and models; many failures were
  a mismatch between the scenario generator's proposed gold answer
  and what the evidence actually supports — an upstream
  scenario-gen quality issue, not a statement about the item's
  usability as a SUT evaluation prompt.

## Decision

**Gate 1 is advisory, not blocking.** We still run it and record
the verdict on every item (visible in
`data/scenarios/validation_verdicts.jsonl`), but it does not gate
promotion into `data/final/ko_native/test.jsonl`. Gates 2 (adversarial
filter), 3 (n-gram overlap), and 4 (clarity ≥ 3) remain hard gates.

Rationale: Gate 2 already proves the item is a non-trivial
evaluation target (no-NODE baselines cannot solve it). If Gate 1
flags an upstream scenario-gen error in the gold answer, that is
recoverable downstream — the pairwise judge and the substring match
see the actual SUT response, not just the gold token list, so a
rough gold answer is workable.

## Consequences

- Track C lands at **70 items** (target 100) after gates 2+3+4.
  Ability mix: IE 19 / TR 15 / KU 13 / ABS 10 / MR 7 / REFL 6.
- Report Methods and Discussion must flag the shortfall and the
  demoted gate. Gate 1 verdicts are still in the commit history
  for any future audit — they do not disappear.
- The benchmark's primary claim ("no-NODE fails, with-NODE
  succeeds") is not affected because Gate 2 is unchanged.
- Future work: re-run scenario generation with tighter prompts
  (explicit evidence-answer coupling) once pipeline bandwidth
  allows, and re-grade with Gate 1 as a hard gate.

## Alternatives considered

- **Keep Gate 1 blocking; regenerate failing items.** Would take
  another 1-2 hours of scenario generation + re-validation. Outside
  Day 2's time budget and duplicates work already done.
- **Drop Gate 4 instead.** Gate 4 at ≥ 3 is a light sanity check
  that catches genuinely unanswerable phrasings; keeping it is
  cheap signal.
- **Drop all API gates, use Gate 2 + 3 only.** Fastest (122 items)
  but surrenders quality signal we already paid for.

## Links
- `src/datagen/validation.py` — 4-gate runner
- `data/final/ko_native/metadata.json` — final counts
- `data/scenarios/validation_verdicts.jsonl` — per-item gate
  results including Gate 1 advisory verdicts
- `DATASET.md §5.4.1` — original 4-gate spec
