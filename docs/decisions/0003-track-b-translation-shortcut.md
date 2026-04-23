# ADR 0003 — Track B Translation Shortcut

## Status
Accepted (2026-04-24 — Day 2)

## Context

`DATASET.md §2.2` specifies a full 4-generator translation pipeline:
each of the 100 Track A items is translated by four separate LLMs,
the four candidates are cross-evaluated by another LLM, and the
author spot-checks every item. That costs roughly:

- 100 items × 4 generators × ~15 K input + 15 K output tokens
  ≈ 12 M tokens across the pool
- At the cost-tier mix (haiku / mini / flash) ≈ **USD 8–12**

Day 2's remaining API budget after Day 1 (USD 1.88) and the Track C
scenario run (USD 0.29) is ~USD 2.0 allocated to Track B by
`configs/models.yaml: budget.per_phase_cap_usd`. Four-way
translation does not fit.

## Decision

Reduce Track B to:

1. **Translate only the `question` and `gold_answer` fields; keep
   `history_sessions` in English.** Originally we attempted
   full-item translation with Gemini 2.5 Flash (the only model whose
   1 M input context fits a full LongMemEval_S item). That failed
   on every attempt for two different reasons: (a) Gemini caps its
   output around 16 K tokens so a 47-session haystack truncates
   mid-JSON; (b) Korean output contained unescaped `"` inside JSON
   strings (quoted phrases in source text like `"교육은 어땠습니까?"`),
   breaking the parser. Even with an 8 K-token haystack trim +
   prompt-level quote-avoidance rules, the failure rate stayed high
   and Day 2 was burning.
   MVP path: one 600-token call per item. Rotation runs across G1/
   G2/G3 normally, and every item records
   `metadata.translation_scope = "question_and_gold_answer_only"` +
   `metadata.history_language = "en"` so reviewers know the scope.
2. **Back-translation BLEU on a 10 % sample** (10 items) as a
   lightweight quality gate. We compute EN → KO → EN and report
   BLEU against the original EN. Items below threshold are
   flagged but not automatically regenerated — the sample size is
   too small for per-item remediation within Day 2.
3. **Spot check by the author** on ~20 items in Day 3. Any systemic
   issue found there triggers a targeted regeneration run.
4. **History stays in English.** The MVP scope in point 1 above
   supersedes the earlier "haystack trim" attempt; we do not
   translate sessions at all. Discussion section of the report will
   flag this explicitly: Track B measures Korean question
   understanding against an English haystack, which is a weaker
   but still interesting comparison.

## Consequences

- Cost drops to ~USD 1.5–2.0 at cost-tier pricing.
- We lose inter-generator agreement as a quality signal. Mitigation:
  report translation_generator_family on every item so downstream
  analysis can check whether any family correlates with model
  accuracy degradation.
- Parse failures (truncation, invalid JSON) will leave fewer than
  100 items. We accept a shortfall down to ~90 items — below that
  we re-run with a shorter chunking strategy.
- Documented as a limitation in the eventual report's Methods
  section: "Track B used single-generator translation; the full
  4-way cross-check called for by our dataset plan was deferred
  owing to API budget."

## Alternatives considered

- **Trim haystack to evidence + 10 distractors.** Cheap but breaks
  the comparability between Track A and Track B (different haystack
  sizes would confound the language-only axis).
- **Use only Gemini Flash (~USD 0.6).** Cheapest, but a single
  family across all Track B items maximises generator fingerprint.
- **Skip Track B entirely for Day 2.** Kills the core contrast the
  dataset was designed to support.

## Links
- `DATASET.md §2.2` — the original four-way pipeline
- `src/datagen/translate.py` — single-generator implementation
- `scripts/05_translate_track_b.py` — CLI
- ADR 0002 — cost-constrained model tier (this ADR extends the
  same cost reasoning)
