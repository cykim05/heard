# ADR 0001 — OpenRouter as Unified LLM Gateway

## Status
Accepted (2026-04-23)

## Context

`DATASET.md §12 #7` originally called for three direct-provider SDKs
(`anthropic`, `openai`, `google-generativeai`) with the note "OpenRouter
미사용 — 단순화". The rationale was that a single API surface was
simpler than a multi-provider aggregator.

In practice the inverse is true:
- Three separate API keys → three separate billing dashboards, three
  separate rate-limit policies, three separate error surfaces.
- Each provider SDK has a different request/response shape; a wrapper
  is needed regardless. That wrapper ends up reimplementing what
  OpenRouter already provides.
- Per-provider rate limits tend to be tighter than OpenRouter's
  aggregated quota (OpenRouter routes through multiple upstream
  accounts for the same model).

## Decision

Route all generator and judge traffic through OpenRouter via a single
HTTP client (`src/utils/openrouter.py`). Model IDs stay in the
provider-prefixed form (`anthropic/claude-sonnet-4.5`, `openai/gpt-4o`,
`google/gemini-2.5-pro`) so the target provider is explicit in every
call.

**Exception: G4 (Kanana 1.5 8B) remains local HF inference.** OpenRouter
does not serve Kanana, and even if it did, the Apache-2.0 license
reasoning in `DATASET §4.1` specifically requires local inference so
the generated text is unambiguously free of upstream license
constraints. G4 is wired through `src/utils/llm_backend.py` (Day 2,
separate from OpenRouter client).

Tokenization cost is tracked from OpenRouter's `usage.cost` response
field rather than a local price table — provider prices move too
often for a hand-maintained table to stay accurate.

## Consequences

- Single env var (`OPENROUTER_API_KEY`) covers all API generators and
  judges, simplifying `.env.example` and user onboarding.
- OpenRouter margin (~0–5 %) is absorbed into the USD 30 total budget.
  Measured Day 1 smoke test: USD 0.0381 for 16 utterances — well
  inside the generation sub-cap of USD 10.
- Rate limits are effectively one aggregate pool; tenacity backs off
  on 429 with `Retry-After`.
- One integration surface to test. `tests/test_openrouter_offline.py`
  covers the budget and cache logic without hitting the network.

## Alternatives considered

- **Stay with direct SDKs as originally planned.** Rejected — triples
  the integration and billing surface for marginal cost savings.
- **LiteLLM as the aggregator.** Similar shape to OpenRouter-as-proxy
  but adds another Python dependency and its billing still goes back
  to provider accounts. OpenRouter's single-billing property is the
  key simplification.
- **Per-provider SDKs with a thin adapter.** Half-measure — the
  adapter becomes what OpenRouter already is, with worse reliability
  because we do not get the aggregator's upstream failover.

## Links
- `DATASET.md §12 #7` — original decision (now revised in place)
- `src/utils/openrouter.py` — implementation
- `configs/models.yaml` — per-model transport and role assignments
