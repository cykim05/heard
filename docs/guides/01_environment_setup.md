# Environment Setup

This is the 5-minute tutorial for getting a fresh checkout to the point
where `python scripts/01_generate_data.py --dry-run` succeeds.

## 1. System requirements

- Linux (tested on Ubuntu 20.04) with CUDA 12.x driver
- NVIDIA L40S 48GB (or any GPU with ≥16GB VRAM for Day 2+ inference)
- Python 3.10–3.12 recommended. Python 3.13 works for Day 1 API work
  but `bitsandbytes` wheels may lag — if you hit ImportError at the
  int4 quantization step, downgrade to 3.11.
- Disk ≥50 GB for model caches (HF weights land in `~/.cache/huggingface`)

## 2. Clone and install

```bash
git clone <repo-url> heard
cd heard

# Optional but recommended — isolated env
python -m venv .venv && source .venv/bin/activate

# Day 1 deps only (skip torch until Day 2 if you want quick setup):
pip install httpx orjson diskcache tenacity pyyaml python-dotenv pydantic pytest tqdm numpy

# Full install (adds torch, transformers, chromadb, etc.):
pip install -r requirements.txt
```

## 3. Secrets

```bash
cp .env.example .env
$EDITOR .env
```

Fill in:
- `OPENROUTER_API_KEY` — from <https://openrouter.ai/keys>. Funds must
  cover at least USD 10; the full Day 1–3 run is budgeted for USD 30
  (see `configs/models.yaml: budget`).
- `HF_TOKEN` — from <https://huggingface.co/settings/tokens>. A
  **write-scope** token is required for the Day 4 dataset upload;
  read-scope is enough until then (only needed for gated models like
  Gemma-2).
- `OPENROUTER_SITE_URL` (optional) — shows up on the OpenRouter
  leaderboard. Leave blank to opt out.

## 4. Git hooks

```bash
git config core.hooksPath .githooks
```

This enables `.githooks/pre-commit` (scans staged diff for AI
authorship attribution) and `.githooks/commit-msg` (scans commit
message). The hooks block accidental `Co-Authored-By: ...AI` trailers
or `🤖 Generated with ...` footers — see `docs/plans/GIT_WORKFLOW.md
§1.2–1.4`.

## 5. GPU selection

The default assignment is GPU 6 with fallback 7
(`configs/models.yaml: hardware`). Override for a single run:

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/03_run_sweep.py ...
```

## 6. Verify

```bash
# Offline unit tests, no API calls
python -m pytest tests/

# Timeline generation (no API either — deterministic)
python scripts/01_generate_data.py --dry-run
```

Expected output:
```
timeline[yejin_florist]: 32 events across 60 days
timeline[minseok_cafe]: 39 events across 60 days
timeline[sunhee_hair]: 38 events across 60 days
dry-run: skipping LLM calls
```

If this passes you are ready for `02_data_generation.md` (which runs
the actual LLM calls).

## Troubleshooting

**`RuntimeError: Missing required env vars: ['OPENROUTER_API_KEY']`**
→ `.env` is missing or the shell did not load it.
`python-dotenv` loads `.env` from the repo root automatically, but
only inside the loader — confirm the file is at `heard/.env`.

**`httpx.ConnectError`** → check proxy / corporate firewall; OpenRouter
needs outbound HTTPS to `openrouter.ai`.

**`ProviderError: 401 Unauthorized`** → key typo or insufficient funds.

**`BudgetExceeded`** → `configs/models.yaml: budget.total_cap_usd`
is lower than your cumulative spend. Raise the cap or reset
`.api_cache/budget.json` if you intentionally want to burn through
more budget. The cap is a soft fuse, not a provider-side limit.
