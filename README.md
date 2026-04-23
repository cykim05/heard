# Heard

A Korean long-term-memory benchmark and on-device LLM pipeline for
solo-business nightly monologue. Companion to the *Heard* proposal —
MIC (speech, out-of-scope for v0.1) → **NODE** (domain memory) →
**MIRROR** (reflective response).

**Dataset:** https://huggingface.co/datasets/chanyoungkim/heard-bench
**Report:** [`report/report.md`](report/report.md) · build PDF with
`bash report/build/build.sh` (requires pandoc + xelatex + Noto CJK)

## Headline results (v0.1, single L40S)

| | ko_native pass rate (advisory) | | |
|---|---:|---:|---:|
| SUT | no-NODE | retrieval | oracle |
| kanana_nano (2.1 B) | 4.7 % | **10.9 %** | 15.6 % |
| qwen25_3b  (3.0 B)  | 3.1 % | 12.5 % | 10.9 % |

Reflective vs advisory — 96 pairwise judge decisions per rubric:

| rubric | advisory wins | reflective wins | tie |
|---|---:|---:|---:|
| specificity          | 31 | 60 | 5 |
| non_directive        | 29 | 67 | 0 |
| emotional_attunement | **6** | **82** | 8 |
| open_question        | **5** | **88** | 3 |

## Quickstart

```bash
git clone git@github.com:cykim05/heard.git
cd heard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in OPENROUTER_API_KEY and HF_TOKEN
```

### Reproduce from scratch

The numbered scripts mirror the 4-day pipeline. Each step logs its
run dir and per-call cost so you can resume if a step fails.

```bash
# Day 1 — persona-driven utterance corpus (2,046 utts, ~USD 2)
python scripts/01_generate_data.py --out data/raw/utterances.jsonl

# Day 2 — three tracks
python scripts/02_subsample_longmemeval.py               # Track A
python scripts/03_generate_scenarios.py                  # Track C candidates
CUDA_VISIBLE_DEVICES=6 python scripts/04_adversarial_filter.py
python scripts/05_translate_track_b.py                   # Track B (KO Q+A)
python scripts/06_auto_validation.py                     # 4-gate validation

# Day 3 — embedding indices + SUT sweep + judge
CUDA_VISIBLE_DEVICES=6 python scripts/07a_build_indices.py
CUDA_VISIBLE_DEVICES=6 python scripts/07_run_sweep.py
python scripts/08_run_judge.py --run-dir experiments/<run_id>/

# Day 4 — metrics, figures, HF release
python scripts/09_aggregate_metrics.py --run-dir experiments/<run_id>/
python scripts/99_make_figures.py --run-dir experiments/<run_id>/
python scripts/10_upload_hf_dataset.py

# Optional — MIC demo (records mic input + Korean transcription)
python scripts/00_record_and_transcribe.py
```

Total API spend reproducing from scratch: ~USD 4 at the cost tier
declared in `configs/models.yaml`. The disk cache (`.api_cache/`)
makes a second run effectively free.

## Repository layout

```
heard/
├── configs/
│   ├── models.yaml           # generator/judge/SUT registry
│   └── personas/             # yejin_florist, minseok_cafe, sunhee_hair
├── src/
│   ├── datagen/              # utterance / scenario / translate / filter / validate
│   ├── mic/                  # recorder + faster-whisper transcriber
│   ├── node/                 # embedding retriever (store + retriever)
│   ├── mirror/               # advisory / reflective prompts + generator
│   ├── eval/                 # metrics, runner, judge
│   └── utils/                # openrouter client, llm_backend, config, logging
├── scripts/                  # numbered entrypoints 00..99
├── data/
│   ├── raw/                  # utterances + timelines (gitignored heavy files)
│   ├── scenarios/            # Track C survivors, verdicts
│   └── final/                # heard-bench tracks (en_subset, ko_translated, ko_native)
├── experiments/              # run artifacts (config, results, metrics, api_log)
├── report/
│   ├── report.md             # technical report body
│   ├── figures/              # fig_overview.jpg + 2×21:9 matplotlib PNGs
│   └── build/                # pandoc preamble, metadata, build.sh → PDF
└── docs/
    ├── plans/                # PLAN, IMPL_DETAILS, DATASET, GIT_WORKFLOW
    ├── lab_notebook/         # day1..day4 entries
    ├── guides/               # 01_environment_setup
    └── decisions/            # 4 ADRs
```

## Dataset at a glance

| track | items | language | haystack | abilities covered |
|---|---:|---|---|---|
| `en_subset`     | 100 | EN | EN | IE/MR/KU/TR/ABS × 20 |
| `ko_translated` | 100 | KO question + answer; EN haystack | EN | same as en_subset |
| `ko_native`     |  70 | KO | KO | IE 19 / TR 15 / KU 13 / ABS 10 / MR 7 / REFL 6 |

See the dataset card on HF for the full item schema and
construction pipeline.

## Design decisions

Living ADRs under `docs/decisions/`:

- **0001** — OpenRouter as unified LLM gateway
- **0002** — Cost-constrained model tier (Haiku / mini / flash)
- **0003** — Track B MVP scope (question + gold_answer only)
- **0004** — Validation Gate 1 demoted to advisory

## License

Code: **MIT**. Dataset: **CC-BY-4.0**. LongMemEval_S (Track A source)
is MIT and attributed per-item in the `en_subset` metadata.

## Citation

```
@misc{heard-bench-2026,
  title        = {heard-bench: A Korean Long-Term Memory Benchmark for Solo-Business Monologue},
  author       = {Kim, Chanyoung},
  year         = {2026},
  howpublished = {huggingface.co/datasets/chanyoungkim/heard-bench}
}
```
