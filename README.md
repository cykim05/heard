# Heard v0.1

**A Korean on-device LLM that reflects your past self back at you at decision moments.**

Natural Language Processing (2150534701) · Term Project #1 · 2026 · Chanyoung Kim (20243053)

[![Part 1 Proposal](https://img.shields.io/badge/Part_1-Proposal_PDF-6b7280?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](report/260419_NLP_termproject_1.pdf)
[![Part 2 Report](https://img.shields.io/badge/Part_2-Report_PDF-EF4444?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](report/report.pdf)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-heard--bench-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/chanyoungkim/heard-bench)
[![Code MIT](https://img.shields.io/badge/Code-MIT-0A84FF?style=for-the-badge)](LICENSE)
[![Dataset CC-BY-4.0](https://img.shields.io/badge/Dataset-CC--BY--4.0-34C759?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

![Heard v0.1 overview](report/figures/fig_overview.jpg)

---

## The idea

It's 23:40. The flower-shop owner is wiping down the counter, 
thinking about whether to raise the price on rose stems. There's
no coworker to check in with, no staff meeting tomorrow, and no
one who would hear the worry the way it was said. Korea has roughly
**1.4 million** people like her — solo-business owners whose whole
workday is a conversation with themselves.

Heard is the Part 1 NLP proposal for an on-device Korean assistant
built for that conversation. Its architecture has three pillars:

- **MIC** — always-off tap-to-talk speech-to-text. Implemented in
  v0.1 as `sounddevice` + `faster-whisper`, not benchmarked yet.
- **NODE** — domain-specific typed memory over the user's past
  utterances.
- **MIRROR** — a reflective response that **quotes** the user's
  past words back to them, avoids imperatives, and ends with an
  **open question** instead of advice.

The goal isn't to give advice. It's to hand back a sentence the
user once said to themselves, and let them make their own call.

---

## What v0.1 shipped

**`heard-bench` — a 270-item Korean long-term-memory benchmark**
for the under-studied domain of nightly solo-business monologue.
No existing benchmark (LongMemEval, PerLTQA, LoCoMo) meets the
three criteria of Korean, monologue, and solo-business, so we
built one. Three tracks — `en_subset` (100), `ko_translated`
(100), and `ko_native` (70) — cover six long-term-memory
abilities (IE / MR / KU / TR / ABS + REFL). Items are
**adversarially filtered** so that no-memory baselines cannot
solve them, **4-gate validated**, and **author-reviewed**.

**An end-to-end on-device retrieval pipeline** — `multilingual-e5-small`
cosine retriever, two cost-tier SUTs (Kanana 2.1B, Qwen 2.5 3B),
advisory vs reflective prompt policies, and a pairwise LLM-as-judge
for REFL quality. All runs stay on a single L40S, and per-response
latency is under 2.7 s.

**A one-page technical report** at
[`report/report.pdf`](report/report.pdf) with three 21 : 9 result
figures and a single Paperbanana overview (shown above). Raw
source at [`report/report.md`](report/report.md); pandoc + xelatex
build toolchain at [`report/build/`](report/build/).

---

## Headline results

**NODE lift on ko_native (advisory policy).** Memory doubles
Kanana-2.1B's pass rate on the hardest subset of items, and an
oracle ceiling extends another 1.4× beyond dense retrieval.

| SUT | no-NODE | retrieval | oracle |
|---|---:|---:|---:|
| **Kanana 2.1 B** | 4.7% | **10.9%** | 15.6% |
| Qwen 2.5 3 B     | 3.1% | 12.5%    | 10.9% |

**Language-axis decay (Kanana retrieval).** Korean-native data is
necessary, not a preference.

| Track | Pass rate |
|---|---:|
| `en_subset` (EN haystack)              |  0.0% |
| `ko_translated` (KO query, EN haystack)|  5.0% |
| `ko_native`   (KO throughout)          | 10.9% |

**Reflective vs advisory.** 96 pairwise judge decisions per rubric
(two SUTs × two conditions × two judges × two A/B swaps × six
REFL items).

| Rubric | Advisory wins | Reflective wins | Tie |
|---|---:|---:|---:|
| specificity          | 31 | 60 | 5 |
| non-directive        | 29 | 67 | 0 |
| emotional attunement |  6 | **82** | 8 |
| open question        |  5 | **88** | 3 |

The "mirror, do not advise" thesis comes through cleanly on
emotional attunement (85%) and open-question framing (92%).

---

## Quickstart

```bash
git clone https://github.com/cykim05/heard
cd heard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill OPENROUTER_API_KEY + HF_TOKEN
```

Try the MIC demo (records a clip, transcribes it with Whisper):

```bash
python scripts/00_record_and_transcribe.py
```

Load the dataset:

```python
from datasets import load_dataset
ko_native     = load_dataset("chanyoungkim/heard-bench", "ko_native",     split="test")
ko_translated = load_dataset("chanyoungkim/heard-bench", "ko_translated", split="test")
en_subset     = load_dataset("chanyoungkim/heard-bench", "en_subset",     split="test")
```

---

## Reproduce from scratch

Every script is an independent CLI. Disk-cache on
`(model, messages, temperature, seed)` makes a second pass
effectively free.

```bash
# Day 1 — 2,046-utterance corpus (~USD 2)
python scripts/01_generate_data.py --out data/raw/utterances.jsonl

# Day 2 — three tracks
python scripts/02_subsample_longmemeval.py              # Track A
python scripts/03_generate_scenarios.py                 # Track C candidates
CUDA_VISIBLE_DEVICES=6 python scripts/04_adversarial_filter.py
python scripts/05_translate_track_b.py                  # Track B
python scripts/06_auto_validation.py                    # 4-gate validation

# Day 3 — indices + sweep + judge
CUDA_VISIBLE_DEVICES=6 python scripts/07a_build_indices.py
CUDA_VISIBLE_DEVICES=6 python scripts/07_run_sweep.py
python scripts/08_run_judge.py --run-dir experiments/<run_id>/

# Day 4 — metrics, figures, HF release
python scripts/09_aggregate_metrics.py --run-dir experiments/<run_id>/
python scripts/99_make_figures.py      --run-dir experiments/<run_id>/
python scripts/10_upload_hf_dataset.py
```

Total API spend reproducing from scratch: **~USD 4** at the
`haiku-4.5 / gpt-4o-mini / gemini-2.5-flash` tier.

---

## Repository layout

```
heard/
├── configs/
│   ├── models.yaml           # generator / judge / SUT registry
│   └── personas/             # yejin_florist, minseok_cafe, sunhee_hair
├── src/
│   ├── datagen/              # utterance, scenario, translate, filter, validate
│   ├── mic/                  # recorder + faster-whisper transcriber
│   ├── node/                 # embedding index + cosine retriever
│   ├── mirror/               # advisory / reflective / listening prompts
│   ├── eval/                 # runner, metrics, pairwise judge
│   └── utils/                # openrouter client, llm_backend, config
├── scripts/                  # numbered entrypoints 00..99
├── data/
│   ├── raw/                  # utterance corpus + timelines
│   ├── scenarios/            # Track C survivors + verdicts
│   └── final/                # heard-bench tracks
├── experiments/              # config, results.jsonl, metrics, api_log
├── report/
│   ├── report.pdf            # submission PDF
│   ├── report.md             # source
│   ├── figures/              # fig_overview.jpg + two 21:9 matplotlib PNGs
│   └── build/                # pandoc + xelatex toolchain
└── docs/
    ├── plans/                # PLAN, IMPL_DETAILS, DATASET, GIT_WORKFLOW
    ├── decisions/            # 4 ADRs
    ├── guides/               # 01_environment_setup
    └── lab_notebook/         # Day 1–4 narrative
```

---

## Design decisions

Four ADRs under [`docs/decisions/`](docs/decisions/) capture the
non-obvious calls:

- **0001** OpenRouter as the unified LLM gateway (vs one SDK per provider)
- **0002** Cost-constrained model tier (haiku / mini / flash instead of Sonnet / GPT-4o / Gemini Pro)
- **0003** Track B scope (translate question + gold answer only; haystack stays English)
- **0004** Gate 1 of 4-gate validation demoted to advisory

---

## Citation

```bibtex
@misc{heard-bench-2026,
  title        = {heard-bench: A Korean Long-Term Memory Benchmark for Solo-Business Monologue},
  author       = {Kim, Chanyoung},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/chanyoungkim/heard-bench}}
}
```

---

## License

- **Code** — [MIT](LICENSE)
- **Dataset** — [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Upstream** — Track A inherits LongMemEval_S (MIT); attribution
  appears per-item in `en_subset/metadata.json`.

---

## Acknowledgments

Claude Code assisted with pipeline implementation. Figure 1
(overview) was drawn in Paperbanana. All research framing, dataset
design, hypothesis formulation, and writing were performed by the
author.
