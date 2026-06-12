# Heard v0.1 → v0.2

**A Korean on-device LLM that reflects your past self back at you at decision moments.**

Natural Language Processing (2150534701) · Term Project #1 · 2026 · Chanyoung Kim (20243053)

[![Part 1 Proposal](https://img.shields.io/badge/Part_1-Proposal_PDF-6b7280?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](report/260419_NLP_termproject_1.pdf)
[![Part 2 Technical Report](https://img.shields.io/badge/Part_2-Technical_Report-EF4444?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](report/report.pdf)
[![Final Report](https://img.shields.io/badge/Submission-Final_Report_PDF-1F6FEB?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](20243053.pdf)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-heard--bench-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/chanyoungkim/heard-bench)
[![Code Apache 2.0](https://img.shields.io/badge/Code-Apache_2.0-0A84FF?style=for-the-badge)](LICENSE)
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

> 📄 Full Part-1 framing (motivation, prior art, architecture
> sketch) is in the **proposal PDF**:
> [`report/260419_NLP_termproject_1.pdf`](report/260419_NLP_termproject_1.pdf).

---

## What v0.1 → v0.2 ships

**`heard-bench` — a 270-item Korean long-term-memory benchmark**
for the under-studied domain of nightly solo-business monologue.
No existing benchmark (LongMemEval, PerLTQA, LoCoMo) meets the
three criteria of Korean, monologue, and solo-business, so we
built one. Three tracks — `en_subset` (100), `ko_translated`
(100), and `ko_native` (70) — cover six long-term-memory
abilities (IE / MR / KU / TR / ABS + REFL). Items are
**adversarially filtered** so that no-memory baselines cannot
solve them, **4-gate validated**, and **author-reviewed**.

**An end-to-end on-device retrieval pipeline.** A
`multilingual-e5-small` cosine retriever shared by every SUT, and
a v0.2 SUT lineup of **11 configurations** across **four model
families** and a **4-bit-NF4 quantization axis** — Kakao Kanana
1.5 (2.1 B / 8 B, fp16 + int4), NAVER HyperCLOVA-X SEED (1.5 B,
fp16 + int4), Alibaba Qwen 2.5 (3 B fp16 + int4 / 7 B fp16), and
the Korean-tuned Llama 3 derivatives Bllossom-8B and Open-Ko-8B.
Two MIRROR policies (advisory baseline vs reflective constrained)
on every track. All runs stay on a single L40S 48 GB; per-response
latency stays in 1.8 – 5.2 s.

**The full Part 2 technical report** at
[`report/report.pdf`](report/report.pdf) — §3 reorganized around
the v0.2 11-SUT lineup with seven results tables and one results
figure. Raw source at [`report/report.md`](report/report.md);
pandoc + xelatex build toolchain at
[`report/build/`](report/build/).

> 📄 **Technical report →** [`report/report.pdf`](report/report.pdf)
> · Proposal (Part 1) →
> [`report/260419_NLP_termproject_1.pdf`](report/260419_NLP_termproject_1.pdf)

---

## Headline results (v0.2 expanded sweep)

**14,850 SUT generations · 2,112 pairwise judge verdicts.**

### NODE lift on `ko_native` — every SUT improves under retrieval

Advisory pass rate, contains-token, n = 64 non-REFL items.

| SUT | no_node | retrieval | oracle |
|---|---:|---:|---:|
| kanana_nano_int4      | 6.2% | **20.3%** | 20.3% |
| kanana_8b             | 1.6% |    18.8%  |  7.8% |
| kanana_8b_int4        | 0.0% |    18.8%  | 10.9% |
| open_ko_8b            | 6.2% |    17.2%  | 12.5% |
| bllossom_8b           | 3.1% |    15.6%  |  9.4% |
| hclova_seed_15b_int4  | 0.0% |    15.6%  | 14.1% |
| qwen25_3b             | 3.1% |    12.5%  | 10.9% |
| qwen25_3b_int4        | 1.6% |    12.5%  | 10.9% |
| hclova_seed_15b       | 3.1% |    10.9%  |  7.8% |
| **kanana_nano** *(v0.1)* | **4.7%** | **10.9%** | **15.6%** |
| qwen25_7b             | 4.7% |    10.9%  |  6.2% |

The retrieval lift is universal: every configuration records a
positive `retrieval`−`no_node` delta (mean +11.8 pp, minimum
+6.2 pp, no negative deltas). The on-device 2.1 B Part-1 target
(`kanana_nano`) anchors the table at the canonical
4.7 % → 10.9 % → 15.6 % trajectory.

### Reflective vs advisory — pairwise judge replicates v0.1 dominance under 5.5× sample

528 decisions per rubric · 11 SUTs × 2 conditions × 6 REFL items
× 2 judges × 2 A/B swaps.

| Rubric | Advisory wins | Reflective wins | Tie | Reflective share |
|---|---:|---:|---:|---:|
| specificity            | 158 | 348 | 22 | 65.9 % |
| non_directive          | 178 | 350 |  0 | 66.3 % |
| emotional_attunement   |  32 | **465** | 31 | **88.1 %** |
| open_question          |  33 | **466** | 29 | **88.3 %** |

The "mirror, do not advise" thesis comes through cleanly on
emotional attunement (88 %) and open-question framing (88 %),
recovering the v0.1 numbers (82 / 96 ≈ 85 % and 88 / 96 ≈ 92 %)
to within a percentage point.

### Per-SUT reflective share — `kanana_8b` leads the lineup

192 decisions per SUT · summed across both conditions and four
rubrics.

| SUT | Reflective | Advisory | Tie | Share |
|---|---:|---:|---:|---:|
| kanana_8b             | **168** | 22 |  2 | **87.5 %** |
| kanana_8b_int4        | 165 | 26 |  1 | 85.9 % |
| hclova_seed_15b       | 160 | 28 |  4 | 83.3 % |
| qwen25_7b             | 158 | 28 |  6 | 82.3 % |
| kanana_nano           | 149 | 35 |  8 | 77.6 % |
| kanana_nano_int4      | 148 | 38 |  6 | 77.1 % |
| hclova_seed_15b_int4  | 148 | 35 |  9 | 77.1 % |
| qwen25_3b             | 147 | 37 |  8 | 76.6 % |
| qwen25_3b_int4        | 140 | 40 | 12 | 72.9 % |
| bllossom_8b           | 134 | 49 |  9 | 69.8 % |
| open_ko_8b            | 112 | 63 | 17 | 58.3 % |

Combining retrieval pass, reflective share, latency (3.7 s on
`ko_native` retrieval), and quantization sensitivity, **the v0.2
recommended deployment SUT for *Heard* is `kanana_8b`** (or its
int4 quantization for memory-constrained devices).

### int4 vs fp16 — the marginal gains do not survive paired analysis

The 11-SUT marginals suggest +9.4 pp on Kanana 2.1B and +4.7 pp
on HCX-SEED 1.5B from int4 quantization. A paired item-level
McNemar test shows three of four pairs have z ≤ 1.13, and only
Kanana 2.1B reaches a marginal z = 1.90 (p ≈ 0.06):

| pair | both pass | fp16 only | int4 only | discordant z |
|---|---:|---:|---:|---:|
| kanana_nano        |  5 | 2 | 8 | 1.90 (marginal) |
| hclova_seed_15b    |  5 | 2 | 5 | 1.13 (n.s.) |
| kanana_8b          | 11 | 1 | 1 | 0.00 |
| qwen25_3b          |  5 | 3 | 3 | 0.00 |

The HCX-SEED int4 pass-rate gain is further confounded by a
1.5× response-length increase (mean 261 → 401 chars on
`ko_native` retrieval) — a substring metric reads longer
responses as mechanically more likely to hit. We treat int4 as a
**memory-bound deployment option** rather than a quality
improvement; latencies on this hardware run 1.3 – 2.0× *longer*
than fp16 for three of four families.

### Language-axis decay holds across the lineup

Advisory retrieval pass rate, ten of eleven SUTs preserve
`en_subset ≤ ko_translated < ko_native`. Mean across the lineup:

| Track | Mean pass rate |
|---|---:|
| `en_subset` (EN haystack)               | 0.6 % |
| `ko_translated` (KO query, EN haystack) | 2.9 % |
| `ko_native` (KO throughout)             | 14.8 % |

A 25× gap between the endpoints; multilingual baselines (Qwen)
do not close the en_subset gap despite English being their
native training distribution.

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

# Day 3 — indices + v0.2 sweep + judge
CUDA_VISIBLE_DEVICES=6 python scripts/07a_build_indices.py
CUDA_VISIBLE_DEVICES=6 python scripts/07_run_sweep.py \
    --run-dir experiments/<run_id>_v0.2_sweep
python scripts/08_run_judge.py --run-dir experiments/<run_id>_v0.2_sweep

# Day 4 — metrics, figures, HF release
python scripts/09_aggregate_metrics.py --run-dir experiments/<run_id>_v0.2_sweep
python scripts/99_make_figures.py      --run-dir experiments/<run_id>_v0.2_sweep
python scripts/10_upload_hf_dataset.py
```

Total API spend reproducing from scratch: **~USD 4** at the
`haiku-4.5 / gpt-4o-mini / gemini-2.5-flash` tier.

The v0.2 sweep is published in full at
[`experiments/20260426_1242_v0.2_sweep_merged/`](experiments/20260426_1242_v0.2_sweep_merged/):
14,850 generations in `results.jsonl`, aggregated metrics in
`metrics.csv` / `metrics.json`, and 2,112 pairwise judge verdicts
in `judge_verdicts.jsonl` / `judge_aggregate.json`.

---

## Repository layout

```
heard/
├── configs/
│   ├── models.yaml           # generator / judge / SUT registry (12 SUT configs)
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
├── experiments/
│   ├── 20260423_1610_day3_sweep/        # v0.1 reference (2 SUTs, archive)
│   └── 20260426_1242_v0.2_sweep_merged/ # v0.2 expanded sweep (11 SUTs)
├── report/
│   ├── report.pdf            # submission PDF
│   ├── report.md             # source
│   ├── figures/              # fig_overview.jpg + fig_v02_results.png
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

- **Code** — [Apache 2.0](LICENSE) · attribution + explicit patent grant
- **Dataset** — [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Upstream** — Track A inherits LongMemEval_S (MIT); attribution
  appears per-item in `en_subset/metadata.json`. See [NOTICE](NOTICE)
  for third-party attributions covering all eleven v0.2 SUTs.

---

## Acknowledgments

Claude Code assisted with pipeline implementation. Figure 1
(overview) was drawn in Paperbanana. All research framing, dataset
design, hypothesis formulation, and writing were performed by the
author.
