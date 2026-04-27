# Heard v0.1 → v0.2: A Korean Long-Term Memory Benchmark and On-Device Retrieval Pipeline for Solo-Business Monologue

**Chanyoung Kim** · NLP Term Project Part 2 · 2026-04-28

| | |
|---|---|
| Website | https://cykim05.github.io/ |
| Code | https://github.com/cykim05/heard |
| Dataset | https://huggingface.co/datasets/chanyoungkim/heard-bench |

---

## Summary

Heard (Part 1 NLP proposal) is an on-device Korean LLM assistant
that reflects a user's past self back at them at decision moments.
Its three architectural pillars are MIC (speech-to-text,
implemented as a sounddevice + faster-whisper pipeline but not
evaluated in v0.1), NODE (domain-specific typed memory), and
MIRROR (reflective response). Part 2 asks whether a lightweight
on-device memory pipeline yields measurable gains for a small
Korean LM on nightly-self-talk questions, and whether a reflective
prompt policy beats a generic advisory one.

Evaluating this requires a long-term memory benchmark in Korean,
covering monologue (not dialogue), in the solo-business domain. No
existing benchmark — LongMemEval, PerLTQA, or LoCoMo — satisfies
all three criteria, so we constructed heard-bench: 270 items across
three tracks (en_subset, ko_translated, ko_native), generated from
a 2,046-utterance synthetic corpus over three solo-business
personas, adversarially filtered against no-memory baselines,
4-gate auto-validated, and author-reviewed. The dataset is released
on HuggingFace under CC-BY-4.0.

On the resulting benchmark, the v0.2 sweep evaluates eleven SUT
configurations across four model families (Kanana 1.5, Qwen 2.5,
HyperCLOVA-X SEED, and the Korean-tuned Llama 3 derivatives
Bllossom and Open-Ko), three parameter scales (1.5–8 B), and a
4-bit-NF4 quantization axis applied wherever the model fits within
a 30 GB memory budget. The sweep totals 14,850 SUT generations and
2,112 pairwise judge verdicts.

Three results are robust across the lineup. First, dense retrieval
lifts contains-pass on `ko_native` for every configuration (mean
+11.8 pp `retrieval`−`no_node`, minimum +6.2 pp, no negative
deltas), with the on-device 2.1 B target named in Part 1
(`kanana_nano`) following the canonical 4.7 % → 10.9 % → 15.6 %
trajectory. Second, the reflective policy dominates the advisory
baseline on every rubric across 528 pairwise judge decisions per
rubric, with the two "mirror, do not advise" rubrics — emotional
attunement and open-question framing — landing at 88.1 % and
88.3 % reflective share. Third, the language-axis ordering
en_subset ≤ ko_translated < ko_native is preserved by ten of
eleven SUTs, establishing Korean-native data as necessary rather
than optional. The strongest single configurations on `ko_native`
retrieval are `kanana_nano_int4` (20.3 %, but borderline-noise
under paired analysis), `kanana_8b` and `kanana_8b_int4` (18.8 %),
and `open_ko_8b` (17.2 %); combining retrieval pass, reflective
share, latency, and quantization sensitivity, we recommend
`kanana_8b` as the v0.2 default deployment SUT for the *Heard*
product.

![Heard v0.1 overview — pipeline and headline results.](figures/fig_overview.jpg){#fig:overview}

**Figure 1.** *Heard v0.1 pipeline and headline results at a
glance.* The **top band** shows the three-pillar architecture —
MIC for speech-to-text (dashed, not evaluated in v0.1), NODE for
the five typed node categories (customer / stock / pricing / mood
/ decision), and MIRROR for the reflective response style (cite
past self, no imperative, open question). The **middle band** is
split into two halves. The left half compresses the dataset
construction pipeline: three persona cards and the LongMemEval_S
source feed into a 2,046-utterance corpus and a
146 → 122 → 70 ko_native funnel, converging on heard-bench v0.1
(270 items). The right half is a three-condition filmstrip that
shows what each condition delivers to the SUT: question only
(`no_node`, 4.7%), question + top-5 retrieved (`retrieval`, 10.9%),
and question + gold evidence (`oracle`, 15.6%). The **bottom
band** summarises the three headline findings — the NODE lift on
`ko_native` (×3.3 end-to-end for Kanana), language-axis decay
across tracks, and the reflective-vs-advisory judge outcomes on
emotional attunement (82 wins) and open-question framing (88 wins)
out of 96 pairwise decisions.

## 1 Introduction

### 1.1 Background

Heard is the Part 1 NLP proposal for an on-device Korean LLM
assistant. Its architecture has three pillars. MIC handles
always-off tap-to-talk speech-to-text; it is implemented in v0.1
but not evaluated in Part 2 (§2.1). NODE is the domain-specific
typed memory built over utterances. MIRROR produces reflective
responses that quote the user's past words and end with an open
question rather than giving advice.

The proposal targets Korea's 1.4M-member solo-business community,
where nightly self-talk arrives from people who have no coworkers,
no staff, and no one to check in with. Part 2 asks two empirical
questions: (1) does a lightweight retrieval-based memory pipeline
yield measurable gains for a small Korean on-device LM on nightly
self-talk questions, and (2) does a reflective prompt policy beat a
generic advisory policy when memory is available?

### 1.2 The benchmark gap

Answering these empirical questions requires a long-term memory
benchmark meeting three criteria: (a) Korean, (b) monologue
(nightly self-talk, not dialogue), and (c) solo-business
decision-making. A survey of recent long-term memory benchmarks
returns no candidate that satisfies all three.

| Benchmark | Language | Format | Domain |
|---|---|---|---|
| LongMemEval (Wu et al., ICLR 2025) | EN | user–AI dialogue | generic chat |
| PerLTQA (Du et al., 2024) | ZH / EN | human–human dialogue | social events |
| LoCoMo (Maharana et al., 2024) | EN | long dialogue | generic personal |

We therefore construct **heard-bench**: a 270-item benchmark across
three tracks. The construction pipeline is: (1) generate a
2,046-utterance synthetic corpus over three solo-business personas
(florist, roaster-café owner, one-chair hair salon), (2) compose
scenarios with today-questions and evidence utterances, (3)
adversarially filter against no-memory baselines (3 SUTs × 3 trials,
keeping only items where all fail), (4) 4-gate auto-validate
(evidence consistency, history-only fail, cross-generator
duplication, ambiguity score), and (5) author-review the survivors
for naturalness and gold-answer correctness. The dataset is
released on HuggingFace under CC-BY-4.0.

### 1.3 Contributions

1. **Dataset construction as necessary infrastructure** (§2.2). We
   construct, adversarially filter, 4-gate validate, and
   author-review heard-bench, a 270-item set across en_subset,
   ko_translated, and ko_native tracks, and release it on
   HuggingFace under CC-BY-4.0. No existing long-term memory
   benchmark meets our three criteria of Korean / monologue /
   solo-business.

2. **Empirical NODE lift** (§3.1). Dense retrieval lifts
   Kanana-2.1B's ko_native pass rate from 4.7% (no memory) to
   10.9% (retrieval) to 15.6% (oracle), a 3.3× end-to-end gain
   over the memoryless baseline.

3. **Language axis decay confirms Korean-native necessity**
   (§3.2). Kanana's retrieval pass rate decays monotonically:
   ko_native 10.9% → ko_translated 5.0% → en_subset 0.0%.
   Korean-native data is necessary, not optional, for a Korean
   on-device assistant.

4. **Reflective policy dominates on emotional and open-question
   axes** (§3.3). Across 96 pairwise judge decisions per rubric,
   reflective beats advisory on emotional attunement 82/96 and
   on open-question framing 88/96, supporting the Part 1 MIRROR
   thesis.

## 2 Methods

All code, configurations, and experiment artifacts are available at
**https://github.com/cykim05/heard** (Apache 2.0). The dataset is published
at **https://huggingface.co/datasets/chanyoungkim/heard-bench**
(CC-BY-4.0). Per-call API logs under `experiments/_api_log/` and
per-run folders under `experiments/<run_id>/` enable full
reproducibility; diskcache keyed on `(model, messages, seed)` makes a
cold rerun free.

### 2.1 MIC — recording and transcription

Heard's first pillar turns spoken nightly monologue into text
before NODE ingests it. We implement it as a lightweight
tap-to-talk recorder plus a Whisper-family transcriber; the code
ships with v0.1 but is not exercised by the evaluation in §3.

**Recorder** (`src/mic/recorder.py`). A `sounddevice` wrapper
captures 16 kHz mono PCM in float32, which is the format
Whisper-family models expect natively. Two usage modes are
supported: `record_fixed(seconds)` for scripted capture, and a
`Recorder` context manager for tap-to-talk (start, wait for the
user, stop). The captured array is either passed directly to the
transcriber or written to a 16-bit WAV via `save_wav`.

**Transcriber** (`src/mic/transcribe.py`). A thin `faster-whisper`
wrapper (CTranslate2 backend), defaulting to the Whisper `small`
model (244 M parameters). Language is fixed to Korean; VAD filtering
and beam search are exposed as kwargs. Heavy model loading is
lazy, so importing the module is cheap even in environments that
never run STT. `transcribe()` returns a single concatenated
transcript; `transcribe_segments()` returns per-segment timestamps
for later utterance splitting.

**Integration.** `scripts/00_record_and_transcribe.py` is the
reference integration: it records via `Recorder`, optionally
saves a WAV, and routes the audio through `Transcriber`. The
script ships as a working demo rather than a v0.1 evaluation
target, since measuring WER on Korean casual monologue requires a
dedicated audio benchmark that we leave to v2. All subsequent
evaluation in §3 operates on gold utterance text so that the
memory-pipeline question is isolated from STT error.

### 2.2 Dataset construction

The heard-bench construction pipeline has five stages: corpus
generation, scenario composition, adversarial filtering, 4-gate
auto-validation, and author review. We describe each in turn.

**Personas.** Three synthetic Korean solo-business owners anchor
the corpus: Yejin (florist, Mapo), Minseok (roaster-café, Seongsu),
and Sunhee (one-chair hair salon, Hongje). Each persona card
specifies regulars, stock or services, recurring stressors, and
timestamped historical anchors (`days_ago`-indexed past events) so
that temporal-reasoning and multi-session items have grounded
evidence. The persona set is synthetic to avoid privacy concerns;
all names and establishments are fictitious.

**Utterance corpus.** One generator call per (persona, day) block
across 60 days produces 12 utterances per day on average, rotating
across `anthropic/claude-haiku-4.5`, `openai/gpt-4o-mini`, and
`google/gemini-2.5-flash` so that no single generator's fingerprint
dominates. 43% of the resulting utterances reference a persona
historical anchor, and these references become retrieval gold
hooks in later stages.

| Persona | Role / location | Days | Utterances | Historical-ref share |
|---|---|---:|---:|---:|
| yejin_florist | Florist, Mapo             | 60 | 686 | 43% |
| minseok_cafe  | Roaster-café, Seongsu     | 60 | 687 | 43% |
| sunhee_hair   | One-chair hair salon, Hongje | 60 | 673 | 43% |
| **Total** | | | **2,046** | **881 / 2,046** |

**Three tracks.** heard-bench is organized into three tracks.

| Track | Items | Language | Haystack | Source | License |
|---|---:|---|---|---|---|
| `en_subset`     | 100 | EN            | EN            | LongMemEval_S stratified (20 per ability) | MIT |
| `ko_translated` | 100 | KO Q + answer | EN (preserved)| Gemini 2.5 Flash translation              | CC-BY-4.0 |
| `ko_native`     |  70 | KO            | KO (our corpus) | persona-driven gen + adversarial filter | CC-BY-4.0 |
| **Total** | **270** | | | | |

**Ability distribution (ko_native).** The native-track items are
stratified by the six long-term-memory abilities we target. IE, MR,
KU, TR, and ABS follow LongMemEval's taxonomy; REFL (reflective
quality) is specific to Heard.

| Ability | Count | What it tests |
|---|---:|---|
| IE  | 19 | Specific fact recall |
| TR  | 15 | Temporal reference resolution |
| KU  | 13 | Knowledge update (latest state) |
| ABS | 10 | Abstention (never mentioned) |
| MR  |  7 | Multi-session aggregation |
| REFL|  6 | Reflective response quality (LLM-judged) |

**Validation pipeline.** ko_native candidates pass through four
stages before inclusion:

1. **Adversarial filtering**: 150 candidates generated → 146
   successfully parsed → 122 survived (all three no-memory
   baselines, Kanana-2.1B / Qwen-1.5B / Qwen-3B, fail across three
   trials each). This ensures that memory is structurally
   necessary to solve the retained items.

2. **4-gate auto-validation**: evidence-answer consistency,
   history-only fail reproduction, cross-generator 4-gram overlap
   < 15%, and ambiguity score ≥ 3/5 from a third-party LLM rater
   (ADR 0004).

3. **Author review**: the 70 surviving items were reviewed by the
   author for naturalness, evidence alignment, and gold-answer
   correctness. Minor edits to gold substrings and fluency fixes
   were applied as needed.

4. **Final count**: 70 ko_native items, shorter than the originally
   planned 100. The rejection rate at Gate 4 is discussed in §4.5.

Algorithm 1 summarises the adversarial filter. Any candidate that
is solvable without memory by any of the three SUTs in any of three
trials is discarded; only candidates where memory is structurally
required survive.

```text
Algorithm 1 — Adversarial filter
---------------------------------------------------------
input  : candidate set C, SUT pool M = {m1, m2, m3},
         trials T = 3
output : hard subset H ⊆ C

H ← ∅
for each candidate c ∈ C:
    any_pass ← False
    for each m ∈ M:
        for t in 1..T:
            r ← m.generate(c.question)      // no memory
            if gold_criterion(r, c.gold):
                any_pass ← True
                break
        if any_pass: break
    if not any_pass:
        H ← H ∪ {c}
return H
```

**ADR 0003 summary.** Track B keeps only question and gold_answer
in Korean; haystack stays English, because full-session translation
exceeded Gemini-Flash's output token cap. Chunked-session
translation is a concrete v2 path.

**ADR 0004 summary.** Gate 1 was demoted to advisory after a
scenario-generation regression produced false negatives that did
not correlate with SUT evaluation value.

### 2.3 Memory pipeline

**Retriever.** We use `intfloat/multilingual-e5-small` (118 M
parameters), mean-pool the final hidden states, L2-normalize, and
take cosine top-5. The index is per-persona for ko_native
(~680 utterances each) and per-item for en_subset and
ko_translated (~50 sessions per item).

### 2.4 Systems under test

The v0.1 sweep evaluated two SUTs (`kanana_nano` and `qwen25_3b`,
both fp16), and the v0.2 expansion this report describes runs the
strict superset shown in Table A. The two v0.1 rows are retained
verbatim in the table for continuity; reading the table top-to-
bottom recovers the v0.1 sweep, while reading all eleven rows
gives the v0.2 sweep. All configurations run sequentially on a
single NVIDIA L40S 48 GB, with the embedder and the SUT
co-resident in memory; the listed memory budget for int4 is the
observed peak during the sweep run.

**Table A.** *SUT lineup (v0.2 = strict superset of v0.1).
Configurations marked v0.1 are the two-SUT reference set from the
v0.1 release (results archived under
`experiments/20260423_1610_day3_sweep/`); all eleven rows are
evaluated in §3.1–§3.5 of this report.*

| Logical | HF id | Params | Quant | License | Sweep |
|---|---|---:|---|---|:---:|
| kanana_nano          | `kakaocorp/kanana-1.5-2.1b-instruct-2505`             | 2.09 B | fp16 | Apache-2.0 | v0.1, v0.2 |
| qwen25_3b            | `Qwen/Qwen2.5-3B-Instruct`                            | 3.09 B | fp16 | Apache-2.0 | v0.1, v0.2 |
| kanana_nano_int4     | `kakaocorp/kanana-1.5-2.1b-instruct-2505`             | 2.09 B | int4 | Apache-2.0 | v0.2 |
| kanana_8b            | `kakaocorp/kanana-1.5-8b-instruct-2505`               | 8.03 B | fp16 | Apache-2.0 | v0.2 |
| kanana_8b_int4       | `kakaocorp/kanana-1.5-8b-instruct-2505`               | 8.03 B | int4 | Apache-2.0 | v0.2 |
| hclova_seed_15b      | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | 1.59 B | fp16 | HCX-SEED-Public (gated) | v0.2 |
| hclova_seed_15b_int4 | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | 1.59 B | int4 | HCX-SEED-Public (gated) | v0.2 |
| qwen25_3b_int4       | `Qwen/Qwen2.5-3B-Instruct`                            | 3.09 B | int4 | Apache-2.0 | v0.2 |
| qwen25_7b            | `Qwen/Qwen2.5-7B-Instruct`                            | 7.62 B | fp16 | Apache-2.0 | v0.2 |
| bllossom_8b          | `MLP-KTLim/llama-3-Korean-Bllossom-8B`                | 8.03 B | fp16 | Llama-3 Community | v0.2 |
| open_ko_8b           | `beomi/Llama-3-Open-Ko-8B-Instruct-preview`           | 8.03 B | fp16 | Llama-3 Community | v0.2 |

The v0.2 sweep extends the v0.1 lineup along three axes: a third
parameter decade (`hclova_seed_15b` at 1.5 B and `kanana_8b` at
8 B), a model-architecture axis (the Korean-tuned Llama 3
derivatives `bllossom_8b` and `open_ko_8b`), and a 4-bit-NF4
quantization axis on the four configurations that fit under a
30 GB memory budget. HyperCLOVA-X SEED 1.5B was admitted once
HuggingFace gated-access approval was granted. Three further
SUTs were attempted but dropped: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`
(the installed `transformers` 4.57.6 lacks the `RopeParameters`
symbol the EXAONE config imports), `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`,
and `upstage/SOLAR-10.7B-Instruct-v1.0` (fp16 throughput too low
under the reflective policy to finish within the wall-clock budget,
≈47 and ≈200 generations per hour respectively); all three remain
v0.3 candidates.

**Per-family description.** Four model families participate in the
v0.2 sweep, each chosen for a specific role.

- **Kakao Kanana 1.5 (Apache-2.0).** Kanana-1.5-2.1B is the
  primary on-device target for the *Heard* product and the SUT
  whose v0.1 numbers anchor §3.1; the 8B sibling Kanana-1.5-8B is
  included as an in-family reference ceiling. Both are
  Korean-native instruction-tuned models from Kakao, available
  under a permissive Apache-2.0 license that does not restrict
  redistribution of model output.
- **NAVER HyperCLOVA-X SEED (HCX-SEED-Public, gated).**
  HyperCLOVAX-SEED-Text-Instruct-1.5B is NAVER's smallest
  Korean-native instruction model and the lightest configuration
  in the lineup at 1.59 B. Its license requires HuggingFace
  gated-access approval, which was granted to the runtime account
  on 2026-04-26; outputs are retained in `experiments/` and the
  model is not redistributed by this report.
- **Alibaba Qwen 2.5 (Apache-2.0).** Qwen 2.5-3B and Qwen 2.5-7B
  are multilingual instruction-tuned baselines, included to
  measure how much a strong general multilingual model can close
  the gap against a Korean-native model on Korean-language
  evaluation. The 3B variant anchors v0.1 alongside Kanana 2.1B;
  the 7B variant tests whether scaling the multilingual baseline
  catches up to Korean-native specialists at similar parameter
  count.
- **Korean-tuned Llama 3 derivatives (Llama-3 Community License).**
  Bllossom-8B and Open-Ko-8B are independent Korean fine-tunes of
  Meta Llama 3 8B, released by MLP-KTLim and `beomi` respectively.
  They test whether a Korean fine-tune of a stronger non-Korean
  base outperforms a Korean-native model at the same parameter
  count; the Llama-3 Community License permits research use and
  redistribution under attribution and naming rules that we
  preserve in the repository's `NOTICE` file.

The retriever is shared across all SUTs:
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT, 118 M parameters), with mean-pooled L2-normalized
embeddings and a top-5 cosine policy as described in §2.3.

### 2.5 MIRROR policies

We implement two policies in `src/mirror/prompts.py`. The advisory
policy serves as a generic-assistant baseline, permitting
imperatives and targeting three to eight sentences of concise
bullet advice. The reflective policy imposes three hard
constraints: each response must cite at least one retrieved memory
verbatim, avoid imperatives entirely, and end with an open
question, within a three- to six-sentence target. Advisory is the
baseline against which we measure the Part 1 MIRROR style; a third
listening policy is implemented but unused in v0.1.

### 2.6 Conditions

We compare three memory conditions. The *no_node* baseline
presents only the today-question. The *retrieval* condition
augments the question with the top-5 neighbors from the embedding
index. The *oracle* condition replaces retrieval with every
document whose id appears in the item's `evidence_*_ids` lists,
capped at k × 3 (= 15) for prompt length. Algorithm 2 gives the
test-time pipeline shared by all three.

```text
Algorithm 2 — Retrieval-augmented inference
---------------------------------------------------------
input  : item x, SUT m, policy π ∈ {advisory, reflective},
         condition c ∈ {no_node, retrieval, oracle},
         index I, embedder E, top-k = 5
output : response string r

if c = no_node:
    ctx ← ∅
elif c = retrieval:
    q   ← E.encode(x.question)
    ctx ← top_k(I, q, k = 5)          // cosine similarity
elif c = oracle:
    ctx ← { u ∈ I.docs | u.id ∈ x.evidence_ids }
    ctx ← ctx[0 : 3·k]                 // cap at 15

prompt ← format(π, ctx, x.question)    // policy prompt template
r      ← m.generate(prompt, max_tokens = 200, temperature = 0.3)
return r
```

### 2.7 Judging

Reflective-quality pairs (advisory vs reflective response) were
scored by two cost-tier judges, `openai/gpt-4o-mini` and
`anthropic/claude-haiku-4.5`, on four rubrics (specificity,
non_directive, emotional_attunement, open_question) with A/B swap
for position-bias mitigation. 6 REFL items × 2 SUTs × 2 conditions
× 2 judges × 2 swaps = **96 pairwise decisions per rubric**
aggregated in `experiments/.../judge_aggregate.json`. Algorithm 3
details the pairwise protocol.

```text
Algorithm 3 — Pairwise reflective-quality judging
---------------------------------------------------------
input  : REFL items X_REFL, SUT pool S, judges J,
         rubrics R = {specificity, non_directive,
                      emotional_attunement, open_question}
output : win counts W[s, c, rubric] → {adv, refl, tie}

for each x ∈ X_REFL, s ∈ S, c ∈ {retrieval, oracle}:
    r_adv  ← results[x, s, c, advisory].response
    r_refl ← results[x, s, c, reflective].response
    for each j ∈ J:
        for swap ∈ {False, True}:
            (A, B) ← (r_refl, r_adv) if swap else (r_adv, r_refl)
            v     ← j.score(x.question, A, B, R)   // per-rubric
            if swap:
                v ← {A↔B, tie↔tie}                 // invert
            for each ρ ∈ R:
                W[s, c, ρ][v[ρ]] += 1
return W
```

Each rubric therefore accumulates `|X_REFL| · |S| · 2 · |J| · 2`
= 6 · 2 · 2 · 2 · 2 = 96 decisions.

### 2.8 Budget and reproducibility

| Phase | API cost (USD) |
|---|---:|
| Day 1 utterance corpus | 1.88 |
| Day 2 scenarios + translation + 4-gate | 2.04 |
| Day 3 judge | 0.06 |
| Misc / smoke / retries | ~0.02 |
| **Total** | **≈ 4.00 / 30.00 cap (13%)** |

Every API call logs `(model, temperature, seed, usage.cost)` to
`experiments/_api_log/api_calls.jsonl`. Diskcache keyed on
`(model, messages, temperature, seed, response_format)` makes a
cold rerun free. Run artifacts (config, results.jsonl, metrics,
per-SUT model revisions) are committed under
`experiments/<run_id>/`.

## 3 Results

We report results from the v0.2 expanded sweep, which evaluates
all eleven SUT configurations of Table A on the full v0.1 dataset
under both advisory and reflective policies. Numbers are organized
around the five Tables 1–6 introduced below; Figure 2 visualizes
the three primary findings.

![v0.2 results across eleven SUTs: (a) NODE lift on `ko_native`, (b) reflective-share per SUT from the pairwise judge, (c) fp16 vs int4 paired contains-pass with item-level discordance.](figures/fig_v02_results.png){#fig:v02}

**Figure 2.** *v0.2 results, three side-by-side panels.*
**Panel (a) — NODE lift on `ko_native`.** Eleven v0.2 SUTs,
sorted left-to-right by retrieval pass rate, with grouped bars
for the three memory conditions (`no_node` grey, `retrieval` blue,
`oracle` sage). Every SUT records a positive
`retrieval`−`no_node` delta; the strongest configuration
(`kanana_nano_int4`) ties retrieval and oracle at 20.3 %.
**Panel (b) — Reflective share per SUT.** Each horizontal bar
aggregates 192 pairwise judge decisions per SUT (4 rubrics × 6
REFL items × 2 conditions × 2 judges × 2 A/B swaps); the red
segment is reflective wins, the tan segment advisory, the light
grey segment ties. Reflective share spans 58 % (`open_ko_8b`) to
88 % (`kanana_8b`). **Panel (c) — fp16 vs int4 paired.** For each
of the four SUT families with both quantizations, the two bars
show contains-pass on `ko_native` retrieval (n = 64 paired items);
the annotation above each pair gives the item-level discordance —
items where exactly one quantization passed — together with the
McNemar z-score. Three of four pairs have z ≤ 1.13; only Kanana
2.1B reaches a marginal z = 1.90.

### 3.1 NODE lift across the v0.2 lineup (Table 1)

The first headline result is that dense embedding retrieval lifts
contains-pass on `ko_native` for every SUT in the lineup. Table 1
gives the per-SUT, per-condition pass rate under the advisory
policy (n = 64 non-REFL items per cell, sorted by retrieval pass
rate descending); Figure 2(a) is a graphical rendering of the
same data.

**Table 1.** *Advisory pass rate on `ko_native`, contains-token
metric, n = 64. Sorted by `retrieval`.*

| SUT | no_node | retrieval | oracle | Δ retrieval−no_node |
|---|---:|---:|---:|---:|
| kanana_nano_int4      | 6.2% | **20.3%** | 20.3% | +14.1 pp |
| kanana_8b             | 1.6% |    18.8%  |  7.8% | +17.2 pp |
| kanana_8b_int4        | 0.0% |    18.8%  | 10.9% | +18.8 pp |
| open_ko_8b            | 6.2% |    17.2%  | 12.5% | +11.0 pp |
| bllossom_8b           | 3.1% |    15.6%  |  9.4% | +12.5 pp |
| hclova_seed_15b_int4  | 0.0% |    15.6%  | 14.1% | +15.6 pp |
| qwen25_3b             | 3.1% |    12.5%  | 10.9% |  +9.4 pp |
| qwen25_3b_int4        | 1.6% |    12.5%  | 10.9% | +10.9 pp |
| hclova_seed_15b       | 3.1% |    10.9%  |  7.8% |  +7.8 pp |
| kanana_nano           | 4.7% |    10.9%  | 15.6% |  +6.2 pp |
| qwen25_7b             | 4.7% |    10.9%  |  6.2% |  +6.2 pp |

The retrieval lift is universal across the lineup. The mean
`retrieval`−`no_node` delta is +11.8 pp and the minimum is +6.2 pp
(`kanana_nano` and `qwen25_7b`); no configuration regresses with
retrieval relative to the memoryless baseline. The Korean-native
families (Kanana 1.5 and HyperCLOVA-X SEED) and the Korean-tuned
Llama 3 derivatives populate the top half of the table; the
multilingual Qwen 2.5 baselines remain mid-pack and never lead.
The strongest single configuration is `kanana_nano_int4` at
20.3 % retrieval, followed by `kanana_8b` and `kanana_8b_int4`
at 18.8 %. The on-device 2.1 B target named in Part 1 of the
proposal — `kanana_nano` — sits at 10.9 % retrieval (4.7 % no_node,
15.6 % oracle), preserving the v0.1 headline trajectory of
4.7 % → 10.9 % → 15.6 % as a within-table reference point.

A second pattern in Table 1 is that retrieval and oracle no
longer move in lockstep across the lineup. Five of eleven
configurations record `oracle` strictly below `retrieval`,
including the strongest large model `kanana_8b` (7.8 % oracle vs
18.8 % retrieval). The regression is concentrated in the 7–8 B
SUTs and is absent in the smallest (`kanana_nano`,
`hclova_seed_15b`); we read this as evidence that long oracle
prompts (up to k × 3 = 15 evidence documents) push larger SUTs
into a long-context summarization mode that recapitulates the
evidence rather than answering the question. We previously
discussed this effect, on Qwen 2.5 3B alone, in §4.2 of v0.1; the
v0.2 expansion shows it generalizes well beyond Qwen and is
worth treating as a first-class artifact rather than a model-
specific anomaly. A natural mitigation is an "answer-first"
prompt guardrail; we leave it to v0.3 because its evaluation
requires re-running the oracle column rather than re-aggregating
existing data.

### 3.2 Quantization axis (Table 2; Figure 2c)

Four SUT families admit a 4-bit-NF4 quantization variant under
the 30 GB memory budget; Table 2 contrasts their fp16 and int4
pass rates and latencies on the `ko_native` retrieval cell, and
Figure 2(c) renders the corresponding paired item-level
analysis. The aggregate reading from the marginal table is that
the two smallest SUTs gain materially under int4 (Kanana 2.1B
+9.4 pp, HyperCLOVA-X SEED 1.5B +4.7 pp) while the two larger
SUTs are unchanged. We argue below that this aggregate reading
overstates the int4 effect.

**Table 2.** *int4 vs fp16 on `ko_native` retrieval (advisory).
Δpass = int4 − fp16 in percentage points; latencies are mean
seconds per generation across n = 64.*

| SUT family | fp16 pass | int4 pass | Δpass | fp16 latency | int4 latency |
|---|---:|---:|---:|---:|---:|
| kanana_nano (2.1 B)        | 10.9 % | 20.3 % | **+9.4 pp** | 2.54 s | 5.22 s |
| hclova_seed_15b (1.5 B)    | 10.9 % | 15.6 % | **+4.7 pp** | 1.83 s | 3.45 s |
| kanana_8b (8.0 B)          | 18.8 % | 18.8 % |   0.0 pp    | 3.73 s | 3.60 s |
| qwen25_3b (3.0 B)          | 12.5 % | 12.5 % |   0.0 pp    | 2.66 s | 4.81 s |

A paired item-level analysis modulates the marginal +9.4 pp and
+4.7 pp gains substantially. For each of the four SUT families,
we count the items where fp16 and int4 disagree — items where
exactly one of the two quantizations passed contains-token — and
apply a McNemar two-sided z test. The discordance counts and
z-scores reported in Figure 2(c) are: Kanana 2.1B int4 8 / fp16 2
(z = 1.90, p ≈ 0.06), HyperCLOVA-X SEED 1.5B int4 5 / fp16 2
(z = 1.13), Kanana 8B int4 1 / fp16 1 (z = 0), Qwen 2.5 3B int4 3
/ fp16 3 (z = 0). Only Kanana 2.1B reaches marginal significance,
and even that result rests on a 6-item swing within a 64-item
denominator with base pass rate near 15 %, where the standard
error of a single configuration's pass rate is approximately
±5 pp. The +4.7 pp HyperCLOVA-X SEED gain is not significant.

A second confound is decoding behavior. Mean response length on
`ko_native` retrieval is 261 characters for `hclova_seed_15b`
fp16 and 401 characters for `hclova_seed_15b_int4`; the int4
variant produces responses ≈ 1.5× longer on average, while
Kanana 8B and Qwen 3B produce responses of statistically
indistinguishable length under the two quantizations (243 vs
226 chars and 244 vs 239 chars respectively). Because contains-
token is a substring metric, longer responses are mechanically
more likely to contain the gold tokens by coincidence; the
HyperCLOVA-X SEED int4 gain is therefore at least partially a
length artifact rather than evidence of better question
answering. We do not have an analogous length explanation for
the Kanana 2.1B gain (mean lengths 316 vs 322 chars), where the
discordance is genuinely paired but borderline-significant.

The latency story is uncomplicated: int4 is *not* faster than
fp16 on this hardware. Three of four SUT families show int4
latency 1.3–2.0× longer than fp16, and only `kanana_8b` is
neutral (3.60 s vs 3.73 s). The slowdown is consistent with the
known dequantization overhead of bitsandbytes-NF4 on short
generations — the per-token saving from the smaller weight
matrix is offset by the per-generation fixed cost of unpacking
4-bit integers into half-precision compute — and is in line with
the published benchmarks for this quantizer. For decision-moment
assistants where the relevant deployment budget is GPU memory
rather than wall-clock latency (an L40S can otherwise hold a
fp16 8 B SUT and the retriever simultaneously, so the memory
saving matters most below the L40S class), int4 remains an
attractive option for the small Korean-native SUTs and a neutral
option for the larger ones. We do *not* read the small-model
contains-pass gains as evidence of better question answering
under int4; they are best read as borderline-significant decoding
artifacts in a 64-item benchmark.

### 3.3 Cross-track language axis (Table 4)

Table 4 reports advisory retrieval pass rate across the three
heard-bench tracks for all eleven SUTs, generalizing the v0.1
single-SUT language-axis report. The denominators are the
per-track non-REFL counts established in §2.2: 64 on `ko_native`,
100 on `en_subset` and `ko_translated`.

**Table 4.** *Cross-track advisory retrieval pass rate, contains-
token metric. Sorted by `ko_native`.*

| SUT | en_subset | ko_translated | ko_native |
|---|---:|---:|---:|
| kanana_nano_int4      | 0.0 % | 6.0 % | **20.3 %** |
| kanana_8b             | 2.0 % | 5.0 % |   18.8 %   |
| kanana_8b_int4        | 0.0 % | 2.0 % |   18.8 %   |
| open_ko_8b            | 2.0 % | 2.0 % |   17.2 %   |
| bllossom_8b           | 2.0 % | 4.0 % |   15.6 %   |
| hclova_seed_15b_int4  | 0.0 % | 2.0 % |   15.6 %   |
| qwen25_3b             | 0.0 % | 2.0 % |   12.5 %   |
| qwen25_3b_int4        | 0.0 % | 1.0 % |   12.5 %   |
| hclova_seed_15b       | 0.0 % | 1.0 % |   10.9 %   |
| kanana_nano           | 0.0 % | 5.0 % |   10.9 %   |
| qwen25_7b             | 1.0 % | 2.0 % |   10.9 %   |

The v0.1 language-axis ordering en_subset ≤ ko_translated <
ko_native is preserved by ten of eleven configurations
(`open_ko_8b` ties en_subset and ko_translated at 2.0 %). Mean
pass rates over the lineup are 0.6 % on en_subset, 2.9 % on
ko_translated, and 14.8 % on ko_native — a 25× gap between the
two endpoints. The multilingual Qwen 2.5 baselines do not close
the en_subset gap despite English being their native training
distribution, which we attribute to the LongMemEval haystack
length (sessions span hundreds of utterances per item) and the
absence of session-level retrieval in our pipeline; this is the
v0.1 limitation we discuss in §4.4 and is unchanged in v0.2.
A second observation is that ko_translated sits closer to
en_subset than to ko_native across the board even though the
question and gold answer are in Korean — confirming the
ADR-0003 hypothesis that haystack language dominates for this
kind of task and motivating a future `ko_translated_full` track
that translates the haystack as well as the question.

### 3.4 Reflective policy: contains-token neutrality and pairwise dominance (Tables 3, 5, 6; Figure 2b)

We measure the reflective policy along two axes: contains-token
accuracy (does the reflective response still surface the gold
substring?) and the pairwise reflective-quality judge described
in §2.7 (does it sound more like a mirror?). The contains-token
view (Table 3) shows the policy is approximately accuracy-
neutral; the pairwise view (Tables 5 and 6, Figure 2b) shows it
dominates on the qualitative rubrics that operationalize the
"mirror, do not advise" thesis.

**Table 3.** *Reflective − Advisory contains-pass on `ko_native`
retrieval, percentage points.*

| SUT | advisory | reflective | Δ (refl − adv) |
|---|---:|---:|---:|
| bllossom_8b           | 15.6 % | **28.1 %** | **+12.5 pp** |
| kanana_nano           | 10.9 % | **20.3 %** | **+9.4 pp**  |
| qwen25_7b             | 10.9 % | 14.1 %     |  +3.1 pp     |
| kanana_8b             | 18.8 % | 21.9 %     |  +3.1 pp     |
| kanana_8b_int4        | 18.8 % | 18.8 %     |   0.0 pp     |
| qwen25_3b_int4        | 12.5 % | 10.9 %     |  −1.6 pp     |
| hclova_seed_15b       | 10.9 % |  9.4 %     |  −1.6 pp     |
| hclova_seed_15b_int4  | 15.6 % | 12.5 %     |  −3.1 pp     |
| qwen25_3b             | 12.5 % |  9.4 %     |  −3.1 pp     |
| kanana_nano_int4      | 20.3 % | 17.2 %     |  −3.1 pp     |
| open_ko_8b            | 17.2 % | 10.9 %     |  −6.2 pp     |

Across eleven SUTs the mean contains-pass shift from advisory to
reflective is +0.9 pp with median 0.0 pp, four positive deltas,
two zero, and five negative; the spread (−6.2 to +12.5 pp) is
within the noise band one would expect from a 64-item
denominator. We read the policy as accuracy-neutral on the
contains-token metric — the hard constraints (cite a memory
verbatim, no imperatives, end with an open question) do not
prevent the SUT from surfacing the gold substring, they merely
route it through a different stylistic shell.

The pairwise judge measures the value of that shell directly.
Each (item, SUT, condition) triple where both advisory and
reflective responses exist is scored under two judges (`gpt-4o-
mini`, `claude-haiku-4.5`) with A/B-swap symmetrization on each
of four rubrics (specificity, non-directive, emotional
attunement, open question). The v0.2 sweep yields 132 valid
triples (eleven SUTs × six REFL items × two retrieval-augmented
conditions; reflective × `no_node` is excluded by the runner
because the two policies degenerate without retrieved context),
and 132 × 2 × 2 × 4 = 2,112 verdicts in total. Table 5 sums
verdicts across all SUTs and conditions per rubric; Table 6 sums
across all conditions and rubrics per SUT (n = 192 decisions per
SUT).

**Table 5.** *Pairwise judge totals per rubric, summed across all
eleven SUTs and both retrieval-augmented conditions. Each rubric
column sums to 528 = 11 SUTs × 2 conditions × 6 REFL items × 2
judges × 2 A/B swaps.*

| Rubric | Advisory wins | Reflective wins | Tie | Reflective share |
|---|---:|---:|---:|---:|
| specificity            | 158 | 348 | 22 | 65.9 % |
| non_directive          | 178 | 350 |  0 | 66.3 % |
| emotional_attunement   |  32 | **465** | 31 | **88.1 %** |
| open_question          |  33 | **466** | 29 | **88.3 %** |

**Table 6.** *Reflective share per SUT, summed across both
conditions and all four rubrics (n = 192 decisions per SUT).
Sorted by reflective share.*

| SUT | Advisory wins | Reflective wins | Tie | Reflective share |
|---|---:|---:|---:|---:|
| kanana_8b             | 22 | **168** |  2 | **87.5 %** |
| kanana_8b_int4        | 26 | 165     |  1 | 85.9 %     |
| hclova_seed_15b       | 28 | 160     |  4 | 83.3 %     |
| qwen25_7b             | 28 | 158     |  6 | 82.3 %     |
| kanana_nano           | 35 | 149     |  8 | 77.6 %     |
| kanana_nano_int4      | 38 | 148     |  6 | 77.1 %     |
| hclova_seed_15b_int4  | 35 | 148     |  9 | 77.1 %     |
| qwen25_3b             | 37 | 147     |  8 | 76.6 %     |
| qwen25_3b_int4        | 40 | 140     | 12 | 72.9 %     |
| bllossom_8b           | 49 | 134     |  9 | 69.8 %     |
| open_ko_8b            | 63 | 112     | 17 | 58.3 %     |

The reflective policy wins every rubric by a clear margin; the
two rubrics that operationalize the "mirror, do not advise"
thesis — emotional attunement and open-question framing — land at
88.1 % and 88.3 % reflective share respectively, recovering the
v0.1 numbers (82/96 ≈ 85 % and 88/96 ≈ 92 %) under a 5.5×-larger
sample. Specificity and non-directive remain tighter at 66 %; the
advisory policy occasionally wins on specificity by virtue of
being less constrained in surface form, and on non-directive by
accident when the advisory generation phrases a recommendation as
a question.

The per-SUT view is more discriminating than the per-rubric
average. Reflective share spans 58.3 % (`open_ko_8b`) to 87.5 %
(`kanana_8b`); every SUT has the reflective policy ahead, but the
margin varies considerably. The two strongest SUTs on this
metric are the Kanana 8B family (fp16 87.5 %, int4 85.9 %), and
the two weakest are the Korean-tuned Llama 3 derivatives
(`bllossom_8b` 69.8 %, `open_ko_8b` 58.3 %). We attribute the
Llama-3-derivative weakness to a stronger advisory baseline
rather than weaker reflective output: `bllossom_8b` and
`open_ko_8b` accumulate the highest advisory-win counts in the
lineup (49 and 63 respectively, against ≤ 40 elsewhere),
indicating these models follow advisory imperative templates
more willingly than the Kanana / Qwen / HyperCLOVA-X families.
The reflective policy still wins, but it is competing against a
more imperatively-styled advisory response and the margin
narrows.

### 3.5 Latency and SUT recommendation

Table 7 summarizes mean per-generation latency on `ko_native`
across all eleven SUTs and three conditions, on a single L40S 48
GB. The table is the v0.2 analogue of the v0.1 Pareto plot, but
without the figure: with eleven SUTs in two parameter decades, a
scatter plot is no longer a clean visual.

**Table 7.** *Mean wall-clock latency per response (s) on
`ko_native`, advisory policy.*

| SUT | no_node | retrieval | oracle |
|---|---:|---:|---:|
| hclova_seed_15b      | 1.78 | 1.83 | 1.78 |
| kanana_nano          | 2.25 | 2.54 | 2.40 |
| qwen25_3b            | 1.90 | 2.66 | 2.16 |
| qwen25_7b            | 2.74 | 3.15 | 2.58 |
| kanana_8b            | 3.25 | 3.73 | 3.27 |
| kanana_8b_int4       | 3.17 | 3.60 | 3.29 |
| hclova_seed_15b_int4 | 3.37 | 3.45 | 3.37 |
| open_ko_8b           | 3.41 | 3.87 | 3.63 |
| qwen25_3b_int4       | 4.07 | 4.81 | 4.01 |
| bllossom_8b          | 4.90 | 4.94 | 4.91 |
| kanana_nano_int4     | 4.83 | 5.22 | 5.05 |

Latencies range from 1.78 s (`hclova_seed_15b` no_node) to 5.22 s
(`kanana_nano_int4` retrieval). The lineup splits roughly into a
sub-3 s tier (the Korean-native fp16 SUTs at 1.5–8 B) and a 3–5 s
tier (most int4 variants and the Llama 3 derivatives). All SUTs
remain within the rough on-device latency budget of 5–7 s for a
decision-moment assistant operating at human conversational
pacing.

Combining the four orthogonal v0.2 metrics — retrieval pass rate
(§3.1), reflective share (§3.4), latency (this section), and
quantization sensitivity (§3.2) — singles out `kanana_8b` as the
v0.2 recommended SUT for the *Heard* product. It places second
on retrieval pass rate (18.8 %, behind only `kanana_nano_int4`'s
likely-noise 20.3 %), first on reflective share (87.5 %), in the
mid-latency tier (3.73 s retrieval), and is one of the two SUT
families where int4 quantization is exactly latency-neutral, so
the int4 variant offers a clean memory trade for deployment on
hardware below the 48 GB L40S class. The on-device 2.1 B target
named in Part 1 — `kanana_nano` — remains the right choice for
the lightest deployment envelope (1.83 s retrieval, 77.6 %
reflective share), but at the 8 B class the in-family scale-up
preserves all four properties.

## 4 Discussion

### 4.1 Absolute pass rates are low by construction

Adversarial filtering retained only items where three different
no-memory baselines failed across three trials. The surviving
items are, by construction, hard. Kanana's 10.9% retrieval pass
rate on this subset is meaningful evidence of memory contribution;
the same SUT on unfiltered questions would look much better on
paper and teach less. The headline interpretation of §3.1 is the
relative lift (×3.3 end-to-end), not the absolute rate.

### 4.2 The retrieval > oracle regression

Five of eleven v0.2 configurations record `oracle` strictly below
`retrieval` on `ko_native` (Table 1), with the regression
concentrated in the 7–8 B parameter tier (`kanana_8b`,
`kanana_8b_int4`, `qwen25_7b`, `bllossom_8b`) and absent in the
two smallest SUTs (`kanana_nano`, `hclova_seed_15b`). We read
this as a long-context summarization bias: a 15-document oracle
prompt approximately triples the prompt length of a 5-neighbor
retrieval prompt, and the larger SUTs respond by recapitulating
the evidence sessions rather than answering the today-question.
A concrete fix is an "answer-first" guardrail in the system
prompt — *"Answer the question. Do not re-narrate the past."* —
which we expect to close most of the retrieval-vs-oracle gap on
the affected SUTs. Validating that prediction is a v0.3 task
because it requires re-running the oracle column rather than
re-aggregating existing data; it reads as a prompt-design
artifact rather than a capability ceiling.

### 4.3 What dense retrieval does not measure

The v0.1 5-ability benchmark measures substring recall, namely
whether the SUT reproduces the correct string given retrieved
evidence. Dense embedding retrieval handles this well: when the
evidence session is in the top-5, substring recall usually
succeeds.

Three axes fall outside what this benchmark scores cleanly, each
connecting to a Part 1 NODE claim. First, absence queries of the
form "the user has never mentioned a vegan menu" resist dense
retrieval because cosine similarity always returns some neighbor,
encouraging hallucinated confirmation rather than refusal. Second, *entity-aggregation* queries such as "every
memory about Grandma Park" require exhaustive graph traversal for
completeness, whereas cosine top-k is both approximate and bounded
in coverage. Third, *temporal-latest* queries such as "the current
permanent-wave product" need exact timestamp filtering, whereas
semantic similarity conflates old and new mentions. A NODE-native
benchmark of 30 to 50 items targeting these three axes is the
natural v2 extension.

### 4.4 Track B scope limitation

ko_translated keeps the haystack in English (ADR 0003). The
"language only" axis is thus partially confounded, because
Kanana's ko_translated number reflects both translation and
haystack-language effects. Re-running with Korean haystacks is a
tractable extension; the cost analysis in ADR 0003 identifies
chunked session translation as the right path.

### 4.5 Validation shortfall

The final ko_native set is 70 items, below the planned 100.
Gate 4 (question clarity ≥ 3/5) was the dominant rejection cause:
LLM raters are systematically conservative on isolated questions
shown without persona context. Giving the rater that context in a
future run is a one-line prompt change likely to lift acceptance
closer to what a human reviewer would accept.

### 4.6 What the int4 axis does and does not show

The marginal +9.4 pp and +4.7 pp gains on `kanana_nano_int4` and
`hclova_seed_15b_int4` (Table 2) are easy to misread as evidence
that 4-bit-NF4 quantization *helps* contains-token accuracy on
small Korean-native SUTs. The paired item-level analysis in
§3.2 and Figure 2(c) does not support that reading. Of the four
fp16/int4 SUT pairs, three have McNemar z-scores at or below
1.13 (i.e., not significant at any conventional level), and the
fourth (Kanana 2.1B) reaches z = 1.90, which is borderline
significant in a 64-item denominator at base rate ≈ 15 %. The
HyperCLOVA-X SEED int4 gain is moreover confounded by a
decoding-behavior shift: mean response length on `ko_native`
retrieval grows from 261 characters (fp16) to 401 characters
(int4) for that SUT alone, while the other three pairs produce
responses of indistinguishable length under the two
quantizations. Because contains-token is a substring metric,
longer responses are mechanically more likely to contain the
gold tokens by coincidence; we therefore treat the
HyperCLOVA-X SEED int4 advantage as a length artifact rather than
a quality gain.

We retain the int4 axis in the sweep because the latency and
memory analyses still favor it for two specific deployment
profiles. Three of four SUT families show int4 *latency* 1.3–
2.0× longer than fp16 on the L40S (§3.2), so int4 is not a
throughput optimization on this hardware; the relevant deployment
profiles are memory-bound, where int4 quarters the weight memory
of an 8 B SUT and lets it co-reside with the retriever on devices
below the 48 GB L40S class, and `kanana_8b` specifically, where
the int4 variant is exactly latency-neutral and incurs no
contains-pass cost. We do not recommend int4 as a quality
improvement, only as a memory option with a known small accuracy
risk.

### 4.7 What the v0.2 expansion leaves unmeasured

The v0.2 sweep expands the SUT axis from two to eleven
configurations and adds a quantization axis, but does not
expand the dataset axis. The three v0.1 dataset limitations
articulated in §4.3 (NODE-native abilities not measured by
substring recall), §4.4 (English haystack on `ko_translated`),
and §4.5 (70 items rather than the planned 100) are unchanged.

Three SUT-axis items also remained out of reach. The two
Korean-tuned 10 B+ models we wanted to include — EEVE-Korean
10.8B and SOLAR 10.7B — were dropped from the sweep at runtime
when their fp16 throughput proved too low to finish within the
wall-clock budget (≈ 47 and ≈ 200 generations per hour
respectively under the reflective policy); they remain v0.3
candidates once an int4 path with adequate generation speed is
identified. EXAONE-3.5 2.4B remained excluded by a `transformers`
import incompatibility (`RopeParameters`) that the v0.2
environment did not resolve. And a BM25 baseline that would
isolate the contribution of dense retrieval against a sparse
counterpart is not yet in the lineup.

## 5 Conclusion

Heard contributes, to our knowledge, the first Korean
long-term memory benchmark targeting solo-business monologue,
built because no existing benchmark meets the three criteria of
Korean / monologue / solo-business. On this benchmark, the v0.2
sweep evaluates eleven SUT configurations across four model
families and a 4-bit-NF4 quantization axis under both advisory
and reflective response policies, generating 14,850 SUT
generations and 2,112 pairwise judge verdicts.

Three results are robust across the lineup. First, dense
retrieval lifts contains-pass on `ko_native` for every
configuration (mean +11.8 pp, minimum +6.2 pp, no negative
deltas), with the on-device 2.1 B target named in Part 1
(`kanana_nano`) following the canonical 4.7 % → 10.9 % → 15.6 %
trajectory. Second, the reflective policy dominates the
advisory baseline on every rubric across 528 pairwise judge
decisions per rubric, with the two "mirror, do not advise"
rubrics — emotional attunement and open-question framing —
landing at 88.1 % and 88.3 % reflective share. Third, the
language-axis ordering en_subset ≤ ko_translated < ko_native is
preserved by ten of eleven SUTs, with multilingual baselines
unable to close the gap on Korean tracks. Combining the four
v0.2 metrics (retrieval pass, reflective share, latency, int4
sensitivity) we recommend `kanana_8b` as the default *Heard*
deployment SUT, with its int4 variant as a memory-bound
alternative.

Two results require careful framing. The marginal int4
contains-pass gains on the two smallest SUTs (Kanana 2.1B
+9.4 pp, HyperCLOVA-X SEED 1.5B +4.7 pp) reduce to one
borderline-significant pair (z = 1.90) and one confounded by a
1.5× response-length increase under paired item-level analysis;
we explicitly do not read these as a quality improvement from
quantization (§4.6). And five of eleven SUTs show retrieval >
oracle regression on `ko_native`, concentrated in the 7–8 B
parameter tier, which we attribute to long-context
summarization bias and propose to fix with an "answer-first"
guardrail in v0.3 (§4.2).

Several extensions follow from the limitations identified above,
ordered by rough value-to-effort priority.

1. A NODE-native capability benchmark of 30 to 50 items
   targeting absence-strict, entity-aggregation, and temporal-
   latest queries, the three axes §4.3 argues dense retrieval
   cannot reach.
2. Full-haystack Korean translation of `ko_translated` via
   chunked session translation (ADR 0003), closing the confound
   in §4.4.
3. An "answer-first" prompt guardrail to test the §4.2
   hypothesis that the retrieval > oracle regression on five of
   eleven SUTs is a long-context summarization artifact.
4. SUT expansion to EXAONE-3.5 (pending a `transformers` import
   fix) and to EEVE-Korean 10.8B / SOLAR 10.7B (pending an int4
   path with adequate generation speed).
5. A BM25 baseline to isolate the dense-retrieval contribution
   against a sparse counterpart.

## 6 Artifacts and References

### 6.1 Artifacts

The author website, code repository, and dataset repository links
appear in the header of this report. The remaining artifacts used
for the results in §3 are located as follows.

| | Location |
|---|---|
| v0.2 expanded sweep (merged, 11 SUTs) | `experiments/20260426_1242_v0.2_sweep_merged/` |
| v0.2 sweep results (raw) | `experiments/20260426_1242_v0.2_sweep_merged/results.jsonl` |
| v0.2 aggregate metrics | `experiments/20260426_1242_v0.2_sweep_merged/metrics.csv`, `metrics.json` |
| v0.2 judge verdicts | `experiments/20260426_1242_v0.2_sweep_merged/judge_verdicts.jsonl` |
| v0.2 judge aggregate | `experiments/20260426_1242_v0.2_sweep_merged/judge_aggregate.json` |
| v0.1 reference sweep (2 SUTs, archive) | `experiments/20260423_1610_day3_sweep/` |
| Per-call API log | `experiments/_api_log/api_calls.jsonl` |
| ADRs | `docs/decisions/0001–0004*.md` |

### 6.2 References

- Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., & Yu, D.
  (2025). LongMemEval: Benchmarking chat assistants on long-term
  interactive memory. *ICLR 2025*.
- Du, Y., et al. (2024). PerLTQA: A personal long-term memory
  dataset for memory classification, retrieval, and synthesis in
  question answering. *SIGHAN-10*.
- Maharana, A., et al. (2024). Evaluating very long-term
  conversational memory of LLM agents. *ACL 2024*.
- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench
  and chatbot arena. *NeurIPS 2023*.
- Pennebaker, J. W. (2011). *The Secret Life of Pronouns: What
  Our Words Say About Us*. Bloomsbury Press.
- Kakao Kanana team. (2025). Kanana 1.5 technical report.
- Alibaba Qwen team. (2024). Qwen2.5 technical report.
- Wang, L., et al. (2024). Multilingual E5 text embeddings: A
  technical report. arXiv:2402.05672.
- Lewis, P., et al. (2020). Retrieval-augmented generation for
  knowledge-intensive NLP tasks. *NeurIPS 2020*.

## 7 Acknowledgments and AI Use

Claude Code was used as a coding assistant for implementing the
pipeline and scripts in this repository. Figure 1
(`fig_overview.jpg`) was drawn in Paperbanana. All design
decisions, including research framing, dataset methodology,
hypothesis formulation, result interpretation, and writing
direction, were performed by the author.

## Appendix A — System prompts used during dataset construction

Every dataset-building stage that calls an LLM does so under a
short, fixed system prompt. The prompts are reproduced verbatim
below for transparency; they live in the repository under
`src/datagen/` and `src/eval/`. Persona cards and per-call user
messages are built around these system prompts at run time.

### A.1 Utterance generator (`src/datagen/utterance_gen.py`)

Used by `scripts/01_generate_data.py` to expand each (persona,
day) block into 10–15 Korean nightly-self-talk utterances.
Generators rotate across Haiku 4.5 / GPT-4o-mini / Gemini-2.5-flash.

```text
당신은 한국 1인 자영업자의 **혼잣말 생성기**입니다.
페르소나 카드와 오늘의 이벤트를 보고, 그 사람이 실제로 내뱉을
법한 짧은 혼잣말을 만드세요.
규칙:
 (1) 각 발화는 1–3문장, 구어체, 자연스러운 줄임말 허용.
 (2) 감정·판단·의문이 섞여야 함. 뉴스 요약조 금지.
 (3) 같은 날의 발화들은 주제가 이어지되 시점(아침·영업 중·
     마감 후)이 달라야 함.
 (4) 이벤트의 topic_key 를 1개 이상 intended_topics 에 남겨
     주세요 — 나중에 retrieval gold label 로 씁니다.
 (5) 응답은 반드시 JSON 객체 {"utterances": [...]} 형식으로만.
 (6) 각 utterance 객체의 필드: time (HH:MM, 24h), text,
     intended_categories (customer/stock/pricing/mood/decision
     중 해당되는 것), intended_topics (topic_key 리스트),
     references_historical_event (기억하는 과거 사건 힌트,
     없으면 빈 리스트).
```

### A.2 Scenario generator (`src/datagen/scenario_gen.py`)

Used by `scripts/03_generate_scenarios.py` to compose 150 Track-C
candidates from sampled evidence utterances. The prompt forbids
fact invention and forces the gold answer to be derivable from
the evidence already in the corpus.

```text
당신은 한국어 long-term memory benchmark 의 **시나리오 생성기**
입니다. 주어진 페르소나의 과거 발화(evidence)를 토대로, 오늘
시점의 질문 하나를 만듭니다.
사실을 **새로 지어내지 마세요** — 질문과 정답은 evidence 에서
도출되어야 합니다.

출력은 반드시 JSON 객체:
{
  "question_text": "오늘 페르소나가 스스로에게 던지는 질문",
  "question_timestamp_day": 55~60 사이의 정수,
  "question_timestamp_hhmm": "HH:MM",
  "gold_answer_text": "정답 문장 (간결)",
  "gold_contains_tokens": ["정답에 반드시 포함되어야 할
                           짧은 substring (1~3개)"],
  "gold_excludes_tokens": ["hallucination 탐지용 (선택)"],
  "evidence_summary": "왜 이 evidence 가 정답의 근거인지 1문장",
  "reasoning": "생성 근거 (짧게)"
}
```

### A.3 Korean translator (`src/datagen/translate.py`)

Used by `scripts/05_translate_track_b.py` for the Track B (KO
question + gold answer) translation. ADR 0003 records why the
haystack stays in English.

```text
You are a professional English→Korean translator. Translate the
user's question and the gold answer into natural Korean. Preserve
numbers, proper nouns, and relative time expressions faithfully.
Use casual Korean (해요체).
Inside JSON strings, do NOT insert literal double-quote characters
— use Korean 「」 or single ' ' quotes if you need to quote
something.
Output ONLY a JSON object:
  {"question_ko": "...", "gold_answer_ko": "..."}
```

### A.4 Validation gate 1 — evidence-answer consistency
(`src/datagen/validation.py`)

Asks a third-party LLM whether the gold answer is derivable from
the candidate's evidence alone. Demoted to advisory by ADR 0004,
but the verdict is still recorded per item.

```text
You are a strict grader. Given evidence utterances (past monologues
of a Korean shop owner) and a question, decide whether the gold
answer can be reasonably derived from the evidence alone.
Reply ONLY with a JSON object:
  {"consistent": true|false, "reason": "1 sentence"}.
```

### A.5 Validation gate 4 — question clarity
(`src/datagen/validation.py`)

Rates an isolated Korean question on a 1–5 Likert scale. Threshold
≥ 3 / 5 retained 70 of 122 items (§2.2).

```text
You are a strict question-clarity grader. Rate how clear and
unambiguous a Korean monologue question is on a 1–5 Likert scale
(5 = perfectly clear, 1 = unanswerable).
Reply ONLY with JSON:
  {"clarity": <1-5 integer>, "reason": "1 sentence"}.
```

### A.6 Pairwise reflective-quality judge (`src/eval/judge.py`)

Used by `scripts/08_run_judge.py` to score advisory-vs-reflective
response pairs across four rubrics. The hard constraint is the
ternary {"A","B","tie"} output; the post-hoc swap-and-invert step
in Algorithm 3 mitigates position bias.

```text
You are a strict grader of Korean reflective-dialogue quality.

You compare two candidate responses (A and B) to the same user's
utterance (오늘의 고민). Rate which one is better on each of four
rubrics, on a ternary scale: "A" / "B" / "tie".

Rubrics:
  specificity         Does it cite a specific past event with detail?
  non_directive       Does it AVOID imperatives / prescriptive advice?
  emotional_attunement  Does it acknowledge the speaker's feeling?
  open_question       Does it end with an open (not yes/no) question?

Output ONLY JSON:
  {"specificity":"A","non_directive":"tie", ...}
```

