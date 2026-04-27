#!/usr/bin/env python3
"""Upload heard-bench dataset to HuggingFace Hub.

Stages files under /tmp/heard_bench_stage/ mirroring the HF repo layout
(DATASET §8.1), writes a dataset card, and uploads with huggingface_hub.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import REPO_ROOT
from src.utils.logging import get_logger

log = get_logger("hf_upload")


DATASET_CARD_TEMPLATE = """\
---
license: cc-by-4.0
language:
  - ko
  - en
pretty_name: heard-bench
size_categories:
  - n<1K
task_categories:
  - question-answering
tags:
  - long-term-memory
  - korean
  - monologue
  - on-device
  - reflective
configs:
  - config_name: en_subset
    data_files:
      - split: test
        path: data/en_subset/test.jsonl
  - config_name: ko_translated
    data_files:
      - split: test
        path: data/ko_translated/test.jsonl
  - config_name: ko_native
    data_files:
      - split: test
        path: data/ko_native/test.jsonl
---

# heard-bench

A Korean long-term-memory benchmark for the under-studied domain of
**solo-business nightly monologue**, companion to the *Heard* system
(MIC → NODE → MIRROR). v0.1 ships 270 items across three tracks and
six memory abilities.

## Tracks

| track | items | language | haystack | source |
|---|---:|---|---|---|
| `en_subset`     | 100 | English | English | LongMemEval_S stratified sample (20 per ability) |
| `ko_translated` | 100 | Korean question & answer, English haystack | English | LongMemEval_S translated to Korean via `google/gemini-2.5-flash` |
| `ko_native`     |  70 | Korean | Korean | Generated from a 2,046-utterance synthetic corpus of 3 Korean solo-business personas, adversarially filtered, 4-gate validated, and author-reviewed |

## Abilities

Following LongMemEval's taxonomy (+ one addition):

- **IE** — information extraction
- **MR** — multi-session reasoning
- **KU** — knowledge update
- **TR** — temporal reasoning
- **ABS** — abstention (answer should be "I don't know")
- **REFL** — reflective quality (ours — evaluated with pairwise LLM-as-judge)

ko_native ability counts: **IE 19 / TR 15 / KU 13 / ABS 10 / MR 7 / REFL 6**

## Personas (ko_native only)

- `yejin_florist` — 38yo florist, Mapo. 27 items.
- `minseok_cafe` — 42yo roaster-café owner, Seongsu. 21 items.
- `sunhee_hair` — 45yo hair salon owner, Hongje. 22 items.

Persona cards are in `personas/<id>.yaml`, covering regulars,
stock/services, stressors, and historical events (days-ago-indexed).

## Item schema

```json
{
  "item_id": "ko_native_042",
  "track": "ko_native",
  "persona_id": "yejin_florist",
  "ability": "IE",
  "question": {"text": "…", "timestamp": "…"},
  "gold_answer": {
    "text": "…",
    "contains_tokens": ["…"],
    "excludes_tokens": ["…"],
    "evidence_utt_ids": ["…"],        // ko_native
    "evidence_session_ids": ["…"]     // en_subset / ko_translated
  },
  "reflective_rubric": { "criteria": ["specificity", "non_directive",
                                       "emotional_attunement", "open_question"] },
  "metadata": { "…": "…" }
}
```

## Baseline results (v0.2 expanded sweep)

Eleven SUT configurations across four model families and a 4-bit-NF4
quantization axis, evaluated on `ko_native` with an embedding
retriever (`intfloat/multilingual-e5-small`, top-5). Numbers below
are advisory pass rate (contains-token, n = 64) at each memory
condition; the v0.1 Kanana-1.5-2.1B headline (`kanana_nano` row) is
preserved verbatim and shown for continuity.

| SUT | params | quant | no_node | retrieval | oracle |
|---|---:|---|---:|---:|---:|
| kanana_nano          | 2.1 B | fp16 | 4.7 % | 10.9 % | 15.6 % |
| kanana_nano_int4     | 2.1 B | int4 | 6.2 % | **20.3 %** | 20.3 % |
| kanana_8b            | 8.0 B | fp16 | 1.6 % | 18.8 % |  7.8 % |
| kanana_8b_int4       | 8.0 B | int4 | 0.0 % | 18.8 % | 10.9 % |
| hclova_seed_15b      | 1.5 B | fp16 | 3.1 % | 10.9 % |  7.8 % |
| hclova_seed_15b_int4 | 1.5 B | int4 | 0.0 % | 15.6 % | 14.1 % |
| qwen25_3b            | 3.0 B | fp16 | 3.1 % | 12.5 % | 10.9 % |
| qwen25_3b_int4       | 3.0 B | int4 | 1.6 % | 12.5 % | 10.9 % |
| qwen25_7b            | 7.6 B | fp16 | 4.7 % | 10.9 % |  6.2 % |
| bllossom_8b          | 8.0 B | fp16 | 3.1 % | 15.6 % |  9.4 % |
| open_ko_8b           | 8.0 B | fp16 | 6.2 % | 17.2 % | 12.5 % |

The dense-retrieval lift from §3.5 of the technical report
generalizes across the lineup: every configuration records a
positive `retrieval`−`no_node` delta (mean +11.8 pp). The
reflective response policy beats advisory on every rubric across
528 pairwise judge decisions per rubric; emotional attunement
lands at 88.1 % reflective and open-question framing at 88.3 %
(see Tables 5–6 in the technical report).

Full sweep (11 SUTs × 3 tracks × 3 conditions × advisory/reflective)
is in the code repo under
`experiments/20260426_1242_v0.2_sweep_merged/`. The v0.1 reference
2-SUT sweep used for the dataset's initial release remains at
`experiments/20260423_1610_day3_sweep/`.

## Models evaluated

The v0.2 sweep covers four families across three parameter
decades. Each entry below names the upstream HuggingFace
checkpoint, the license under which the model is redistributed,
and the role the model plays in the *Heard* evaluation.

- **Kakao Kanana 1.5 series.**
  [`kakaocorp/kanana-1.5-2.1b-instruct-2505`](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)
  (Apache-2.0) is a Korean-native 2.1 B instruction-tuned model and
  the primary on-device target for the *Heard* product. The v0.1
  baseline numbers in the previous section are taken on this model.
  [`kakaocorp/kanana-1.5-8b-instruct-2505`](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505)
  (Apache-2.0) is the 8 B sibling, included as a reference ceiling
  for the family.
- **NAVER HyperCLOVA-X SEED.**
  [`naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)
  (HCX-SEED-Public, gated) is the 1.5 B Korean-native model from
  NAVER, included once gated-access approval was granted. It is
  the smallest Korean-native SUT in the lineup and the strongest
  small-model int4 case.
- **Alibaba Qwen 2.5.**
  [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
  and
  [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  (both Apache-2.0) are multilingual baselines, included to
  establish whether a non-Korean-native multilingual model is
  competitive on the Korean tracks. Both place mid-pack on
  `ko_native` retrieval.
- **Korean-tuned Llama 3 derivatives.**
  [`MLP-KTLim/llama-3-Korean-Bllossom-8B`](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)
  and
  [`beomi/Llama-3-Open-Ko-8B-Instruct-preview`](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview)
  (both Llama-3 Community License) are Korean fine-tunes of Meta
  Llama 3 8B, included to test whether a fine-tune of a stronger
  base beats a Korean-native model at the same parameter count. On
  `ko_native` retrieval they place above the multilingual Qwen
  baselines and below the Kanana 8B reference.

Three additional Korean SUTs were attempted but dropped from the
final v0.2 lineup: `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`
(`transformers` 4.57.6 lacks the `RopeParameters` symbol the
EXAONE config imports), `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`
and `upstage/SOLAR-10.7B-Instruct-v1.0` (fp16 throughput too low
to finish the sweep within the wall-clock budget). These remain
candidates for a v0.3 expansion.

The retriever
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT) is shared across all SUT runs; the same top-5 cosine policy
is applied to every model in the table above.

## Construction notes

- Scenario generators were rotated across `anthropic/claude-haiku-4.5`,
  `openai/gpt-4o-mini`, and `google/gemini-2.5-flash` with
  consecutive-item anti-fingerprint.
- `ko_native` candidates passed, in order: (1) a 3-SUT × 3-trial
  adversarial filter where all no-NODE baselines must fail,
  (2) 4-gate validation covering n-gram overlap, LLM-judged
  clarity, evidence–answer consistency (advisory per ADR 0004),
  and history-only-fail reproduction, and (3) author review by
  the dataset creator for naturalness, evidence alignment, and
  gold-answer correctness.
- Translation: Track B translates only `question` and `gold_answer`
  to Korean; the haystack stays English due to Gemini's
  output-token cap on full items (ADR 0003).

## License

- Dataset: **CC-BY-4.0**. Compatible with LongMemEval_S (MIT) —
  attribution appears on each `en_subset` item's metadata.
- Code: **Apache 2.0** with an accompanying NOTICE file — see the
  repo at `github.com/cykim05/heard`.

## Ethical considerations

- All personas and dialogues are **synthetic**; no real personal
  information is included.
- Place and brand names are fictional.
- The dataset is intended for research on Korean long-term-memory
  language models and small on-device assistants.

## Known limitations

- `ko_native` is 70 items; ability-level statistics are
  underpowered.
- `ko_translated` uses an English haystack (see ADR 0003); the
  "language-only" comparison is partial.
- Gate 1 of the validation pipeline is advisory rather than
  blocking (ADR 0004); some `ko_native` items may therefore have
  gold answers that are imprecise relative to their evidence,
  mitigated by the author-review step.

## Citation

```
@misc{{heard-bench-2026,
  title        = {{heard-bench: A Korean Long-Term Memory Benchmark for Solo-Business Monologue}},
  author       = {{Kim, Chanyoung}},
  year         = {{2026}},
  howpublished = {{huggingface.co/datasets/chanyoungkim/heard-bench}}
}}
```
"""


def stage(stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    (stage_dir / "README.md").write_text(DATASET_CARD_TEMPLATE, encoding="utf-8")
    shutil.copy(REPO_ROOT / "LICENSE", stage_dir / "LICENSE")

    # data/
    for track in ("en_subset", "ko_translated", "ko_native"):
        src = REPO_ROOT / "data" / "final" / track / "test.jsonl"
        if not src.exists():
            log.warning("missing %s — skipping track %s", src, track)
            continue
        dst = stage_dir / "data" / track
        dst.mkdir(parents=True)
        shutil.copy(src, dst / "test.jsonl")
        meta_src = src.with_name("metadata.json")
        if meta_src.exists():
            shutil.copy(meta_src, dst / "metadata.json")
        log.info("staged data/%s (%.1f MB)", track, src.stat().st_size / 1e6)

    # personas/
    p_out = stage_dir / "personas"
    p_out.mkdir()
    for p in (REPO_ROOT / "configs" / "personas").glob("*.yaml"):
        shutil.copy(p, p_out / p.name)
    log.info("staged personas")


def upload(stage_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or _read_env_hf_token()
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        log.info("repo ready: %s", repo_id)
    except Exception as e:
        log.warning("create_repo: %s", e)

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(stage_dir),
        commit_message="Initial release — heard-bench v0.1",
    )
    log.info("upload complete → huggingface.co/datasets/%s", repo_id)


def _read_env_hf_token() -> str:
    env = (REPO_ROOT / ".env").read_text(encoding="utf-8")
    for line in env.splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("HF_TOKEN not set")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", type=Path, default=Path("/tmp/heard_bench_stage"))
    ap.add_argument("--repo-id", default="chanyoungkim/heard-bench")
    ap.add_argument("--no-upload", action="store_true",
                    help="Stage only (useful for dry-run)")
    args = ap.parse_args()

    stage(args.stage_dir)
    log.info("staged at %s", args.stage_dir)
    if not args.no_upload:
        upload(args.stage_dir, args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
