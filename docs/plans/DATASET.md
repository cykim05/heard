# Heard — Dataset Construction & Release Plan

> 이 문서는 `PLAN.md` / `IMPL_DETAILS.md` 의 **데이터셋 제작·검증·공개** 부분만 분리한 문서입니다. HuggingFace에 공개될 artifact의 사양과 제작 절차를 정의합니다.

---

## 0. Why a Separate Dataset Document

원래 plan은 단일 페르소나 기반 50 scenarios였습니다. 본 확장은 다음 3가지 이유로 별도 artifact를 만듭니다:

1. **학술적 비교가 가능해짐**: LongMemEval(ICLR 2025)과 같은 축에서 한국어 성능을 측정 → 리포트의 Related Work와 Results가 둘 다 강화됨.
2. **Huggingface 공개 → 재사용 가능한 contribution**: 단일 텀프로젝트를 넘어 "한국어 long-term memory benchmark"라는 reusable artifact가 됨.
3. **편향 통제가 체계적으로 필요함**: multi-source 데이터(원본/번역/네이티브) × multi-LLM 생성이 섞이면 제작 규약이 복잡해지므로 별도 문서로 분리.

**Dataset 이름**: `heard-bench` (확정, §12)

---

## 1. Relationship with LongMemEval

우리는 LongMemEval의 **memory ability 분류와 평가 프로토콜을 차용**하고, **도메인(한국 1인 자영업 혼잣말)과 language(한국어)를 새로** 만듭니다.

### 1.1 What we inherit (LongMemEval에서 가져옴)

| LongMemEval 요소 | 우리 사용 여부 | 비고 |
|---|---|---|
| 5 core abilities (IE / MR / KU / TR / ABS) | ✅ 전부 차용 | 우리 Probe 1–4가 여기에 매핑됨 (§3 참조) |
| 7 question types | ✅ 부분 차용 | 3개 track에 분포시킴 |
| Needle-in-haystack 방식 (evidence session + distractor) | ✅ 차용 | 단, dialogue가 아닌 **monologue**로 재구성 |
| LLM-as-judge (gpt-4o 기반, 97% human agreement) | ✅ 차용 | 단, 우리는 judge 2개로 agreement까지 측정 |
| Recall@k / NDCG@k | ✅ 차용 | 우리 retriever 평가에 동일 적용 |

### 1.2 What we change (우리의 novelty)

| 요소 | LongMemEval | 우리 |
|---|---|---|
| Interaction format | User ↔ AI dialogue | **User monologue (혼잣말)** — Heard의 핵심 차이 |
| Language | English | **Korean** (+ English subset for comparison) |
| Domain | Generic chat (lifestyle, belongings, life events...) | **Solo-business decision-making** (꽃집·카페 등) |
| Schema | Flat attribute ontology (164 attributes) | **Domain-specific 5-node schema** (customer/stock/pricing/mood/decision) |
| Response evaluation | Factual QA only | Factual QA **+ Reflective response quality** |
| Adversarial filtering | — | ✅ No-NODE baseline이 확실히 실패하도록 filter |

---

## 2. Three-Track Dataset Structure

총 **300 items**. 세 track이 각각 100 items.

### 2.1 Track A — `en_subset`: LongMemEval 원본 subset (영어, 100 items)

**목적**: (1) 우리 시스템이 기존 benchmark에서도 작동하는지 sanity check, (2) 언어간 비교의 baseline.

**구성**:
- LongMemEval_S의 500 questions 중 **100개를 ability-stratified sampling**.
- 각 ability (IE/MR/KU/TR/ABS)별 20개씩 비율 유지.
- 원본 그대로 사용 (재가공 없음). MIT 라이선스 유지.

**작업량**: ~1시간 (HuggingFace에서 다운로드 → subsample → 로컬 저장).

### 2.2 Track B — `ko_translated`: 번역 track (한국어, 100 items)

**목적**: Track A를 한국어로 번역. **"언어만 바뀌면 성능이 얼마나 떨어지는가"**를 측정 → 한국어 특화 필요성의 직접 증거.

**번역 절차**:
1. **기계 번역**: Track A의 100 items를 4개 LLM(§4)으로 각각 번역 → 4개 후보.
2. **Cross-validation**: 각 item마다 4개 번역본을 **다른 LLM에 평가**시켜 best selection + 불일치 flagging.
3. **사람 검수 필수**: 본인(김찬영)이 전체 100개 **1회 패스 검수**. 특히:
   - 시간 표현(dates, durations)의 한국어 관용구 정합성
   - Named entity 번역 (외국 고유명사 → 한국식 혹은 음차)
   - Abstention 질문의 "모른다" 표현 자연스러움
4. **Back-translation check**: 10% 샘플에 대해 역번역 → 의미 drift 정량화 (BLEU 보고).

**주의**: 문화적 부적절 항목(예: 영어권 생활문화 특화 질문)은 translation이 아닌 **localization** (문화 치환)으로 처리하고, 해당 사실을 metadata `is_localized: true`로 표시.

### 2.3 Track C — `ko_native`: 한국어 네이티브 생성 (100 items)

**목적**: **우리 도메인(한국 1인 자영업 혼잣말)**에서의 성능 측정. 프로젝트의 핵심 기여.

**페르소나 확장**: 단일 예진으로는 200+ history 세션을 커버하기 어렵고, 일반화 주장도 약함. 따라서 **3개 페르소나**로 확장:

| Persona ID | 직업 | 특성 | 할당 items |
|---|---|---|---|
| `yejin_florist` | 꽃집 사장 (38, 마포) | Part 1 그대로 | 40 |
| `minseok_cafe` | 카페 사장 (42, 성수) | 로스터리, 원두 단가 민감 | 30 |
| `sunhee_hair` | 미용실 원장 (45, 홍제) | 예약 시스템, 단골 컷 스타일 관리 | 30 |

**3 페르소나 이유**:
- 스키마 transferability 실증 (Part 1 Slide 6 Tier 3의 주장)
- 100 items를 단일 페르소나로 커버하면 history가 비현실적으로 길어짐
- ability별 자연스러운 분포가 가능해짐 (미용실 쪽은 customer/relation이 강하고, 카페 쪽은 pricing/stock이 강함)

**Ability 분포** (Track C 100 items):
- IE: 25 (specific fact recall)
- MR: 20 (multi-session reasoning across days)
- KU: 15 (knowledge update, 예: 재료 공급처 변경)
- TR: 20 (temporal reasoning, 예: "지난 달" / "작년 4월")
- ABS: 10 (false premise, 정답은 "혼잣말에 없음")
- **+ REFL: 10** (reflective quality, LongMemEval에 없는 우리 고유)

---

## 3. Probe Type × Ability Mapping

우리 이전 문서(IMPL_DETAILS.md §3.2)의 4-probe 분류를 LongMemEval의 5-ability에 정렬합니다.

| Our Probe | LongMemEval Ability | Scoring |
|---|---|---|
| Probe 1 (Factual Recall) | IE (Information Extraction) | Substring match + synonym list |
| Probe 2 (Entity Linking) | IE + MR (multi-session aggregation) | Entity match |
| Probe 3 (Temporal Consistency) | TR (Temporal Reasoning) | Keyword rubric + time window check |
| — | KU (Knowledge Update) | "New info overrides old" check |
| — | ABS (Abstention) | Binary: answered "모른다" vs hallucinated |
| Probe 4 (Reflective Quality) | **(Ours, new)** | LLM-judge pairwise, 4 rubrics |

→ Reporting table는 ability별로 줄이 생기며, 총 6개 row가 Results 표에 찍힘.

---

## 4. Multi-LLM Generation Strategy

### 4.1 Role assignment — **4 generators, 2 judges**

**단순화 원칙**: 모두 **직접 API 호출**로 통일. OpenRouter 불필요. 각 provider의 native SDK 사용.

**생성 담당 (Generator Pool, 4개)**:

| Slot | Model | API | 호출 방식 | 강점 | 비율 |
|---|---|---|---|---|---|
| G1 | Claude Sonnet 4.5 (or 4.6) | Anthropic API | `anthropic` SDK | 한국어 자연스러움, 감정 표현 | 30% |
| G2 | GPT-4o | OpenAI API | `openai` SDK | 논리 일관성, structured output | 30% |
| G3 | Gemini 2.5 Pro | Google AI Studio | `google-generativeai` SDK | 장문 맥락, 시간 추론 | 20% |
| G4 | **Kanana 1.5 8B** | **로컬 inference** (L40S) | `transformers` | **한국 문화·관용구 네이티브** + **Apache 2.0** | 20% |

**G4 결정 이유**:
- **License 최우선**: Kanana 1.5 8B (`kakaocorp/kanana-1.5-8b-instruct-2505`)는 **Apache 2.0**이라 output 제약 없음 → 우리 dataset을 CC-BY-4.0 (commercial OK)으로 공개 가능.
- EXAONE 3.5 32B가 품질은 더 좋을 수 있으나 **EXAONE License 1.1-NC** 이라 output이 dataset에 포함되면 우리 dataset도 non-commercial 제약을 상속받을 가능성이 큼. 학술 목적은 OK지만 future reuse 범위가 좁아짐.
- Kanana 8B는 L40S에 fp16으로 ~16GB → 여유롭게 로딩 + 32B보다 추론 3~4배 빠름.
- **만약 Kanana 품질이 Track C에서 부족하면** EXAONE 32B로 fallback 가능. 단 이 경우 dataset license를 CC-BY-NC-4.0으로 내림 (ADR 작성 필수).

**검증·평가 담당 (Judge Pool, 2개)**:

| Slot | Model | API | 역할 |
|---|---|---|---|
| J1 | GPT-4o | OpenAI API | Primary judge (LongMemEval이 human 97% agreement 보고한 것과 동일) |
| J2 | Claude Opus 4.6 | Anthropic API | Secondary judge, agreement 측정 |

**생성자 ≠ 평가자** 제약: Track C 생성에 쓰인 모델은 **같은 item의 judge에서 제외**. G1(Claude)로 생성 → J1(GPT-4o)이 judge. G2(GPT-4o)로 생성 → J2(Claude)가 judge. G3(Gemini)와 G4(EXAONE)은 judge pool에 없으니 둘 다 사용 가능.

### 4.2 Generator rotation rule (Track C)

100 items를 4-way rotation. 각 페르소나 내에서 generator를 균등 분산:

- Yejin 40 items: G1×12, G2×12, G3×8, G4×8
- Minseok 30 items: G1×9, G2×9, G3×6, G4×6
- Sunhee 30 items: G1×9, G2×9, G3×6, G4×6

**동일 페르소나 내에서 consecutive item은 반드시 서로 다른 generator**. → topic drift가 한 generator의 패턴에 종속되지 않음.

### 4.3 Anti-contamination checks

1. **N-gram overlap**: 같은 페르소나 내 item 간 4-gram 중복률 < 15% 강제. 초과 시 regenerate.
2. **Vocab diversity**: 페르소나별 unique token 수 / total tokens 비율 ≥ 0.35.
3. **Generator fingerprint detection**: 각 generator가 즐겨쓰는 phrase (예: Claude의 "한편으로는", GPT의 "~하는 경향이 있다") 탐지 → item당 top-3 distinctive phrase 로깅 후 편향 시 regenerate.

---

## 5. Generation Pipeline (Track C 상세)

### 5.1 Stage 1 — Persona × Timeline 생성

각 페르소나마다:
- 60일 타임라인 (Track C의 MR/TR 질문이 multi-day span을 요구하므로 최소 60일).
- 하루 10–15 utterance × 60일 ≈ **600–900 utterances per persona**.
- 총 corpus ≈ **2,000–2,700 utterances** (3 persona 합).

**중요**: 이 utterance corpus 자체가 test scenario의 "haystack"이 됩니다. 즉 각 scenario의 history는 이 corpus에서 sampling.

### 5.2 Stage 2 — Scenario construction (per item)

LongMemEval 방식 차용:

```
1. Pick ability (IE/MR/KU/TR/ABS/REFL)
2. Draft "evidence statement" (정답이 들어 있는 utterance) — 1~N개
3. Embed evidence into utterance corpus (natural position, timestamp)
4. Construct question utterance (오늘의 고민·질문)
5. Annotate gold answer + gold retrieval IDs + gold contains/excludes tokens
6. Sample distractor sessions from the same persona's corpus
   (LongMemEval은 ShareGPT/UltraChat에서 가져오지만, 우리는 같은 페르소나 내 다른 날짜에서)
```

### 5.3 Stage 3 — Adversarial Filtering (IMPL_DETAILS.md §3.4 재사용)

100개 candidate → **filtering pass** → 살아남은 것만 최종:

```
for each candidate:
    run no-NODE baseline with {Kanana-2.1B, HyperCLOVA-SEED-3B, Qwen2.5-3B}
    × 3 trials each (temp=0.7)
    if any of 9 runs passes gold criterion:
        → discard or regenerate with harder distractors
    keep only items where all 9 runs fail
```

**예상 discard rate**: 30–40%. 즉 초기 candidate는 **~150개 생성** 필요 → 100 유효 items.

### 5.4 Stage 4 — Auto-validation primary, spot check만 수동

**기본 원칙**: 300 items를 사람이 전수 검수할 필요 없도록 **multi-LLM cross-validation이 primary quality gate**. 사람 검수는 ~1시간 spot check만.

#### 5.4.1 Auto-validation (Claude Code가 자동 실행)

각 item은 다음 4개 게이트를 **모두** 통과해야 최종 데이터셋에 포함:

1. **Evidence-answer consistency check**
   - G4가 아닌 **다른 LLM** (보통 Claude 또는 GPT-4o)이 "evidence utterance만 보고 gold answer를 도출할 수 있는가?"를 판정.
   - 5회 시도 중 4회 이상 정답이 일치해야 통과. 아니면 discard.

2. **History-only fail check** (= adversarial filtering, §5.3에서 이미 수행)
   - no-NODE 조건에서 3 SUT × 3 trials 모두 fail.

3. **Cross-generator duplication check**
   - 같은 페르소나 내 item 간 4-gram 중복률 < 15%.
   - 초과 시 regenerate (다른 generator로).

4. **Ambiguity check**
   - Question utterance를 **세 번째 LLM** (예: Gemini)에 보여주고 "이 질문이 명확한가 (1–5)"를 점수화.
   - 3점 이하면 discard.

이 4 게이트를 통과한 item만 `data/final/ko_native/test.jsonl` 에 들어감. 게이트별 reject rate를 `data/final/validation_report.json` 에 기록.

#### 5.4.2 Human spot check (~1시간, 선택이지만 권장)

본인(김찬영)이 **랜덤 20 items (Track C에서)** 을 확인:
- 질문이 페르소나에 자연스러운가
- Gold answer가 실제로 evidence에서 도출되는가
- 한국어가 어색하지 않은가

결과를 `data/final/spot_check.md` 에 간단히 기록 (통과/탈락 개수 + 대표 issue).

**리포트 표기**:
- "All items passed 4-gate multi-LLM auto-validation."
- "A random 20% sample was spot-checked by the author; X/20 items accepted."

이 방식은 LongMemEval이 400 human hours 투자한 것과 비교하면 약하지만, (a) scale이 다르고 (500 vs 100) (b) 우리는 ADR에 방법론을 명시하여 transparent하게 처리. Discussion에 한계로 언급.

### 5.5 Track B 번역 품질 자동 검증

Track B는 별도 검수 필요:
- **Back-translation BLEU**: 원본 EN → KO → EN 역번역 후 BLEU ≥ 30 기준.
- **다른 LLM cross-check**: G1이 번역한 것을 G2가 "의미 일치 (1–5)" 평가. 4점 이상만 통과.
- 미달 items은 다른 generator로 재번역.

---

## 6. Target SUT Selection (Korean on-device LMs)

Part 1의 "on-device" 주장을 유지하려면 **target SUT는 한국어 특화 small LM**이어야 합니다. 2026년 4월 기준 **HuggingFace에서 바로 로딩 가능하고 인증 불필요한** 모델들로 확정:

### 6.1 Primary Korean SUTs (3개, fp16)

| Model | Params | HF ID | License | 근거 |
|---|---|---|---|---|
| **Kanana 1.5 Nano** | 2.1B | `kakaocorp/kanana-1.5-2.1b-instruct-2505` | Apache 2.0 | Kakao의 경량 한국어 on-device 모델, 2025년 5월 갱신 |
| **EXAONE 3.5** | 2.4B | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` | EXAONE (연구 목적 OK) | LG의 "resource-constrained device 최적화" 모델, 32K context |
| **HyperCLOVA X SEED** | 1.5B | `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | 공개 (확인 필요) | NAVER의 최경량 한국어 모델 |

**대안**: 위 중 로딩이 안 되는 경우 Claude Code는 아래 fallback을 자동 시도:
- Kanana 실패 → `kakaocorp/kanana-nano-2.1b-instruct` (1.0 버전)
- EXAONE 실패 → `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-AWQ` (사전 양자화 버전)
- HyperCLOVA 실패 → `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` (더 작은 모델)

### 6.2 Non-Korean baselines (2개, 비교군)

| Model | Params | HF ID | 이유 |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 3B | `Qwen/Qwen2.5-3B-Instruct` | Multilingual이지만 한국어 특화 아님 → "왜 한국어 모델이 필요한가" 증명 |
| Gemma-2-2B-it | 2B | `google/gemma-2-2b-it` | 서구 small LM 대표 baseline (gated — HF login + license 동의 필요) |

**Gemma-2 접근 주의**: gated model이라 HF 계정에 license 동의 필요. Claude Code가 로딩 실패하면 `meta-llama/Llama-3.2-3B-Instruct` 로 자동 대체 (역시 gated지만 더 흔함).

### 6.3 Reference ceiling (1개)

| Model | Params | HF ID | 이유 |
|---|---|---|---|
| Kanana 1.5 8B | 8B | `kakaocorp/kanana-1.5-8b-instruct-2505` | 상한선 참조 — "더 큰 모델이면 얼마나 좋아지나" |

### 6.4 Quantization axis

Primary Korean 3개 × {fp16, int4 (bnb-nf4)} = 6 configs
+ Non-Korean baselines 2개 × fp16 = 2 configs
+ Reference ceiling 1개 × fp16 = 1 config
= **총 9 configs**, L40S 1장에서 하룻밤 완료 가능.

### 6.5 Model loading wrapper (Claude Code가 작성)

`src/utils/llm_backend.py` 에 다음 로직 포함:
```python
def load_sut(hf_id, quantization=None):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            quantization_config=BNBConfig(...) if quantization == "int4" else None,
            trust_remote_code=True,  # EXAONE 계열 필요
            device_map="auto",
        )
        return model
    except Exception as e:
        # ADR에 기록 후 fallback 사용
        log_and_fallback(hf_id, e)
```

모든 실제 로딩된 모델 ID와 revision hash를 `experiments/{run_id}/model_registry.json` 에 기록 (GIT_WORKFLOW §5.1).

---

## 7. Evaluation Matrix (Reduced)

3 tracks × 9 SUTs × 3 retrieval conditions × 2 policies × ~100 items = **~16,200 response generations**

L40S 48GB 기준 ~2–3B 모델 throughput은 fp16에서 50–100 tok/s. 평균 응답 300 tokens 가정 시 item당 3–6초. 16,200개 × 5초 = **~22.5 시간**. Day 3 야간에 돌려서 Day 4 아침 완료.

**Reduced further (min viable)**:
- Track A/B/C 각 100 items
- SUT: Kanana-2.1B (fp16+int4), HyperCLOVA-SEED-3B fp16, Qwen2.5-3B fp16, Qwen2.5-7B fp16 (ceiling) = **5 configs**
- Conditions: no-NODE, joint-retrieval, oracle = 3
- Policies: advisory, reflective = 2
- Total: 300 × 5 × 3 × 2 = **9,000 runs** → ~12.5 시간 가능.

---

## 8. HuggingFace Upload Plan

### 8.1 Repository structure

```
heard-bench/                          # HF repo name
├── README.md                         # Dataset card (ko + en)
├── data/
│   ├── en_subset/
│   │   ├── test.jsonl
│   │   └── metadata.json
│   ├── ko_translated/
│   │   ├── test.jsonl
│   │   └── metadata.json
│   └── ko_native/
│       ├── test.jsonl
│       └── metadata.json
├── personas/
│   ├── yejin_florist.yaml
│   ├── minseok_cafe.yaml
│   └── sunhee_hair.yaml
├── dataset_script.py                 # HF datasets loader
└── LICENSE                           # MIT
```

### 8.2 Each item schema (JSONL)

```json
{
  "item_id": "ko_native_001",
  "track": "ko_native",
  "persona_id": "yejin_florist",
  "ability": "TR",
  "question_type": "temporal-reasoning",
  "history_sessions": [
    {"session_id": "s_042", "timestamp": "2026-02-14T22:40:00+09:00",
     "utterances": ["...", "..."], "is_evidence": false},
    {"session_id": "s_067", "timestamp": "2026-03-07T23:10:00+09:00",
     "utterances": ["...", "..."], "is_evidence": true}
  ],
  "question": {
    "text": "작년에 장미값 올렸을 때 얼마나 올렸더라?",
    "timestamp": "2026-04-20T22:40:00+09:00"
  },
  "gold_answer": {
    "text": "500원",
    "contains_tokens": ["500"],
    "excludes_tokens": [],
    "evidence_session_ids": ["s_067"],
    "evidence_utterance_indices": [[2]]
  },
  "reflective_rubric": null,   // REFL ability일 때만 채움
  "metadata": {
    "generator_model": "anthropic/claude-sonnet-4.5",
    "generator_run_id": "gen_20260424_0421",
    "filtering_failed_sut_ids": ["kanana-2.1b", "hclova-seed-3b", "qwen2.5-3b"],
    "human_validated": true,
    "validator": "kim_cy_20243053",
    "validation_date": "2026-04-26"
  }
}
```

### 8.3 Dataset card (README.md) 구성

- **Summary**: Korean long-term memory benchmark for solo-business monologue.
- **Construction pipeline**: §5 요약.
- **Statistics**: item count by (track × ability × persona).
- **License**: MIT (LongMemEval 호환).
- **Ethical considerations**:
  - 페르소나는 **실존 인물이 아님** (합성 데이터).
  - 상호명, 지명, 인명은 모두 가명/허구.
  - 데이터에 개인정보 포함 없음.
- **Known limitations**:
  - Synthetic monologue, not real transcription.
  - 3 personas only; generalization to other trades requires further data.
  - Korean standard dialect; 방언 미포함.
- **Citation**: 리포트에 BibTeX.
- **Baseline results** 테이블: 우리 실험 결과 그대로.

### 8.4 Upload workflow

```bash
# Day 4 작업
huggingface-cli login
huggingface-cli repo create heard-bench --type dataset
git lfs track "*.jsonl"
git add .
git commit -m "Initial release of Heard-Bench v0.1"
git push
```

- HF username 결정 필요 (본인 계정 또는 SSU/KAIST lab 계정).
- License: MIT로 시작, 추후 논문 발표 시 CC-BY 고려.

---

## 9. Validation & Quality Assurance

### 9.1 Inter-generator agreement (Track C)

각 item candidate를 4 generators 모두에게 주고, 만들어진 question-answer pair 사이 **의미 일치율** 측정:
- 2/4 이상 generators가 같은 evidence를 지목하면 해당 item은 "unambiguous"로 표시.
- 1/4 이하면 ambiguous → discard.

### 9.2 Judge-judge agreement

GPT-4o와 Claude Opus 4.6의 reflective quality 점수 Pearson correlation 보고. **r > 0.7** 미달 시 rubric 재설계.

### 9.3 Retrieval gold label sanity check

각 gold_retrieval_ids가 실제로 오늘의 question과 의미적으로 연결되는지, embedding similarity로 확인. 하위 5%는 수동 재검토.

### 9.4 Human acceptance rate target

본인 검수 시 reject rate < 15% 목표. 초과 시 prompt engineering으로 돌아가 regenerate.

---

## 10. Timeline (revised)

이 확장은 기존 4일 scope을 **유지하되 auto-validation으로 human 부담을 최소화**합니다.

| Day | Morning | Afternoon | Night |
|---|---|---|---|
| **Thu (Day 1)** | Repo init + GIT_WORKFLOW.md 규약 적용, Persona × 3 확정 | Utterance corpus 생성 (~2,400, 4 generators) | Extraction/retrieval 코드 스켈레톤, day1 lab notebook |
| **Fri (Day 2)** | Track A subsample + Track B 번역 (auto-cross-check) | Track C scenario 생성 (~150 candidates) | Adversarial filter + auto-validation 4-gate |
| **Sat (Day 3)** | **옵션**: 20 items spot check (~1h) | SUT 다운로드, full sweep 시작 (9 configs × ~2,700 = ~24K runs) | Sweep 진행, judge 병렬 |
| **Sun (Day 4)** | Sweep 종료, metric 집계 | Figures + 리포트 작성 | HF dataset upload + 리포트 최종 |
| **Mon (Day 5)** | 버퍼 + retries | — | 제출 (23:59) |

**Day 3 spot check는 선택**: 건너뛰어도 프로젝트는 성립. 리포트에 "auto-validated only"로 표기. 1시간 투자로 방어력이 올라가니 가능하면 수행.

---

## 11. Revised Data Scale Summary

| 데이터 | 수량 |
|---|---|
| Personas | 3 |
| Utterance corpus (histories 재료) | ~2,400 utterances |
| Test items (총계) | **300** (en 100 + ko_translated 100 + ko_native 100) |
| Abilities covered | 6 (IE, MR, KU, TR, ABS, REFL) |
| Unique LLMs in pipeline | 6 (4 generators + 2 judges) |
| Target SUTs | 5–9 configs |
| Total generation calls (evaluation) | 9,000–16,200 |

이는 LongMemEval(500 questions)의 60% scale이고, workshop short paper 혹은 **datasets & benchmarks track** 수준의 contribution입니다.

---

## 12. Decisions (확정됨)

이전 대화를 통해 확정된 결정사항:

| # | 항목 | 결정 | 근거 |
|---|---|---|---|
| 1 | HF upload 계정 | **사용자 본인 계정** (Claude Code가 실행 시 토큰 요청) | 개인 portfolio 가치 |
| 2 | Dataset 이름 | **`heard-bench`** (GitHub/HF 동일) | 간결, 프로젝트명과 일관 |
| 3 | License | **Code: MIT** / **Dataset: CC-BY-4.0** | 학술·상업 호환, generator 제약 상속 회피 |
| 4 | G4 (Korean generator) | **Kanana 1.5 8B** (Apache 2.0) / EXAONE 32B 대안 | License 호환성 우선 |
| 5 | Human validation | **Auto-validation primary + 20 items spot check 선택** | 시간 절약 |
| 6 | Personas 수 | **3개 (Yejin, Minseok, Sunhee)** | Schema transfer 주장 강화 |
| 7 | Generator APIs | **OpenRouter 단일 게이트웨이** (revised 2026-04-23) | 키·빌링·rate limit 일원화; 3 provider 개별 관리 오버헤드 > OpenRouter 마진 |
| 8 | Git workflow | **GIT_WORKFLOW.md** 규약 적용, Claude attribution 전면 금지 | §GIT_WORKFLOW.md |

### 12.1 사용자가 Claude Code 시작 전 준비할 것

1. **GitHub repo 생성**: `heard-bench` (private 권장, 추후 public 전환)
2. **API keys 준비** (`.env` 파일에 저장, `.env.example` 참고):
   - `OPENROUTER_API_KEY` (Claude/GPT/Gemini 통합 접근)
   - `HF_TOKEN` (upload + gated model 접근, write 권한 필요)
3. **Git user config**: `user.name`, `user.email` 본인 정보
4. **L40S 환경 확인**: CUDA 12.x, driver, 디스크 50GB+ 여유
5. **예산 상한**: API 비용 USD 30 상한 권장 (`budget_guard` 자동 중단)

---

**End of dataset plan. PLAN.md §5의 Day-by-day를 이 문서 §10으로 대체합니다.**
