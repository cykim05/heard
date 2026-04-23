# Heard — NLP Term Project Part 2 · Implementation Plan

> 이 문서는 Claude Code가 **병렬로** 구현할 수 있도록 작성된 macro-level plan입니다. 각 Track은 독립적으로 시작 가능하며, 의존성은 명시되어 있습니다. 상세 구현 결정은 Claude Code가 코드 작성 중에 판단합니다.

---

## 0. Project Context (읽고 시작)

**What we're building.** "Heard"는 한국 1인 자영업자(예: 꽃집 사장 "예진")의 혼잣말을 듣고, 도메인 특화 메모리로 누적한 뒤, 미래의 의사결정 순간에 **과거 자기 자신의 발화를 되돌려주는** on-device LLM 에이전트입니다. Part 1 제안서의 3대 컴포넌트는 MIC(STT) → NODE(memory) → MIRROR(reflective response).

**What this Part 2 covers.**
- STT는 **out-of-scope** (성공적 전사 가정, 텍스트 입력).
- **NODE + MIRROR**의 파이프라인을 구현하고 정량·정성 평가.
- "On-device 시뮬레이션" 관점에서 **small LM (1.5B–3B급) + 4-bit quant** 까지 실험 축으로 포함.

**Hardware / Environment.**
- GPU: **NVIDIA L40S (48GB VRAM)** × 1
- Python 3.10+, PyTorch 2.x, transformers, bitsandbytes, vLLM (optional), chromadb
- 평가용 큰 judge 모델은 API (Claude Sonnet 4.6 또는 GPT-4o) 사용 허용

**Deadline.** 4 days (report due Monday 23:59). 즉 **구현 3일 + 리포트 1일**이 현실적 배분.

---

## 1. Success Criteria (리포트 쓸 때 채워야 할 빈칸)

리포트의 Results 섹션이 성립하려면 아래 표의 셀들이 채워져야 합니다. 구현의 *goal* 은 이 표를 채우는 것입니다.

### 1.1 Main result table (to fill)

| Model | Params | Quant | Extraction F1 | Retrieval Recall@5 | Reflective-win-rate vs advisory | Latency / utt (s) | VRAM (GB) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | fp16 | | | | | |
| Qwen2.5-1.5B-Instruct | 1.5B | int4 | | | | | |
| Gemma-2-2B-it | 2B | fp16 | | | | | |
| Qwen2.5-3B-Instruct | 3B | fp16 | | | | | |
| Qwen2.5-7B-Instruct (reference) | 7B | fp16 | | | | | |

### 1.2 Ablations (to fill)

- **Memory ablation**: NODE 없이 vs. NODE 포함 → MIRROR 응답 품질 차이.
- **Retrieval axis ablation**: time-only vs. relation-only vs. time×relation joint.
- **Response policy ablation**: advisory baseline vs. reflective policy.

---

## 2. Repository Layout (Claude Code가 만들 구조)

```
heard/
├── README.md
├── requirements.txt
├── configs/
│   ├── models.yaml             # 실험할 모델들 선언
│   └── personas/
│       └── yejin.yaml          # 예진 페르소나 스펙
├── data/
│   ├── raw/                    # 생성된 원본 utterance
│   ├── annotated/              # extraction ground truth
│   └── scenarios/              # MIRROR 평가용 decision moments
├── src/
│   ├── datagen/                # Track A
│   │   ├── persona.py
│   │   ├── utterance_gen.py
│   │   ├── annotation_gen.py
│   │   └── scenario_gen.py
│   ├── node/                   # Track B
│   │   ├── schema.py
│   │   ├── extractor.py
│   │   ├── store.py            # chromadb wrapper + relation graph
│   │   └── retriever.py
│   ├── mirror/                 # Track C
│   │   ├── policies.py         # advisory / reflective
│   │   ├── prompts.py
│   │   └── generator.py
│   ├── eval/                   # Track D
│   │   ├── metrics.py          # F1, Recall@k, etc.
│   │   ├── judge.py            # LLM-as-judge
│   │   └── runner.py
│   └── utils/
│       ├── llm_backend.py      # HF transformers / vLLM / API 통합
│       └── logging.py
├── scripts/
│   ├── 01_generate_data.py
│   ├── 02_run_extraction.py
│   ├── 03_build_memory.py
│   ├── 04_run_mirror.py
│   ├── 05_evaluate.py
│   └── 99_make_figures.py
├── experiments/
│   └── (run 결과 저장)
└── report/
    └── figures/
```

---

## 3. Parallel Tracks — Claude Code 병렬 작업 단위

> **원칙**: Track A → {B, C, D}는 data contract 만 고정되면 동시에 시작 가능. 각 Track 내부도 file-level로 쪼개서 병렬 작성 가능.

### Track A — Synthetic Data Generation (블로커, 최우선)

**왜 먼저?** B/C/D가 모두 A의 출력을 전제로 하므로, A의 **데이터 스키마 JSON 샘플 1개**가 나오는 즉시 B/C/D를 병렬 시작.

**A.1. Persona & scenario scaffold**
- `configs/personas/yejin.yaml`: Part 1의 프로필 그대로 — 38세, 꽃집 Bosong, 단골 목록(Grandma Park 등), 재고 품목, 가격 이력, 스트레스 포인트 등을 구조화.
- 배경 이벤트 타임라인 4–8주 (e.g., "3주차 화요일: 장미 도매가 인상", "5주차 목요일: 박 할머니 빈손 귀가") — **MIRROR 평가의 ground truth는 여기서 나옴**.

**A.2. Utterance generation (`src/datagen/utterance_gen.py`)**
- Claude Sonnet 4.6 API로 생성 (품질이 결과 전체를 좌우하므로 아끼지 말 것).
- 하루 8–15 utterances × 28–56일 → **총 300–800 utterances**.
- 각 utterance는 다음 JSON schema:
  ```json
  {
    "utt_id": "u_0001",
    "day": 12,
    "timestamp": "2026-03-14T22:40:00+09:00",
    "location": "shop_closing",
    "text": "내일 장미값 올릴까... 단골들 발길 끊길지도",
    "gold_categories": ["pricing", "mood", "decision"],
    "gold_entities": {
      "customer": [],
      "stock": ["rose"],
      "pricing": {"item": "rose", "direction": "up", "amount_krw": null},
      "mood": ["anxious"],
      "decision": {"type": "price_change", "status": "deliberating"}
    }
  }
  ```
- **다양성 요구**: 같은 이슈가 여러 날에 걸쳐 등장해야 retrieval이 의미 있음 (예: "장미 가격 고민"이 3주차·5주차·7주차에 반복).

**A.3. Decision-moment scenarios (`src/datagen/scenario_gen.py`)**
- MIRROR 평가용 30–50개 시나리오. 각 시나리오 = `(과거 utterance 리스트, 오늘의 의사결정 발화, gold reflective response의 핵심 포인트)`.
- 예: 오늘 "장미값 올릴까"라고 말할 때, 과거 "지난 4월에 500원 올렸고 이탈 거의 없었다"가 회상되어야 함.

**A.4. Annotation gold set**
- Utterance의 최소 100개는 **수동으로 검수**(= Claude Code가 생성하되, 본인이 한 번 더 검토 패스). Extraction F1 계산용.

**Output contract (Track B/C/D가 의존하는 것)**
- `data/raw/utterances.jsonl`
- `data/annotated/utterances_gold.jsonl` (100개)
- `data/scenarios/decision_moments.jsonl` (30–50개)

---

### Track B — NODE: Memory Extraction & Retrieval

**B.1. Schema (`src/node/schema.py`)**
- Part 1에서 선언한 5개 노드 타입을 pydantic 모델로: `CustomerNode`, `StockNode`, `PricingNode`, `MoodNode`, `DecisionNode`.
- 각 노드는 `created_at`, `source_utt_id`, `embedding`, `relations: List[RelationEdge]`.

**B.2. Extractor (`src/node/extractor.py`)**
- Small LM (Qwen2.5-1.5B/3B, Gemma-2-2B)에 few-shot 프롬프트 → JSON 출력 → 파싱.
- Output: `List[Node]` per utterance.
- **반드시 structured output** (json-mode 또는 outlines 라이브러리). 파싱 실패율도 metric으로 기록.

**B.3. Store (`src/node/store.py`)**
- ChromaDB로 벡터 저장 (embedding model: `jhgan/ko-sbert-sts` 또는 `intfloat/multilingual-e5-small` — 한국어 품질 확인 후 선택).
- **Relation graph**는 별도로 networkx DiGraph로 유지 (customer↔decision, stock↔pricing 등).
- 시간 인덱스는 단순 metadata filter.

**B.4. Retriever (`src/node/retriever.py`)**
- Query = 오늘의 utterance. 세 가지 전략:
  - `time_only`: 최근 N일 + cosine similarity
  - `relation_only`: graph traversal (같은 customer/stock 노드에 연결된 것)
  - `joint`: time-decay weighted × relation hop × semantic similarity
- Top-k (k=5) 반환.

**B.5. Metrics**
- Extraction: category classification F1 (multi-label), entity-level F1.
- Retrieval: gold scenario의 "반드시 회상되어야 할 과거 utterance" 기준 **Recall@5, MRR**.

---

### Track C — MIRROR: Response Generation

**C.1. Policies (`src/mirror/policies.py`)**
- **Advisory baseline** (= 일반 챗봇 mimicking): "You should raise the price by X%."
- **Reflective policy**: Part 1의 22:40 예시처럼 *"Last May you faced the same dilemma. You did X. What feels different this time?"* — 과거 인용 + 열린 질문.
- **Pure listening mode** (optional, Slide 5의 23:15): 짧은 수용 발화만, 조언 금지.

**C.2. Prompts (`src/mirror/prompts.py`)**
- System prompt는 한국어. Reflective policy는 명시적으로 `(a) 과거 발화를 최소 1개 인용, (b) 결정을 내리지 말 것, (c) 열린 질문 1개로 끝낼 것` 을 하드 제약.

**C.3. Generator (`src/mirror/generator.py`)**
- 입력: 오늘의 utterance + retrieved memory nodes → 출력: response text.
- 같은 LM 집합으로 실험 (NODE와 모델 공유).

---

### Track D — Evaluation Harness

**D.1. Automatic metrics (`src/eval/metrics.py`)**
- Extraction F1, Retrieval Recall@k/MRR, latency (tokens/sec, end-to-end), VRAM (nvidia-smi 샘플링).

**D.2. LLM-as-judge (`src/eval/judge.py`)**
- Judge: Claude Sonnet 4.6 또는 GPT-4o (API).
- **Pairwise comparison**: 같은 상황에서 `advisory vs. reflective` 응답을 주고, 아래 rubric으로 채점:
  - Specificity (과거 맥락을 구체적으로 인용했는가)
  - Non-directive (명령/조언조 회피)
  - Emotional attunement (감정 수용)
  - Question quality (열린 질문의 깊이)
- Position bias 완화: 순서 swap 2회 평균.

**D.3. Runner (`src/eval/runner.py`)**
- 전체 모델 × 시나리오 × 정책 × quant 설정을 sweep. `experiments/{run_id}/` 에 jsonl로 저장.

**D.4. Figure generation (`scripts/99_make_figures.py`)**
- 리포트용 5개 figure:
  1. Extraction F1 by model × quant
  2. Retrieval Recall@5 by retrieval strategy
  3. Reflective-win-rate (LLM judge) by model
  4. Latency–quality Pareto (x: latency, y: win-rate, marker: model)
  5. Case study 1–2개 (qualitative, 표 형태)

---

## 4. Lightweight Experiments (L40S 48GB에서의 축)

| Axis | Values | 근거 |
|---|---|---|
| Model size | 1.5B / 2B / 3B / 7B | on-device 현실성 ↔ 품질 trade-off |
| Quantization | fp16 / int4 (bnb-nf4) | 진짜 모바일 환경 proxy |
| Retrieval strategy | time / relation / joint | NODE의 핵심 주장 검증 |
| Response policy | advisory / reflective | MIRROR의 핵심 주장 검증 |
| (optional) Context size | short (k=3) / long (k=10) | retrieval noise 영향 |

**Full sweep은 크지 않음**: 5 models × 2 quant × 3 retrieval × 2 policy ≈ 60 셀이지만, extraction과 retrieval은 policy에 독립이라 실제 unique run은 약 **30개**. L40S 1장으로 하루면 끝남.

---

## 5. Execution Timeline (Day-by-day)

- **Day 1 (오늘, Thu)**
  - [A.1 A.2] 데이터 생성 파이프라인 + 전체 corpus 1차 생성
  - [B.1 B.2] Schema 확정, extractor 프롬프트 초안 + 1개 모델에서 동작 확인
  - [C.1 C.2] 두 정책 프롬프트 초안
  - [D.1] Metric 함수 스켈레톤
- **Day 2 (Fri)**
  - [A.3 A.4] Scenario set + gold annotation 확정
  - [B.3 B.4 B.5] Store/Retriever 완성, extraction 전체 모델 sweep
  - [C.3] Generator 완성, 소량 sanity check
  - [D.2] Judge 프롬프트 확정 + pilot 10 case
- **Day 3 (Sat)**
  - [D.3] Full sweep 실행 (L40S 야간 돌림)
  - Qualitative case study 2–3개 픽업
- **Day 4 (Sun)**
  - [D.4] Figure 생성
  - 리포트 작성 (Summary → Intro → Methods → Results → Discussion → Conclusion)
- **Day 5 (Mon)**: 버퍼 + 제출 (23:59)

---

## 6. Known Risks & Mitigations

- **Synthetic data가 reflective evaluation을 over-fit할 위험**: utterance 생성기와 judge 모델을 다르게(Claude vs GPT-4o) 사용해 일부 분리. 한계로 Discussion에 명시.
- **Small LM의 JSON 파싱 실패**: `outlines` 또는 grammar-constrained decoding 사용. 파싱 실패율 자체를 reportable metric으로.
- **한국어 embedding 품질**: `multilingual-e5`와 `ko-sbert` 둘 다 10개 샘플로 즉석 비교 후 선택. 시간 여유 없으면 multilingual-e5 고정.
- **LLM-judge bias**: pairwise + 순서 swap. 가능하면 judge 2개 agreement도 리포트.
- **4일 타임박스 초과**: Day 2 끝에 minimum viable 실험셋(1.5B fp16 + 3B fp16 + 7B fp16, joint retrieval, reflective vs advisory)만으로도 리포트 성립하도록 Day 1부터 이 subset을 우선 실행.

---

## 7. Claude Code에게 주는 운영 지침

1. **이 PLAN.md를 repo 루트에 두고, 각 커밋/PR 설명에 어느 섹션 번호를 진행했는지 명시**.
2. **Track A의 output contract(§3 맨 끝 3개 파일)가 먼저 확정되어야** B/C/D가 의미 있는 코드를 쓸 수 있음. 따라서 Day 1 오전은 A에 집중.
3. 모델 로딩은 `src/utils/llm_backend.py`에 통일. HF transformers로 시작, 필요시 vLLM으로 교체.
4. **모든 실험 결과는 `experiments/{run_id}/config.yaml + results.jsonl + metrics.json`** 형태로 저장. `runner.py`가 이걸 일관되게 쓰도록.
5. 리포트 figure는 반드시 `scripts/99_make_figures.py` 하나로 재생성 가능해야 함 (재현성).
6. README에는 "How to reproduce"만 명료하게: `bash scripts/run_all.sh` 한 방에.
7. **막히면 추측하지 말고 stub + TODO 남기고 다음 Track으로 이동**. 4일 안에 끝내려면 전체 파이프라인이 연결된 상태(end-to-end 먼저 성립, 정교화는 나중)가 partial 최적화보다 훨씬 가치 있음.

---

## 8. Out of Scope (리포트에 명시할 것)

- STT / 음성 입력 (v2)
- 진짜 on-device 배포, mobile SDK 통합 (v3)
- 실제 사용자 파일럿 (IRB 부담, 4일 내 불가)
- 꽃집 외 도메인 일반화 (schema transfer는 Discussion에서 언급만)

---

## 9. Report Mapping (구현 결과 → 리포트 섹션)

| 리포트 섹션 | 의존하는 결과물 |
|---|---|
| Summary | §1.1, §1.2 핵심 숫자 2–3개 |
| Introduction | Part 1 PDF + §0 |
| Methods | §2 repo layout, §3 Track B/C 설계, §4 실험 축, GitHub 링크 |
| Results | §1.1 표, Figure 1–4 (§D.4) |
| Discussion | §6 risks, Case study (Figure 5), ablation 해석 |
| Conclusion | §1 핵심 숫자 요약 + future work (§8) |
| References | Pennebaker, JSBM 2025, MOHW 2024, FSC 2023, Whisper, Gemma, Qwen 논문, ChromaDB |

---

**End of plan. Claude Code, start with Track A §3.A.1.**
