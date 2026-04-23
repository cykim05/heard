# Heard — Implementation Details

> 이 문서는 `PLAN.md` 의 보조 문서입니다. PLAN은 **무엇을** 만들지를, 이 문서는 **어떻게** 만들지를 다룹니다. 특히 (1) 데이터 편향을 피하기 위한 multi-LLM 전략, (2) "NODE 없으면 틀리고 있으면 맞도록" 설계된 adversarial test set, (3) 재현 가능한 평가 프로토콜에 집중합니다.

---

## 0. Guiding Principle

**"If we remove NODE, the LLM must fail. If we add NODE, it must succeed."**

이 한 문장이 데이터 설계·평가 설계·모델 선택 전체의 북극성입니다. 이 원칙이 깨지는 test item은 test set에서 제외합니다 (§4.4 참조).

---

## 1. Multi-LLM Strategy via OpenRouter

### 1.1 왜 multi-LLM인가

단일 LLM으로 utterance를 생성하고, **다른 단일 LLM으로 응답 품질을 judge**하면 다음 3가지 편향이 리포트의 신뢰도를 무너뜨립니다:

1. **Generator bias**: 한 모델의 문체·vocabulary·논리 패턴이 test set 전체에 각인됨.
2. **Judge self-preference**: Judge가 자기 family 모델의 응답을 선호함 (well-known; Zheng et al., 2023).
3. **Capability ceiling bias**: Judge가 약하면 미묘한 reflective quality 차이를 구분 못 함.

→ **세 역할(generator / target-SUT / judge)을 서로 다른 provider 모델**로 분리합니다.

### 1.2 Role assignment

| Role | Purpose | 제안 모델 (OpenRouter ID) | 대안 |
|---|---|---|---|
| **Data Generator** | Utterance / scenario / gold answer 생성 | `anthropic/claude-sonnet-4.5` | `openai/gpt-4o` |
| **Gold Annotator** | Generator와 **다른 family**로 gold label 검증 | `openai/gpt-4o` | `google/gemini-2.5-pro` |
| **Judge (primary)** | Reflective quality pairwise 채점 | `google/gemini-2.5-pro` | `openai/gpt-4o` |
| **Judge (secondary, agreement 측정용)** | Inter-judge agreement 계산 | `anthropic/claude-opus-4.6` | `openai/gpt-4o` |
| **Target SUT (Systems Under Test)** | 실제 실험 대상 small LMs | HF local: Qwen2.5-1.5B/3B/7B, Gemma-2-2B | (on-device simulation이므로 API 금지) |

**중요 제약**:
- Generator ≠ Judge (family level에서 분리).
- Target SUT는 반드시 **로컬 HF 모델**. OpenRouter 호출하면 "on-device" 주장이 무너집니다.

### 1.3 Generator diversity

단일 generator도 위험하므로, utterance 생성을 **3개 provider에 분산**:

- 40% `claude-sonnet-4.5`
- 40% `gpt-4o`
- 20% `gemini-2.5-pro`

같은 페르소나 카드 + 같은 "오늘의 이벤트" 프롬프트를 주되 **생성 모델만 다르게** 해서 섞습니다. 이렇게 하면 judge가 어느 한 family에 치우쳐도 generator bias와 상쇄됩니다.

### 1.4 OpenRouter 연동 (`src/utils/openrouter.py`)

```python
# 의사코드
class OpenRouterClient:
    def __init__(self, api_key, default_headers):
        self.base = "https://openrouter.ai/api/v1"
    
    def chat(self, model: str, messages: list, **kwargs) -> dict:
        # rate limit 재시도, temperature, response_format=json 지원
        ...

# 사용 시
client.chat("anthropic/claude-sonnet-4.5", messages=[...], response_format={"type":"json_object"})
client.chat("openai/gpt-4o", messages=[...])
client.chat("google/gemini-2.5-pro", messages=[...])
```

- **모든 API 호출에 `seed` / `temperature` 기록**. 재현성을 위해 `experiments/{run_id}/api_calls.jsonl`로 덤프.
- 비용 상한: total budget USD 30 가정, 초과 시 자동 중단 (`budget_guard`).
- Caching: 같은 (model, messages, temperature, seed) 조합은 디스크 캐시 (`diskcache`). 재실행 시 0원.

---

## 2. Data Preparation — End-to-End

### 2.1 Pipeline overview

```
[persona.yaml] 
    ↓
(A) Event timeline 생성 (28일, 하루 2–4 이벤트)
    ↓
(B) Utterance 생성 (3 generators 분산, 하루 8–15개)
    ↓
(C) Gold extraction annotation (Generator-A ≠ Annotator)
    ↓
(D) Decision-moment scenario 생성 (§4 핵심)
    ↓
(E) Sanity check & dedup
    ↓
data/{raw, annotated, scenarios}/*.jsonl
```

### 2.2 Persona card (`configs/personas/yejin.yaml`)

```yaml
persona:
  name: "예진"
  age: 38
  occupation: "꽃집 사장 (4년차)"
  shop_name: "보송"
  location: "서울 마포구"
  family: ["husband", "child (초등)"]
  work_hours: "05:00-24:00, 주 6일"
  personality: ["perfectionist", "ruminative_at_night", "avoids_burdening_family"]

regulars:
  - id: "grandma_park"
    name: "박 할머니"
    preference: ["pink_rose", "baby_breath"]
    visit_frequency: "weekly"
    backstory: "매주 수요일 오전, 손녀 결혼식 장식 이후 단골"
  - id: "office_kim"
    name: "김 과장"
    preference: ["white_lily", "eucalyptus"]
    visit_frequency: "monthly"
    backstory: "회사 리셉션용 정기 납품"
  - id: "couple_lee"
    ...

stock_items:
  - {name: "장미(빨강)", cost_per_stem: 1500, price: 3000}
  - {name: "장미(핑크)", cost_per_stem: 1800, price: 3500}
  - {name: "안개꽃", cost_per_stem: 500, price: 1200}
  - ...

recurring_stressors:
  - "도매가 변동"
  - "단골 이탈 불안"
  - "SNS 관리 피로"
  - "남편에게 솔직하지 못함"

historical_events:  # 28일 이전에 일어난 중요 사건 — 과거 회상 훅
  - {days_ago: 280, event: "작년 4월 장미값 500원 인상, 이탈 거의 없음"}
  - {days_ago: 120, event: "안개꽃 도매처 바꾼 뒤 품질 개선"}
  - {days_ago: 45, event: "박 할머니 손녀 출산, 한 달간 방문 안 함"}
```

### 2.3 Event timeline (`src/datagen/timeline.py`)

28일 × 3 이벤트/일 = 84 이벤트. **같은 이슈가 의도적으로 여러 날에 반복**되도록 강제:

- `rose_price_concern`: Day 3, 12, 19, 26 (retrieval 테스트 훅)
- `grandma_park_visit`: Day 2, 9, 16, 23 (relation 테스트 훅)
- `stock_shortage_pink_rose`: Day 14 (단발, 그러나 Day 23에 영향)

Event 예시:
```json
{
  "day": 12, "time": "22:40", 
  "event_type": "pricing_deliberation",
  "trigger": "도매가 +800원 인상 통보",
  "links_back_to": ["day_3_rose_price_concern", "past_event:4월_인상_경험"]
}
```

### 2.4 Utterance generation prompt (§1.3의 3 providers에 rotate)

```
[SYSTEM]
당신은 1인 꽃집을 운영하는 예진의 **혼잣말 생성기**입니다. 다음 페르소나 카드와 오늘의 이벤트를 보고, 예진이 실제로 내뱉을 법한 짧은 혼잣말 3–5개를 생성하세요.

규칙:
- 각 발화는 1–3문장, 구어체, 자연스러운 줄임말 허용.
- 감정이 드러날 것. 판단과 의문을 섞을 것.
- 과거 경험을 자연스럽게 흘릴 것 (timeline의 links_back_to 참조).
- JSON 배열로만 출력.

[USER]
페르소나: {persona_yaml}
오늘의 이벤트: {event_json}
오늘이 며칠째인지: {day}

[RESPONSE FORMAT]
[{"timestamp": "...", "text": "...", "intended_categories": [...], "intended_links_back": [...]}]
```

**`intended_links_back`**: generator가 "이 utterance는 Day 3의 발화와 연결되어야 한다"를 명시 → 나중에 retrieval gold label로 쓰임.

### 2.5 Gold annotation 분리 (`src/datagen/annotation_gen.py`)

Generator가 `intended_categories`를 말했어도 그대로 gold로 쓰면 안 됩니다 (generator가 틀렸을 수 있음). **Gold Annotator (§1.2) 가 독립적으로 다시 라벨링**:

```
[Annotator prompt]
다음 utterance를 읽고, 5개 카테고리 {customer, stock, pricing, mood, decision} 중 해당하는 것을 모두 고르고, 각 카테고리의 세부 entity를 추출하세요.

Generator의 주장은 **보지 말고** 당신만의 판단으로 라벨링하세요.
```

- Generator label과 Annotator label의 **agreement**를 계산 → Cohen's kappa.
- 불일치 항목은 **3rd LLM (tie-breaker)** 또는 **사람(본인) 확인**으로 resolve. 100개 gold는 반드시 사람이 최종 승인.

---

## 3. Test Set Design — The Core of Evaluation

### 3.1 두 가지 trial 유형

모든 평가는 **동일한 시나리오**를 두 조건에 넣고 비교합니다:

| Condition | 입력 |
|---|---|
| **(A) Memoryless (no-NODE)** | system prompt + **오늘의 utterance만** |
| **(B) With NODE (ours)** | system prompt + retrieved past utterances (top-k) + 오늘의 utterance |

Baseline SUT (A)와 Full system (B)가 **같은 LM**을 씁니다. 차이는 오직 NODE의 존재. → 성능 delta = NODE의 contribution.

### 3.2 Test item의 4가지 probe 유형

> 이 섹션이 질문하신 "NODE 없으면 틀리게" 설계의 핵심입니다.

각 decision-moment scenario는 아래 4개 probe 중 하나 이상을 포함합니다:

#### Probe 1: **Factual Recall Probe** (객관 채점)

과거 utterance에 **구체적 숫자/고유명사**가 등장하고, 오늘 그걸 묻는 형태.

**예시**:
- Day 7 utterance: "작년 4월에 장미값 500원 올렸는데 단골 이탈은 거의 없었어."
- Day 26 scenario (오늘의 발화): "작년에 장미값 올렸을 때 얼마나 올렸더라?"
- **Gold answer contains**: `"500"` (substring).
- **No-NODE 예상 성능**: hallucinate or "모르겠음" → 정답률 ~0%.
- **With-NODE 예상 성능**: retriever가 Day 7을 회수 → ~90%.

**채점**: exact substring match on gold tokens (e.g., `"500원"` or `"500 원"`).

#### Probe 2: **Entity-Linking Probe** (객관 채점)

과거에 등장한 **고유 entity** (단골 이름·꽃 품종)를 오늘 소환. No-NODE는 해당 entity를 만들어내거나 일반화.

**예시**:
- Day 2: "박 할머니가 오늘도 핑크 장미만 한 다발 사 가셨네."
- Day 9: "박 할머니 또 오셨네. 역시 핑크 장미."
- Day 16 scenario: "박 할머니 내일 오시면 뭐 추천할까?"
- **Gold answer contains**: `"핑크"` AND `"장미"` (둘 다).
- **No-NODE**: "꽃을 추천하세요" 수준의 generic → 0%.
- **With-NODE**: "핑크 장미" 언급 → 거의 100%.

#### Probe 3: **Temporal Consistency Probe** (객관+semi-objective)

과거의 결정·경험을 오늘의 결정에 연결.

**예시**:
- Day 3: "장미값 올려야 하나 고민되네."
- Day 7: "작년에 올렸을 때 이탈 없었잖아."
- Day 12 scenario: "내일 장미값 올릴지 말지 결정해야 해."
- **Gold**: 응답이 `"작년"` OR `"지난번"` OR `"이전"` + `"이탈"` OR `"단골"` 관련 키워드 포함.
- **No-NODE**: general pricing advice ("수요와 공급을 고려하세요") → fails keyword check.
- **With-NODE**: 과거 경험 인용 → passes.

#### Probe 4: **Reflective Quality Probe** (LLM judge)

LLM-as-judge pairwise. §1.2의 2 judge로 채점.

Rubric (1–5 Likert, 4개 축):
1. **Specificity**: 과거 구체적 사건을 인용했는가
2. **Non-directive**: 명령·조언조를 피했는가
3. **Emotional attunement**: 감정을 수용했는가
4. **Open-ended question**: 열린 질문으로 마무리했는가

### 3.3 Test set 구성 비율 (총 50 scenarios 목표)

| Probe type | Count | 채점 방식 | 목적 |
|---|---|---|---|
| Probe 1 (Factual) | 15 | Substring match | 가장 극명한 delta |
| Probe 2 (Entity) | 15 | Entity match | Relation-axis retrieval 검증 |
| Probe 3 (Temporal) | 12 | Keyword rubric | Time-axis retrieval 검증 |
| Probe 4 (Reflective) | 8 | LLM judge pairwise | Reflective policy 검증 |

### 3.4 **"No-NODE로 절대 못 맞추는" 보장 절차** (중요)

Test item이 실제로 NODE 없이 fail하는지 **사전에 확인**해야 test set이 의미 있습니다.

**Filtering loop**:

```
for each candidate scenario:
    for each target SUT (Qwen-1.5B, Qwen-7B, Gemma-2-2B):
        run in no-NODE condition, 3 trials (temperature=0.7)
        if any trial passes the gold criterion:
            discard this scenario   ← "쉬운" 문제 제거
    keep scenario only if all 3 SUTs fail in all 3 trials
```

이 filter를 통과한 scenario만 test set에 포함합니다. 이렇게 하면:

- No-NODE baseline의 정답률이 ~0%에 수렴 → **bar가 분명한 표**.
- 우연히 맞추는 케이스가 제거되어 **NODE의 contribution이 깨끗하게 측정**됨.
- 리포트 Methods 섹션에 이 "adversarial filtering"을 명시하면 평가의 엄격성이 강조됨.

**주의**: filtering에 쓴 SUT 셋과 본평가 SUT 셋이 동일하면 over-fitting. 따라서:
- Filtering 시: Qwen-1.5B, Qwen-7B, Gemma-2-2B (fp16)
- 평가 시: 위 셋 + **int4 variants** + **Qwen-3B** 추가 → int4·3B는 held-out 체크 역할.

### 3.5 **Upper-bound sanity check**

반대로, NODE가 있으면 "반드시" 맞아야 합니다. Retrieval oracle (retriever가 완벽하게 동작, 즉 gold 관련 utterance를 그냥 context에 넣어줌) 조건도 같이 측정:

- **Oracle retrieval + SUT**: 이론상 상한선
- **Our retrieval + SUT**: 실제 시스템
- Gap = retriever의 loss

---

## 4. Retrieval Gold Labels

각 scenario는 "**반드시 회상되어야 할 과거 utterance의 utt_id 리스트**"를 gold로 가집니다.

예:
```json
{
  "scenario_id": "s_026",
  "today_utterance": "작년에 장미값 올렸을 때 얼마나 올렸더라?",
  "probe_type": "factual_recall",
  "gold_retrieval_ids": ["u_0047", "u_0048"],  // Day 7의 해당 발화들
  "gold_answer_contains": ["500"],
  "gold_answer_excludes": []  // 있으면 hallucination 체크
}
```

**Retrieval metric**: Recall@5, Recall@10, MRR — gold_retrieval_ids 기준.

**답변 metric**: `all(tok in response for tok in gold_answer_contains) and not any(tok in response for tok in gold_answer_excludes)`.

---

## 5. Experimental Matrix

### 5.1 Full factorial

```
SUT models:         [Qwen2.5-1.5B, Gemma-2-2B, Qwen2.5-3B, Qwen2.5-7B]   (4)
Quantization:       [fp16, int4]                                          (2)
Condition:          [no-NODE, time-only, relation-only, joint, oracle]    (5)
Policy:             [advisory, reflective]                                (2)
```

**Total**: 4 × 2 × 5 × 2 = 80 cells × 50 scenarios = 4,000 responses.

### 5.2 Reduced matrix (if time-constrained)

Minimum viable:
- SUT: Qwen2.5-1.5B (fp16, int4), Qwen2.5-7B (fp16)  → 3 configs
- Condition: no-NODE, joint, oracle  → 3
- Policy: advisory, reflective  → 2
- Total: 18 cells × 50 = 900 responses. L40S 1장에서 2–3시간.

### 5.3 Expected result shape (pre-registered hypothesis)

리포트 Discussion에 넣을 "예측된" 표 (실제 숫자는 실험 후 채움):

| Condition | Probe 1 Acc | Probe 2 Acc | Probe 3 Acc | Probe 4 Reflective-win |
|---|---|---|---|---|
| no-NODE, advisory | ~0–5% | ~0–10% | ~10–20% | 40% (baseline) |
| no-NODE, reflective | ~0–5% | ~0–10% | ~15–25% | 50% |
| joint, advisory | ~60% | ~65% | ~55% | 55% |
| **joint, reflective** | **~80%** | **~85%** | **~75%** | **~75%** |
| oracle, reflective | ~95% | ~95% | ~90% | ~80% |

핵심 주장: **(a) NODE 자체의 gain이 모든 probe에서 크고, (b) reflective policy는 NODE가 있을 때만 의미가 있다** (상호작용 효과).

---

## 6. Reproducibility Checklist

- [ ] 모든 OpenRouter 호출: model, seed, temperature, timestamp 로깅
- [ ] 모든 HF 추론: model hash, quantization config, seed 로깅
- [ ] Test set JSON 파일은 git에 commit (private repo ok)
- [ ] Filter loop (§3.4)의 판정 로그도 커밋
- [ ] Judge prompt 버전 관리 (`prompts/judge_v1.md`, `v2.md` ...)
- [ ] `make reproduce`: 캐시 hit 시 API 호출 0원으로 전체 재현

---

## 7. Ethical & Validity Notes (리포트 Discussion에 명시할 것)

1. **Synthetic data limitation**: 합성 데이터이므로 실제 자영업자 발화의 분포와 다를 수 있음. Multi-generator로 완화했으나 제거 불가.
2. **Judge bias**: GPT-4o가 "structured response"를 선호하는 경향 + Claude가 "empathetic response"를 선호하는 경향이 문헌에 있음. Two-judge agreement로 완화.
3. **Adversarial filtering의 양날**: test set이 "NODE 없으면 어려운" 문제들로만 구성되므로, 실제 배포 시의 평균 성능과는 다름. 이 점을 Results 섹션에 명시.
4. **On-device claim**: OpenRouter를 generator·judge에 썼지만 SUT는 로컬 HF. Part 1의 on-device 주장은 SUT에만 해당.

---

## 8. What Goes Where (파일 매핑)

| Module | Owner file | Inputs | Outputs |
|---|---|---|---|
| Persona | `configs/personas/yejin.yaml` | — | dict |
| Timeline | `src/datagen/timeline.py` | persona | `data/raw/timeline.jsonl` |
| Utterances | `src/datagen/utterance_gen.py` | persona, timeline | `data/raw/utterances.jsonl` |
| Annotation | `src/datagen/annotation_gen.py` | utterances | `data/annotated/gold.jsonl` |
| Scenarios | `src/datagen/scenario_gen.py` | utterances, timeline | `data/scenarios/candidates.jsonl` |
| Filter | `src/datagen/adversarial_filter.py` | candidates, SUT small set | `data/scenarios/final.jsonl` |
| NODE extract | `src/node/extractor.py` | utterances, SUT | `experiments/.../nodes.jsonl` |
| NODE store | `src/node/store.py` | nodes | ChromaDB + networkx pickle |
| Retriever | `src/node/retriever.py` | query utt, store | top-k utt_ids |
| MIRROR | `src/mirror/generator.py` | query + retrieved, SUT, policy | response text |
| Eval | `src/eval/runner.py` | responses, scenarios | `metrics.json` |
| Judge | `src/eval/judge.py` | response pairs | `judge.jsonl` |
| Figures | `scripts/99_make_figures.py` | metrics + judge | `report/figures/*.pdf` |

---

**End of details. 실행 순서는 PLAN.md §5의 Day-by-day를 따르되, §3.4 filtering loop 는 Day 2 안에 반드시 끝낼 것 (이게 전체 평가의 토대).**
