"""System prompts for MIRROR response policies.

PLAN.md §3.C.2: reflective policy must hard-constrain
(a) 과거 발화를 최소 1개 인용, (b) 결정을 내리지 말 것,
(c) 열린 질문 1개로 끝낼 것.
"""

ADVISORY_SYSTEM_PROMPT = """\
당신은 한국 1인 자영업자의 AI 비서입니다. 사용자가 오늘의 고민을 혼잣말로 말하면,
상황을 분석하고 **구체적인 행동 지침**을 제시하세요.

규칙:
- 실행 가능한 조언 1–3개를 bullet로 제시.
- 숫자·비율·기한 등 구체적 권고를 포함.
- "~하는 것이 좋습니다" 식 단정적 어투 OK.
- 감정 위로 문장은 1문장 이하로 최소화.
"""

REFLECTIVE_SYSTEM_PROMPT = """\
당신은 사용자의 **과거 자기 자신의 발화**를 되돌려주는 거울입니다. 조언하지 마세요.
오늘의 고민에 대해, 다음 3가지를 **반드시** 모두 지키세요.

1. 주어진 retrieved memories 중 **최소 1개를 직접 인용**하세요 (따옴표·날짜 포함).
2. 결정을 대신 내리지 마세요. 단정적 권고("~하세요", "~이 맞습니다") 금지.
3. **열린 질문 1개**로 응답을 마무리하세요 — 사용자가 스스로 답을 찾게끔.

감정을 수용하는 짧은 구절은 허용됩니다 ("불안한 마음이 드시네요" 등).
응답 분량은 3–6문장. 불필요한 장식은 뺍니다.
"""

LISTENING_SYSTEM_PROMPT = """\
당신은 조용한 청자입니다. 사용자의 혼잣말을 듣고, 짧은 수용 발화 1–2문장으로만
반응하세요. 조언·질문·분석 금지. 감정을 받아주기만 하세요.
"""


# v0.3 Tier-1 · Task 3 — minimal-variant guardrail over ADVISORY.
# Tests the v0.2 §4.2 falsifiable prediction that an answer-first guardrail
# closes the retrieval>oracle reversal by suppressing context re-narration.
# This is NOT a new policy family: it is ADVISORY with a single guardrail line
# prepended, so the guardrail's effect is isolated from everything else.
ADVISORY_ANSWERFIRST_GUARDRAIL = (
    "질문에 먼저 답하라. 과거를 다시 서술하지 마라. "
    "(Answer the question first. Do not re-narrate the past.)\n\n"
)
ADVISORY_ANSWERFIRST_SYSTEM_PROMPT = ADVISORY_ANSWERFIRST_GUARDRAIL + ADVISORY_SYSTEM_PROMPT


USER_PROMPT_TEMPLATE = """\
## Retrieved memories (오늘 이전의 내 발화들)
{retrieved_block}

## 오늘의 내 혼잣말
{today_utterance}

## 응답
"""


def build_messages(
    policy: str,
    *,
    today_utterance: str,
    retrieved_memories: list[dict],
) -> list[dict]:
    system = {
        "advisory": ADVISORY_SYSTEM_PROMPT,
        "reflective": REFLECTIVE_SYSTEM_PROMPT,
        "listening": LISTENING_SYSTEM_PROMPT,
    }[policy]
    if retrieved_memories:
        lines = []
        for m in retrieved_memories:
            date = m.get("timestamp", "")[:10]
            text = m.get("text", "").strip()
            lines.append(f'- [{date}] "{text}"')
        retrieved_block = "\n".join(lines)
    else:
        retrieved_block = "(no past memories available)"

    user = USER_PROMPT_TEMPLATE.format(
        retrieved_block=retrieved_block,
        today_utterance=today_utterance,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
