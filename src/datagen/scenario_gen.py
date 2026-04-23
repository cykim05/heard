"""Track C scenario generator — builds heard-bench candidates from the
yejin/minseok/sunhee utterance corpus.

Per DATASET.md §5.2, each candidate needs:
  1. ability ∈ {IE, MR, KU, TR, ABS, REFL}
  2. evidence utterance(s) sampled from the persona's corpus
  3. a plausible 'today' question whose answer requires recalling the
     evidence (ABS: a question about something the persona never said)
  4. gold answer + contains/excludes substring tokens
  5. evidence utt_ids for retrieval gold

The scenario LLM must not invent facts — it only composes a question
around the already-existing evidence. Output is JSON we parse into
ScenarioCandidate.

Generation is rotated across G1/G2/G3 the same way utterances were, but
we additionally tag each item with `generator_model_family` so the
Day 2 judge-exclusion rule (DATASET §4.1) works downstream.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from ..utils.logging import get_logger
from ..utils.openrouter import OpenRouterClient, extract_text
from .utterance_gen import GeneratorRotation, _extract_json

log = get_logger(__name__)


ABILITIES: tuple[str, ...] = ("IE", "MR", "KU", "TR", "ABS", "REFL")

# Candidate target distribution for Track C (pre-filter, 150 items total).
# Scaled from DATASET §2.3 Track C post-filter target (100 items) by ~1.5×
# to absorb the expected 30–40% adversarial-filter discard rate.
CANDIDATE_DISTRIBUTION: dict[str, int] = {
    "IE":   38,
    "MR":   30,
    "KU":   23,
    "TR":   30,
    "ABS":  15,
    "REFL": 14,
}
assert sum(CANDIDATE_DISTRIBUTION.values()) == 150

# Persona distribution follows DATASET §2.3 Track C: yejin 40 / minseok 30 / sunhee 30.
PERSONA_DISTRIBUTION = {
    "yejin_florist": 0.40,
    "minseok_cafe":  0.30,
    "sunhee_hair":   0.30,
}


# Family tag used by the judge-exclusion rule (DATASET §4.1).
MODEL_FAMILY_BY_ID_PREFIX = {
    "anthropic/": "claude",
    "openai/":    "openai",
    "google/":    "gemini",
}


def _model_family(model_id: str) -> str:
    for prefix, fam in MODEL_FAMILY_BY_ID_PREFIX.items():
        if model_id.startswith(prefix):
            return fam
    return "other"


@dataclass
class ScenarioCandidate:
    candidate_id: str
    track: str
    persona_id: str
    ability: str
    question_text: str
    question_timestamp: str
    gold_answer_text: str
    gold_contains_tokens: list[str]
    gold_excludes_tokens: list[str]
    evidence_utt_ids: list[str]
    evidence_summary: str
    generator_name: str
    generator_model_id: str
    generator_model_family: str
    seed: int
    raw_reasoning: str = ""


# ---------------------- evidence sampling --------------------------

def _utterances_by_persona(corpus_path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out.setdefault(o["persona_id"], []).append(o)
    return out


def _sample_evidence(
    corpus: list[dict[str, Any]],
    *,
    ability: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Pick evidence utterances to seed the scenario generator."""
    # Prefer utterances whose model-annotated topics are non-empty — they carry
    # retrievable hooks.
    seeded = [u for u in corpus if u.get("intended_topics")]
    seeded = seeded or corpus

    if ability in ("IE", "TR"):
        n = 1
    elif ability in ("MR", "KU"):
        n = rng.randint(2, 3)
    elif ability == "REFL":
        n = rng.randint(1, 2)
    elif ability == "ABS":
        return []  # abstention — no evidence
    else:
        n = 1

    return rng.sample(seeded, min(n, len(seeded)))


# ---------------------- prompt ------------------------------------

_ABILITY_BRIEFING = {
    "IE": "구체적 사실(숫자·고유명사·날짜 중 하나) 을 묻는 질문. Gold answer 는 짧고 substring match 가능해야 한다.",
    "MR": "여러 날에 걸친 정보를 종합해야 답이 나오는 질문. 증거가 2~3개 발화에 흩어져 있음.",
    "KU": "정보가 시간이 지나며 업데이트된 상황. 최신 상태를 묻고, 과거 상태를 답하면 오답.",
    "TR": "명시적인 시간 표현('작년 4월', '지난 주', '3개월 전')이 정답의 핵심 힌트.",
    "ABS": "페르소나가 corpus 에서 한 번도 언급한 적 없는 주제에 대한 질문. Gold answer 는 '모른다/말한 적 없다' 의 한국어 자연스러운 표현.",
    "REFL": "오늘의 감정적 고민. Reflective quality 를 평가하는 item. Gold answer 는 '인용해야 할 과거 발화의 요지' 로 기록 (자유 채점).",
}


SYSTEM_PROMPT = """당신은 한국어 long-term memory benchmark 의 **시나리오 생성기**입니다.
주어진 페르소나의 과거 발화 (evidence) 를 토대로, 오늘 시점의 질문 하나를 만듭니다.
사실을 **새로 지어내지 마세요** — 질문과 정답은 evidence 에서 도출되어야 합니다.

출력은 반드시 JSON 객체:
{
  "question_text": "오늘 페르소나가 스스로에게 던지는 질문 (1~2문장, 구어체)",
  "question_timestamp_day": 55~60 사이의 정수,
  "question_timestamp_hhmm": "HH:MM",
  "gold_answer_text": "정답 문장 (간결)",
  "gold_contains_tokens": ["정답에 반드시 포함되어야 할 짧은 substring (1~3개)"],
  "gold_excludes_tokens": ["hallucination 탐지용 (선택, 없으면 [])"],
  "evidence_summary": "왜 이 evidence 가 정답의 근거인지 1문장",
  "reasoning": "생성 근거 (짧게)"
}
"""


def _user_prompt(
    *,
    persona: dict[str, Any],
    ability: str,
    evidence: list[dict[str, Any]],
) -> str:
    ev_lines = []
    for u in evidence:
        ev_lines.append(
            f"- utt_id={u['utt_id']} day={u['day']} time={u['timestamp'][11:16]} "
            f"topics={u.get('intended_topics') or '-'} text={u['text']!r}"
        )
    ev_block = "\n".join(ev_lines) or "(evidence 없음 — ABS 질문을 만드세요)"

    return f"""## 페르소나
{yaml.safe_dump(persona, allow_unicode=True, sort_keys=False)}

## Ability 요구사항
- ability = {ability}
- 지시문: {_ABILITY_BRIEFING[ability]}

## Evidence utterances
{ev_block}

## 요청
위 evidence 기반으로 scenario 하나를 만드세요. JSON 만 출력.
"""


# ---------------------- generation ---------------------------------

def _parse(raw: str) -> dict[str, Any]:
    obj = json.loads(_extract_json(raw))
    required = {
        "question_text", "question_timestamp_day", "question_timestamp_hhmm",
        "gold_answer_text", "gold_contains_tokens",
    }
    if not required.issubset(obj):
        missing = required - set(obj.keys())
        raise ValueError(f"missing keys: {missing}")
    return obj


def generate_candidate(
    client: OpenRouterClient,
    *,
    persona: dict[str, Any],
    corpus: list[dict[str, Any]],
    ability: str,
    rotation: GeneratorRotation,
    base_date: datetime,
    rng: random.Random,
    candidate_idx: int,
    sleep_between_calls: float = 0.3,
) -> ScenarioCandidate | None:
    persona_id = persona["persona"]["id"]
    evidence = _sample_evidence(corpus, ability=ability, rng=rng)

    gen_name, model_id = rotation.pick()
    seed = rng.randint(1_000, 9_999_999)

    user_msg = _user_prompt(persona=persona, ability=ability, evidence=evidence)

    reply = client.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        seed=seed,
        max_tokens=1500,
        tag=f"scenario_gen:{persona_id}:{ability}:{gen_name}",
    )
    raw = extract_text(reply)
    try:
        obj = _parse(raw)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("scenario parse failed [%s|%s|%s]: %s", persona_id, ability, gen_name, e)
        Path("experiments/_parse_failures").mkdir(parents=True, exist_ok=True)
        Path(f"experiments/_parse_failures/scenario_{candidate_idx}_{persona_id}_{ability}_{gen_name}.txt").write_text(raw, encoding="utf-8")
        time.sleep(sleep_between_calls)
        return None

    day = int(obj.get("question_timestamp_day", 60))
    day = max(1, min(60, day))
    hhmm = str(obj.get("question_timestamp_hhmm", "22:00"))
    try:
        hh, mm = [int(x) for x in hhmm.split(":")[:2]]
    except Exception:
        hh, mm = 22, 0
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    ts = (base_date + timedelta(days=day - 1)).replace(hour=hh, minute=mm).isoformat()

    candidate = ScenarioCandidate(
        candidate_id=f"ko_native_cand_{candidate_idx:04d}",
        track="ko_native",
        persona_id=persona_id,
        ability=ability,
        question_text=str(obj["question_text"]).strip(),
        question_timestamp=ts,
        gold_answer_text=str(obj["gold_answer_text"]).strip(),
        gold_contains_tokens=list(obj.get("gold_contains_tokens") or []),
        gold_excludes_tokens=list(obj.get("gold_excludes_tokens") or []),
        evidence_utt_ids=[u["utt_id"] for u in evidence],
        evidence_summary=str(obj.get("evidence_summary", "")).strip(),
        generator_name=gen_name,
        generator_model_id=model_id,
        generator_model_family=_model_family(model_id),
        seed=seed,
        raw_reasoning=str(obj.get("reasoning", ""))[:500],
    )
    time.sleep(sleep_between_calls)
    return candidate


def save_jsonl(items: list[ScenarioCandidate], path: Path, *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


__all__ = [
    "ABILITIES",
    "CANDIDATE_DISTRIBUTION",
    "PERSONA_DISTRIBUTION",
    "ScenarioCandidate",
    "generate_candidate",
    "save_jsonl",
]
