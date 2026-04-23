"""Track B — translate LongMemEval items to Korean (MVP version).

See ADR 0003. Day 2 final approach:
  - Translate ONLY the question and the gold_answer to Korean.
  - history_sessions keep the original English text.
  - One call per item (small payload, always parseable).

The "language-only" comparison axis is weaker than a full-text
translation would support, but this version (a) lands in minutes,
(b) costs <$0.20, and (c) still exercises the SUT's Korean
question-understanding + Korean answer-production skills against
an English haystack — an interesting sub-axis in its own right.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from ..utils.openrouter import OpenRouterClient, extract_text
from .utterance_gen import GeneratorRotation, _extract_json

log = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are a professional English→Korean translator. Translate the user's "
    "question and the gold answer into natural Korean. Preserve numbers, "
    "proper nouns, and relative time expressions faithfully. "
    "Use casual Korean (해요체). "
    "Inside JSON strings, do NOT insert literal double-quote characters — "
    "use Korean 「」 or single ' ' quotes if you need to quote something. "
    "Output ONLY a JSON object:\n"
    '{"question_ko": "...", "gold_answer_ko": "..."}'
)


USER_PROMPT_TEMPLATE = """Translate these two fields to Korean.

question (EN): {question_en}
gold_answer (EN): {gold_answer_en}

Return the JSON object described."""


@dataclass
class TranslationResult:
    item_id: str
    question_ko: str
    gold_answer_ko: str
    generator_name: str
    generator_model_id: str
    generator_model_family: str
    cost_usd: float
    parse_failed: bool = False


def translate_item(
    client: OpenRouterClient,
    item: dict[str, Any],
    *,
    rotation: GeneratorRotation,
    seed: int = 42,
    max_new_tokens: int = 600,
    sleep_between_calls: float = 0.3,
) -> TranslationResult:
    gen_name, model_id = rotation.pick()
    family = ("claude" if model_id.startswith("anthropic/")
              else "openai" if model_id.startswith("openai/")
              else "gemini" if model_id.startswith("google/") else "other")

    user_msg = USER_PROMPT_TEMPLATE.format(
        question_en=item["question"]["text"],
        gold_answer_en=item["gold_answer"]["text"],
    )

    reply = client.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        seed=seed,
        max_tokens=max_new_tokens,
        tag=f"translate:{item['item_id']}:{gen_name}",
    )
    raw = extract_text(reply)
    cost = float((reply.get("usage") or {}).get("cost", 0.0))

    result = TranslationResult(
        item_id=item["item_id"],
        question_ko="",
        gold_answer_ko="",
        generator_name=gen_name,
        generator_model_id=model_id,
        generator_model_family=family,
        cost_usd=cost,
    )

    try:
        obj = json.loads(_extract_json(raw))
        result.question_ko = str(obj.get("question_ko", "")).strip()
        result.gold_answer_ko = str(obj.get("gold_answer_ko", "")).strip()
        if not result.question_ko or not result.gold_answer_ko:
            raise ValueError("empty translation field")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("translate parse failed %s gen=%s: %s", item["item_id"], gen_name, e)
        Path("experiments/_parse_failures").mkdir(parents=True, exist_ok=True)
        (Path("experiments/_parse_failures") /
            f"translate_{item['item_id']}_{gen_name}.txt").write_text(raw, encoding="utf-8")
        result.parse_failed = True

    time.sleep(sleep_between_calls)
    return result


def build_track_b_item(source: dict[str, Any], tr: TranslationResult) -> dict[str, Any]:
    """Track B item: KO question + KO gold_answer, EN history_sessions preserved."""
    return {
        "item_id": source["item_id"].replace("en_subset_", "ko_translated_"),
        "track": "ko_translated",
        "persona_id": None,
        "ability": source["ability"],
        "question_type": source.get("question_type"),
        # Haystack stays in English — see ADR 0003 for why.
        "history_sessions": source["history_sessions"],
        "question": {
            "text": tr.question_ko,
            "timestamp": source["question"]["timestamp"],
        },
        "gold_answer": {
            "text": tr.gold_answer_ko,
            "contains_tokens": [],
            "excludes_tokens": [],
            "evidence_session_ids": source["gold_answer"].get("evidence_session_ids", []),
            "evidence_utterance_indices": [],
        },
        "reflective_rubric": None,
        "metadata": {
            "source": "xiaowu0162/longmemeval:longmemeval_s",
            "derived_from_item_id": source["item_id"],
            "translation_generator": tr.generator_model_id,
            "translation_generator_family": tr.generator_model_family,
            "translation_cost_usd": tr.cost_usd,
            "translation_scope": "question_and_gold_answer_only",
            "history_language": "en",
        },
    }
