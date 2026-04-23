"""Pairwise LLM-as-judge for reflective-quality comparison.

Given two responses A and B to the same REFL item, ask a judge to
pick which is 'more reflective' on four rubrics:
  1. specificity    — cites past context in specifics
  2. non_directive  — avoids imperatives/advice
  3. emotional_attunement  — acknowledges feeling
  4. open_question  — ends with an open question

Run each pair twice with A/B swapped (position-bias mitigation) and
with two judges (J1 gpt-4o-mini, J2 claude-haiku-4.5).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger
from ..utils.openrouter import OpenRouterClient, extract_text
from ..datagen.utterance_gen import _extract_json

log = get_logger(__name__)


RUBRICS = ("specificity", "non_directive", "emotional_attunement", "open_question")

JUDGES = {
    "j1_gpt4o_mini": "openai/gpt-4o-mini",
    "j2_claude_haiku": "anthropic/claude-haiku-4.5",
}


SYSTEM_PROMPT = """You are a strict grader of Korean reflective-dialogue quality.

You compare two candidate responses (A and B) to the same user's utterance
(오늘의 고민). Rate which one is better on each of four rubrics, on a ternary
scale: "A" / "B" / "tie".

Rubrics:
  specificity         Does it cite a specific past event with detail?
  non_directive       Does it AVOID imperatives / prescriptive advice?
  emotional_attunement  Does it acknowledge the speaker's feeling?
  open_question       Does it end with an open (not yes/no) question?

Output ONLY JSON: {"specificity":"A","non_directive":"tie", ...}
"""

USER_TEMPLATE = """## 오늘의 발화
{question}

## Response A
{resp_a}

## Response B
{resp_b}

Judge A vs B on the four rubrics. Output the JSON only."""


@dataclass
class JudgeVerdict:
    item_id: str
    sut: str
    condition: str
    judge: str
    swap: bool
    scores: dict[str, str] = field(default_factory=dict)   # rubric -> "A" | "B" | "tie"
    cost_usd: float = 0.0
    parse_failed: bool = False


def _call(client: OpenRouterClient, judge_name: str, judge_model: str,
          question: str, resp_a: str, resp_b: str,
          seed: int, tag: str) -> tuple[dict[str, str], float, bool]:
    reply = client.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=question, resp_a=resp_a, resp_b=resp_b,
            )},
        ],
        temperature=0.0,
        seed=seed,
        max_tokens=200,
        tag=tag,
    )
    raw = extract_text(reply)
    cost = float((reply.get("usage") or {}).get("cost", 0.0))
    try:
        obj = json.loads(_extract_json(raw))
        scores = {r: str(obj.get(r, "tie")).strip() for r in RUBRICS}
        if all(v in {"A", "B", "tie"} for v in scores.values()):
            return scores, cost, False
    except (json.JSONDecodeError, ValueError):
        pass
    log.warning("judge parse failed %s %s: %s", judge_name, tag, raw[:120])
    return {r: "tie" for r in RUBRICS}, cost, True


def judge_pair(
    client: OpenRouterClient,
    *,
    item_id: str,
    sut: str,
    condition: str,
    question: str,
    advisory_response: str,
    reflective_response: str,
    seed_base: int = 0,
) -> list[JudgeVerdict]:
    """Score one (SUT, condition) pair with each judge, twice (swap A/B)."""
    results: list[JudgeVerdict] = []
    for judge_name, judge_model in JUDGES.items():
        for swap in (False, True):
            if not swap:
                a, b = advisory_response, reflective_response
            else:
                a, b = reflective_response, advisory_response
            scores, cost, failed = _call(
                client, judge_name, judge_model,
                question=question, resp_a=a, resp_b=b,
                seed=seed_base + (1 if swap else 0),
                tag=f"judge:{item_id}:{sut}:{condition}:{judge_name}:swap={swap}",
            )
            # When we swap, invert the meaning of A/B in the scores.
            if swap:
                inv = {"A": "B", "B": "A", "tie": "tie"}
                scores = {k: inv[v] for k, v in scores.items()}
            results.append(JudgeVerdict(
                item_id=item_id, sut=sut, condition=condition,
                judge=judge_name, swap=swap, scores=scores,
                cost_usd=cost, parse_failed=failed,
            ))
            time.sleep(0.2)
    return results


def aggregate_wins(verdicts: list[JudgeVerdict]) -> dict[str, dict[str, Any]]:
    """Return {(sut, condition): rubric -> {win_reflective, win_advisory, tie}}."""
    out: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    for v in verdicts:
        if v.parse_failed:
            continue
        key = (v.sut, v.condition)
        slot = out.setdefault(key, {r: {"advisory_win": 0, "reflective_win": 0, "tie": 0}
                                    for r in RUBRICS})
        for r, verdict_char in v.scores.items():
            # After swap-inversion, 'A' means advisory (original position 0),
            # 'B' means reflective. 'tie' unchanged.
            if verdict_char == "A":
                slot[r]["advisory_win"] += 1
            elif verdict_char == "B":
                slot[r]["reflective_win"] += 1
            else:
                slot[r]["tie"] += 1
    return {f"{s}|{c}": v for (s, c), v in out.items()}
