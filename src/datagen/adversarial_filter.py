"""Adversarial filter — keep only candidates that no-NODE baselines fail.

Per IMPL_DETAILS.md §3.4 and DATASET.md §5.3, each Track C candidate
is pushed through N SUTs × M trials in a memoryless condition. Any
candidate where even one of those N×M runs passes gold is discarded
("too easy"). The survivors form the benchmark's hard subset.

REFL candidates are exempt — their gold is free-form reflective
quality, so substring scoring would misfire. They bypass the filter.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..eval.metrics import answer_contains_tokens
from ..utils.logging import get_logger
from ..utils.llm_backend import LoadedModel, generate

log = get_logger(__name__)


SYSTEM_PROMPT_NO_NODE = (
    "당신은 질문에 답하는 한국어 비서입니다. 주어진 질문에 대해 "
    "당신이 아는 범위에서 간결하게 답하세요. 모르면 '모른다'고 말해도 됩니다. "
    "1~2문장으로 답해 주세요."
)


@dataclass
class FilterRun:
    sut_name: str
    trial: int
    response: str
    passed_gold: bool


@dataclass
class FilterVerdict:
    candidate_id: str
    ability: str
    persona_id: str
    kept: bool
    reason: str              # "refl_bypass" | "all_failed" | "sut_passed"
    runs: list[FilterRun] = field(default_factory=list)


def run_filter(
    candidates: list[dict[str, Any]],
    suts: list[LoadedModel],
    *,
    trials: int = 3,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    skip_abilities: frozenset[str] = frozenset({"REFL"}),
    seed_base: int = 12345,
    log_every: int = 10,
) -> list[FilterVerdict]:
    if not suts:
        raise ValueError("need at least one SUT for the filter")
    verdicts: list[FilterVerdict] = []
    t_start = time.time()

    for idx, cand in enumerate(candidates):
        cid = cand["candidate_id"]
        ability = cand["ability"]

        if ability in skip_abilities:
            verdicts.append(FilterVerdict(
                candidate_id=cid, ability=ability,
                persona_id=cand["persona_id"], kept=True, reason="refl_bypass",
            ))
            continue

        question = cand["question_text"]
        contains = cand.get("gold_contains_tokens", []) or []
        excludes = cand.get("gold_excludes_tokens", []) or []

        runs: list[FilterRun] = []
        any_pass = False
        for sut in suts:
            if any_pass:
                break
            for t in range(trials):
                resp = generate(
                    sut,
                    question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    seed=seed_base + idx * 97 + t,
                    system=SYSTEM_PROMPT_NO_NODE,
                )
                passed = answer_contains_tokens(
                    resp, contains=contains, excludes=excludes
                ) if contains else False
                runs.append(FilterRun(
                    sut_name=sut.name, trial=t, response=resp, passed_gold=passed,
                ))
                if passed:
                    any_pass = True
                    break

        verdicts.append(FilterVerdict(
            candidate_id=cid,
            ability=ability,
            persona_id=cand["persona_id"],
            kept=not any_pass,
            reason=("sut_passed" if any_pass else "all_failed"),
            runs=runs,
        ))

        if (idx + 1) % log_every == 0:
            kept = sum(1 for v in verdicts if v.kept)
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (len(candidates) - idx - 1)
            log.info(
                "filter %d/%d  kept=%d  discard=%d  elapsed=%.0fs  eta=%.0fs",
                idx + 1, len(candidates), kept, (idx + 1) - kept, elapsed, eta,
            )

    return verdicts


def save_verdicts_and_survivors(
    verdicts: list[FilterVerdict],
    candidates: list[dict[str, Any]],
    *,
    verdicts_path: Path,
    survivors_path: Path,
    report_path: Path,
) -> None:
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    survivors_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Verdicts jsonl — one per candidate with all 9 runs.
    with verdicts_path.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")

    # Survivors — filtered original candidate rows in same schema as input.
    kept_ids = {v.candidate_id for v in verdicts if v.kept}
    with survivors_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            if c["candidate_id"] in kept_ids:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Aggregate report.
    by_ability = {}
    for v in verdicts:
        slot = by_ability.setdefault(
            v.ability, {"total": 0, "kept": 0, "discarded": 0, "bypass": 0}
        )
        slot["total"] += 1
        if v.reason == "refl_bypass":
            slot["bypass"] += 1
            slot["kept"] += 1
        elif v.kept:
            slot["kept"] += 1
        else:
            slot["discarded"] += 1

    report = {
        "candidates_in": len(candidates),
        "survivors": sum(1 for v in verdicts if v.kept),
        "discarded_by_filter": sum(1 for v in verdicts if not v.kept),
        "refl_bypassed": sum(1 for v in verdicts if v.reason == "refl_bypass"),
        "by_ability": by_ability,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
