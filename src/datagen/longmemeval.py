"""LongMemEval_S subsampling for heard-bench Track A.

DATASET.md §2.1 — 100 items stratified 20 per ability across
{IE, MR, KU, TR, ABS}. Items keep LongMemEval's original schema and
are licensed under MIT per the upstream repo.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# question_type -> ability mapping. Abstention (`_abs` suffix in
# question_id) is handled independently.
_ABILITY_BY_TYPE = {
    "single-session-user": "IE",
    "single-session-assistant": "IE",
    "multi-session": "MR",
    "knowledge-update": "KU",
    "single-session-preference": "KU",
    "temporal-reasoning": "TR",
}
ABILITIES = ("IE", "MR", "KU", "TR", "ABS")


def load_longmemeval_s() -> list[dict[str, Any]]:
    """Download longmemeval_s from HF and return the 500 items."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval",
        filename="longmemeval_s",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify_ability(item: dict[str, Any]) -> str:
    if "_abs" in item.get("question_id", ""):
        return "ABS"
    return _ABILITY_BY_TYPE.get(item.get("question_type", ""), "IE")


def stratified_sample(
    items: list[dict[str, Any]],
    *,
    per_ability: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {a: [] for a in ABILITIES}
    for it in items:
        buckets[classify_ability(it)].append(it)

    sampled: list[dict[str, Any]] = []
    for ability in ABILITIES:
        pool = buckets[ability]
        if len(pool) < per_ability:
            raise ValueError(
                f"Not enough items for ability={ability}: have {len(pool)}, want {per_ability}"
            )
        picked = rng.sample(pool, per_ability)
        for it in picked:
            record = dict(it)
            record["_ability"] = ability
            sampled.append(record)
    return sampled


def normalize_item(upstream: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    """Convert an upstream LongMemEval item to heard-bench Track A schema
    (DATASET §8.2, with track='en_subset')."""
    sessions = []
    dates = upstream.get("haystack_dates", [])
    sess_ids = upstream.get("haystack_session_ids", [])
    sess_bodies = upstream.get("haystack_sessions", [])
    evidence_sess_ids = set(upstream.get("answer_session_ids", []))

    for i, body in enumerate(sess_bodies):
        sid = sess_ids[i] if i < len(sess_ids) else f"s_{i:03d}"
        date = dates[i] if i < len(dates) else ""
        sessions.append({
            "session_id": sid,
            "timestamp": date,
            "turns": body,
            "is_evidence": sid in evidence_sess_ids,
        })

    return {
        "item_id": f"en_subset_{ordinal:03d}",
        "track": "en_subset",
        "persona_id": None,
        "ability": upstream["_ability"],
        "question_type": upstream.get("question_type"),
        "history_sessions": sessions,
        "question": {
            "text": upstream.get("question", ""),
            "timestamp": upstream.get("question_date", ""),
        },
        "gold_answer": {
            "text": str(upstream.get("answer", "")),
            "contains_tokens": [],
            "excludes_tokens": [],
            "evidence_session_ids": list(evidence_sess_ids),
            "evidence_utterance_indices": [],
        },
        "reflective_rubric": None,
        "metadata": {
            "source": "xiaowu0162/longmemeval:longmemeval_s",
            "license": "MIT",
            "upstream_question_id": upstream.get("question_id"),
        },
    }
