"""Automatic metrics — extraction F1, retrieval Recall@k / MRR, answer match.

No ML deps here; pure Python + numpy so Day 1 can import this without
torch loaded. LLM-as-judge lives in src/eval/judge.py (Day 2).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import numpy as np


def multilabel_f1(
    gold: Sequence[set[str]],
    pred: Sequence[set[str]],
    *,
    average: str = "micro",
) -> dict[str, float]:
    """Multi-label F1 over set-valued labels. Supports micro and macro."""
    if len(gold) != len(pred):
        raise ValueError(f"length mismatch: gold={len(gold)} pred={len(pred)}")
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if average == "micro":
        tp = fp = fn = 0
        for g, p in zip(gold, pred):
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1}

    if average == "macro":
        all_labels: set[str] = set().union(*gold, *pred)
        per_label = []
        for label in all_labels:
            g = [label in s for s in gold]
            p = [label in s for s in pred]
            tp = sum(1 for a, b in zip(g, p) if a and b)
            fp = sum(1 for a, b in zip(g, p) if b and not a)
            fn = sum(1 for a, b in zip(g, p) if a and not b)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_label.append((prec, rec, f1))
        mean = np.mean(per_label, axis=0)
        return {"precision": float(mean[0]), "recall": float(mean[1]), "f1": float(mean[2])}

    raise ValueError(f"unknown average={average!r}")


def recall_at_k(
    gold_ids: Iterable[str], retrieved: Sequence[str], *, k: int
) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(gold & top_k) / len(gold)


def reciprocal_rank(gold_ids: Iterable[str], retrieved: Sequence[str]) -> float:
    gold = set(gold_ids)
    for i, rid in enumerate(retrieved, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    gold: Sequence[Iterable[str]], retrieved: Sequence[Sequence[str]]
) -> float:
    if len(gold) != len(retrieved):
        raise ValueError(f"length mismatch: {len(gold)} vs {len(retrieved)}")
    if not gold:
        return 0.0
    return float(np.mean([reciprocal_rank(g, r) for g, r in zip(gold, retrieved)]))


def answer_contains_tokens(
    response: str,
    *,
    contains: Sequence[str],
    excludes: Sequence[str] = (),
) -> bool:
    """Substring match: all `contains` present, none of `excludes`."""
    lo = response.lower()
    if any(tok.lower() in lo for tok in excludes):
        return False
    return all(tok.lower() in lo for tok in contains)
