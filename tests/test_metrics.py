"""Metric unit tests."""
from __future__ import annotations

from src.eval.metrics import (
    answer_contains_tokens,
    mean_reciprocal_rank,
    multilabel_f1,
    recall_at_k,
    reciprocal_rank,
)


def test_multilabel_f1_micro_perfect():
    gold = [{"customer", "mood"}, {"pricing"}]
    pred = [{"customer", "mood"}, {"pricing"}]
    m = multilabel_f1(gold, pred, average="micro")
    assert m["f1"] == 1.0


def test_multilabel_f1_micro_partial():
    gold = [{"customer", "mood"}, {"pricing"}]
    pred = [{"customer"}, {"pricing", "stock"}]
    m = multilabel_f1(gold, pred, average="micro")
    # tp=2 (customer, pricing), fp=1 (stock), fn=1 (mood)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "c", "d"], k=3) == 0.5
    assert recall_at_k(["a"], ["a"], k=5) == 1.0
    assert recall_at_k([], ["a"], k=5) == 0.0


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank(["b"], ["a", "b", "c"]) == 0.5
    assert reciprocal_rank(["x"], ["a", "b", "c"]) == 0.0
    assert mean_reciprocal_rank([["b"], ["x"]], [["a", "b"], ["a", "b", "c"]]) == 0.25


def test_answer_contains_tokens():
    assert answer_contains_tokens("작년에 500원 올렸어", contains=["500"])
    assert not answer_contains_tokens("잘 모르겠어", contains=["500"])
    assert not answer_contains_tokens(
        "500원이지만 2000원은 아니야", contains=["500"], excludes=["2000"]
    )
