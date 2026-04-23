"""Offline unit tests for OpenRouter utilities — no real API calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.openrouter import BudgetExceeded, BudgetGuard, _hash_request


def test_hash_request_is_deterministic_and_order_independent():
    a = _hash_request({"model": "x", "messages": [{"a": 1, "b": 2}]})
    b = _hash_request({"messages": [{"b": 2, "a": 1}], "model": "x"})
    assert a == b


def test_budget_guard_records_and_blocks(tmp_path: Path):
    ledger = tmp_path / "budget.json"
    g = BudgetGuard(ledger_path=ledger, cap_usd=1.0)
    assert g.current() == 0.0
    g.record(0.3)
    g.record(0.4)
    assert abs(g.current() - 0.7) < 1e-9

    g.check(projected_cost_usd=0.2)  # under cap, ok

    with pytest.raises(BudgetExceeded):
        g.check(projected_cost_usd=0.5)  # 0.7 + 0.5 > 1.0


def test_budget_guard_persists_across_instances(tmp_path: Path):
    ledger = tmp_path / "budget.json"
    BudgetGuard(ledger_path=ledger, cap_usd=10.0).record(2.5)

    g2 = BudgetGuard(ledger_path=ledger, cap_usd=10.0)
    assert g2.current() == 2.5
    raw = json.loads(ledger.read_text())
    assert raw["calls"] == 1
