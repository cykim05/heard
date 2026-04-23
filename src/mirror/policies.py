"""Response policies — advisory / reflective / listening.

Day 1 ships configuration only. The Day 2 generator in
src/mirror/generator.py consumes these policies to build prompts and
drive local HF inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PolicyName = Literal["advisory", "reflective", "listening"]


@dataclass(frozen=True)
class PolicyConfig:
    name: PolicyName
    description: str
    target_length_sentences: tuple[int, int]
    must_cite_past_utterances: bool = False
    must_end_with_open_question: bool = False
    forbid_imperative: bool = False


POLICIES: dict[PolicyName, PolicyConfig] = {
    "advisory": PolicyConfig(
        name="advisory",
        description="Generic AI-assistant baseline: bullet advice, specific numbers.",
        target_length_sentences=(3, 8),
    ),
    "reflective": PolicyConfig(
        name="reflective",
        description="Mirror policy — cite past self, no decision, open question.",
        target_length_sentences=(3, 6),
        must_cite_past_utterances=True,
        must_end_with_open_question=True,
        forbid_imperative=True,
    ),
    "listening": PolicyConfig(
        name="listening",
        description="Pure acceptance — short reflection, no advice.",
        target_length_sentences=(1, 2),
        forbid_imperative=True,
    ),
}
