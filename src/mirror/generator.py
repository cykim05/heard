"""MIRROR generator — compose prompt from (question, retrieved, policy)
and call a local SUT for the response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.llm_backend import LoadedModel, generate
from .prompts import (
    ADVISORY_SYSTEM_PROMPT,
    ADVISORY_ANSWERFIRST_SYSTEM_PROMPT,
    REFLECTIVE_SYSTEM_PROMPT,
    LISTENING_SYSTEM_PROMPT,
)


_POLICY_SYSTEM = {
    "advisory": ADVISORY_SYSTEM_PROMPT,
    "advisory_answerfirst": ADVISORY_ANSWERFIRST_SYSTEM_PROMPT,
    "reflective": REFLECTIVE_SYSTEM_PROMPT,
    "listening": LISTENING_SYSTEM_PROMPT,
}


def format_prompt(
    question: str,
    retrieved: list[dict[str, Any]] | None,
    *,
    policy: str,
) -> tuple[str, str]:
    """Return (system, user) strings for a single SUT call."""
    system = _POLICY_SYSTEM[policy]
    if retrieved:
        ctx_lines = []
        for r in retrieved:
            ts = r.get("timestamp", "")
            ctx_lines.append(f"- [{ts}] {r.get('text', '')}")
        ctx = "\n".join(ctx_lines)
        user = (
            f"## Retrieved context (과거 기억)\n{ctx}\n\n"
            f"## Today's question\n{question}\n\n## Response"
        )
    else:
        user = f"## Today's question\n{question}\n\n## Response"
    return system, user


def run_once(
    sut: LoadedModel,
    *,
    question: str,
    retrieved: list[dict[str, Any]] | None,
    policy: str,
    max_new_tokens: int = 200,
    temperature: float = 0.3,
    seed: int = 0,
) -> str:
    system, user = format_prompt(question, retrieved, policy=policy)
    return generate(
        sut,
        user,
        system=system,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        do_sample=True,
    )
