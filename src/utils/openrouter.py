"""OpenRouter client with disk cache, budget guard, and per-call logging.

Usage:
    client = OpenRouterClient.from_settings(load_settings())
    reply = client.chat(
        model="anthropic/claude-sonnet-4.5",
        messages=[{"role": "user", "content": "..."}],
        temperature=0.7,
        seed=42,
        response_format={"type": "json_object"},
    )

Design notes (IMPL_DETAILS §1.4):
- Every call records model/seed/temperature/usage/cost to experiments/_api_log/api_calls.jsonl.
- Identical (model, messages, temperature, seed, response_format) requests hit diskcache — 0 cost.
- BudgetGuard reads cumulative cost from a JSON ledger and aborts before exceeding cap.
- Cost comes from OpenRouter's own `usage` field (no hardcoded pricing).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import diskcache
import httpx
import orjson
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import REPO_ROOT, Settings, load_settings


OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


class RateLimited(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    """Tracks cumulative spend in a JSON ledger and refuses calls past the cap."""

    ledger_path: Path
    cap_usd: float
    _lock: threading.Lock = threading.Lock()

    def __post_init__(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists():
            self.ledger_path.write_text(json.dumps({"spent_usd": 0.0, "calls": 0}))

    def current(self) -> float:
        return json.loads(self.ledger_path.read_text())["spent_usd"]

    def check(self, projected_cost_usd: float = 0.0) -> None:
        spent = self.current()
        if spent + projected_cost_usd > self.cap_usd:
            raise BudgetExceeded(
                f"Budget cap {self.cap_usd:.2f} USD would be exceeded: "
                f"spent={spent:.4f}, projected_add={projected_cost_usd:.4f}"
            )

    def record(self, cost_usd: float) -> None:
        with self._lock:
            data = json.loads(self.ledger_path.read_text())
            data["spent_usd"] = float(data["spent_usd"]) + float(cost_usd)
            data["calls"] = int(data["calls"]) + 1
            data["last_update"] = datetime.now(timezone.utc).isoformat()
            self.ledger_path.write_text(json.dumps(data, indent=2))


def _hash_request(payload: dict[str, Any]) -> str:
    blob = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(blob).hexdigest()


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: Path,
        api_log_dir: Path,
        budget_guard: BudgetGuard,
        site_url: str = "",
        app_name: str = "heard-bench",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._cache = diskcache.Cache(str(cache_dir))
        self._log_path = api_log_dir / "api_calls.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._budget = budget_guard
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        self._http = httpx.Client(
            base_url=OPENROUTER_BASE, headers=headers, timeout=timeout
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "OpenRouterClient":
        settings = settings or load_settings()
        return cls(
            api_key=settings.openrouter_api_key,
            cache_dir=settings.cache_dir,
            api_log_dir=settings.api_log_dir,
            budget_guard=BudgetGuard(
                ledger_path=settings.cache_dir / "budget.json",
                cap_usd=settings.budget_total_usd,
            ),
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )

    def close(self) -> None:
        self._http.close()
        self._cache.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type((RateLimited, httpx.TransportError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post("/chat/completions", json=payload)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "5"))
            time.sleep(min(retry_after, 30))
            raise RateLimited(resp.text)
        if resp.status_code >= 500:
            raise ProviderError(f"{resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tag: str = "",
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        """Send a chat completion. Returns the raw OpenRouter response dict.

        Access the text via reply["choices"][0]["message"]["content"].
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "usage": {"include": True},  # request usage stats from OpenRouter
        }
        if seed is not None:
            payload["seed"] = seed
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        cache_key = _hash_request(payload)

        if not bypass_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            self._log_call(payload, cached, tag=tag, cache_hit=True, cost_usd=0.0)
            return cached

        self._budget.check()
        try:
            reply = self._post_chat(payload)
        except RetryError as e:
            raise ProviderError(f"retries exhausted: {e}") from e

        usage = reply.get("usage", {}) or {}
        cost_usd = float(usage.get("cost", 0.0))

        self._budget.record(cost_usd)
        self._cache[cache_key] = reply
        self._log_call(payload, reply, tag=tag, cache_hit=False, cost_usd=cost_usd)
        return reply

    def _log_call(
        self,
        payload: dict[str, Any],
        reply: dict[str, Any],
        *,
        tag: str,
        cache_hit: bool,
        cost_usd: float,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "seed": payload.get("seed"),
            "tag": tag,
            "cache_hit": cache_hit,
            "cost_usd": cost_usd,
            "usage": reply.get("usage", {}),
            "finish_reason": (reply.get("choices", [{}])[0] or {}).get("finish_reason"),
            "input_hash": _hash_request({"messages": payload["messages"]})[:12],
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_text(reply: dict[str, Any]) -> str:
    return reply["choices"][0]["message"]["content"]
