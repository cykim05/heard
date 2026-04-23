"""Utterance generator with G1/G2/G3 rotation through OpenRouter.

Design (DATASET.md §4–5, IMPL_DETAILS.md §2.4):
- One generator call per (persona, day) block. The generator produces
  10–15 utterances covering the day's events.
- Round-robin across G1/G2/G3 per persona; consecutive days must use
  different generators (anti-fingerprint, DATASET §4.2).
- G4 Kanana is deferred — its 20% share is redistributed across G1/G2/G3.
- Response format is a JSON object with `{"utterances": [...]}` so
  response_format json_object works on all three providers.
- Throttle: sleep_between_calls defaults to 0.5s to stay far under
  per-minute rate caps. Tenacity in OpenRouterClient already backs off
  on 429 with Retry-After.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..utils.logging import get_logger
from ..utils.openrouter import OpenRouterClient, extract_text

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")

log = get_logger(__name__)

SEOUL = timezone(timedelta(hours=9))

SYSTEM_PROMPT = (
    "당신은 한국 1인 자영업자의 **혼잣말 생성기**입니다. "
    "페르소나 카드와 오늘의 이벤트를 보고, 그 사람이 실제로 내뱉을 법한 "
    "짧은 혼잣말을 만드세요. "
    "규칙: (1) 각 발화는 1–3문장, 구어체, 자연스러운 줄임말 허용. "
    "(2) 감정·판단·의문이 섞여야 함. 뉴스 요약조 금지. "
    "(3) 같은 날의 발화들은 주제가 이어지되 시점(아침·영업 중·마감 후)이 달라야 함. "
    "(4) 이벤트의 topic_key를 1개 이상 intended_topics 에 남겨 주세요 — "
    "나중에 retrieval gold label로 씁니다. "
    "(5) 응답은 반드시 JSON 객체 `{\"utterances\": [...]}` 형식으로만 주세요. "
    "(6) 각 utterance 객체의 필드: time (HH:MM, 24h), text, intended_categories "
    "(customer/stock/pricing/mood/decision 중 해당되는 것), intended_topics "
    "(topic_key 리스트), references_historical_event (기억하는 과거 사건 힌트, 없으면 빈 리스트)."
)


USER_TEMPLATE = """\
## 페르소나 카드
```yaml
{persona_yaml}
```

## 오늘이 며칠째인지
Day {day} of 60 (base date {base_date}, so today = {today_date})

## 오늘의 이벤트 블록
```json
{events_json}
```

## 오늘까지 누적 topic_key 목록 (intended_topics 후보)
{topic_keys_str}

## 과거 사건 참조용 (persona.historical_events, days_ago 기준)
```json
{historical_json}
```

## 요청
오늘 하루의 혼잣말 {target_count}개를 생성하세요.
- 이벤트 트리거당 1–3개의 발화를 분산시키되, 이벤트와 무관한 배경 발화도 2–3개 포함 가능.
- 반드시 이벤트의 links_back_to_day / links_back_historical 을 **일부 발화가 자연스럽게 회상**하도록 하세요 (그래야 retrieval test hook이 성립합니다).
- 응답은 JSON 객체 `{{"utterances": [...]}}` 만.
"""


@dataclass
class Utterance:
    utt_id: str
    persona_id: str
    day: int
    timestamp: str
    text: str
    intended_categories: list[str] = field(default_factory=list)
    intended_topics: list[str] = field(default_factory=list)
    references_historical_event: list[Any] = field(default_factory=list)
    source_event_topic_keys: list[str] = field(default_factory=list)
    generator: str = ""
    generator_model_id: str = ""
    seed: int = 0


@dataclass
class GeneratorRotation:
    """Round-robin rotation ensuring consecutive calls use different models."""

    generators: list[tuple[str, str]]  # (name, openrouter_model_id)
    weights: list[float]
    _last_idx: int = -1
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    @classmethod
    def from_models_yaml(
        cls,
        models: dict[str, Any],
        *,
        include: Iterable[str] | None = None,
        seed: int = 0,
    ) -> "GeneratorRotation":
        """Build a rotation. Passing include=None auto-picks every generator
        with transport=='openrouter', so renaming keys in models.yaml no longer
        silently drops them.
        """
        gens = models["generators"]
        keys = list(include) if include is not None else list(gens.keys())
        pairs, weights = [], []
        total_share = 0.0
        for name in keys:
            spec = gens.get(name)
            if spec is None or spec.get("transport") != "openrouter":
                continue
            pairs.append((name, spec["model_id"]))
            weights.append(float(spec.get("share", 1.0)))
            total_share += float(spec.get("share", 1.0))
        if not pairs:
            raise ValueError("No OpenRouter generators available in models.yaml")
        weights = [w / total_share for w in weights]
        return cls(generators=pairs, weights=weights, _rng=random.Random(seed))

    def pick(self) -> tuple[str, str]:
        # Weighted choice, but reject if same as last. If all generators
        # are the last one (shouldn't happen with len>=2), just return.
        for _ in range(8):
            idx = self._rng.choices(
                range(len(self.generators)), weights=self.weights, k=1
            )[0]
            if idx != self._last_idx:
                self._last_idx = idx
                return self.generators[idx]
        idx = (self._last_idx + 1) % len(self.generators)
        self._last_idx = idx
        return self.generators[idx]


def _extract_json(raw: str) -> str:
    """Strip markdown fences and trailing prose, keep first JSON object."""
    s = _FENCE_RE.sub("", raw).strip()
    # Some models wrap in prose; grab the first {...} block.
    m = _JSON_OBJ_RE.search(s)
    if m:
        return m.group(0)
    return s


def _parse_utterances(
    raw: str,
    *,
    persona_id: str,
    day: int,
    base_date: datetime,
    source_events: list[dict[str, Any]],
    generator_name: str,
    generator_model_id: str,
    seed: int,
    start_index: int,
) -> list[Utterance]:
    obj = json.loads(_extract_json(raw))
    items = obj.get("utterances", [])
    if not isinstance(items, list):
        raise ValueError(f"utterances field is not a list: {type(items)}")

    day_date = base_date + timedelta(days=day - 1)
    source_topic_keys = [ev["topic_key"] for ev in source_events]
    out: list[Utterance] = []
    for i, it in enumerate(items):
        time_str = str(it.get("time", "22:00"))
        try:
            hh, mm = [int(x) for x in time_str.split(":")[:2]]
        except Exception:
            hh, mm = 22, 0
        # Models sometimes emit "24:00" (meant as midnight) or out-of-range minutes.
        if hh >= 24:
            hh, day_shift = hh - 24, 1
        else:
            day_shift = 0
        hh = max(0, min(23, hh))
        mm = max(0, min(59, mm))
        ts = (day_date + timedelta(days=day_shift)).replace(
            hour=hh, minute=mm, second=0, microsecond=0
        )
        out.append(
            Utterance(
                utt_id=f"{persona_id}_u_{start_index + i:05d}",
                persona_id=persona_id,
                day=day,
                timestamp=ts.isoformat(),
                text=str(it.get("text", "")).strip(),
                intended_categories=list(it.get("intended_categories") or []),
                intended_topics=list(it.get("intended_topics") or []),
                references_historical_event=list(it.get("references_historical_event") or []),
                source_event_topic_keys=source_topic_keys,
                generator=generator_name,
                generator_model_id=generator_model_id,
                seed=seed,
            )
        )
    return out


def generate_for_day(
    client: OpenRouterClient,
    *,
    persona: dict[str, Any],
    day: int,
    events: list[dict[str, Any]],
    rotation: GeneratorRotation,
    base_date: datetime,
    target_count: int = 12,
    seed: int = 42,
    start_index: int = 0,
    sleep_between_calls: float = 0.5,
) -> list[Utterance]:
    persona_id = persona["persona"]["id"]
    gen_name, model_id = rotation.pick()
    historical = persona.get("historical_events", [])

    topic_keys = sorted({ev["topic_key"] for ev in events})
    user_msg = USER_TEMPLATE.format(
        persona_yaml=yaml.safe_dump(persona, allow_unicode=True, sort_keys=False),
        day=day,
        base_date=base_date.date().isoformat(),
        today_date=(base_date + timedelta(days=day - 1)).date().isoformat(),
        events_json=json.dumps(events, ensure_ascii=False, indent=2),
        topic_keys_str=", ".join(topic_keys) or "(none — produce background utterances only)",
        historical_json=json.dumps(historical, ensure_ascii=False, indent=2),
        target_count=target_count,
    )

    reply = client.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
        seed=seed,
        max_tokens=4000,
        tag=f"utt_gen:{persona_id}:day{day}:{gen_name}",
    )
    raw = extract_text(reply)
    try:
        utterances = _parse_utterances(
            raw,
            persona_id=persona_id,
            day=day,
            base_date=base_date,
            source_events=events,
            generator_name=gen_name,
            generator_model_id=model_id,
            seed=seed,
            start_index=start_index,
        )
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("parse failed for %s day=%d gen=%s: %s", persona_id, day, gen_name, e)
        dump_dir = Path("experiments/_parse_failures")
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / f"{persona_id}_day{day}_{gen_name}.txt").write_text(raw, encoding="utf-8")
        utterances = []

    time.sleep(sleep_between_calls)
    return utterances


def save_jsonl(items: list[Utterance], path: Path, *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    from dataclasses import asdict

    with path.open(mode, encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
