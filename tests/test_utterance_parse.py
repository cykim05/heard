"""Offline parse tests for the utterance generator — no API."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from src.datagen.utterance_gen import (
    GeneratorRotation,
    _extract_json,
    _parse_utterances,
)


SEOUL = timezone(timedelta(hours=9))


def test_extract_json_strips_markdown_fence():
    raw = '```json\n{"utterances":[]}\n```'
    assert _extract_json(raw) == '{"utterances":[]}'


def test_extract_json_picks_first_object_from_prose():
    raw = 'Sure, here is the answer:\n{"utterances":[{"a":1}]}\nHope this helps.'
    obj = _extract_json(raw)
    assert obj.startswith("{") and obj.endswith("}")


def test_parse_handles_24_00_time():
    """Models sometimes emit '24:00' meaning midnight of the next day."""
    raw = (
        '{"utterances":[{"time":"24:00","text":"오늘 끝","intended_categories":["mood"],'
        '"intended_topics":[],"references_historical_event":[]}]}'
    )
    base = datetime(2026, 2, 23, tzinfo=SEOUL)
    out = _parse_utterances(
        raw,
        persona_id="yejin_florist",
        day=7,
        base_date=base,
        source_events=[],
        generator_name="g2_gpt4o",
        generator_model_id="openai/gpt-4o",
        seed=42,
        start_index=0,
    )
    assert len(out) == 1
    # 24:00 on day 7 rolls to 00:00 on day 8 (calendar day, still tied to utt day=7).
    expected = base + timedelta(days=7)  # day-7 base is base+6 days; +1 more for wrap = +7
    assert out[0].timestamp.startswith(expected.date().isoformat())


def test_rotation_auto_discovers_renamed_generators():
    """Regression: default include= hardcoded old names and silently dropped
    renamed generators, so all traffic went to whichever was still named
    'g1_claude'. Auto-discovery now covers every openrouter-transport gen."""
    models = {
        "generators": {
            "g1_claude": {"transport": "openrouter", "model_id": "a/b", "share": 0.3},
            "g2_gpt4o_mini": {"transport": "openrouter", "model_id": "c/d", "share": 0.3},
            "g3_gemini_flash": {"transport": "openrouter", "model_id": "e/f", "share": 0.4},
            "g4_local": {"transport": "local_hf", "model_id": "x/y", "share": 0.1},
        }
    }
    rot = GeneratorRotation.from_models_yaml(models, seed=0)
    assert len(rot.generators) == 3
    names = {name for name, _ in rot.generators}
    assert names == {"g1_claude", "g2_gpt4o_mini", "g3_gemini_flash"}
    # weights renormalize to sum 1
    assert abs(sum(rot.weights) - 1.0) < 1e-9
    # Weighted, consecutive-different picks over 200 draws should hit all 3.
    picks = {rot.pick()[0] for _ in range(200)}
    assert picks == names


def test_parse_clamps_bad_minute_and_hour():
    raw = '{"utterances":[{"time":"25:99","text":"x","intended_categories":[],"intended_topics":[]}]}'
    base = datetime(2026, 2, 23, tzinfo=SEOUL)
    out = _parse_utterances(
        raw,
        persona_id="p",
        day=1,
        base_date=base,
        source_events=[],
        generator_name="g",
        generator_model_id="x/y",
        seed=0,
        start_index=0,
    )
    assert len(out) == 1  # does not raise
