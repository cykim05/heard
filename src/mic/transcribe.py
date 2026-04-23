"""Whisper-based Korean transcription for Heard's MIC pillar.

Uses `faster-whisper` (CTranslate2 backend) — 4–5× faster than the
reference implementation on the same GPU and runs cleanly on CPU
for on-device simulation. The default model is `small` (244 M
parameters); `medium` is a drop-in upgrade for harder audio.

Usage:

    from src.mic import Recorder, Transcriber
    tr = Transcriber(model_size="small", device="cuda")
    with Recorder() as rec:
        rec.start(); input(); audio = rec.stop()
    text = tr.transcribe(audio)

No evaluation of WER is performed in v0.1 — the text-level
benchmark in §3 of the report operates directly on gold utterance
text to isolate the memory-pipeline question from STT error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_MODEL = "small"
DEFAULT_LANG = "ko"


def _require_faster_whisper():
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is required for MIC transcription. "
            "Install via `pip install faster-whisper`. "
            "On GPU it also needs CUDA libraries loadable by CTranslate2."
        ) from e
    return WhisperModel


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


class Transcriber:
    """Thin wrapper over faster-whisper.

    The heavy model load happens lazily on first call so importing
    this module is cheap even in environments that will not use STT.
    """
    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str | None = None,
        language: str = DEFAULT_LANG,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type or ("float16" if device != "cpu" else "int8")
        self.language = language
        self._model: Any | None = None

    def _lazy_load(self) -> Any:
        if self._model is None:
            WhisperModel = _require_faster_whisper()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> str:
        """Return a single concatenated Korean transcript for the audio."""
        model = self._lazy_load()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_segments(
        self,
        audio: np.ndarray,
        *,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> list[TranscriptSegment]:
        """Return per-segment timestamps + text (useful for utterance split)."""
        model = self._lazy_load()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        return [
            TranscriptSegment(
                start_s=float(seg.start),
                end_s=float(seg.end),
                text=seg.text.strip(),
            )
            for seg in segments
        ]
