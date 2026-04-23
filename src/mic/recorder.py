"""Microphone recording — tap-to-talk and fixed-duration modes.

Uses `sounddevice` (PortAudio wrapper) to capture 16 kHz mono PCM,
which is what whisper-family STT models expect.

Two usage modes:

    # Fixed duration
    audio = record_fixed(seconds=10)

    # Tap-to-talk (press Enter twice)
    with Recorder() as rec:
        rec.start()
        input("press Enter to stop… ")
        audio = rec.stop()

The returned array is float32 with shape (N,) at 16 kHz. Save to
WAV via `scipy.io.wavfile.write` or pass directly to
`Transcriber.transcribe(audio)`.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"


def _require_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except ImportError as e:  # pragma: no cover — surfaced at runtime
        raise RuntimeError(
            "sounddevice is required for MIC recording. "
            "Install via `pip install sounddevice` (PortAudio must be on PATH)."
        ) from e
    return sd


def record_fixed(seconds: float, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Block for `seconds` and return a (N,) float32 array at 16 kHz mono."""
    sd = _require_sounddevice()
    buf = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    return buf.reshape(-1)


@dataclass
class Recorder:
    """Tap-to-talk recorder.

    Call `start()` to begin, `stop()` to finalize and receive the
    recorded audio as float32 numpy. Safe to reuse — each
    start/stop pair is an independent clip.
    """
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        self._stream = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._lock = threading.Lock()
        self._running = False

    def _callback(self, indata, frames, time, status):  # sounddevice signature
        if status:  # pragma: no cover — diagnostic only
            pass
        self._queue.put(indata.copy())

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        if self._running:
            self.stop()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            sd = _require_sounddevice()
            self._queue = queue.Queue()
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._callback,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> np.ndarray:
        with self._lock:
            if not self._running or self._stream is None:
                return np.zeros(0, dtype=np.float32)
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._running = False
        chunks: list[np.ndarray] = []
        while not self._queue.empty():
            chunks.append(self._queue.get_nowait())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0).reshape(-1)


def save_wav(audio: np.ndarray, path: str, *, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a (N,) float32 array to a 16-bit PCM WAV file."""
    from scipy.io import wavfile  # type: ignore

    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, pcm16)
