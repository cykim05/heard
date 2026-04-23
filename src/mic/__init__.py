"""MIC — Heard's speech-to-text front-end.

Tap-to-talk sounddevice recorder + faster-whisper transcription.
Code ships with v0.1; STT WER evaluation is out of scope for the
v0.1 technical report (see report §2.2).
"""
from .recorder import Recorder, record_fixed
from .transcribe import Transcriber

__all__ = ["Recorder", "record_fixed", "Transcriber"]
