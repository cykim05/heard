#!/usr/bin/env python3
"""Tap-to-talk demo for Heard's MIC pillar.

This script is shipped as the reference integration between
`src.mic.Recorder` and `src.mic.Transcriber`. It is **not** executed
in the v0.1 evaluation pipeline — the benchmark in §3 of the report
operates on gold utterance text to isolate the memory-pipeline
question from STT error.

Usage:
    python scripts/00_record_and_transcribe.py \
        --model small \
        --device auto \
        --out-wav /tmp/utterance.wav \
        --out-text /tmp/utterance.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mic import Recorder, Transcriber
from src.mic.recorder import save_wav
from src.utils.logging import get_logger

log = get_logger("mic")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small",
                    help="Whisper model size: tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="auto",
                    help='"auto" | "cuda" | "cpu"')
    ap.add_argument("--language", default="ko")
    ap.add_argument("--out-wav", type=Path, default=None,
                    help="Optional — save raw WAV alongside transcript")
    ap.add_argument("--out-text", type=Path, default=None,
                    help="Optional — write transcript to this file")
    args = ap.parse_args()

    log.info("press Enter to start recording…")
    input()
    rec = Recorder()
    rec.start()
    log.info("recording… press Enter to stop")
    input()
    audio = rec.stop()
    log.info("captured %.2f s of audio", len(audio) / rec.sample_rate)

    if args.out_wav is not None:
        save_wav(audio, str(args.out_wav), sample_rate=rec.sample_rate)
        log.info("wav → %s", args.out_wav)

    log.info("loading whisper %s on %s…", args.model, args.device)
    tr = Transcriber(model_size=args.model, device=args.device, language=args.language)
    text = tr.transcribe(audio)
    log.info("transcript: %s", text)

    if args.out_text is not None:
        args.out_text.write_text(text, encoding="utf-8")
        log.info("text → %s", args.out_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
