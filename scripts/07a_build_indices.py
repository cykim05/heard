#!/usr/bin/env python3
"""Build embedding indices for the three tracks.

Outputs:
  data/indices/ko_native/<persona>.npy + .meta.jsonl    (per-persona corpus)
  data/indices/en_subset/<item_id>.npy + .meta.jsonl    (per-item haystack)
  data/indices/ko_translated/<item_id>.npy + .meta.jsonl (EN haystack, same shape)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.node.store import Embedder, build_index, save_index
from src.utils.config import REPO_ROOT
from src.utils.logging import get_logger

log = get_logger("build_idx")


def _session_text(session: dict) -> str:
    turns = session.get("turns") or session.get("turns_ko") or []
    lines = []
    for t in turns:
        if isinstance(t, dict):
            lines.append(f"{t.get('role','user')}: {t.get('content','')}")
        else:
            lines.append(str(t))
    return "\n".join(lines)


def build_ko_native_indices(embedder: Embedder, out_dir: Path, corpus_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_persona: dict[str, list[dict]] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by_persona.setdefault(o["persona_id"], []).append(o)
    for pid, utts in by_persona.items():
        docs = [u["text"] for u in utts]
        meta = [{"utt_id": u["utt_id"], "day": u["day"], "timestamp": u["timestamp"]} for u in utts]
        idx = build_index(embedder, name=pid, docs=docs, metadata=meta)
        save_index(idx, out_dir / pid)
        log.info("ko_native[%s]: %d utts embedded", pid, len(docs))


def build_haystack_indices(
    embedder: Embedder, out_dir: Path, items_path: Path, track_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with items_path.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    for item in items:
        iid = item["item_id"]
        sessions = item.get("history_sessions", [])
        docs = [_session_text(s) for s in sessions]
        meta = [{
            "session_id": s.get("session_id"),
            "timestamp": s.get("timestamp"),
            "is_evidence": s.get("is_evidence", False),
        } for s in sessions]
        idx = build_index(embedder, name=iid, docs=docs, metadata=meta)
        save_index(idx, out_dir / iid)
    log.info("%s: %d items, avg %.1f sessions", track_name, len(items),
             sum(len(i.get("history_sessions", [])) for i in items) / max(1, len(items)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "data/raw/utterances.jsonl")
    ap.add_argument("--en-subset", type=Path, default=REPO_ROOT / "data/final/en_subset/test.jsonl")
    ap.add_argument("--ko-translated", type=Path, default=REPO_ROOT / "data/final/ko_translated/test.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/indices")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    log.info("loading embedder…")
    embedder = Embedder()
    log.info("device: %s", embedder._device)  # noqa: SLF001

    if "ko_native" not in args.skip:
        build_ko_native_indices(embedder, args.out_dir / "ko_native", args.corpus)
    if "en_subset" not in args.skip and args.en_subset.exists():
        build_haystack_indices(embedder, args.out_dir / "en_subset",
                               args.en_subset, "en_subset")
    if "ko_translated" not in args.skip and args.ko_translated.exists():
        build_haystack_indices(embedder, args.out_dir / "ko_translated",
                               args.ko_translated, "ko_translated")
    log.info("DONE — indices under %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
