"""BM25 sparse retriever — drop-in alternative to the dense cosine retriever.

v0.3 Tier-1 · Task 2. Isolates the dense retriever's contribution against a
classic sparse baseline (the gap flagged in v0.2 §4.7 "BM25 baseline absent").

Everything downstream of retrieval is unchanged: the BM25 hits are emitted in
the SAME (score, text, timestamp, doc_id) dict shape the dense path produces in
src/eval/runner.py::_retrieve_for_item, so the runner's prompt assembly,
advisory policy, top-k=5, and scoring all stay identical.

Korean tokenization: kiwipiepy morphemes (same tokenizer for corpus + query).
If kiwipiepy is unavailable, falls back to char-bigrams and SETS .fallback so
the caller can log the downgrade (global constraint: log the fallback).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi


class _Tokenizer:
    """Morpheme tokenizer (kiwipiepy) with a char-bigram fallback."""

    def __init__(self) -> None:
        self.kind = "kiwi_morpheme"
        self.fallback = False
        self._kiwi = None
        try:
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi()
        except Exception as e:  # pragma: no cover - environment dependent
            self.kind = "char_bigram"
            self.fallback = True
            self._fallback_reason = repr(e)

    def __call__(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self._kiwi is not None:
            return [t.form for t in self._kiwi.tokenize(text)]
        # char-bigram fallback (whitespace-stripped)
        s = "".join(text.split())
        if len(s) < 2:
            return [s] if s else []
        return [s[i:i + 2] for i in range(len(s) - 1)]


@dataclass
class BM25Index:
    name: str
    docs: list[str]
    metadata: list[dict[str, Any]]
    bm25: BM25Okapi
    tokenizer: _Tokenizer


def build_bm25_index(name: str, docs: list[str], metadata: list[dict[str, Any]],
                     tokenizer: _Tokenizer) -> BM25Index:
    corpus_tokens = [tokenizer(d) for d in docs]
    # rank_bm25 needs every doc to have >=1 token; guard empties.
    corpus_tokens = [t if t else ["<empty>"] for t in corpus_tokens]
    bm25 = BM25Okapi(corpus_tokens)
    return BM25Index(name=name, docs=docs, metadata=metadata, bm25=bm25,
                     tokenizer=tokenizer)


def _doc_id(meta: dict[str, Any]) -> str:
    return meta.get("utt_id") or meta.get("session_id") or ""


def retrieve_bm25(idx: BM25Index, query: str, *, k: int = 5) -> list[dict[str, Any]]:
    """Return top-k BM25 hits in the dense-retriever dict shape."""
    if not idx.docs:
        return []
    q_tokens = idx.tokenizer(query) or ["<empty>"]
    scores = idx.bm25.get_scores(q_tokens)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    out = []
    for i in order:
        m = idx.metadata[i]
        out.append({
            "score": float(scores[i]),
            "text": idx.docs[i],
            "timestamp": m.get("timestamp", ""),
            "doc_id": _doc_id(m),
        })
    return out
