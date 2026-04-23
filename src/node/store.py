"""Embedding-based retrieval store.

Day 3 MVP keeps this simple — no ChromaDB, no networkx relation graph.
Each track gets an in-memory (matrix, metadata) pair:

  ko_native:       per-persona indices of utterances
  en_subset:       per-item index of haystack sessions
  ko_translated:   per-item index of (EN) haystack sessions

Retrieval = cosine top-k against the matrix. The 5-node NODE schema
of PLAN §3.B is deferred to a follow-up (see report Discussion).

Embedding model: `intfloat/multilingual-e5-small` — 118 M params,
~230 MB, handles KO + EN cross-lingual similarity well enough for
our haystack sizes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
_E5_DIM = 384


@dataclass
class Index:
    name: str
    docs: list[str]
    metadata: list[dict[str, Any]] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # (n, dim) normalized

    def __len__(self) -> int:
        return len(self.docs)


class Embedder:
    """Small wrapper over sentence-transformers-style e5 call.

    We load the HF model directly via transformers (no sentence-transformers
    dep) to keep the stack thin. e5 models expect "query: …" / "passage: …"
    prefixes for best performance.
    """
    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
        self._model = AutoModel.from_pretrained(EMBEDDING_MODEL_ID).to(self._device)
        self._model.eval()

    def _mean_pool(self, last_hidden: "Tensor", attn_mask: "Tensor") -> "Tensor":
        mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def embed(self, texts: Iterable[str], *, kind: str = "passage", batch_size: int = 64) -> np.ndarray:
        torch = self._torch
        prefix = f"{kind}: "
        batch_in = [prefix + (t or "").strip() for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(batch_in), batch_size):
                chunk = batch_in[i : i + batch_size]
                enc = self._tok(chunk, padding=True, truncation=True, max_length=512,
                                return_tensors="pt").to(self._device)
                h = self._model(**enc).last_hidden_state
                emb = self._mean_pool(h, enc["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out.append(emb.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, _E5_DIM), dtype=np.float32)


def build_index(
    embedder: Embedder,
    name: str,
    docs: list[str],
    metadata: list[dict[str, Any]] | None = None,
) -> Index:
    meta = metadata or [{} for _ in docs]
    if len(meta) != len(docs):
        raise ValueError("docs/metadata length mismatch")
    embeddings = embedder.embed(docs, kind="passage") if docs else np.zeros((0, _E5_DIM))
    return Index(name=name, docs=docs, metadata=meta, embeddings=embeddings)


def save_index(idx: Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), idx.embeddings)
    meta_path = path.with_suffix(".meta.jsonl")
    with meta_path.open("w", encoding="utf-8") as f:
        for i, (doc, m) in enumerate(zip(idx.docs, idx.metadata)):
            f.write(json.dumps({"doc": doc, "meta": m}, ensure_ascii=False) + "\n")


def load_index(path: Path, name: str) -> Index:
    emb = np.load(path.with_suffix(".npy"))
    docs, meta = [], []
    with path.with_suffix(".meta.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs.append(o["doc"])
            meta.append(o["meta"])
    return Index(name=name, docs=docs, metadata=meta, embeddings=emb)
