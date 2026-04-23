"""Top-k cosine retriever over a single Index."""
from __future__ import annotations

import numpy as np

from .store import Embedder, Index


def retrieve(
    idx: Index,
    query: str,
    *,
    embedder: Embedder,
    k: int = 5,
) -> list[tuple[float, str, dict]]:
    """Return the top-k (score, doc, metadata) for a single query string."""
    if idx.embeddings is None or len(idx) == 0:
        return []
    q = embedder.embed([query], kind="query")[0]
    scores = idx.embeddings @ q  # both L2-normalized -> dot product is cosine
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), idx.docs[i], idx.metadata[i]) for i in top]
