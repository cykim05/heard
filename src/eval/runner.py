"""Sweep runner — iterate track × SUT × condition × policy × item.

Each run emits a line to results.jsonl:
  {track, item_id, sut, condition, policy, question, response,
   retrieved_ids, gold, metric_flags, latency_s}

Scoring is lightweight and fast — substring match for
contains/excludes tokens. REFL items go to the judge pipeline in
a second pass (scripts/08_run_judge.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.metrics import answer_contains_tokens
from ..mirror.generator import run_once
from ..node.retriever import retrieve
from ..node.store import Embedder, Index, load_index
from ..utils.llm_backend import LoadedModel
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RunResult:
    track: str
    item_id: str
    persona_id: str | None
    ability: str
    sut: str
    condition: str        # "no_node" | "retrieval" | "oracle"
    policy: str           # "advisory" | "reflective"
    question: str
    response: str
    retrieved_doc_count: int
    retrieved_doc_ids: list[str]
    gold_answer: str
    passed_contains: bool
    latency_s: float
    generated_at: str


def _abs_passed(response: str) -> bool:
    """Crude 'abstention' check — the response indicates 'don't know'."""
    low = response.lower()
    markers = ["모른다", "모르겠", "없어", "없습니다", "don't know", "do not know",
               "not sure", "i'm not sure", "no information", "unable to answer"]
    return any(m in low for m in markers)


@dataclass
class SweepPlan:
    track: str
    items_path: Path
    index_dir: Path
    index_key: str         # "persona_id" or "item_id" — how to look up the right index


TRACKS = {
    "ko_native":     SweepPlan("ko_native", Path("data/final/ko_native/test.jsonl"),
                               Path("data/indices/ko_native"), index_key="persona_id"),
    "en_subset":     SweepPlan("en_subset", Path("data/final/en_subset/test.jsonl"),
                               Path("data/indices/en_subset"), index_key="item_id"),
    "ko_translated": SweepPlan("ko_translated", Path("data/final/ko_translated/test.jsonl"),
                               Path("data/indices/ko_translated"), index_key="item_id"),
}


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _retrieve_for_item(
    item: dict[str, Any],
    *,
    plan: SweepPlan,
    embedder: Embedder,
    index_cache: dict[str, Index],
    k: int,
    condition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (retrieved_docs, telemetry)."""
    if condition == "no_node":
        return [], {}
    # Choose index key
    if plan.index_key == "persona_id":
        key = item["persona_id"]
    else:
        key = item["item_id"]

    if key not in index_cache:
        idx_path = plan.index_dir / key
        if not idx_path.with_suffix(".npy").exists():
            return [], {}
        index_cache[key] = load_index(idx_path, key)

    idx = index_cache[key]
    question = item["question"]["text"]

    if condition == "retrieval":
        hits = retrieve(idx, question, embedder=embedder, k=k)
        out = []
        for score, doc, meta in hits:
            out.append({
                "score": score,
                "text": doc,
                "timestamp": meta.get("timestamp", ""),
                "doc_id": meta.get("utt_id") or meta.get("session_id") or "",
            })
        return out, {}

    if condition == "oracle":
        gold = item.get("gold_answer", {})
        gold_ids = set(
            gold.get("evidence_utt_ids", [])  # ko_native
            + gold.get("evidence_session_ids", [])  # en_subset / ko_translated
        )
        out = []
        for d, m in zip(idx.docs, idx.metadata):
            did = m.get("utt_id") or m.get("session_id")
            if did in gold_ids:
                out.append({
                    "score": 1.0,
                    "text": d,
                    "timestamp": m.get("timestamp", ""),
                    "doc_id": did,
                })
        return out[:k * 3], {}  # oracle can exceed k; cap for prompt length

    raise ValueError(f"unknown condition {condition!r}")


def run_sweep(
    *,
    sut: LoadedModel,
    embedder: Embedder,
    out_path: Path,
    tracks: list[str],
    conditions: list[str],
    policies: list[str],
    k: int = 5,
    max_items_per_track: int | None = None,
    log_every: int = 20,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx_cache: dict[str, Index] = {}

    total_runs = 0
    t_start = time.time()
    with out_path.open("a", encoding="utf-8") as fout:
        for track in tracks:
            plan = TRACKS[track]
            items = load_items(plan.items_path)
            if max_items_per_track:
                items = items[:max_items_per_track]
            log.info("track=%s items=%d SUT=%s", track, len(items), sut.name)

            for i, item in enumerate(items):
                for cond in conditions:
                    retrieved, telemetry = _retrieve_for_item(
                        item, plan=plan, embedder=embedder,
                        index_cache=idx_cache, k=k, condition=cond,
                    )
                    for policy in policies:
                        # Reflective policy on no_node is a degenerate case; skip.
                        if policy == "reflective" and cond == "no_node":
                            continue
                        t0 = time.time()
                        resp = run_once(
                            sut,
                            question=item["question"]["text"],
                            retrieved=retrieved,
                            policy=policy,
                            seed=42,
                        )
                        latency = time.time() - t0

                        gold = item.get("gold_answer", {})
                        contains = gold.get("contains_tokens", []) or []
                        excludes = gold.get("excludes_tokens", []) or []
                        ability = item.get("ability", "")
                        if ability == "ABS":
                            passed = _abs_passed(resp)
                        elif contains:
                            passed = answer_contains_tokens(
                                resp, contains=contains, excludes=excludes,
                            )
                        else:
                            passed = False  # REFL / other — judged separately

                        result = RunResult(
                            track=track,
                            item_id=item["item_id"],
                            persona_id=item.get("persona_id"),
                            ability=ability,
                            sut=sut.name,
                            condition=cond,
                            policy=policy,
                            question=item["question"]["text"],
                            response=resp,
                            retrieved_doc_count=len(retrieved),
                            retrieved_doc_ids=[r["doc_id"] for r in retrieved],
                            gold_answer=str(gold.get("text", "")),
                            passed_contains=passed,
                            latency_s=latency,
                            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        )
                        fout.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                        fout.flush()
                        total_runs += 1

                        if total_runs % log_every == 0:
                            elapsed = time.time() - t_start
                            log.info(
                                "runs=%d  elapsed=%.0fs  rate=%.2f/s",
                                total_runs, elapsed, total_runs / max(1, elapsed),
                            )

    log.info("sweep DONE: %d total runs, %.0fs wall", total_runs, time.time() - t_start)
    return total_runs
