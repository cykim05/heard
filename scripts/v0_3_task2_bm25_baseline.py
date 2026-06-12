#!/usr/bin/env python3
"""v0.3 Tier-1 · Task 2 — BM25 sparse retrieval baseline (anchor SUTs).

Isolates the dense retriever's contribution against a classic BM25 baseline
(v0.2 §4.7 flagged "BM25 baseline absent"). Only the retriever changes;
Algorithm-2 prompt assembly, advisory policy, top-k=5, per-persona ko_native
indices, and the contains-token gold criterion are all reused verbatim.

Scope: SUTs kanana_nano, kanana_8b · ko_native · 64 non-REFL items.
Conditions per item: no_node / bm25_retrieval / dense_retrieval (advisory).

INTEGRITY: before the sweep, re-run dense retrieval for kanana_nano and confirm
the v0.2 reported pass rate (0.1094 == 7/64) reproduces; also checks that the
dense top-5 doc-ids match v0.2's stored retrieved_doc_ids item-for-item. If the
pass rate does not reproduce, the script STOPS — that is a wiring problem.

Local inference only · 0 paid API calls. Run on GPU 7.

Out: experiments/20260612_v0.3_tier1/bm25_baseline.{jsonl,csv}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import answer_contains_tokens, recall_at_k
from src.eval.runner import TRACKS, _abs_passed, _retrieve_for_item, load_items
from src.mirror.generator import run_once
from src.node.bm25_retriever import build_bm25_index, retrieve_bm25, _Tokenizer
from src.node.store import Embedder, load_index
from src.utils.llm_backend import load_sut, unload
from src.utils.logging import get_logger

log = get_logger("task2_bm25")

OUT_DIR = Path("experiments/20260612_v0.3_tier1")
V02_RESULTS = Path("experiments/20260426_1242_v0.2_sweep_merged/results.jsonl")
K = 5
SEED = 42  # identical to v0.2 sweep (runner hardcodes seed=42)

ANCHOR_SUTS = [
    ("kanana_nano", "kakaocorp/kanana-1.5-2.1b-instruct-2505", "fp16"),
    ("kanana_8b",   "kakaocorp/kanana-1.5-8b-instruct-2505",   "fp16"),
]

# v0.2 reported dense-retrieval advisory pass rate, ko_native non-REFL (n=64).
V02_DENSE_RETRIEVAL = {"kanana_nano": 7 / 64, "kanana_8b": 12 / 64}  # 0.1094, 0.1875


def score_passed(item: dict, resp: str) -> bool:
    """Exact reuse of the v0.2 runner gold criterion (runner.py:186-197)."""
    gold = item.get("gold_answer", {})
    contains = gold.get("contains_tokens", []) or []
    excludes = gold.get("excludes_tokens", []) or []
    ability = item.get("ability", "")
    if ability == "ABS":
        return _abs_passed(resp)
    if contains:
        return answer_contains_tokens(resp, contains=contains, excludes=excludes)
    return False


def gold_evidence_ids(item: dict) -> list[str]:
    g = item.get("gold_answer", {})
    return list(g.get("evidence_utt_ids", []) + g.get("evidence_session_ids", []))


def load_v02_dense_ids() -> dict[str, list[str]]:
    """item_id -> retrieved_doc_ids from v0.2 kanana_nano dense retrieval (advisory)."""
    out: dict[str, list[str]] = {}
    with V02_RESULTS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r["track"] == "ko_native" and r["sut"] == "kanana_nano"
                    and r["condition"] == "retrieval" and r["policy"] == "advisory"):
                out[r["item_id"]] = r["retrieved_doc_ids"]
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = TRACKS["ko_native"]
    items = [it for it in load_items(plan.items_path) if it.get("ability") != "REFL"]
    log.info("ko_native non-REFL items: %d", len(items))
    assert len(items) == 64, f"expected 64 non-REFL items, got {len(items)}"

    log.info("loading embedder (dense)…")
    embedder = Embedder()
    index_cache: dict = {}

    # Build BM25 per persona from the SAME docs/metadata the dense index holds.
    tok = _Tokenizer()
    if tok.fallback:
        log.warning("kiwipiepy UNAVAILABLE — BM25 tokenizer fell back to %s (reason: %s)",
                    tok.kind, getattr(tok, "_fallback_reason", "?"))
    else:
        log.info("BM25 tokenizer = %s", tok.kind)
    bm25_cache: dict = {}
    personas = sorted(set(it["persona_id"] for it in items))
    for p in personas:
        idx = load_index(plan.index_dir / p, p)
        bm25_cache[p] = build_bm25_index(p, idx.docs, idx.metadata, tok)
        index_cache.setdefault(p, idx)  # prime dense cache too (same Index object)
    log.info("built BM25 indices for personas: %s", personas)

    # Precompute retrieval (SUT-independent): dense + bm25 top-5 + gold-hit@5.
    retr: dict[str, dict] = {}
    v02_dense_ids = load_v02_dense_ids()
    dense_id_match = dense_id_total = 0
    for it in items:
        gid = set(gold_evidence_ids(it))
        dense_docs, _ = _retrieve_for_item(
            it, plan=plan, embedder=embedder, index_cache=index_cache,
            k=K, condition="retrieval")
        dense_ids = [d["doc_id"] for d in dense_docs]
        bm25_docs = retrieve_bm25(bm25_cache[it["persona_id"]], it["question"]["text"], k=K)
        bm25_ids = [d["doc_id"] for d in bm25_docs]
        retr[it["item_id"]] = {
            "dense_docs": dense_docs, "dense_ids": dense_ids,
            "bm25_docs": bm25_docs, "bm25_ids": bm25_ids,
            "gold_ids": sorted(gid),
            "dense_hit5": int(len(gid & set(dense_ids)) > 0) if gid else None,
            "bm25_hit5": int(len(gid & set(bm25_ids)) > 0) if gid else None,
            "dense_recall5": recall_at_k(gid, dense_ids, k=K) if gid else None,
            "bm25_recall5": recall_at_k(gid, bm25_ids, k=K) if gid else None,
        }
        # retriever-wiring check vs v0.2 stored ids
        if it["item_id"] in v02_dense_ids:
            dense_id_total += 1
            dense_id_match += int(v02_dense_ids[it["item_id"]] == dense_ids)

    wiring_frac = dense_id_match / dense_id_total if dense_id_total else 0.0
    log.info("retriever-wiring check: dense top-5 ids match v0.2 on %d/%d items (%.1f%%)",
             dense_id_match, dense_id_total, 100 * wiring_frac)
    n_gold = sum(1 for it in items if gold_evidence_ids(it))
    dense_hit = sum(retr[it["item_id"]]["dense_hit5"] for it in items if gold_evidence_ids(it))
    bm25_hit = sum(retr[it["item_id"]]["bm25_hit5"] for it in items if gold_evidence_ids(it))
    log.info("gold-hit@5  dense=%d/%d (%.1f%%)  bm25=%d/%d (%.1f%%)",
             dense_hit, n_gold, 100 * dense_hit / n_gold,
             bm25_hit, n_gold, 100 * bm25_hit / n_gold)

    # ---- sweep ------------------------------------------------------------
    results_path = OUT_DIR / "bm25_baseline.jsonl"
    results_path.unlink(missing_ok=True)
    agg: dict = {}            # (sut,cond) -> [n_pass, n_total]
    fout = results_path.open("a", encoding="utf-8")

    for name, hf_id, quant in ANCHOR_SUTS:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_sut = time.time()
        log.info("loading SUT %s (%s)…", name, quant)
        sut = load_sut(hf_id, name=name, quantization=quant)
        log.info("  loaded params=%.2fB device=%s", sut.params / 1e9, sut.model.device)

        for it in items:
            r = retr[it["item_id"]]
            conds = [
                ("no_node", [], None, None),
                ("bm25_retrieval", r["bm25_docs"], r["bm25_ids"], r["bm25_hit5"]),
                ("dense_retrieval", r["dense_docs"], r["dense_ids"], r["dense_hit5"]),
            ]
            for cond_name, docs, ids, hit5 in conds:
                t0 = time.time()
                resp = run_once(sut, question=it["question"]["text"],
                                retrieved=docs, policy="advisory", seed=SEED)
                lat = time.time() - t0
                passed = score_passed(it, resp)
                agg.setdefault((name, cond_name), [0, 0])
                agg[(name, cond_name)][0] += int(passed)
                agg[(name, cond_name)][1] += 1
                fout.write(json.dumps({
                    "track": "ko_native", "item_id": it["item_id"],
                    "persona_id": it["persona_id"], "ability": it["ability"],
                    "sut": name, "condition": cond_name, "policy": "advisory",
                    "question": it["question"]["text"], "response": resp,
                    "retrieved_doc_ids": ids,
                    "gold_evidence_ids": r["gold_ids"],
                    "gold_hit_at5": hit5,
                    "gold_answer": str(it.get("gold_answer", {}).get("text", "")),
                    "passed_contains": passed, "latency_s": lat,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }, ensure_ascii=False) + "\n")
                fout.flush()

        peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        log.info("[%s] DONE  wall=%.0fs  GPU_peak=%.2fGB", name, time.time() - t_sut, peak)
        unload(sut)

    fout.close()

    # ---- aggregate + sanity ----------------------------------------------
    import csv
    sanity_ok = True
    rows_csv = []
    print("\n=== Task 2: BM25 vs dense (advisory, ko_native non-REFL n=64) ===")
    print(f"{'sut':<14s} {'no_node':>8s} {'bm25':>8s} {'dense':>8s}  "
          f"{'d-nn':>6s} {'bm-nn':>6s} {'d-bm':>6s}")
    for name, _, _ in ANCHOR_SUTS:
        nn = agg[(name, "no_node")][0] / agg[(name, "no_node")][1]
        bm = agg[(name, "bm25_retrieval")][0] / agg[(name, "bm25_retrieval")][1]
        de = agg[(name, "dense_retrieval")][0] / agg[(name, "dense_retrieval")][1]
        print(f"{name:<14s} {nn:>8.4f} {bm:>8.4f} {de:>8.4f}  "
              f"{de-nn:>+6.3f} {bm-nn:>+6.3f} {de-bm:>+6.3f}")
        for cond, val in (("no_node", nn), ("bm25_retrieval", bm), ("dense_retrieval", de)):
            ghit = ""
            if cond == "bm25_retrieval":
                ghit = round(bm25_hit / n_gold, 4)
            elif cond == "dense_retrieval":
                ghit = round(dense_hit / n_gold, 4)
            rows_csv.append([name, cond, agg[(name, cond)][1], round(val, 4),
                             agg[(name, cond)][0], ghit])
        # sanity: dense reproduces v0.2
        exp = V02_DENSE_RETRIEVAL[name]
        match = abs(de - exp) < 1e-9
        flag = "OK" if match else "MISMATCH"
        print(f"   sanity[{name}] dense={de:.4f} vs v0.2={exp:.4f}  [{flag}]")
        if name == "kanana_nano" and not match:
            sanity_ok = False

    with (OUT_DIR / "bm25_baseline.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sut", "condition", "n", "pass_rate", "n_pass", "gold_hit_at5"])
        w.writerows(rows_csv)
        w.writerow([])
        w.writerow(["retriever_quality", "metric", "value", "n_gold", "", ""])
        w.writerow(["dense", "gold_hit_at5", round(dense_hit / n_gold, 4), n_gold, "", ""])
        w.writerow(["bm25", "gold_hit_at5", round(bm25_hit / n_gold, 4), n_gold, "", ""])
        w.writerow(["dense", "wiring_match_vs_v0.2", round(wiring_frac, 4), dense_id_total, "", ""])

    summary = {
        "bm25_tokenizer": tok.kind, "tokenizer_fallback": tok.fallback,
        "k": K, "seed": SEED, "n_items_nonREFL": len(items), "n_gold_items": n_gold,
        "retriever_wiring_match_vs_v02": wiring_frac,
        "gold_hit_at5": {"dense": dense_hit / n_gold, "bm25": bm25_hit / n_gold},
        "pass_rate": {f"{s}|{c}": agg[(s, c)][0] / agg[(s, c)][1] for (s, c) in agg},
        "v02_dense_reference": V02_DENSE_RETRIEVAL,
        "sanity_dense_reproduces_v02": sanity_ok,
    }
    (OUT_DIR / "bm25_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwiring match vs v0.2 = {wiring_frac:.3f}  "
          f"gold-hit@5 dense={dense_hit/n_gold:.3f} bm25={bm25_hit/n_gold:.3f}")
    if not sanity_ok:
        log.error("SANITY FAILED: kanana_nano dense did not reproduce v0.2 — wiring problem.")
        return 2
    log.info("Task 2 done. sanity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
