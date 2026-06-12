#!/usr/bin/env python3
"""v0.3 Tier-1 · Task 3 — answer-first guardrail on oracle (reversal SUTs).

Tests the v0.2 §4.2 falsifiable prediction: an answer-first guardrail should
close the retrieval>oracle reversal by stopping the model from re-narrating the
(now perfect) oracle context instead of answering.

Minimal-variant design: policy 'advisory_answerfirst' == ADVISORY with one
guardrail line prepended (src/mirror/prompts.py). Nothing else changes — oracle
condition, top-k cap, contains-token gold criterion, seed=42, max_tokens=200,
temperature=0.3 are all reused from v0.2.

Scope: condition=oracle only, policy=advisory_answerfirst, 4 reversal SUTs
(kanana_8b, kanana_8b_int4, qwen25_7b, bllossom_8b), ko_native 64 non-REFL.
Baseline (oracle advisory) and retrieval columns come from v0.2 results.jsonl
(no re-run needed). Local inference · 0 paid API. Run on GPU 7.

Out: experiments/20260612_v0.3_tier1/answerfirst_oracle.{jsonl,csv}
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import answer_contains_tokens
from src.eval.runner import TRACKS, _abs_passed, _retrieve_for_item, load_items
from src.mirror.generator import run_once
from src.node.store import Embedder
from src.utils.llm_backend import load_sut, unload
from src.utils.logging import get_logger

log = get_logger("task3_af")

OUT_DIR = Path("experiments/20260612_v0.3_tier1")
V02_RESULTS = Path("experiments/20260426_1242_v0.2_sweep_merged/results.jsonl")
K = 5
SEED = 42

REVERSAL_SUTS = [
    ("kanana_8b",      "kakaocorp/kanana-1.5-8b-instruct-2505", "fp16"),
    ("kanana_8b_int4", "kakaocorp/kanana-1.5-8b-instruct-2505", "int4"),
    ("qwen25_7b",      "Qwen/Qwen2.5-7B-Instruct",              "fp16"),
    ("bllossom_8b",    "MLP-KTLim/llama-3-Korean-Bllossom-8B",  "fp16"),
]


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


def char_4grams(text: str) -> set[str]:
    s = "".join((text or "").split())
    return {s[i:i + 4] for i in range(len(s) - 3)} if len(s) >= 4 else set()


def recap_share(resp: str, evidence: str) -> float:
    """Share of response 4-grams that also appear in oracle evidence."""
    rg = char_4grams(resp)
    if not rg:
        return 0.0
    eg = char_4grams(evidence)
    return len(rg & eg) / len(rg)


def load_v02(suts: set[str]) -> dict:
    """item_id -> {sut: {oracle_resp, oracle_pass, retrieval_pass}} from v0.2."""
    base = defaultdict(dict)        # (sut,item)->oracle response/pass
    pass_acc = defaultdict(lambda: [0, 0])  # (sut,cond)->[npass,n] over non-REFL
    with V02_RESULTS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["track"] != "ko_native" or r["sut"] not in suts:
                continue
            if r["policy"] != "advisory":
                continue
            if r["ability"] == "REFL":
                continue
            if r["condition"] in ("oracle", "retrieval"):
                pass_acc[(r["sut"], r["condition"])][0] += int(r["passed_contains"])
                pass_acc[(r["sut"], r["condition"])][1] += 1
            if r["condition"] == "oracle":
                base[(r["sut"], r["item_id"])] = {
                    "resp": r["response"], "passed": r["passed_contains"]}
    return base, pass_acc


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = TRACKS["ko_native"]
    items = [it for it in load_items(plan.items_path) if it.get("ability") != "REFL"]
    assert len(items) == 64, f"expected 64 non-REFL items, got {len(items)}"
    log.info("ko_native non-REFL items: %d", len(items))

    sut_names = {n for n, _, _ in REVERSAL_SUTS}
    v02_base, v02_pass = load_v02(sut_names)

    log.info("loading embedder (for oracle evidence assembly)…")
    embedder = Embedder()
    index_cache: dict = {}

    # Precompute oracle evidence docs per item (SUT-independent).
    oracle_docs: dict[str, list] = {}
    oracle_text: dict[str, str] = {}
    for it in items:
        docs, _ = _retrieve_for_item(it, plan=plan, embedder=embedder,
                                     index_cache=index_cache, k=K, condition="oracle")
        oracle_docs[it["item_id"]] = docs
        oracle_text[it["item_id"]] = " ".join(d.get("text", "") for d in docs)

    results_path = OUT_DIR / "answerfirst_oracle.jsonl"
    results_path.unlink(missing_ok=True)
    fout = results_path.open("a", encoding="utf-8")

    af_pass = defaultdict(lambda: [0, 0])       # sut -> [npass, n]
    af_len = defaultdict(list)                   # sut -> response lengths
    base_len = defaultdict(list)
    af_recap = defaultdict(list)
    base_recap = defaultdict(list)

    for name, hf_id, quant in REVERSAL_SUTS:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_sut = time.time()
        log.info("loading SUT %s (%s)…", name, quant)
        sut = load_sut(hf_id, name=name, quantization=quant)
        log.info("  loaded params=%.2fB device=%s", sut.params / 1e9, sut.model.device)

        for it in items:
            iid = it["item_id"]
            docs = oracle_docs[iid]
            t0 = time.time()
            resp = run_once(sut, question=it["question"]["text"],
                            retrieved=docs, policy="advisory_answerfirst", seed=SEED)
            lat = time.time() - t0
            passed = score_passed(it, resp)
            af_pass[name][0] += int(passed)
            af_pass[name][1] += 1
            af_len[name].append(len(resp))
            af_recap[name].append(recap_share(resp, oracle_text[iid]))

            # baseline mechanism from v0.2 stored response
            b = v02_base.get((name, iid))
            if b is not None:
                base_len[name].append(len(b["resp"]))
                base_recap[name].append(recap_share(b["resp"], oracle_text[iid]))

            fout.write(json.dumps({
                "track": "ko_native", "item_id": iid, "persona_id": it["persona_id"],
                "ability": it["ability"], "sut": name,
                "condition": "oracle", "policy": "advisory_answerfirst",
                "question": it["question"]["text"], "response": resp,
                "passed_contains": passed,
                "resp_len_chars": len(resp),
                "recap_share_4gram": recap_share(resp, oracle_text[iid]),
                "baseline_oracle_passed": (b["passed"] if b else None),
                "baseline_resp_len_chars": (len(b["resp"]) if b else None),
                "baseline_recap_share_4gram": (recap_share(b["resp"], oracle_text[iid]) if b else None),
                "gold_answer": str(it.get("gold_answer", {}).get("text", "")),
                "latency_s": lat,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, ensure_ascii=False) + "\n")
            fout.flush()

        peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        log.info("[%s] DONE wall=%.0fs GPU_peak=%.2fGB", name, time.time() - t_sut, peak)
        unload(sut)

    fout.close()

    # ---- aggregate + prediction verdict -----------------------------------
    import csv
    import statistics as st

    rows = []
    closed_fracs = []
    print("\n=== Task 3: answer-first guardrail on oracle (ko_native non-REFL n=64) ===")
    print(f"{'sut':<16s} {'or_base':>8s} {'or_af':>8s} {'retr':>8s}  "
          f"{'af-base':>8s} {'closed%':>8s}  {'len_b':>6s} {'len_af':>6s}  "
          f"{'rec_b':>6s} {'rec_af':>6s}")
    for name, _, _ in REVERSAL_SUTS:
        ob = v02_pass[(name, "oracle")][0] / v02_pass[(name, "oracle")][1]
        re_ = v02_pass[(name, "retrieval")][0] / v02_pass[(name, "retrieval")][1]
        oaf = af_pass[name][0] / af_pass[name][1]
        gap = re_ - ob
        closed = (oaf - ob) / gap if abs(gap) > 1e-9 else float("nan")
        if abs(gap) > 1e-9:
            closed_fracs.append(closed)
        lb = st.mean(base_len[name]) if base_len[name] else float("nan")
        laf = st.mean(af_len[name])
        rb = st.mean(base_recap[name]) if base_recap[name] else float("nan")
        raf = st.mean(af_recap[name])
        print(f"{name:<16s} {ob:>8.4f} {oaf:>8.4f} {re_:>8.4f}  "
              f"{oaf-ob:>+8.3f} {closed*100:>7.1f}%  {lb:>6.0f} {laf:>6.0f}  "
              f"{rb:>6.3f} {raf:>6.3f}")
        rows.append([name, round(ob, 4), round(oaf, 4), round(re_, 4),
                     round(oaf - ob, 4), round(closed, 4),
                     round(lb, 1), round(laf, 1), round(rb, 4), round(raf, 4)])

    # verdict heuristic
    deltas = [r[4] for r in rows]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    mean_closed = st.mean(closed_fracs) if closed_fracs else float("nan")
    if n_pos == len(rows) and mean_closed >= 0.5:
        verdict = "CONFIRMED"
    elif n_pos >= 1 and n_neg <= len(rows) - n_pos and any(d > 0 for d in deltas):
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"
    if n_pos == 0:
        verdict = "REFUTED"

    print(f"\nprediction verdict: {verdict}  "
          f"(deltas>0: {n_pos}/{len(rows)}, mean gap-closed: {mean_closed*100:.1f}%)")

    with (OUT_DIR / "answerfirst_oracle.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sut", "oracle_base", "oracle_af", "retrieval",
                    "delta_af_minus_base", "gap_closed_frac",
                    "mean_len_base", "mean_len_af",
                    "mean_recap_base", "mean_recap_af"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["prediction_verdict", verdict, f"deltas_pos={n_pos}/{len(rows)}",
                    f"mean_gap_closed={round(mean_closed,4)}", "", "", "", "", "", ""])

    summary = {
        "guardrail": "질문에 먼저 답하라. 과거를 다시 서술하지 마라. "
                     "(Answer the question first. Do not re-narrate the past.)",
        "policy": "advisory_answerfirst", "condition": "oracle",
        "seed": SEED, "k": K, "n_items_nonREFL": len(items),
        "prediction_verdict": verdict,
        "mean_gap_closed_frac": mean_closed,
        "per_sut": {r[0]: {"oracle_base": r[1], "oracle_af": r[2], "retrieval": r[3],
                           "delta": r[4], "gap_closed_frac": r[5],
                           "mean_len_base": r[6], "mean_len_af": r[7],
                           "mean_recap_base": r[8], "mean_recap_af": r[9]} for r in rows},
    }
    (OUT_DIR / "answerfirst_oracle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Task 3 done. verdict=%s", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
