#!/usr/bin/env python3
"""v0.3 Tier-1 · Task 1 — Judge reliability (position-consistency) report.

Pure re-analysis of the v0.2 pairwise judge verdicts. NO new judge calls,
NO GPU, NO paid API. Anchored to Zheng et al. 2023 (MT-Bench) — position
bias in LLM-as-a-judge.

Input  : experiments/20260426_1242_v0.2_sweep_merged/judge_verdicts.jsonl
Output : experiments/20260612_v0.3_tier1/judge_reliability.{csv,json}

Stored-verdict semantics (see src/eval/judge.py::judge_pair):
  Both swap=False and swap=True rows are stored in CANONICAL space, i.e.
  the swap=True row already had inv({A<->B}) applied before writing. So in
  the file, for EVERY row:  "A" == advisory wins, "B" == reflective wins.

  - position-consistency: for a given (item,sut,condition,judge,rubric),
    the two presentation orders (swap False/True) AGREE on the canonical
    label.  consistent  <=>  canonical_false == canonical_true.
  - positional preference: reconstruct the PHYSICAL choice the judge made
    (which on-screen slot, A=first or B=second). swap=False physical ==
    canonical; swap=True physical == inv(canonical). Share picking slot A
    (among non-ties); 0.5 == unbiased.
  - inter-judge agreement: consolidate each judge to one label per
    (item,sut,condition,rubric) — the agreed label if both orders agree,
    else 'tie' (inconclusive) — then raw agreement + Cohen's kappa, j1 vs j2.

NB: repetition-stability is intentionally NOT computed — identical judge
calls hit the diskcache, so a re-run is a tautology (global constraint).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.judge import RUBRICS  # reuse the canonical rubric tuple

V02_VERDICTS = Path("experiments/20260426_1242_v0.2_sweep_merged/judge_verdicts.jsonl")
OUT_DIR = Path("experiments/20260612_v0.3_tier1")

INV = {"A": "B", "B": "A", "tie": "tie"}


def cohen_kappa(labels_a: list[str], labels_b: list[str], categories: list[str]) -> float:
    """Cohen's kappa for two raters over a fixed category set."""
    n = len(labels_a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    # marginal probabilities
    pa = {c: labels_a.count(c) / n for c in categories}
    pb = {c: labels_b.count(c) / n for c in categories}
    pe = sum(pa[c] * pb[c] for c in categories)
    if pe == 1.0:
        return 1.0  # perfect-and-degenerate; avoid div-by-zero
    return (po - pe) / (1.0 - pe)


def main() -> int:
    if not V02_VERDICTS.exists():
        raise SystemExit(f"missing input: {V02_VERDICTS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in V02_VERDICTS.open(encoding="utf-8") if l.strip()]
    n_parse_failed = sum(1 for r in rows if r.get("parse_failed"))

    # Pair the two swap orders per (item,sut,condition,judge).
    pairs: dict[tuple, dict[bool, dict]] = defaultdict(dict)
    for r in rows:
        key = (r["item_id"], r["sut"], r["condition"], r["judge"])
        pairs[key][bool(r["swap"])] = r["scores"]

    # ---- (a) position-consistency + (b) positional preference -------------
    consist_hit = defaultdict(int)   # bucket -> n consistent
    consist_tot = defaultdict(int)
    posA = defaultdict(int)          # bucket -> n physical picks of slot A (non-tie)
    pos_decided = defaultdict(int)   # bucket -> n non-tie physical picks

    for (item, sut, cond, judge), by_swap in pairs.items():
        if False not in by_swap or True not in by_swap:
            continue  # incomplete pair — skip (none expected)
        s_false, s_true = by_swap[False], by_swap[True]
        for rub in RUBRICS:
            cf, ct = s_false[rub], s_true[rub]
            # consistency buckets
            for b in ("overall", f"rubric:{rub}", f"judge:{judge}",
                      f"judge_rubric:{judge}:{rub}"):
                consist_tot[b] += 1
                consist_hit[b] += int(cf == ct)
            # positional preference — reconstruct physical slot choices
            phys_false = cf            # swap=False: stored == physical
            phys_true = INV[ct]        # swap=True : physical = inv(stored)
            for phys in (phys_false, phys_true):
                if phys == "tie":
                    continue
                for b in ("overall", f"rubric:{rub}", f"judge:{judge}"):
                    pos_decided[b] += 1
                    posA[b] += int(phys == "A")

    # ---- (c) inter-judge agreement (consolidated label per judge) ---------
    # consolidated[(item,sut,cond,rubric)][judge] = label | 'tie'(inconsistent)
    consolidated: dict[tuple, dict[str, str]] = defaultdict(dict)
    for (item, sut, cond, judge), by_swap in pairs.items():
        if False not in by_swap or True not in by_swap:
            continue
        for rub in RUBRICS:
            cf, ct = by_swap[False][rub], by_swap[True][rub]
            consolidated[(item, sut, cond, rub)][judge] = cf if cf == ct else "tie"

    cats = ["A", "B", "tie"]
    inter_by_rubric: dict[str, dict] = {}
    all_a, all_b = [], []
    per_rub_lists: dict[str, tuple[list, list]] = {r: ([], []) for r in RUBRICS}
    for (item, sut, cond, rub), jl in consolidated.items():
        if "j1_gpt4o_mini" in jl and "j2_claude_haiku" in jl:
            la, lb = jl["j1_gpt4o_mini"], jl["j2_claude_haiku"]
            per_rub_lists[rub][0].append(la)
            per_rub_lists[rub][1].append(lb)
            all_a.append(la)
            all_b.append(lb)
    for rub in RUBRICS:
        la, lb = per_rub_lists[rub]
        raw = sum(1 for x, y in zip(la, lb) if x == y) / len(la) if la else float("nan")
        inter_by_rubric[rub] = {
            "raw_agreement": raw,
            "cohen_kappa": cohen_kappa(la, lb, cats),
            "n": len(la),
        }
    inter_overall = {
        "raw_agreement": sum(1 for x, y in zip(all_a, all_b) if x == y) / len(all_a),
        "cohen_kappa": cohen_kappa(all_a, all_b, cats),
        "n": len(all_a),
    }

    # ---- assemble ----------------------------------------------------------
    def rate(hit, tot, b):
        return hit[b] / tot[b] if tot[b] else None

    result = {
        "source": str(V02_VERDICTS),
        "n_verdict_rows": len(rows),
        "n_swap_pairs": len(pairs),
        "n_parse_failed": n_parse_failed,
        "judges": ["j1_gpt4o_mini", "j2_claude_haiku"],
        "rubrics": list(RUBRICS),
        "conditions_pooled": sorted(set(r["condition"] for r in rows)),
        "position_consistency": {
            "overall": rate(consist_hit, consist_tot, "overall"),
            "by_rubric": {r: rate(consist_hit, consist_tot, f"rubric:{r}") for r in RUBRICS},
            "by_judge": {j: rate(consist_hit, consist_tot, f"judge:{j}")
                         for j in ("j1_gpt4o_mini", "j2_claude_haiku")},
            "by_judge_rubric": {
                f"{j}:{r}": rate(consist_hit, consist_tot, f"judge_rubric:{j}:{r}")
                for j in ("j1_gpt4o_mini", "j2_claude_haiku") for r in RUBRICS
            },
            "n_decisions_per_order": consist_tot["overall"],
        },
        "positional_preference": {
            "note": "share of PHYSICAL slot-A picks among non-tie decisions; 0.5 == unbiased",
            "overall_position_A_share": (posA["overall"] / pos_decided["overall"]
                                         if pos_decided["overall"] else None),
            "n_decided": pos_decided["overall"],
            "by_judge": {j: (posA[f"judge:{j}"] / pos_decided[f"judge:{j}"]
                             if pos_decided[f"judge:{j}"] else None)
                         for j in ("j1_gpt4o_mini", "j2_claude_haiku")},
            "by_rubric": {r: (posA[f"rubric:{r}"] / pos_decided[f"rubric:{r}"]
                              if pos_decided[f"rubric:{r}"] else None) for r in RUBRICS},
        },
        "inter_judge_agreement": {
            "overall": inter_overall,
            "by_rubric": inter_by_rubric,
        },
    }

    (OUT_DIR / "judge_reliability.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # tidy long CSV
    with (OUT_DIR / "judge_reliability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "group_type", "group", "value", "n"])
        pc = result["position_consistency"]
        w.writerow(["position_consistency", "overall", "ALL", round(pc["overall"], 4),
                    pc["n_decisions_per_order"]])
        for r, v in pc["by_rubric"].items():
            w.writerow(["position_consistency", "rubric", r, round(v, 4),
                        consist_tot[f"rubric:{r}"]])
        for j, v in pc["by_judge"].items():
            w.writerow(["position_consistency", "judge", j, round(v, 4),
                        consist_tot[f"judge:{j}"]])
        pp = result["positional_preference"]
        w.writerow(["positional_preference_A_share", "overall", "ALL",
                    round(pp["overall_position_A_share"], 4), pp["n_decided"]])
        for j, v in pp["by_judge"].items():
            w.writerow(["positional_preference_A_share", "judge", j, round(v, 4),
                        pos_decided[f"judge:{j}"]])
        for r, v in pp["by_rubric"].items():
            w.writerow(["positional_preference_A_share", "rubric", r, round(v, 4),
                        pos_decided[f"rubric:{r}"]])
        ia = result["inter_judge_agreement"]
        w.writerow(["inter_judge_raw_agreement", "overall", "ALL",
                    round(ia["overall"]["raw_agreement"], 4), ia["overall"]["n"]])
        w.writerow(["inter_judge_cohen_kappa", "overall", "ALL",
                    round(ia["overall"]["cohen_kappa"], 4), ia["overall"]["n"]])
        for r, v in ia["by_rubric"].items():
            w.writerow(["inter_judge_raw_agreement", "rubric", r,
                        round(v["raw_agreement"], 4), v["n"]])
            w.writerow(["inter_judge_cohen_kappa", "rubric", r,
                        round(v["cohen_kappa"], 4), v["n"]])

    # console summary
    print("=== Task 1: Judge reliability ===")
    print(f"rows={len(rows)} swap_pairs={len(pairs)} parse_failed={n_parse_failed}")
    print(f"position-consistency overall = {pc['overall']:.3f} "
          f"(n={pc['n_decisions_per_order']} per order)")
    for r, v in pc["by_rubric"].items():
        print(f"  consistency[{r:<20s}] = {v:.3f}")
    for j, v in pc["by_judge"].items():
        print(f"  consistency[{j:<16s}] = {v:.3f}")
    print(f"positional A-share overall = {pp['overall_position_A_share']:.3f} "
          f"(non-tie n={pp['n_decided']}; 0.5==unbiased)")
    print(f"inter-judge raw agreement = {ia['overall']['raw_agreement']:.3f}  "
          f"kappa = {ia['overall']['cohen_kappa']:.3f}  (n={ia['overall']['n']})")
    for r, v in ia["by_rubric"].items():
        print(f"  agree[{r:<20s}] raw={v['raw_agreement']:.3f} kappa={v['cohen_kappa']:.3f}")
    print(f"wrote {OUT_DIR/'judge_reliability.json'} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
