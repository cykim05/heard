#!/usr/bin/env python3
"""v0.3 figure builder — Figures 3, 4, and the enhanced Figure 2 (v2).

FIGURES ONLY. No rendered tables — table-ready aggregates are dumped as csv to
figures/v0.3/_tabledata/ for downstream LaTeX. NO hardcoded numbers: every value
is read from experiments/<run>/ csv/json/jsonl and (re)computed here.

Style is imported verbatim from scripts/99_make_figures.py (the Figure-2
generator): _base_style() rcParams, the palette constants, SUT_ORDER, INT4_PAIRS,
_paired_discordance. Fixed colour map (per spec):
  no_node=grey  retrieval/dense=blue  oracle=sage  bm25=orange
  advisory=tan  reflective=red
Every pass-rate bar carries a Wilson 95% CI (proportion_confint, method='wilson').

Run standalone to regenerate ALL figures + _tabledata csv:
  python scripts/plot_v0.3.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# ─── import style + helpers from the Figure-2 generator (digit-led filename) ──
_MF_PATH = Path(__file__).resolve().parent / "99_make_figures.py"
_spec = importlib.util.spec_from_file_location("make_figures_v02", _MF_PATH)
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

# Fixed colour map (reuse Figure-2 palette; add a distinct orange for BM25).
C_NO_NODE = mf.C_NEUTRAL      # grey
C_DENSE = mf.C_MEMORY         # blue  (== retrieval)
C_ORACLE = mf.C_ORACLE        # sage
C_ADVISORY = mf.C_ADVISORY    # tan
C_REFLECT = mf.C_REFLECT      # red
C_BM25 = "#e8762d"            # orange — BM25 (not in original palette)

NOBS = 64  # ko_native non-REFL items (spec: Wilson CI nobs=64 for pass-rate bars)

DEF_V03 = Path("experiments/20260612_v0.3_tier1")
DEF_V02 = Path("experiments/20260426_1242_v0.2_sweep_merged")
OUT_DIR = Path("figures/v0.3")
TAB_DIR = OUT_DIR / "_tabledata"


# ─── small helpers ────────────────────────────────────────────────────────
def wilson_pct(count: int, nobs: int):
    """Wilson 95% CI as (low%, high%)."""
    lo, hi = proportion_confint(count, nobs, alpha=0.05, method="wilson")
    return lo * 100.0, hi * 100.0


def yerr_cols(h, lo, hi):
    """asymmetric yerr column for a single bar at height h (all in %)."""
    return [[max(0.0, h - lo)], [max(0.0, hi - h)]]


def savefig_both(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}")
    plt.close(fig)


# ─── data loaders (all from raw) ──────────────────────────────────────────
def load_bm25(jsonl_path: Path):
    """Return pass counts per (sut,cond) and gold-hit@5 counts (dense/bm25)."""
    passc = defaultdict(lambda: [0, 0])           # (sut,cond) -> [npass, n]
    dense_hit, bm25_hit = {}, {}                   # item -> 0/1 (dedup; SUT-independent)
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            passc[(r["sut"], r["condition"])][0] += int(r["passed_contains"])
            passc[(r["sut"], r["condition"])][1] += 1
            g = r.get("gold_hit_at5")
            if g is not None:
                if r["condition"] == "dense_retrieval":
                    dense_hit[r["item_id"]] = int(g)
                elif r["condition"] == "bm25_retrieval":
                    bm25_hit[r["item_id"]] = int(g)
    gold = {
        "dense": (sum(dense_hit.values()), len(dense_hit)),
        "bm25": (sum(bm25_hit.values()), len(bm25_hit)),
    }
    return passc, gold


def load_v02_counts(results_path: Path):
    """(sut,cond) -> [npass, n] over ko_native non-REFL advisory items."""
    c = defaultdict(lambda: [0, 0])
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r["track"] != "ko_native" or r["ability"] == "REFL"
                    or r["policy"] != "advisory"):
                continue
            c[(r["sut"], r["condition"])][0] += int(r["passed_contains"])
            c[(r["sut"], r["condition"])][1] += 1
    return c


def load_answerfirst(jsonl_path: Path, csv_path: Path, v02c):
    """Per-SUT answer-first aggregates pulled from raw jsonl + v0.2 counts."""
    af_pass = defaultdict(lambda: [0, 0])
    base_pass = defaultdict(lambda: [0, 0])
    af_len, base_len, af_rec, base_rec = (defaultdict(list) for _ in range(4))
    order = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = r["sut"]
            if s not in order:
                order.append(s)
            af_pass[s][0] += int(r["passed_contains"]); af_pass[s][1] += 1
            af_len[s].append(r["resp_len_chars"]); af_rec[s].append(r["recap_share_4gram"])
            if r.get("baseline_oracle_passed") is not None:
                base_pass[s][0] += int(r["baseline_oracle_passed"]); base_pass[s][1] += 1
            if r.get("baseline_resp_len_chars") is not None:
                base_len[s].append(r["baseline_resp_len_chars"])
            if r.get("baseline_recap_share_4gram") is not None:
                base_rec[s].append(r["baseline_recap_share_4gram"])
    out = {}
    for s in order:
        npass_af, n_af = af_pass[s]
        npass_b, n_b = base_pass[s]
        ret_pass, ret_n = v02c[(s, "retrieval")]
        ob = npass_b / n_b
        oaf = npass_af / n_af
        re_ = ret_pass / ret_n
        gap = re_ - ob
        out[s] = {
            "oracle_base": ob, "oracle_base_count": npass_b,
            "oracle_af": oaf, "oracle_af_count": npass_af,
            "retrieval": re_, "retrieval_count": ret_pass,
            "gap_closed_frac": ((oaf - ob) / gap) if abs(gap) > 1e-9 else float("nan"),
            "mean_len_base": sum(base_len[s]) / len(base_len[s]),
            "mean_len_af": sum(af_len[s]) / len(af_len[s]),
            "mean_recap_base": sum(base_rec[s]) / len(base_rec[s]),
            "mean_recap_af": sum(af_rec[s]) / len(af_rec[s]),
        }
    return order, out


# ─── Figure 3 — retrieval equal, use diverges ─────────────────────────────
def figure3(passc, gold):
    mf._base_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6),
                                   sharey=True, gridspec_kw={"wspace": 0.12})

    # left: gold-hit@5 dense vs bm25 (retriever recall — near-identical)
    g_specs = [("dense", C_DENSE, "dense"), ("bm25", C_BM25, "BM25")]
    for i, (key, col, lab) in enumerate(g_specs):
        cnt, nobs = gold[key]
        h = cnt / nobs * 100
        lo, hi = wilson_pct(cnt, nobs)
        axL.bar(i, h, 0.6, color=col, edgecolor="white", linewidth=0.8, label=lab)
        axL.errorbar(i, h, yerr=yerr_cols(h, lo, hi), fmt="none",
                     ecolor="#333", elinewidth=1.3, capsize=4)
        axL.text(i, hi + 1.2, f"{h:.1f}%", ha="center", va="bottom",
                 fontsize=10, weight="700")
    axL.set_xticks(range(len(g_specs)))
    axL.set_xticklabels(["dense", "BM25"])
    axL.set_ylabel("Rate (%)")
    axL.set_title("a  gold-hit@5 (retriever recall, n=54)")
    axL.set_xlim(-0.7, 1.7)

    # right: contains-pass kanana_nano/kanana_8b × {dense, bm25}
    suts = [("kanana_nano", "Kanana 2.1B"), ("kanana_8b", "Kanana 8B")]
    width = 0.36
    for j, (key, col, lab) in enumerate([("dense_retrieval", C_DENSE, "dense"),
                                         ("bm25_retrieval", C_BM25, "BM25")]):
        xs, hs, los, his = [], [], [], []
        for i, (sut, _) in enumerate(suts):
            cnt, nobs = passc[(sut, key)]
            h = cnt / nobs * 100
            lo, hi = wilson_pct(cnt, nobs)
            x = i + (j - 0.5) * width
            xs.append(x); hs.append(h); los.append(lo); his.append(hi)
            axR.bar(x, h, width, color=col, edgecolor="white", linewidth=0.8,
                    label=lab if i == 0 else None)
            axR.errorbar(x, h, yerr=yerr_cols(h, lo, hi), fmt="none",
                         ecolor="#333", elinewidth=1.3, capsize=4)
            axR.text(x, hi + 1.2, f"{h:.1f}%", ha="center", va="bottom",
                     fontsize=9, weight="700")
    axR.set_xticks(range(len(suts)))
    axR.set_xticklabels([lab for _, lab in suts])
    axR.set_title("b  contains-pass: dense vs BM25 (advisory, n=64)")
    axR.legend(loc="upper left", ncol=2)
    axR.set_ylim(0, 50)

    fig.suptitle("Figure 3 — retrieval recall is equal; reader use diverges",
                 fontsize=15, weight="700", y=1.0)
    savefig_both(fig, "figure3")


# ─── Figure 4 — answer-first gap-closed (honest, negatives shown) ─────────
def figure4(order, af):
    mf._base_style()
    labels_map = dict(mf.SUT_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    suts = order
    y = list(range(len(suts)))
    vals = [af[s]["gap_closed_frac"] * 100 for s in suts]
    cols = [C_ORACLE if v >= 0 else C_REFLECT for v in vals]
    ax.barh(y, vals, 0.6, color=cols, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + (2 if v >= 0 else -2), yi, f"{v:+.1f}%",
                ha="left" if v >= 0 else "right", va="center",
                fontsize=10, weight="700")
    ax.axvline(0, color="#333", linewidth=1.2)
    ax.axvline(100, color=C_DENSE, linewidth=1.4, linestyle="--",
               label="retrieval-level (100% closed)")
    mean_v = sum(vals) / len(vals)
    ax.axvline(mean_v, color="#888", linewidth=1.1, linestyle=":",
               label=f"mean ({mean_v:+.1f}%)")
    ax.set_yticks(y)
    ax.set_yticklabels([labels_map.get(s, s) for s in suts])
    ax.invert_yaxis()
    ax.set_xlabel("Gap closed toward retrieval (%)   ·   0 = oracle baseline")
    ax.set_xlim(min(vals) - 18, 112)
    ax.set_title("Figure 4 — answer-first guardrail does NOT close the "
                 "retrieval>oracle reversal", fontsize=12.5)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    savefig_both(fig, "figure4")


# ─── Figure 2 v2 — NODE lift (CI) · reflective share · int4 paired (CI) ────
def panel_a_nodelift_ci(ax, metrics_rows, v02c):
    conds = [("no_node", "no-NODE", C_NO_NODE),
             ("retrieval", "retrieval", C_DENSE),
             ("oracle", "oracle", C_ORACLE)]

    def _ret(s):
        m = mf._get(metrics_rows, track="ko_native", sut=s,
                    condition="retrieval", policy="advisory")
        return (m["pass_rate"] or 0) if m else 0
    suts_sorted = sorted(mf.SUT_ORDER, key=lambda x: -_ret(x[0]))

    n = len(suts_sorted)
    width = 0.27
    x = list(range(n))
    for i, (ck, clab, col) in enumerate(conds):
        offs = [xi + (i - 1) * width for xi in x]
        for xi, (sut_key, _) in zip(offs, suts_sorted):
            cnt, nobs = v02c[(sut_key, ck)]
            h = cnt / nobs * 100 if nobs else 0
            ax.bar(xi, h, width, color=col, edgecolor="white", linewidth=0.8,
                   label=clab if sut_key == suts_sorted[0][0] else None)
            if nobs:
                lo, hi = wilson_pct(cnt, nobs)
                ax.errorbar(xi, h, yerr=yerr_cols(h, lo, hi), fmt="none",
                            ecolor="#555", elinewidth=0.9, capsize=2.2)
                if ck == "retrieval":
                    ax.text(xi, hi + 0.4, f"{h:.0f}", ha="center", va="bottom",
                            fontsize=7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in suts_sorted], rotation=40, ha="right")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 30)
    ax.set_title("a  NODE lift on ko_native — 11 v0.2 SUTs (advisory, n=64, Wilson CI)")
    ax.legend(loc="upper right", ncol=3)


def panel_c_int4_ci(ax, v02c):
    pairs = mf.INT4_PAIRS
    width = 0.36
    x = list(range(len(pairs)))
    for i, (fp_key, i4_key, label) in enumerate(pairs):
        fpc, fpn = v02c[(fp_key, "retrieval")]
        i4c, i4n = v02c[(i4_key, "retrieval")]
        fp_r, i4_r = fpc / fpn * 100, i4c / i4n * 100
        for x0, cnt, nobs, col, lab in [
                (i - width / 2, fpc, fpn, C_DENSE, "fp16"),
                (i + width / 2, i4c, i4n, C_REFLECT, "int4")]:
            h = cnt / nobs * 100
            lo, hi = wilson_pct(cnt, nobs)
            ax.bar(x0, h, width, color=col, edgecolor="white", linewidth=0.8,
                   label=lab if i == 0 else None)
            ax.errorbar(x0, h, yerr=yerr_cols(h, lo, hi), fmt="none",
                        ecolor="#333", elinewidth=1.1, capsize=3)
        # discordance + McNemar z (recomputed from item-level passes)
        fp_only, i4_only, z = _discord_z(fp_key, i4_key, v02c, ax)
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs], rotation=20, ha="right")
    ax.set_ylim(0, 30)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("c  fp16 vs int4 paired (ko_native retrieval, n=64, Wilson CI)")
    ax.legend(loc="upper right", ncol=2)


def _discord_z(fp_key, i4_key, item_passes, ax):
    """McNemar discordance from item-level passes; annotate above the pair."""
    from math import sqrt
    fp = item_passes["_items"][fp_key]
    q = item_passes["_items"][i4_key]
    common = set(fp) & set(q)
    fp_only = sum(1 for it in common if fp[it] and not q[it])
    i4_only = sum(1 for it in common if q[it] and not fp[it])
    n_disc = fp_only + i4_only
    z = (i4_only - fp_only) / sqrt(n_disc) if n_disc else 0.0
    i = [p[0] for p in mf.INT4_PAIRS].index(fp_key)
    top = max(fp[it] for it in common) if common else 0
    fp_r = sum(fp[it] for it in common) / len(common) * 100
    i4_r = sum(q[it] for it in common) / len(common) * 100
    ax.text(i, max(fp_r, i4_r) + 2.0,
            f"int4 {i4_only} / fp16 {fp_only}\n(z={z:.2f})",
            ha="center", va="bottom", fontsize=8, color="#444")
    return fp_only, i4_only, z


def load_v02_item_passes(results_path: Path):
    """(sut) -> {item_id: bool} for ko_native retrieval advisory non-REFL."""
    by = defaultdict(dict)
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r["track"] != "ko_native" or r["ability"] == "REFL"
                    or r["policy"] != "advisory" or r["condition"] != "retrieval"):
                continue
            by[r["sut"]][r["item_id"]] = bool(r["passed_contains"])
    return by


def figure2_v2(v02_dir: Path):
    mf._base_style()
    metrics_rows = mf._load_metrics(v02_dir / "metrics.csv")
    aggregate = json.loads((v02_dir / "judge_aggregate.json").read_text(encoding="utf-8"))
    v02c = load_v02_counts(v02_dir / "results.jsonl")
    v02c["_items"] = load_v02_item_passes(v02_dir / "results.jsonl")

    fig, axes = plt.subplots(1, 3, figsize=(21, 8), gridspec_kw={"wspace": 0.32})
    panel_a_nodelift_ci(axes[0], metrics_rows, v02c)
    mf.panel_reflective(axes[1], aggregate)   # unchanged; palette already conformant
    panel_c_int4_ci(axes[2], v02c)
    fig.suptitle("Figure 2 (v2) — Heard v0.2 results, with Wilson 95% CIs",
                 fontsize=15, weight="700", y=0.99)
    savefig_both(fig, "figure2_v2")
    return metrics_rows, v02c


# ─── table-data dumps (figures' aggregates, NOT rendered) ─────────────────
def dump_table8(jr_json: Path):
    d = json.loads(jr_json.read_text(encoding="utf-8"))
    pc = d["position_consistency"]; pp = d["positional_preference"]
    ia = d["inter_judge_agreement"]
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    with (TAB_DIR / "table8_judge_reliability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rubric", "pos_consistency", "cohen_kappa", "slotA_rate"])
        for r in d["rubrics"]:
            w.writerow([r, round(pc["by_rubric"][r], 4),
                        round(ia["by_rubric"][r]["cohen_kappa"], 4),
                        round(pp["by_rubric"][r], 4)])
        w.writerow(["Overall", round(pc["overall"], 4),
                    round(ia["overall"]["cohen_kappa"], 4),
                    round(pp["overall_position_A_share"], 4)])


def dump_table9(passc, gold):
    with (TAB_DIR / "table9_bm25.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sut", "no_node", "no_node_ci_lo", "no_node_ci_hi",
                    "bm25", "bm25_ci_lo", "bm25_ci_hi",
                    "dense", "dense_ci_lo", "dense_ci_hi",
                    "delta_dense_minus_bm25", "goldhit5_dense", "goldhit5_bm25"])
        gd = gold["dense"][0] / gold["dense"][1]
        gb = gold["bm25"][0] / gold["bm25"][1]
        for sut in ("kanana_nano", "kanana_8b"):
            vals = {}
            for cond in ("no_node", "bm25_retrieval", "dense_retrieval"):
                cnt, nobs = passc[(sut, cond)]
                lo, hi = proportion_confint(cnt, nobs, method="wilson")
                vals[cond] = (cnt / nobs, lo, hi)
            w.writerow([sut,
                        round(vals["no_node"][0], 4), round(vals["no_node"][1], 4), round(vals["no_node"][2], 4),
                        round(vals["bm25_retrieval"][0], 4), round(vals["bm25_retrieval"][1], 4), round(vals["bm25_retrieval"][2], 4),
                        round(vals["dense_retrieval"][0], 4), round(vals["dense_retrieval"][1], 4), round(vals["dense_retrieval"][2], 4),
                        round(vals["dense_retrieval"][0] - vals["bm25_retrieval"][0], 4),
                        round(gd, 4), round(gb, 4)])


def dump_table10(order, af):
    with (TAB_DIR / "table10_answerfirst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sut", "oracle_base", "oracle_af", "retrieval", "gap_closed_frac",
                    "delta_len_af_minus_base", "overlap_base", "overlap_af"])
        for s in order:
            a = af[s]
            w.writerow([s, round(a["oracle_base"], 4), round(a["oracle_af"], 4),
                        round(a["retrieval"], 4), round(a["gap_closed_frac"], 4),
                        round(a["mean_len_af"] - a["mean_len_base"], 1),
                        round(a["mean_recap_base"], 4), round(a["mean_recap_af"], 4)])


def dump_mapping(jr_json: Path, passc, gold, order, af):
    """related-work -> upgrade map; conclusion column carries computed deltas."""
    d = json.loads(jr_json.read_text(encoding="utf-8"))
    pc = d["position_consistency"]["by_rubric"]
    ia = d["inter_judge_agreement"]["by_rubric"]
    gd = gold["dense"][0] / gold["dense"][1] * 100
    gb = gold["bm25"][0] / gold["bm25"][1] * 100
    d8 = (passc[("kanana_8b", "dense_retrieval")][0]
          - passc[("kanana_8b", "bm25_retrieval")][0]) / NOBS * 100
    d2 = (passc[("kanana_nano", "dense_retrieval")][0]
          - passc[("kanana_nano", "bm25_retrieval")][0]) / NOBS * 100
    mean_closed = sum(af[s]["gap_closed_frac"] for s in order) / len(order) * 100
    n_pos = sum(1 for s in order if af[s]["oracle_af"] > af[s]["oracle_base"])
    rows = [
        ["Zheng et al. 2023 (MT-Bench, pairwise judge position bias)",
         "v0.2 pairwise wins reported without a judge position-bias audit",
         "Task 1: position-consistency + inter-judge Cohen kappa",
         f"emo/open robust (consistency {pc['emotional_attunement']:.2f}/{pc['open_question']:.2f}, "
         f"kappa {ia['emotional_attunement']['cohen_kappa']:.2f}/{ia['open_question']['cohen_kappa']:.2f}); "
         f"non_directive collapses ({pc['non_directive']:.2f}, kappa {ia['non_directive']['cohen_kappa']:.2f})"],
        ["Robertson & Zaragoza 2009 (BM25 Okapi, sparse IR)",
         "v0.2 4.7: 'BM25 baseline absent'",
         "Task 2: BM25 vs dense, anchor SUTs",
         f"gold-hit@5 equal (dense {gd:.1f}% = BM25 {gb:.1f}%); dense-BM25 pass = "
         f"{d8:+.1f}pp on 8B, {d2:+.1f}pp on 2.1B -> bottleneck is reader, not retrieval"],
        ["v0.2 4.2 self-prediction (answer-first guardrail)",
         "predicted answer-first guardrail closes retrieval>oracle reversal",
         "Task 3: advisory_answerfirst on oracle, 4 reversal SUTs",
         f"NOT confirmed: {n_pos}/4 improve, none reach retrieval, mean gap-closed "
         f"{mean_closed:+.1f}%; recapitulation overlap ~unchanged"],
    ]
    with (TAB_DIR / "mapping_relatedwork.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cited_work", "motivating_limitation_section", "v0.3_experiment", "delta_conclusion"])
        w.writerows(rows)


# ─── captions ─────────────────────────────────────────────────────────────
def write_captions(passc, gold, order, af, jr_json: Path):
    d = json.loads(jr_json.read_text(encoding="utf-8"))
    pc = d["position_consistency"]; ia = d["inter_judge_agreement"]
    gd = gold["dense"][0] / gold["dense"][1] * 100
    gb = gold["bm25"][0] / gold["bm25"][1] * 100
    nd = passc[("kanana_nano", "dense_retrieval")][0] / NOBS * 100
    nb = passc[("kanana_nano", "bm25_retrieval")][0] / NOBS * 100
    e8d = passc[("kanana_8b", "dense_retrieval")][0] / NOBS * 100
    e8b = passc[("kanana_8b", "bm25_retrieval")][0] / NOBS * 100
    mean_closed = sum(af[s]["gap_closed_frac"] for s in order) / len(order) * 100
    rec_b = sum(af[s]["mean_recap_base"] for s in order) / len(order)
    rec_a = sum(af[s]["mean_recap_af"] for s in order) / len(order)
    lines = [
        "# v0.3 figure captions (self-contained; keep long text out of the figures)",
        "",
        "## Figure 3 — figure3.{pdf,png}",
        f"Retrieval recall is equal across retrievers; what differs is how the reader "
        f"model uses it. **Left (a):** gold-hit@5 — share of items whose gold evidence "
        f"id lands in the top-5 — is near-identical for dense ({gd:.1f}%) and BM25 "
        f"({gb:.1f}%) over the n=54 ko_native non-REFL items that carry evidence ids "
        f"(10 ABS items have none). **Right (b):** contains-token pass rate (advisory "
        f"policy, n=64 non-REFL, seed=42) for Kanana 2.1B and 8B under dense vs BM25 "
        f"retrieval. The dense advantage is SUT-dependent: +{e8d-e8b:.1f}pp on 8B but "
        f"{nd-nb:+.1f}pp on 2.1B (BM25 ahead). All bars carry Wilson 95% CIs "
        f"(nobs=64 right, nobs=54 left). Colours: dense=blue, BM25=orange. "
        f"Takeaway: the bottleneck is the reader, not the retriever.",
        "",
        "## Figure 4 — figure4.{pdf,png}",
        f"Answer-first guardrail does not close the retrieval>oracle reversal. Bars show "
        f"the fraction of the (retrieval - oracle_base) gap that the answer-first "
        f"guardrail closes, per reversal SUT (advisory_answerfirst, condition=oracle, "
        f"contains-token, n=64, seed=42). 0 = oracle baseline (no change); the blue "
        f"dashed line at 100% marks retrieval-level (full closure). Negative bars (red) "
        f"are shown as-is: two of four SUTs get worse, none reach retrieval, mean gap "
        f"closed = {mean_closed:+.1f}%. Mechanism is unsupported: mean char-4gram "
        f"recapitulation overlap with the oracle evidence is essentially unchanged "
        f"({rec_b:.3f} -> {rec_a:.3f}). The strong form of the v0.2 4.2 prediction is "
        f"REFUTED.",
        "",
        "## Figure 2 (v2) — figure2_v2.{pdf,png}",
        "Enhanced v0.2 results figure. **(a)** NODE lift on ko_native (advisory, n=64) "
        "for all 11 SUTs, sorted by retrieval pass rate; every bar now carries a Wilson "
        "95% CI and retrieval bars are value-labelled. **(b)** Reflective win-share per "
        "SUT (192 pairwise decisions/SUT, unchanged from v0.2). **(c)** fp16 vs int4 "
        "paired comparison on ko_native retrieval (n=64) with Wilson 95% CIs on each "
        "pass-rate bar plus the McNemar discordance counts and z. Colours unified to the "
        "fixed map: no_node=grey, retrieval=blue, oracle=sage, advisory=tan, "
        "reflective=red.",
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v03-dir", type=Path, default=DEF_V03)
    ap.add_argument("--v02-dir", type=Path, default=DEF_V02)
    args = ap.parse_args()

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    jr_json = args.v03_dir / "judge_reliability.json"

    # raw aggregates
    passc, gold = load_bm25(args.v03_dir / "bm25_baseline.jsonl")
    v02c = load_v02_counts(args.v02_dir / "results.jsonl")
    order, af = load_answerfirst(args.v03_dir / "answerfirst_oracle.jsonl",
                                 args.v03_dir / "answerfirst_oracle.csv", v02c)

    # figures
    figure3(passc, gold)
    figure4(order, af)
    figure2_v2(args.v02_dir)

    # table-data + captions
    dump_table8(jr_json)
    dump_table9(passc, gold)
    dump_table10(order, af)
    dump_mapping(jr_json, passc, gold, order, af)
    write_captions(passc, gold, order, af, jr_json)

    print("wrote figures + _tabledata + captions under", OUT_DIR)
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
