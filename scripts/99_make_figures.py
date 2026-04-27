#!/usr/bin/env python3
"""Render the v0.2 results figure for the report.

Reads `experiments/<run_id>/metrics.csv`, `judge_aggregate.json`,
and `results.jsonl` for paired item-level analysis. Produces a
single 3-panel figure:

  fig_v02_results.png
    ┌ NODE lift × 11  ┐┌ Reflective share ┐┌ int4 vs fp16     ┐
    │ (ko_native)     ││ (per-SUT, judge) ││ (paired discord.)│
    └─────────────────┘└──────────────────┘└──────────────────┘

Usage:
    python scripts/99_make_figures.py \\
        --run-dir experiments/20260426_1242_v0.2_sweep_merged \\
        --out-dir report/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ─── palette (mirrors the Paperbanana overview) ──────────────────
C_DATA      = "#3a7d3a"   # green — ko_native / pass
C_MEMORY    = "#2b6cb0"   # blue  — retrieval / NODE
C_ORACLE    = "#5fa17a"   # sage — oracle ceiling
C_NEUTRAL   = "#9aa0a6"   # grey — no_node / out-of-scope
C_REFLECT   = "#c23b22"   # red — MIRROR / reflective
C_ADVISORY  = "#d89d6a"   # tan — advisory
C_TIE       = "#dddddd"   # light — ties
C_ACCENT    = "#f4b400"   # amber — headline emphasis


# Display order and labels for the 11 v0.2 SUTs.
SUT_ORDER = [
    ("kanana_nano",          "Kanana 2.1B"),
    ("kanana_nano_int4",     "Kanana 2.1B int4"),
    ("kanana_8b",            "Kanana 8B"),
    ("kanana_8b_int4",       "Kanana 8B int4"),
    ("hclova_seed_15b",      "HCX-SEED 1.5B"),
    ("hclova_seed_15b_int4", "HCX-SEED 1.5B int4"),
    ("qwen25_3b",            "Qwen 2.5 3B"),
    ("qwen25_3b_int4",       "Qwen 2.5 3B int4"),
    ("qwen25_7b",            "Qwen 2.5 7B"),
    ("bllossom_8b",          "Bllossom 8B"),
    ("open_ko_8b",           "Open-Ko 8B"),
]

INT4_PAIRS = [
    ("kanana_nano",     "kanana_nano_int4",     "Kanana 2.1B"),
    ("hclova_seed_15b", "hclova_seed_15b_int4", "HCX-SEED 1.5B"),
    ("kanana_8b",       "kanana_8b_int4",       "Kanana 8B"),
    ("qwen25_3b",       "qwen25_3b_int4",       "Qwen 2.5 3B"),
]

RUBRICS = ["specificity", "non_directive", "emotional_attunement", "open_question"]


def _base_style():
    rcParams.update({
        "figure.dpi":       120,
        "savefig.dpi":      160,
        "savefig.bbox":     "tight",
        "figure.facecolor": "white",
        "axes.facecolor":   "#fafafa",
        "axes.edgecolor":   "#cccccc",
        "axes.labelcolor":  "#333333",
        "axes.titlecolor":  "#222222",
        "axes.titleweight": "600",
        "axes.titlesize":   13,
        "axes.titlepad":    10,
        "axes.labelsize":   11,
        "axes.grid":        True,
        "axes.axisbelow":   True,
        "grid.color":       "#e0e0e0",
        "grid.linewidth":   0.8,
        "xtick.color":      "#333333",
        "ytick.color":      "#333333",
        "xtick.labelsize":  9,
        "ytick.labelsize":  10,
        "font.family":      "sans-serif",
        "font.size":        11,
        "legend.frameon":   False,
        "legend.fontsize":  10,
    })


def _load_metrics(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["n"] = int(r["n"])
            r["pass_rate"] = float(r["pass_rate"]) if r["pass_rate"] else None
            r["avg_latency_s"] = float(r["avg_latency_s"])
            out.append(r)
    return out


def _get(rows, *, track, sut, condition, policy, ability="ALL"):
    for r in rows:
        if (r["track"] == track and r["sut"] == sut
                and r["condition"] == condition and r["policy"] == policy
                and r["ability"] == ability):
            return r
    return None


# ─── panel A : NODE lift × 11 SUTs (ko_native, advisory) ─────────

def panel_node_lift(ax, rows):
    suts = SUT_ORDER
    conds = [("no_node", "no-NODE", C_NEUTRAL),
             ("retrieval", "retrieval", C_MEMORY),
             ("oracle",    "oracle",    C_ORACLE)]

    # Sort by retrieval pass rate (descending) for a clean visual.
    def _retrieval_pass(s):
        m = _get(rows, track="ko_native", sut=s, condition="retrieval", policy="advisory")
        return (m["pass_rate"] or 0) if m else 0

    suts_sorted = sorted(suts, key=lambda x: -_retrieval_pass(x[0]))

    n = len(suts_sorted)
    width = 0.27
    x = list(range(n))
    for i, (cond_key, cond_lab, color) in enumerate(conds):
        ys = []
        for sut_key, _ in suts_sorted:
            m = _get(rows, track="ko_native", sut=sut_key,
                     condition=cond_key, policy="advisory")
            ys.append((m["pass_rate"] or 0) * 100 if m else 0)
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, ys, width, color=color, edgecolor="white",
               linewidth=0.8, label=cond_lab)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in suts_sorted], rotation=40, ha="right")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 26)
    ax.set_title("a  NODE lift on ko_native — 11 v0.2 SUTs (advisory)")
    ax.legend(loc="upper right", ncol=3)


# ─── panel B : reflective share per SUT (judge aggregate) ────────

def panel_reflective(ax, aggregate):
    # Aggregate over both retrieval-augmented conditions and all 4 rubrics.
    sut_tot = defaultdict(lambda: {"adv": 0, "refl": 0, "tie": 0})
    for cell, d in aggregate.items():
        sut = cell.split("|")[0]
        for r in RUBRICS:
            sut_tot[sut]["adv"]  += d[r]["advisory_win"]
            sut_tot[sut]["refl"] += d[r]["reflective_win"]
            sut_tot[sut]["tie"]  += d[r]["tie"]

    items = []
    for sut_key, sut_lab in SUT_ORDER:
        if sut_key not in sut_tot:
            continue
        t = sut_tot[sut_key]
        n = t["adv"] + t["refl"] + t["tie"]
        items.append((sut_lab, t["refl"] / n, t["adv"] / n, t["tie"] / n, n))
    items.sort(key=lambda r: r[1])    # ascending, so largest reflective share at top

    labels = [r[0] for r in items]
    refl   = [r[1] * 100 for r in items]
    adv    = [r[2] * 100 for r in items]
    tie    = [r[3] * 100 for r in items]
    y = list(range(len(items)))

    ax.barh(y, refl,        color=C_REFLECT,  label="reflective wins")
    ax.barh(y, adv, left=refl, color=C_ADVISORY, label="advisory wins")
    ax.barh(y, tie, left=[a + b for a, b in zip(refl, adv)],
            color=C_TIE, label="tie")
    for yi, r in zip(y, refl):
        ax.text(r - 1.5, yi, f"{r:.0f}%", ha="right", va="center",
                color="white", weight="700", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Decisions (%)")
    ax.set_title("b  Reflective share per SUT  ·  192 decisions / SUT")
    ax.legend(loc="lower right", ncol=1)
    ax.grid(axis="y", visible=False)


# ─── panel C : int4 vs fp16 paired discordance ───────────────────

def _paired_discordance(results_path: Path):
    # Build per-SUT item -> passed map for ko_native retrieval advisory.
    by_sut = defaultdict(dict)
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["track"] != "ko_native":      continue
            if r["ability"] == "REFL":         continue
            if r["policy"] != "advisory":      continue
            if r["condition"] != "retrieval":  continue
            by_sut[r["sut"]][r["item_id"]] = bool(r["passed_contains"])

    out = []
    for fp_key, i4_key, label in INT4_PAIRS:
        fp = by_sut[fp_key]; q = by_sut[i4_key]
        common = set(fp) & set(q)
        fp_only = sum(1 for it in common if fp[it] and not q[it])
        i4_only = sum(1 for it in common if q[it] and not fp[it])
        both    = sum(1 for it in common if fp[it] and q[it])
        n_disc  = fp_only + i4_only
        z       = (i4_only - fp_only) / sqrt(n_disc) if n_disc else 0.0
        fp_rate = (both + fp_only) / len(common) * 100 if common else 0
        i4_rate = (both + i4_only) / len(common) * 100 if common else 0
        out.append((label, fp_rate, i4_rate, fp_only, i4_only, z))
    return out


def panel_int4_paired(ax, results_path: Path):
    data = _paired_discordance(results_path)
    labels = [r[0] for r in data]
    fp     = [r[1] for r in data]
    i4     = [r[2] for r in data]
    width = 0.36
    x = list(range(len(labels)))
    ax.bar([xi - width / 2 for xi in x], fp, width,
           color=C_MEMORY, edgecolor="white", linewidth=0.8, label="fp16")
    ax.bar([xi + width / 2 for xi in x], i4, width,
           color=C_REFLECT, edgecolor="white", linewidth=0.8, label="int4")

    # Annotate with discordance counts and z.
    for xi, (_, fp_r, i4_r, fp_only, i4_only, z) in zip(x, data):
        top = max(fp_r, i4_r)
        ax.text(xi, top + 1.4,
                f"int4 wins {i4_only} / fp16 wins {fp_only} (z={z:.1f})",
                ha="center", va="bottom", fontsize=9, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 28)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("c  fp16 vs int4 paired (ko_native retrieval, n=64)")
    ax.legend(loc="upper right", ncol=2)


# ─── composite ───────────────────────────────────────────────────

def make_v02(rows, aggregate, results_path: Path, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), gridspec_kw={"wspace": 0.32})
    panel_node_lift(axes[0], rows)
    panel_reflective(axes[1], aggregate)
    panel_int4_paired(axes[2], results_path)
    fig.suptitle("Figure — Heard v0.2 results across eleven SUTs",
                 fontsize=15, weight="700", y=0.99)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("report/figures"))
    ap.add_argument("--ext", default="png", choices=["png", "pdf"])
    args = ap.parse_args()

    _base_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_metrics(args.run_dir / "metrics.csv")
    aggregate = json.loads((args.run_dir / "judge_aggregate.json").read_text(encoding="utf-8"))
    results = args.run_dir / "results.jsonl"

    out = args.out_dir / f"fig_v02_results.{args.ext}"
    make_v02(rows, aggregate, results, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
