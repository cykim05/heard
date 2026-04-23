#!/usr/bin/env python3
"""Render two 21:9 multi-panel figures for the report.

Reads `experiments/<run_id>/metrics.csv` + `judge_aggregate.json`
and produces a pair of PNGs at 21:9, each carrying three panels
in a single row:

  fig_main_results.png
    ┌ NODE lift  ┐┌ Language axis ┐┌ Reflective win ┐
    │ (ko_native)││ (Kanana ret.) ││ (4 rubrics)    │
    └────────────┘└───────────────┘└────────────────┘

  fig_ability_latency.png
    ┌ Ability    ┐┌ Latency        ┐┌ Pareto        ┐
    │ breakdown  ││ by condition   ││ lat vs acc    │
    └────────────┘└────────────────┘└───────────────┘

Usage:
    python scripts/99_make_figures.py \\
        --run-dir experiments/20260423_1610_day3_sweep \\
        --out-dir report/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "font.family":      "sans-serif",
        "font.size":        11,
        "legend.frameon":   False,
        "legend.fontsize":  10,
    })


def _load_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        out = []
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


def _annotate(ax, bars, ys, *, fmt="{:.1f}", above=2.5):
    for b, y in zip(bars, ys):
        if y is None:
            continue
        ax.text(b.get_x() + b.get_width() / 2, y + above,
                fmt.format(y), ha="center", va="bottom",
                fontsize=10, color="#222222", weight="600")


# ─── panel renderers ─────────────────────────────────────────────

def panel_node_lift(ax, rows):
    """Panel 1: NODE lift by condition × SUT on ko_native."""
    conditions = ["no_node", "retrieval", "oracle"]
    labels = ["no-NODE", "retrieval", "oracle"]
    suts = [("kanana_nano", "Kanana 2.1B"), ("qwen25_3b", "Qwen 2.5 3B")]
    colors = [C_NEUTRAL, C_MEMORY, C_ORACLE]

    x = list(range(len(conditions)))
    width = 0.36
    for i, (sut_key, sut_label) in enumerate(suts):
        ys = []
        for cond in conditions:
            m = _get(rows, track="ko_native", sut=sut_key,
                     condition=cond, policy="advisory")
            ys.append((m["pass_rate"] or 0) * 100 if m else 0)
        offsets = [xi + (i - 0.5) * width for xi in x]
        hatches = ["", "//"][i:i + 1] * len(ys)
        bars = ax.bar(offsets, ys, width,
                      color=[colors[c] for c in range(len(conditions))],
                      edgecolor=("#333" if i == 0 else "white"),
                      linewidth=1.0,
                      alpha=0.95 if i == 0 else 0.6,
                      label=sut_label,
                      hatch=hatches[0] if i else None)
        _annotate(ax, bars, ys)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, max(20, max(ys) * 1.35))
    ax.set_title("a  NODE lift on ko_native (advisory)")
    # annotation arrow: ×3.3
    ax.annotate(
        "×3.3 end-to-end (Kanana)", xy=(2 - 0.18, 15.6), xytext=(0.5, 18),
        fontsize=10, color=C_ACCENT, weight="600",
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
    )
    ax.legend(loc="upper left")


def panel_language_axis(ax, rows):
    """Panel 2: language-axis pass rate for Kanana retrieval."""
    tracks = ["en_subset", "ko_translated", "ko_native"]
    labels = ["en_subset", "ko_translated", "ko_native"]
    ys = []
    for tr in tracks:
        m = _get(rows, track=tr, sut="kanana_nano",
                 condition="retrieval", policy="advisory")
        ys.append((m["pass_rate"] or 0) * 100 if m else 0)

    x = list(range(len(tracks)))
    colors = [C_NEUTRAL, C_MEMORY, C_DATA]
    bars = ax.bar(x, ys, 0.55, color=colors, edgecolor="white", linewidth=1.2)
    _annotate(ax, bars, ys)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, max(16, max(ys) * 1.4))
    ax.set_title("b  Language axis (Kanana retrieval)")
    ax.text(0.5, max(ys) * 1.25,
            "EN → KO-translated → KO-native\nmonotonic",
            ha="left", fontsize=9, color="#666", style="italic")


def panel_reflective_wins(ax, aggregate):
    """Panel 3: reflective vs advisory wins, stacked per rubric."""
    rubrics = ["specificity", "non_directive", "emotional_attunement", "open_question"]
    rubric_labels = ["specificity", "non-directive", "emotional\nattunement", "open\nquestion"]

    adv, refl, tie = [], [], []
    for r in rubrics:
        a = refl_ct = t = 0
        for per_rubric in aggregate.values():
            slot = per_rubric.get(r, {})
            a += slot.get("advisory_win", 0)
            refl_ct += slot.get("reflective_win", 0)
            t += slot.get("tie", 0)
        total = a + refl_ct + t
        adv.append(a / total * 100 if total else 0)
        refl.append(refl_ct / total * 100 if total else 0)
        tie.append(t / total * 100 if total else 0)

    x = list(range(len(rubrics)))
    width = 0.62
    ax.bar(x, adv, width, label="advisory wins",
           color=C_ADVISORY, edgecolor="white", linewidth=1.0)
    ax.bar(x, refl, width, bottom=adv, label="reflective wins",
           color=C_REFLECT, edgecolor="white", linewidth=1.0)
    ax.bar(x, tie, width, bottom=[a + r for a, r in zip(adv, refl)],
           label="tie", color=C_TIE, edgecolor="white", linewidth=1.0)

    # Annotate the reflective share on each bar.
    for xi, (a, r) in enumerate(zip(adv, refl)):
        ax.text(xi, a + r / 2, f"{r:.0f}%", ha="center", va="center",
                fontsize=10, color="white", weight="700")

    ax.set_xticks(x)
    ax.set_xticklabels(rubric_labels)
    ax.set_ylabel("Share of decisions (%)")
    ax.set_ylim(0, 100)
    ax.set_title("c  Reflective dominance (96 pairwise decisions / rubric)")
    ax.legend(loc="lower right")


def panel_ability_breakdown(ax, rows):
    """Panel 4: per-ability pass rate on ko_native retrieval, 2 SUTs."""
    abilities = ["IE", "MR", "KU", "TR", "ABS"]
    suts = [("kanana_nano", "Kanana 2.1B", C_MEMORY),
            ("qwen25_3b", "Qwen 2.5 3B", C_ORACLE)]

    x = list(range(len(abilities)))
    width = 0.38
    for i, (sut_key, sut_label, color) in enumerate(suts):
        ys = []
        for ab in abilities:
            m = _get(rows, track="ko_native", sut=sut_key,
                     condition="retrieval", policy="advisory", ability=ab)
            ys.append((m["pass_rate"] or 0) * 100 if m and m["pass_rate"] is not None else 0)
        offsets = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offsets, ys, width, color=color, alpha=0.9,
                      edgecolor="white", linewidth=1.0, label=sut_label)
        _annotate(ax, bars, ys, fmt="{:.0f}", above=1)

    ax.set_xticks(x)
    ax.set_xticklabels(abilities)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("d  Ability breakdown (ko_native, retrieval, advisory)")
    ax.legend(loc="upper right")


def panel_latency(ax, rows):
    """Panel 5: mean latency per SUT × condition."""
    conditions = ["no_node", "retrieval", "oracle"]
    labels = ["no-NODE", "retrieval", "oracle"]
    suts = [("kanana_nano", "Kanana 2.1B", C_MEMORY),
            ("qwen25_3b", "Qwen 2.5 3B", C_ORACLE)]

    x = list(range(len(conditions)))
    width = 0.38
    for i, (sut_key, sut_label, color) in enumerate(suts):
        ys = []
        for cond in conditions:
            # Average over tracks for a representative mean.
            rows_pick = [r for r in rows
                         if r["sut"] == sut_key and r["condition"] == cond
                         and r["policy"] == "advisory" and r["ability"] == "ALL"]
            ys.append(sum(r["avg_latency_s"] for r in rows_pick) / len(rows_pick)
                      if rows_pick else 0)
        offsets = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offsets, ys, width, color=color, alpha=0.9,
                      edgecolor="white", linewidth=1.0, label=sut_label)
        _annotate(ax, bars, ys, fmt="{:.2f}s", above=0.05)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean latency per response (s)")
    ax.set_title("e  Latency by SUT × condition")
    ax.legend(loc="upper left")


def panel_pareto(ax, rows):
    """Panel 6: latency vs pass_rate (ko_native retrieval)."""
    points = []
    for r in rows:
        if (r["track"] == "ko_native" and r["condition"] == "retrieval"
                and r["policy"] == "advisory" and r["ability"] == "ALL"
                and r["pass_rate"] is not None):
            points.append((r["avg_latency_s"], r["pass_rate"] * 100, r["sut"]))

    colors = {"kanana_nano": C_MEMORY, "qwen25_3b": C_ORACLE}
    names = {"kanana_nano": "Kanana 2.1B", "qwen25_3b": "Qwen 2.5 3B"}
    for lat, acc, sut in points:
        ax.scatter(lat, acc, s=260, color=colors.get(sut, C_DATA),
                   edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(names.get(sut, sut), (lat, acc),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, color="#222", weight="600")

    ax.set_xlabel("Mean latency per response (s)")
    ax.set_ylabel("ko_native pass rate (%)")
    ax.set_title("f  Latency / accuracy Pareto (ko_native retrieval)")
    # generous padding around the cloud
    if points:
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        pad_x = max(0.3, (max(xs) - min(xs)) * 0.6)
        pad_y = max(2.0, (max(ys) - min(ys)) * 0.6)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)


# ─── composite figures ───────────────────────────────────────────

def make_main_results(rows, aggregate, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 9), gridspec_kw={"wspace": 0.25})
    panel_node_lift(axes[0], rows)
    panel_language_axis(axes[1], rows)
    panel_reflective_wins(axes[2], aggregate)
    fig.suptitle("Figure — Main results  ·  Heard v0.1",
                 fontsize=15, weight="700", y=0.97)
    fig.savefig(out_path)
    plt.close(fig)


def make_details(rows, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 9), gridspec_kw={"wspace": 0.25})
    panel_ability_breakdown(axes[0], rows)
    panel_latency(axes[1], rows)
    panel_pareto(axes[2], rows)
    fig.suptitle("Figure — Ability / latency / Pareto  ·  ko_native",
                 fontsize=15, weight="700", y=0.97)
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

    make_main_results(rows, aggregate, args.out_dir / f"fig_main_results.{args.ext}")
    make_details(rows, args.out_dir / f"fig_ability_latency.{args.ext}")
    print(f"wrote 2 figures (21:9) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
