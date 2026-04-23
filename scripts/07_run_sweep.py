#!/usr/bin/env python3
"""Run the Day 3 reduced sweep.

  2 SUTs × 3 tracks × 3 conditions × 1 policy (advisory)
  + ko_native × 3 conditions × 1 extra policy (reflective)

Outputs a single experiments/<run_id>/results.jsonl that downstream
metrics and judge code consume.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.runner import run_sweep
from src.node.store import Embedder
from src.utils.config import REPO_ROOT
from src.utils.llm_backend import append_to_registry, load_sut, unload
from src.utils.logging import get_logger

log = get_logger("sweep")


SUT_CHOICES = [
    ("kanana_nano", "kakaocorp/kanana-1.5-2.1b-instruct-2505", "fp16"),
    ("qwen25_3b",   "Qwen/Qwen2.5-3B-Instruct",               "fp16"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", default=["ko_native", "en_subset", "ko_translated"])
    ap.add_argument("--conditions", nargs="*", default=["no_node", "retrieval", "oracle"])
    ap.add_argument("--policies", nargs="*", default=["advisory"])
    ap.add_argument("--policies-ko-native", nargs="*", default=["advisory", "reflective"],
                    help="Extra policies to run only on ko_native (where REFL lives).")
    ap.add_argument("--max-items-per-track", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir or (REPO_ROOT / "experiments" /
                               f"{datetime.now():%Y%m%d_%H%M}_day3_sweep")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "\n".join([
            f"run_dir: {run_dir}",
            f"tracks: {args.tracks}",
            f"conditions: {args.conditions}",
            f"policies: {args.policies}",
            f"policies_ko_native: {args.policies_ko_native}",
            f"k: {args.k}",
            f"max_items_per_track: {args.max_items_per_track}",
            f"suts: {[s[0] for s in SUT_CHOICES]}",
        ]),
        encoding="utf-8",
    )
    results_path = run_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)

    log.info("loading embedder…")
    embedder = Embedder()

    for name, hf_id, quant in SUT_CHOICES:
        log.info("loading SUT %s (%s, %s)…", name, hf_id, quant)
        sut = load_sut(hf_id, name=name, quantization=quant)
        append_to_registry(run_dir / "model_registry.json", sut)

        # ko_native gets extra policies.
        for track in args.tracks:
            track_policies = list(args.policies)
            if track == "ko_native":
                for p in args.policies_ko_native:
                    if p not in track_policies:
                        track_policies.append(p)
            run_sweep(
                sut=sut,
                embedder=embedder,
                out_path=results_path,
                tracks=[track],
                conditions=args.conditions,
                policies=track_policies,
                k=args.k,
                max_items_per_track=args.max_items_per_track,
            )

        unload(sut)

    log.info("DONE: %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
