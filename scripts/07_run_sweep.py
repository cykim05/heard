#!/usr/bin/env python3
"""Run the v0.2 SUT sweep.

The v0.2 lineup pairs Korean-native, Korean-tuned community,
and multilingual baselines, with an int4 axis on the four
configs that fit comfortably under 30 GB VRAM. Reflective policy
now runs on every track (not just ko_native).

Default sweep size:
  ~13 configs × 3 tracks × 3 conditions × {advisory, reflective}
  with reflective × no_node skipped (degenerate).
  ≈ 13 × 270 items × 5 policy×condition pairs ≈ 17,500 runs.

Outputs experiments/<run_id>/results.jsonl that downstream metrics
and judge code consume.
"""
from __future__ import annotations

import argparse
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


# (logical_name, hf_id, quantization, trust_remote_code)
# The smoke-load step in scripts/00_smoke_load_suts confirmed each
# entry below loads cleanly on a single L40S 48 GB.
SUT_CHOICES = [
    # Korean-native (Apache 2.0)
    ("kanana_nano",   "kakaocorp/kanana-1.5-2.1b-instruct-2505",         "fp16", False),
    ("kanana_nano",   "kakaocorp/kanana-1.5-2.1b-instruct-2505",         "int4", False),
    ("kanana_8b",     "kakaocorp/kanana-1.5-8b-instruct-2505",           "fp16", False),
    ("kanana_8b",     "kakaocorp/kanana-1.5-8b-instruct-2505",           "int4", False),

    # Multilingual baselines (Apache 2.0)
    ("qwen25_3b",     "Qwen/Qwen2.5-3B-Instruct",                        "fp16", False),
    ("qwen25_3b",     "Qwen/Qwen2.5-3B-Instruct",                        "int4", False),
    ("qwen25_7b",     "Qwen/Qwen2.5-7B-Instruct",                        "fp16", False),

    # Korean-native — HyperCLOVA-X SEED (gated, runtime account granted)
    ("hclova_seed_15b", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", "fp16", False),
    ("hclova_seed_15b", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B", "int4", False),

    # Korean-tuned community models
    ("eeve_10b",      "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",         "fp16", False),
    ("bllossom_8b",   "MLP-KTLim/llama-3-Korean-Bllossom-8B",            "fp16", False),
    ("open_ko_8b",    "beomi/Llama-3-Open-Ko-8B-Instruct-preview",       "fp16", False),
    ("solar_10b",     "upstage/SOLAR-10.7B-Instruct-v1.0",               "fp16", False),
]


def _config_id(name: str, quant: str) -> str:
    return name if quant == "fp16" else f"{name}_int4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*",
                    default=["ko_native", "en_subset", "ko_translated"])
    ap.add_argument("--conditions", nargs="*",
                    default=["no_node", "retrieval", "oracle"])
    ap.add_argument("--policies", nargs="*",
                    default=["advisory", "reflective"],
                    help="Policies applied across every track. Reflective × "
                         "no_node is skipped automatically inside the runner.")
    ap.add_argument("--max-items-per-track", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional logical SUT name filter (e.g., kanana_nano "
                         "kanana_8b). When given, runs only those SUTs.")
    args = ap.parse_args()

    run_dir = args.run_dir or (REPO_ROOT / "experiments" /
                               f"{datetime.now():%Y%m%d_%H%M}_v0.2_sweep")
    run_dir.mkdir(parents=True, exist_ok=True)

    suts_to_run = SUT_CHOICES
    if args.only:
        keep = set(args.only)
        suts_to_run = [s for s in SUT_CHOICES if s[0] in keep]
        log.info("filter --only=%s leaves %d configs", args.only, len(suts_to_run))

    (run_dir / "config.yaml").write_text(
        "\n".join([
            f"run_dir: {run_dir}",
            f"tracks: {args.tracks}",
            f"conditions: {args.conditions}",
            f"policies: {args.policies}",
            f"k: {args.k}",
            f"max_items_per_track: {args.max_items_per_track}",
            "suts:",
            *[f"  - {_config_id(n, q)} ({h}, {q})" for n, h, q, _ in suts_to_run],
        ]),
        encoding="utf-8",
    )
    results_path = run_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)

    log.info("loading embedder…")
    embedder = Embedder()

    for name, hf_id, quant, trc in suts_to_run:
        log.info("loading SUT %s (%s, %s)…", _config_id(name, quant), hf_id, quant)
        try:
            sut = load_sut(hf_id, name=_config_id(name, quant),
                           quantization=quant, trust_remote_code=trc)
        except Exception as e:
            log.error("FAILED to load %s: %s", _config_id(name, quant), e)
            continue
        append_to_registry(run_dir / "model_registry.json", sut)
        log.info(
            "  loaded params=%.2fB device=%s",
            sut.params / 1e9, sut.model.device,
        )

        try:
            for track in args.tracks:
                run_sweep(
                    sut=sut,
                    embedder=embedder,
                    out_path=results_path,
                    tracks=[track],
                    conditions=args.conditions,
                    policies=args.policies,
                    k=args.k,
                    max_items_per_track=args.max_items_per_track,
                )
        finally:
            unload(sut)

    log.info("DONE: %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
