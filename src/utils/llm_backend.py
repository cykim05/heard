"""Local HF model loading and generation wrapper.

DATASET.md §6.5 — we load each SUT (system-under-test) with a uniform
interface so sweeps can switch models without changing runner code.
Tracks all loaded models in a per-run `model_registry.json`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


_DEFAULT_MAX_NEW_TOKENS = 512


@dataclass
class LoadedModel:
    name: str            # logical name in models.yaml (e.g. "kanana_nano")
    hf_id: str
    quantization: str    # "fp16" | "int4"
    revision: Optional[str]
    device: str
    model: Any
    tokenizer: Any
    params: int = 0

    def to_registry_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "quantization": self.quantization,
            "revision": self.revision,
            "device": self.device,
            "params_billion": round(self.params / 1e9, 3),
        }


def _pick_device() -> str:
    # Respect CUDA_VISIBLE_DEVICES if the user set it.
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def load_sut(
    hf_id: str,
    *,
    name: str,
    quantization: str = "fp16",
    trust_remote_code: bool = False,
    device: Optional[str] = None,
    revision: Optional[str] = None,
) -> LoadedModel:
    """Load a local HF causal LM.

    quantization:
      - 'fp16' — torch.float16, no bnb
      - 'int4' — bitsandbytes nf4 + float16 compute dtype
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or _pick_device()

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision

    if quantization == "int4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif quantization == "fp16":
        # transformers >=4.56 renamed torch_dtype -> dtype; we follow the new spelling.
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": device} if device != "cpu" else None
    else:
        raise ValueError(f"unknown quantization {quantization!r}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    return LoadedModel(
        name=name,
        hf_id=hf_id,
        quantization=quantization,
        revision=revision,
        device=device,
        model=model,
        tokenizer=tokenizer,
        params=params,
    )


def generate(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    do_sample: bool = True,
    seed: Optional[int] = None,
    system: Optional[str] = None,
) -> str:
    """Single-shot generation. Uses the tokenizer's chat template when
    available (Kanana, HyperCLOVA, Qwen, Gemma, Llama all ship one)."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if hasattr(loaded.tokenizer, "apply_chat_template") and loaded.tokenizer.chat_template:
        prompt_ids = loaded.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(loaded.model.device)
    else:
        # Fallback — concatenate.
        text = (system + "\n\n" if system else "") + prompt
        prompt_ids = loaded.tokenizer(text, return_tensors="pt").input_ids.to(loaded.model.device)

    with torch.no_grad():
        out = loaded.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=loaded.tokenizer.pad_token_id,
        )
    new_ids = out[0, prompt_ids.shape[1]:]
    return loaded.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def unload(loaded: LoadedModel) -> None:
    del loaded.model
    del loaded.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def append_to_registry(path: Path, loaded: LoadedModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.append(loaded.to_registry_entry())
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
