"""Config loader — reads .env + configs/models.yaml into typed settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_site_url: str
    openrouter_app_name: str
    hf_token: str
    models: dict[str, Any] = field(default_factory=dict)
    budget_total_usd: float = 30.0
    cache_dir: Path = REPO_ROOT / ".api_cache"
    api_log_dir: Path = REPO_ROOT / "experiments" / "_api_log"

    @property
    def generators(self) -> dict[str, Any]:
        return self.models.get("generators", {})

    @property
    def judges(self) -> dict[str, Any]:
        return self.models.get("judges", {})

    @property
    def suts(self) -> dict[str, Any]:
        return self.models.get("suts", {})


def load_settings(models_yaml: Path | None = None) -> Settings:
    load_dotenv(REPO_ROOT / ".env")

    missing = [k for k in ("OPENROUTER_API_KEY",) if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}. See .env.example.")

    models_path = models_yaml or (REPO_ROOT / "configs" / "models.yaml")
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))

    settings = Settings(
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        openrouter_site_url=os.environ.get("OPENROUTER_SITE_URL", ""),
        openrouter_app_name=os.environ.get("OPENROUTER_APP_NAME", "heard-bench"),
        hf_token=os.environ.get("HF_TOKEN", ""),
        models=models,
        budget_total_usd=float(models.get("budget", {}).get("total_cap_usd", 30.0)),
    )
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.api_log_dir.mkdir(parents=True, exist_ok=True)
    return settings
