"""
config.py
---------
Loads all environment variables from .env and exposes them as a
typed Settings object. Import `settings` anywhere in the project.

Usage:
    from config import settings
    print(settings.google_api_key)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────
#  Settings model
# ─────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All configuration is read from environment variables (or .env).
    Pydantic-settings will raise a clear ValidationError if a required
    key is missing, rather than silently passing None to an LLM client.
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",            # Ignore unknown env vars
    )

    # ── LLM provider keys ───────────────────────────────────────────────
    google_api_key: str = Field(
        default="",
        description="Google AI Studio key — used by Accuracy Critic & Adjudicator.",
    )
    groq_api_key: str = Field(
        default="",
        description="Groq key — used by Logic Critic (DeepSeek R1).",
    )
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter key — used by Safety Critic (Qwen3 32B).",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama server.",
    )

    # ── Model names (overridable without code changes) ───────────────────
    accuracy_model: str     = Field(default="gemini-2.5-flash")
    logic_model: str        = Field(default="llama-3.3-70b-versatile")
    completeness_model: str = Field(default="gemma3")
    safety_model: str       = Field(default="qwen/qwen3-235b-a22b:free")
    style_model: str        = Field(default="gemma3")
    adjudicator_model: str  = Field(default="gemini-2.5-flash")

    # ── App settings ────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    storage_path: str = Field(default="./storage/arbitrations.db")
    max_concurrent_critics: int = Field(default=5, ge=1, le=10)
    critic_timeout_seconds: int = Field(default=60, ge=5, le=300)
    # Ollama models cold-start slowly on laptop hardware — give them more time
    ollama_timeout_seconds: int = Field(default=180, ge=10, le=600)

    # ── Fallback models ──────────────────────────────────────────────────
    # If a primary provider fails, the critic automatically retries with these
    accuracy_fallback_model: str    = Field(default="llama-3.3-70b-versatile")
    accuracy_fallback_provider: str = Field(default="groq")

    logic_fallback_model: str       = Field(default="llama-3.1-8b-instant")
    logic_fallback_provider: str    = Field(default="groq")

    safety_fallback_model: str      = Field(default="gemma3")
    safety_fallback_provider: str   = Field(default="ollama")

    # Completeness and Style use Ollama as primary — no remote fallback needed

    # ── Orchestration thresholds ─────────────────────────────────────────
    # All scores >= this AND all passed=True → skip adjudicator (fast path)
    fast_path_threshold: int = Field(default=80, ge=0, le=100)
    # Score std-dev >= this → route to adjudicator for conflict resolution
    disagreement_variance_threshold: float = Field(default=15.0, ge=0.0)
    # Confidence is reduced by this amount for each fallback model used
    fallback_confidence_penalty: float = Field(default=0.05, ge=0.0, le=1.0)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    def configured_critics(self) -> list[str]:
        """Return the list of critics whose provider keys are set."""
        critics = []
        if self.google_api_key and self.google_api_key != "your_google_ai_studio_api_key_here":
            critics.append("accuracy")
        if self.groq_api_key and self.groq_api_key != "your_groq_api_key_here":
            critics.append("logic")
        critics.append("completeness")  # Ollama — always available locally
        if self.openrouter_api_key and self.openrouter_api_key != "your_openrouter_api_key_here":
            critics.append("safety")
        critics.append("style")  # Ollama — always available locally
        return critics

    def is_ready(self) -> bool:
        """True if at least one external critic is configured."""
        return len(self.configured_critics()) > 0


# ─────────────────────────────────────────────
#  Singleton accessor
# ─────────────────────────────────────────────

@lru_cache(maxsize=None)
def get_settings() -> Settings:
    """Return the (cached) Settings singleton. Cache is invalidated on process restart."""
    return Settings()


# Module-level convenience alias
settings = get_settings()


# ─────────────────────────────────────────────
#  Logging setup (call once at app startup)
# ─────────────────────────────────────────────

def configure_logging() -> None:
    """Configure root logger based on settings.log_level."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
