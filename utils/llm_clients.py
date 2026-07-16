"""
utils/llm_clients.py
--------------------
Factory functions that build ready-to-use, Instructor-patched LLM clients
for every provider used in the system.

Why Instructor?
  Instructor wraps any OpenAI-compatible client and adds a `.chat.completions.create`
  call that accepts a `response_model=` kwarg, then automatically validates and
  retries until the LLM returns JSON that matches the Pydantic schema.

Provider map:
  - Google AI Studio  →  google-generativeai + instructor
  - Groq              →  groq + instructor (OpenAI-compatible)
  - OpenRouter        →  openai (pointed at openrouter.ai) + instructor
  - Ollama            →  openai (pointed at localhost) + instructor

Usage:
    from utils.llm_clients import get_gemini_client, get_groq_client, ...
    client = get_gemini_client()
    result = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_model=Critique,
        messages=[{"role": "user", "content": prompt}],
    )
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import instructor
from groq import Groq
from openai import OpenAI

from config import settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Google AI Studio (Gemini)
# ─────────────────────────────────────────────

@lru_cache(maxsize=None)
def get_gemini_client() -> Any:
    """
    Returns an Instructor-patched client backed by Google's Gemini API.

    Uses the new google-genai SDK (google.genai). Instructor's GENAI_TOOLS
    mode handles structured output via function-calling.
    """
    try:
        from google import genai  # type: ignore

        raw_client = genai.Client(api_key=settings.google_api_key)
        client = instructor.from_genai(
            client=raw_client,
            mode=instructor.Mode.GENAI_TOOLS,
            model=settings.accuracy_model,
        )
        log.debug("Gemini client created (model=%s)", settings.accuracy_model)
        return client
    except Exception as exc:
        log.error("Failed to create Gemini client: %s", exc)
        raise


# ─────────────────────────────────────────────
#  Groq (DeepSeek R1 / Llama 4 Scout)
# ─────────────────────────────────────────────

@lru_cache(maxsize=None)
def get_groq_client() -> Any:
    """
    Returns an Instructor-patched Groq client.

    Groq is OpenAI-compatible, so Instructor's TOOLS mode works directly.
    """
    try:
        raw_client = Groq(api_key=settings.groq_api_key)
        client = instructor.from_groq(raw_client, mode=instructor.Mode.TOOLS)
        log.debug("Groq client created (model=%s)", settings.logic_model)
        return client
    except Exception as exc:
        log.error("Failed to create Groq client: %s", exc)
        raise


# ─────────────────────────────────────────────
#  OpenRouter (Qwen3 32B and other cloud models)
# ─────────────────────────────────────────────

@lru_cache(maxsize=None)
def get_openrouter_client() -> Any:
    """
    Returns an Instructor-patched OpenAI client pointed at OpenRouter.

    OpenRouter exposes an OpenAI-compatible /chat/completions endpoint,
    so we just override the base_url and pass our OpenRouter API key.
    """
    try:
        raw_client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/llm-arbitration-system",
                "X-Title": "LLM Output Arbitration System",
            },
        )
        client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        log.debug("OpenRouter client created (model=%s)", settings.safety_model)
        return client
    except Exception as exc:
        log.error("Failed to create OpenRouter client: %s", exc)
        raise


# ─────────────────────────────────────────────
#  Ollama (local: Gemma 3, Phi-4 Mini)
# ─────────────────────────────────────────────

@lru_cache(maxsize=None)
def get_ollama_client() -> Any:
    """
    Returns an Instructor-patched OpenAI client pointed at the local Ollama server.

    Ollama v0.1.9+ exposes an OpenAI-compatible API at /v1.
    Instructor's JSON mode uses system-prompt injection to coerce JSON output
    from models that don't support native tool-calling.
    """
    try:
        raw_client = OpenAI(
            api_key="ollama",                          # Placeholder — Ollama ignores the key
            base_url=f"{settings.ollama_base_url}/v1",
        )
        client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        log.debug("Ollama client created (base_url=%s)", settings.ollama_base_url)
        return client
    except Exception as exc:
        log.error("Failed to create Ollama client: %s", exc)
        raise


# ─────────────────────────────────────────────
#  Convenience: get client by provider name
# ─────────────────────────────────────────────

_CLIENT_MAP = {
    "gemini":      get_gemini_client,
    "groq":        get_groq_client,
    "openrouter":  get_openrouter_client,
    "ollama":      get_ollama_client,
}


def get_client(provider: str) -> Any:
    """
    Return the cached client for a named provider.

    Args:
        provider: One of "gemini", "groq", "openrouter", "ollama"

    Raises:
        ValueError: If the provider name is unknown.
    """
    factory = _CLIENT_MAP.get(provider.lower())
    if factory is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(_CLIENT_MAP)}"
        )
    return factory()
