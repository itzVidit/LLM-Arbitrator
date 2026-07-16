"""
critics/base.py
---------------
Abstract base class that every specialist critic inherits from.

Responsibilities of BaseCritic:
  - Define the interface: every critic must implement `_build_prompt` and
    expose a `dimension` and `provider`.
  - Provide a uniform `evaluate()` async entry point with timing, retries,
    and error handling built in.
  - Encapsulate all retry / timeout logic so individual critics stay clean.

How to create a new critic:
    class MyCritic(BaseCritic):
        dimension  = CriticDimension.ACCURACY
        provider   = "gemini"
        model_name = settings.accuracy_model

        def _build_prompt(self, request: ArbitrationRequest) -> str:
            return f"Evaluate accuracy: {request.llm_response}"
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from models.critique import Critique, CriticDimension, ArbitrationRequest
from utils.llm_clients import get_client

log = logging.getLogger(__name__)

# Serialize all Ollama calls so only one runs at a time.
# Two 3GB models competing for laptop RAM/CPU causes timeouts.
# threading.Semaphore works across all event loops (unlike asyncio.Semaphore).
_OLLAMA_LOCK = threading.Semaphore(1)


class BaseCritic(ABC):
    """
    Template for all critic agents.

    Class-level attributes to define on each subclass:
        dimension         (CriticDimension): The axis this critic evaluates.
        provider          (str):             Primary LLM provider key.
        model_name        (str):             Primary model ID.
        fallback_provider (str | None):      Provider to use if primary fails.
        fallback_model    (str | None):      Model ID for the fallback provider.
    """

    dimension:         CriticDimension
    provider:          str
    model_name:        str
    fallback_provider: str | None = None
    fallback_model:    str | None = None

    MAX_RETRIES: int = 3

    def __init__(self) -> None:
        self._client: Any = None
        self._using_fallback: bool = False   # Set True when primary has failed

    # ─────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────

    async def evaluate(self, request: ArbitrationRequest) -> Critique:
        """
        Evaluate the given arbitration request and return a typed Critique.

        Failure strategy:
          1. Try primary provider/model with up to MAX_RETRIES.
          2. If that fails and a fallback is configured, switch to the
             fallback provider and try once more.
          3. If the fallback also fails (or none configured), return a
             clearly-marked fallback Critique with score=0 so the
             orchestrator can continue without this critic.
        """
        log.info(
            "[%s] Starting evaluation (model=%s)",
            self.dimension.value,
            self.model_name,
        )
        t0 = time.perf_counter()
        self._using_fallback = False

        # Ollama models cold-start slowly on laptop hardware — use a longer timeout
        timeout = (
            settings.ollama_timeout_seconds
            if self.provider == "ollama"
            else settings.critic_timeout_seconds
        )

        try:
            critique = await asyncio.wait_for(
                self._call_with_retry(request),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            reason = f"Timeout after {timeout}s on primary model"
            log.warning("[%s] %s", self.dimension.value, reason)
            critique = await self._try_fallback(request, reason)
        except Exception as exc:
            reason = str(exc)
            log.warning("[%s] Primary failed (%s), trying fallback", self.dimension.value, reason)
            critique = await self._try_fallback(request, reason)

        latency = (time.perf_counter() - t0) * 1000
        critique = critique.model_copy(update={"latency_ms": round(latency, 1)})

        log.info(
            "[%s] Done  score=%d  grade=%s  fallback=%s  latency=%.0fms",
            self.dimension.value,
            critique.score,
            critique.grade,
            self._using_fallback,
            latency,
        )
        return critique

    async def _try_fallback(self, request: ArbitrationRequest, primary_reason: str) -> Critique:
        """
        Attempt evaluation with the fallback model.
        Returns a zero-score Critique if fallback is not configured or also fails.
        """
        if not self.fallback_provider or not self.fallback_model:
            log.error("[%s] No fallback configured — returning error Critique", self.dimension.value)
            return self._fallback_critique(reason=primary_reason)

        log.info(
            "[%s] Switching to fallback: provider=%s model=%s",
            self.dimension.value,
            self.fallback_provider,
            self.fallback_model,
        )
        self._using_fallback = True
        # Temporarily swap to fallback client
        original_provider = self.provider
        original_model = self.model_name
        self.provider = self.fallback_provider    # type: ignore[assignment]
        self.model_name = self.fallback_model      # type: ignore[assignment]
        self._client = None                        # Force re-init with fallback provider

        try:
            fallback_timeout = (
                settings.ollama_timeout_seconds
                if self.fallback_provider == "ollama"
                else settings.critic_timeout_seconds
            )
            critique = await asyncio.wait_for(
                self._call_with_retry(request),
                timeout=fallback_timeout,
            )
            # Tag the critique to show the fallback model was used
            critique = critique.model_copy(update={"model_used": f"{self.fallback_model} (fallback)"})
            return critique
        except Exception as exc:
            log.error("[%s] Fallback also failed: %s", self.dimension.value, exc)
            return self._fallback_critique(reason=f"Primary: {primary_reason} | Fallback: {exc}")
        finally:
            # Always restore primary settings for any future calls
            self.provider = original_provider      # type: ignore[assignment]
            self.model_name = original_model       # type: ignore[assignment]
            self._client = None

    # ─────────────────────────────────────────
    #  Abstract methods — implement in subclass
    # ─────────────────────────────────────────

    @abstractmethod
    def _build_prompt(self, request: ArbitrationRequest) -> str:
        """
        Construct the evaluation prompt that will be sent to the LLM.

        The prompt should:
          1. Explain the critic's specific role.
          2. Show the original prompt and LLM response.
          3. Give clear instructions on what to look for.
          4. Specify the desired JSON output format (Instructor handles enforcement).
        """
        ...

    # ─────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────

    async def _call_with_retry(self, request: ArbitrationRequest) -> Critique:
        """Run the LLM call in a thread pool (sync clients) with tenacity retries."""
        loop = asyncio.get_event_loop()
        if self.provider == "ollama":
            # Serialize Ollama calls via a threading.Semaphore acquired inside
            # the worker thread — safe across all event loops.
            def _locked_call():
                log.debug("[%s] Waiting for Ollama lock", self.dimension.value)
                with _OLLAMA_LOCK:
                    log.debug("[%s] Acquired Ollama lock", self.dimension.value)
                    return self._sync_call(request)
            return await loop.run_in_executor(None, _locked_call)
        # Remote providers run fully in parallel (unaffected)
        return await loop.run_in_executor(None, self._sync_call, request)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _sync_call(self, request: ArbitrationRequest) -> Critique:
        """Synchronous LLM call wrapped with tenacity retry logic."""
        if self._client is None:
            self._client = get_client(self.provider)

        prompt = self._build_prompt(request)

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",   "content": prompt},
        ]

        # Gemini: model is baked in at client construction, don't pass it again
        if self.provider == "gemini":
            return self._client.chat.completions.create(
                response_model=Critique,
                messages=messages,
                max_retries=2,
            )

        # Ollama: small local models (gemma3) get confused by Instructor's full
        # Pydantic JSON Schema ($ref/$defs). Instead, inject an explicit flat JSON
        # template into the prompt and parse the raw response ourselves.
        if self.provider == "ollama":
            return self._ollama_call(messages, request)

        # All other providers (Groq, OpenRouter): standard Instructor path
        return self._client.chat.completions.create(
            model=self.model_name,
            response_model=Critique,
            messages=messages,
            max_retries=2,
        )

    def _ollama_call(self, messages: list, request) -> Critique:
        """
        Direct Ollama call without Instructor schema injection.

        Injects a flat JSON template into the system prompt so gemma3 knows
        exactly what to output without being confused by $ref/$defs structures.
        Parses the raw response manually and constructs a Critique.
        """
        import json
        import re

        dim = self.dimension.value

        # Replace system message with one that includes an explicit JSON template
        template = (
            '{\n'
            f'  "dimension": "{dim}",\n'
            '  "score": <integer 0-100>,\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "summary": "<2-4 sentence summary>",\n'
            '  "passed": <true or false>,\n'
            f'  "model_used": "{self.model_name}",\n'
            '  "issues": [\n'
            '    {\n'
            '      "quote": "<exact quote from response, or (general)>",\n'
            '      "explanation": "<why this is a problem>",\n'
            '      "severity": "<critical|major|minor|info>",\n'
            '      "evidence": "<supporting evidence or null>",\n'
            '      "recommendation": "<how to fix it>"\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        system_content = (
            f"You are an expert {dim} critic in an AI quality-assurance system. "
            "Evaluate the response and reply with ONLY a single valid JSON object. "
            "No explanation, no markdown, no code fences — just the raw JSON.\n\n"
            f"Required JSON format:\n{template}"
        )

        ollama_messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": messages[-1]["content"]},
        ]

        raw_client = get_client("ollama")
        # Instructor wraps the OpenAI client — unwrap it to call without schema injection
        actual_client = raw_client.client

        resp = actual_client.chat.completions.create(
            model=self.model_name,
            messages=ollama_messages,
            temperature=0,
            max_tokens=900,
            extra_body={"options": {"num_ctx": 4096, "num_predict": 900}},
        )

        raw_text = resp.choices[0].message.content or ""
        log.debug("[%s] Ollama raw response: %.200s", dim, raw_text)

        # Extract JSON — strip markdown fences if present
        json_text = raw_text.strip()
        if "```" in json_text:
            json_text = re.sub(r"```(?:json)?", "", json_text).strip().strip("`").strip()

        # Find outermost {...}
        start = json_text.find("{")
        end   = json_text.rfind("}") + 1
        if start != -1 and end > start:
            json_text = json_text[start:end]

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama returned invalid JSON: {exc}\nRaw: {raw_text[:300]}") from exc

        # Normalise and validate via Pydantic
        data.setdefault("dimension", dim)
        data.setdefault("model_used", self.model_name)
        data.setdefault("confidence", 0.5)
        data.setdefault("issues", [])
        # Ensure every issue has required fields
        cleaned_issues = []
        for iss in data.get("issues", []):
            if not isinstance(iss, dict):
                continue
            if not iss.get("explanation") or not iss.get("quote"):
                continue
            iss.setdefault("severity", "info")
            iss.setdefault("recommendation", "")
            cleaned_issues.append(iss)
        data["issues"] = cleaned_issues

        return Critique.model_validate(data)

    def _system_prompt(self) -> str:
        """
        Common system prompt preamble injected before the evaluation prompt.
        Subclasses can override if they need a fully custom system message.
        """
        return (
            f"You are an expert {self.dimension.value} critic in an AI quality-assurance system. "
            "Your job is to evaluate AI-generated responses with precision and objectivity. "
            "Always respond with a structured JSON object matching the requested schema exactly. "
            "Be specific: quote actual text, give concrete evidence, and make actionable recommendations. "
            "Do not add commentary outside the JSON structure."
        )

    def _fallback_critique(self, reason: str) -> Critique:
        """
        Returns a minimal, clearly-flagged Critique when the LLM call fails.
        This lets the orchestrator continue even if one critic is unavailable.
        """
        from models.critique import Issue, Severity

        return Critique(
            dimension=self.dimension,
            score=0,
            confidence=0.0,
            summary=f"Critic unavailable: {reason}",
            issues=[
                Issue(
                    quote="(critic failed)",
                    explanation=f"This critic ({self.dimension.value}) failed to respond: {reason}",
                    severity=Severity.INFO,
                    evidence=None,
                    recommendation="Check API keys, model availability, and network connectivity.",
                )
            ],
            passed=False,
            model_used=self.model_name,
            latency_ms=None,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} dimension={self.dimension.value} model={self.model_name}>"

    @property
    def used_fallback(self) -> bool:
        """True if the last evaluate() call used the fallback model."""
        return self._using_fallback
