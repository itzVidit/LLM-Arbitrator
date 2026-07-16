"""
critics/accuracy.py
-------------------
Accuracy Critic — powered by Gemini 2.5 Flash (Google AI Studio)

Responsibilities:
  - Fact-check specific claims in the response
  - Flag hallucinations (plausible-sounding but incorrect information)
  - Detect contradictions within the response or against the original prompt
  - Identify unsupported assertions presented as facts

Why Gemini 2.5 Flash?
  Google's Gemini models have strong world-knowledge recall and excel at
  identifying factually incorrect statements. The Flash variant gives fast
  turnaround suitable for a real-time evaluation pipeline.
"""

from __future__ import annotations

from critics.base import BaseCritic
from config import settings
from models.critique import CriticDimension, ArbitrationRequest


class AccuracyCritic(BaseCritic):
    """
    Evaluates the factual correctness of an LLM response.

    Checks for:
      • Factual errors — claims that contradict established knowledge
      • Hallucinations — specific details (names, dates, numbers) that appear
        invented or cannot be verified
      • Internal contradictions — statements within the response that conflict
        with each other
      • Unsupported assertions — strong claims made without any evidence or
        qualification
      • Misrepresentation of the original prompt — responses that answer a
        different question than was asked
    """

    dimension:          CriticDimension = CriticDimension.ACCURACY
    provider:           str             = "gemini"
    model_name:         str             = settings.accuracy_model
    fallback_provider:  str             = settings.accuracy_fallback_provider
    fallback_model:     str             = settings.accuracy_fallback_model

    def _build_prompt(self, request: ArbitrationRequest) -> str:
        context_block = (
            f"\n\n<context>\n{request.context}\n</context>"
            if request.context
            else ""
        )

        return f"""
You are evaluating an AI response for FACTUAL ACCURACY.

<original_prompt>
{request.original_prompt}
</original_prompt>

<llm_response>
{request.llm_response}
</llm_response>
{context_block}

## Your Task

Carefully read the response and identify every accuracy-related problem.
For each issue you find, you must:
  1. Quote the EXACT text from the response that contains the error.
  2. Explain clearly why it is factually incorrect or unverifiable.
  3. Rate severity:
       - CRITICAL: The error fundamentally misleads the user (wrong medical dose,
         wrong security advice, invented citations, etc.)
       - MAJOR: The claim is clearly wrong and will cause confusion.
       - MINOR: Small factual slip (wrong year, minor name error) that doesn't
         change the core message.
       - INFO: Technically imprecise but not harmful.
  4. Provide evidence or correct information where you know it.
  5. Give a concrete recommendation for how to fix it.

## Scoring Guidance

Score 0–100 reflecting overall factual quality:
  90–100: No factual errors found; all claims are accurate and well-supported.
  75–89:  Mostly accurate with 1–2 minor slips that don't mislead.
  60–74:  Some inaccuracies that could confuse an informed reader.
  40–59:  Multiple factual errors; response is partly unreliable.
  0–39:   Pervasive factual problems; response cannot be trusted.

## Confidence

Set confidence (0.0–1.0) based on how certain you are of your own assessment:
  - High (0.8–1.0): Topic is well within your knowledge; errors are clear.
  - Medium (0.5–0.8): Some uncertainty; the topic may be specialised.
  - Low (0.0–0.5): You cannot reliably verify claims in this domain.

Set `passed = true` only if there are NO critical or major issues.
Dimension must be "accuracy".
Model used: "{self.model_name}"
""".strip()
