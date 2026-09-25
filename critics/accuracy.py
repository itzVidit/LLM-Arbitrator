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
        # G-Eval improvement 3: when context is provided, make it the authoritative
        # source rather than just background material. Instruct the model to anchor
        # its fact-checking to the context first, then fall back to world knowledge.
        if request.context:
            context_block = f"""
<context>
{request.context}
</context>

IMPORTANT: The <context> block above is the authoritative source of truth for
this evaluation. Treat any claim in the response that contradicts the context
as factually incorrect, regardless of what you know from your training data.
Only use your own world knowledge to check claims that the context does not cover.
"""
        else:
            context_block = ""

        return f"""
You are evaluating an AI response for FACTUAL ACCURACY.

<original_prompt>
{request.original_prompt}
</original_prompt>

<llm_response>
{request.llm_response}
</llm_response>
{context_block}
## Evaluation Criteria

Accuracy (0–100) — the degree to which every factual claim in the response is
correct, verifiable, and consistent with established knowledge (and with the
provided context, if any). A high-accuracy response contains no invented details,
no contradictions, and no claims that would mislead an informed reader.

## Evaluation Steps

Work through the following steps in order before producing your final score:

1. Identify every distinct factual claim in the response: specific names, dates,
   numbers, causal relationships, definitions, and attributed statements.

2. For each claim, determine its status:
     - CORRECT: Verifiable and accurate.
     - INCORRECT: Contradicts established knowledge or the provided context.
     - UNVERIFIABLE: Cannot be confirmed or denied with available information.
     - HALLUCINATED: Specific-sounding detail (citation, statistic, name) that
       appears to be invented with no real basis.

3. For every INCORRECT or HALLUCINATED claim:
     a. Quote the exact text.
     b. State what the correct information is and cite your basis.
     c. Classify severity (CRITICAL / MAJOR / MINOR / INFO — see below).
     d. Write a concrete fix recommendation.

4. Consider whether errors compound: a single wrong fact buried in an otherwise
   correct response is less severe than a chain of errors that lead to a wrong
   conclusion.

5. Assign a score from 0–100 based on the proportion and severity of errors found.

## Severity Guide

  CRITICAL: The error fundamentally misleads the user or could cause real harm
            (wrong medical dose, incorrect security advice, invented citations).
  MAJOR:    The claim is clearly wrong and will cause meaningful confusion.
  MINOR:    Small factual slip (wrong year, minor name error) that doesn't change
            the core message.
  INFO:     Technically imprecise but not harmful or misleading.

## Scoring Guidance

  90–100: No factual errors; all claims accurate and well-supported.
  75–89:  Mostly accurate with 1–2 minor slips that don't mislead.
  60–74:  Some inaccuracies that could confuse an informed reader.
  40–59:  Multiple factual errors; response is partly unreliable.
  0–39:   Pervasive factual problems; response cannot be trusted.

## Confidence

Set confidence (0.0–1.0) based on how certain you are of your own assessment:
  High   (0.8–1.0): Topic is well within your knowledge; errors are clear.
  Medium (0.5–0.8): Some uncertainty; topic may be specialised.
  Low    (0.0–0.5): You cannot reliably verify claims in this domain.

Set `passed = true` only if there are NO critical or major issues.
Dimension must be "accuracy".
Model used: "{self.model_name}"
""".strip()
