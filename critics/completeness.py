"""
critics/completeness.py
-----------------------
Completeness Critic — powered by Gemma 3 via Ollama (local, free)

Responsibilities:
  - Verify that all parts of the original prompt/question are answered
  - Identify important topics that are mentioned but not explained sufficiently
  - Flag requirements that appear in the prompt but are absent from the response
  - Check for premature conclusions or cut-off explanations
  - Assess whether examples, edge cases, or caveats are appropriately included

Why Gemma 3 (local)?
  Completeness checking is primarily a gap-finding task — comparing what was
  asked against what was answered. Gemma 3 handles this comprehension task
  well and runs entirely on-device via Ollama, costing nothing and
  preserving privacy.
"""

from __future__ import annotations

from critics.base import BaseCritic
from config import settings
from models.critique import CriticDimension, ArbitrationRequest


class CompletenessCritic(BaseCritic):
    """
    Evaluates whether the LLM response fully addresses the original prompt.

    Checks for:
      • Unanswered questions — explicit questions in the prompt that are ignored
      • Missing requirements — the prompt asked for X, Y, Z; only X was addressed
      • Shallow coverage — a topic is mentioned but not explained to a useful depth
      • Missing examples — the prompt or context implies examples are needed
      • Missing caveats — important limitations or conditions are not mentioned
      • Truncated response — the response appears to end abruptly
    """

    dimension:  CriticDimension = CriticDimension.COMPLETENESS
    provider:   str             = "ollama"
    model_name: str             = settings.completeness_model

    def _build_prompt(self, request: ArbitrationRequest) -> str:
        context_block = (
            f"\n\n<context>\n{request.context}\n</context>"
            if request.context
            else ""
        )

        return f"""
You are evaluating an AI response for COMPLETENESS.
Your job is to check whether the response fully answers everything the prompt asked for.

<original_prompt>
{request.original_prompt}
</original_prompt>

<llm_response>
{request.llm_response}
</llm_response>
{context_block}

## Evaluation Criteria

Completeness (0–100) — the degree to which the response addresses every distinct
requirement, question, and sub-topic raised by the original prompt. A complete
response leaves no explicit requirement unanswered and covers implicit
requirements to a depth the prompt clearly calls for.

## Evaluation Steps

Work through the following steps in order before producing your final score:

1. Decompose the original prompt into its distinct requirements:
     - Explicit questions (anything ending in "?")
     - Explicit tasks ("explain X", "list Y", "compare Z")
     - Implicit expectations (if the prompt asks how something works, an
       example is implicitly expected; if it asks for advice, caveats are
       implicitly expected)

2. For each requirement identified in step 1, check whether the response:
     - Fully addresses it (clear, complete coverage)
     - Partially addresses it (mentioned but not explained sufficiently)
     - Does not address it at all (omitted entirely)

3. For each gap (partial or missing):
     a. State what specific requirement was not met.
     b. Use the response text itself (or "(general)" if no quote applies) as the quote.
     c. Explain why the omission makes the response incomplete.
     d. Rate severity (CRITICAL / MAJOR / MINOR / INFO — see below).
     e. Give a recommendation describing what should be added.

4. Check for truncation: does the response appear to end abruptly before
   finishing a thought or list?

5. Assign a score from 0–100 based on the proportion of requirements met and
   the depth of coverage.

## Severity Guide

  CRITICAL: A core part of the prompt is completely unaddressed.
  MAJOR:    An important requirement is only partially addressed.
  MINOR:    A useful-but-optional elaboration is missing.
  INFO:     The response could be more thorough but is functionally complete.

## Scoring Guidance

  90–100: Every part of the prompt is addressed thoroughly.
  75–89:  Mostly complete; one minor point could use more depth.
  60–74:  Some sections are shallow or a secondary question is missed.
  40–59:  Multiple significant gaps; response only partially helps the user.
  0–39:   The response addresses very little of what was asked.

## Confidence

  High   (0.8–1.0): The prompt's requirements are clear and unambiguous.
  Medium (0.5–0.8): Prompt intent is somewhat implicit.
  Low    (0.0–0.5): The prompt is ambiguous and "completeness" is subjective.

Set `passed = true` only if there are NO critical or major issues.
Dimension must be "completeness".
Model used: "{self.model_name}"
""".strip()
