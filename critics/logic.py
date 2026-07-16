"""
critics/logic.py
----------------
Logic Critic — powered by DeepSeek R1 Distill via Groq

Responsibilities:
  - Evaluate the reasoning chain and argumentation structure
  - Detect logical fallacies (false dichotomies, straw man, circular reasoning)
  - Flag invalid conclusions that don't follow from the stated premises
  - Identify broken or missing reasoning steps
  - Spot self-contradictions within the argument

Why DeepSeek R1?
  R1 is a Chain-of-Thought model trained specifically on reasoning tasks.
  It approaches problems by explicitly working through intermediate steps,
  making it especially well-suited to catch flawed reasoning that other
  models miss. Groq's hardware provides sub-second token generation.
"""

from __future__ import annotations

from critics.base import BaseCritic
from config import settings
from models.critique import CriticDimension, ArbitrationRequest


class LogicCritic(BaseCritic):
    """
    Evaluates the logical soundness and reasoning quality of an LLM response.

    Checks for:
      • Invalid inferences — conclusions that don't follow from the premises
      • Missing reasoning steps — jumps in logic with no justification
      • Logical fallacies — ad hominem, straw man, appeal to authority, etc.
      • Circular reasoning — using a conclusion as its own justification
      • False dichotomies — presenting two options when more exist
      • Internal contradictions — statements that logically cannot both be true
      • Causal errors — confusing correlation with causation
    """

    dimension:          CriticDimension = CriticDimension.LOGIC
    provider:           str             = "groq"
    model_name:         str             = settings.logic_model
    fallback_provider:  str             = settings.logic_fallback_provider
    fallback_model:     str             = settings.logic_fallback_model

    def _build_prompt(self, request: ArbitrationRequest) -> str:
        context_block = (
            f"\n\n<context>\n{request.context}\n</context>"
            if request.context
            else ""
        )

        return f"""
You are evaluating an AI response for LOGICAL CONSISTENCY and sound REASONING.

<original_prompt>
{request.original_prompt}
</original_prompt>

<llm_response>
{request.llm_response}
</llm_response>
{context_block}

## Your Task

Work through the response step by step as a formal reasoner would.
For each logical problem you find:
  1. Quote the EXACT text where the logic breaks down.
  2. Name the specific logical flaw (e.g., "non sequitur", "hasty generalisation").
  3. Explain why the reasoning is invalid.
  4. Rate severity:
       - CRITICAL: The conclusion is completely unsupported; the entire argument fails.
       - MAJOR: A significant reasoning step is flawed, undermining the conclusion.
       - MINOR: A small inferential gap that doesn't break the overall argument.
       - INFO: Imprecise language that could imply a weak argument but isn't wrong.
  5. Provide corrected or stronger reasoning as a recommendation.

## Scoring Guidance

  90–100: Impeccable reasoning; every conclusion follows from stated premises.
  75–89:  Mostly sound with one minor inferential gap.
  60–74:  Some reasoning flaws that weaken the argument.
  40–59:  Multiple flawed steps; conclusions are questionable.
  0–39:   Reasoning is fundamentally broken; conclusions are unjustified.

## Confidence

  High (0.8–1.0): The logical structure is clear and easy to evaluate.
  Medium (0.5–0.8): The argument is complex or domain-specific.
  Low (0.0–0.5): Heavy domain knowledge is required to assess the reasoning.

Set `passed = true` only if there are NO critical or major issues.
Dimension must be "logic".
Model used: "{self.model_name}"
""".strip()
