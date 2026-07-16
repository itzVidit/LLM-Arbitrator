"""
critics/style.py
----------------
Style Critic — powered by Phi-4 Mini via Ollama (local, free)

Responsibilities:
  - Evaluate readability and clarity of expression
  - Check grammar, spelling, and punctuation
  - Assess formatting (headers, lists, code blocks used appropriately)
  - Judge professional tone and appropriateness for the context
  - Flag verbose, redundant, or unclear passages
  - Check consistency of terminology and voice

Why Phi-4 Mini (local)?
  Style evaluation is a language task that doesn't require vast world knowledge —
  it's about prose quality. Phi-4 Mini is a compact, highly capable language
  model that excels at text analysis tasks while running entirely on-device,
  making it the perfect cost-zero style checker.
"""

from __future__ import annotations

from critics.base import BaseCritic
from config import settings
from models.critique import CriticDimension, ArbitrationRequest


class StyleCritic(BaseCritic):
    """
    Evaluates the writing quality, clarity, and professionalism of an LLM response.

    Checks for:
      • Grammar and spelling errors — incorrect word forms, typos, punctuation
      • Clarity — ambiguous pronouns, unclear referents, confusing sentence structure
      • Readability — overly long sentences, passive voice overuse, jargon without explanation
      • Formatting — appropriate use of headers, bullet points, code blocks
      • Tone — appropriate formality for the context; not condescending or overly casual
      • Conciseness — unnecessary repetition, padding, or filler phrases
      • Consistency — uniform terminology, consistent tense, consistent capitalization
    """

    dimension:  CriticDimension = CriticDimension.STYLE
    provider:   str             = "ollama"
    model_name: str             = settings.style_model

    def _build_prompt(self, request: ArbitrationRequest) -> str:
        context_block = (
            f"\n\n<context>\n{request.context}\n</context>"
            if request.context
            else ""
        )

        return f"""
You are a WRITING QUALITY EVALUATOR. Assess the style, clarity, grammar,
and professionalism of the AI response below.

<original_prompt>
{request.original_prompt}
</original_prompt>

<llm_response>
{request.llm_response}
</llm_response>
{context_block}

## Style Dimensions to Evaluate

1. GRAMMAR & MECHANICS: Subject-verb agreement, correct word forms, punctuation,
   spelling, capitalization.

2. CLARITY: Is each sentence easy to understand on first reading? Are pronouns
   and references unambiguous?

3. CONCISENESS: Is every sentence pulling its weight? Flag padding, redundancy,
   and filler phrases (e.g., "it is important to note that…").

4. FORMATTING: Does the response use formatting (headers, bullets, code blocks)
   appropriately for the content type and platform?

5. TONE: Is the tone appropriate for a helpful AI assistant? Flag condescension,
   excessive hedging, or unprofessional language.

6. CONSISTENCY: Are technical terms, abbreviations, and style consistent throughout?

## For Each Issue Found

  1. Quote the EXACT text with the style problem.
  2. Explain the specific style issue.
  3. Rate severity:
       - CRITICAL: Errors that make the response difficult to understand or embarrassing.
       - MAJOR: Significant style problems that hurt readability or professionalism.
       - MINOR: Polish issues — small improvements that would make it better.
       - INFO: Optional suggestion; purely a matter of preference.
  4. Provide a corrected version as your recommendation.

## Scoring Guidance

  90–100: Polished, professional prose with no meaningful style issues.
  75–89:  Well-written with 1–2 minor imperfections.
  60–74:  Readable but with several noticeable style problems.
  40–59:  Multiple issues that make the response harder to read or less professional.
  0–39:   Poor writing quality that significantly degrades the user experience.

Note: Style is judged relative to the context. A casual conversational reply
should be scored differently from a technical documentation response.

Set `passed = true` if there are NO critical or major style issues.
Dimension must be "style".
Model used: "{self.model_name}"
""".strip()
