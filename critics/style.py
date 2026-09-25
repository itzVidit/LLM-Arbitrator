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

## Evaluation Criteria

Style (0–100) — the degree to which the response is well-written, clear,
grammatically correct, appropriately formatted, and professional in tone.
A high-style response communicates effectively with no unnecessary friction
for the reader — it is concise, consistent, and suited to the context.

## Evaluation Steps

Work through the following steps in order before producing your final score:

1. Assess grammar and mechanics:
     - Check subject-verb agreement, correct word forms, punctuation, spelling,
       and capitalization.
     - Identify any sentence fragments or run-on sentences.

2. Assess clarity:
     - Read each sentence and ask: "Would a first-time reader understand this
       immediately?" Flag ambiguous pronoun references, unclear antecedents,
       and confusing sentence structure.

3. Assess conciseness:
     - Flag padding phrases ("it is important to note that", "as mentioned
       above", "in conclusion"), unnecessary repetition, and sentences that
       could be half as long with no loss of meaning.

4. Assess formatting:
     - Is the use of headers, bullet points, numbered lists, and code blocks
       appropriate for the content type and expected rendering context?
     - Are formatting choices consistent throughout the response?

5. Assess tone:
     - Is the tone appropriate for a helpful AI assistant responding to this
       specific prompt? Flag condescension, excessive hedging, unprofessional
       language, or an inappropriately casual/formal register.

6. Assess consistency:
     - Are technical terms, abbreviations, and capitalization consistent
       throughout the response?

7. For each issue found:
     a. Distinguish between a genuine quality problem (reduces clarity or
        professionalism) and a personal preference (valid style choice the
        critic simply dislikes). Only flag genuine quality problems.
     b. Quote the exact text with the issue.
     c. Explain the specific style problem.
     d. Rate severity (CRITICAL / MAJOR / MINOR / INFO — see below).
     e. Provide a corrected version as a recommendation.

8. Assign a score from 0–100 based on the number and severity of genuine
   style issues found. Do not penalise for stylistic preferences.

## Severity Guide

  CRITICAL: Errors that make the response difficult to understand or
            embarrassing to share.
  MAJOR:    Significant style problems that hurt readability or professionalism.
  MINOR:    Polish issues — small improvements that would make it better.
  INFO:     Optional suggestion; purely a matter of preference.

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
