"""
models/critique.py
------------------
Pydantic v2 models that define the structured output contract for every critic.

All LLM responses are coerced into these shapes via the Instructor library,
giving us type-safe, validated critique objects regardless of which model
produced them.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────

class Severity(str, Enum):
    """How badly an issue degrades the response quality."""
    CRITICAL = "critical"   # Must fix — response is wrong / harmful
    MAJOR    = "major"      # Should fix — noticeably hurts quality
    MINOR    = "minor"      # Nice to fix — small polish issue
    INFO     = "info"       # Observation only, no fix needed

    @classmethod
    def _missing_(cls, value: object):
        """Accept uppercase values from LLMs e.g. 'CRITICAL' → Severity.CRITICAL"""
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return None


class CriticDimension(str, Enum):
    """The evaluation axis each critic is responsible for."""
    ACCURACY     = "accuracy"
    LOGIC        = "logic"
    COMPLETENESS = "completeness"
    SAFETY       = "safety"
    STYLE        = "style"

    @classmethod
    def _missing_(cls, value: object):
        """Accept uppercase values from LLMs e.g. 'ACCURACY' → CriticDimension.ACCURACY"""
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return None


# ─────────────────────────────────────────────
#  Sub-models
# ─────────────────────────────────────────────

class Issue(BaseModel):
    """A single problem found by a critic."""

    quote: str = Field(
        description="Exact verbatim excerpt from the evaluated response that "
                    "contains the issue. Use '(general)' if not tied to a specific quote."
    )
    explanation: str = Field(
        description="Clear explanation of why this is a problem."
    )
    severity: str = Field(
        description='How serious this issue is. Must be one of: "critical", "major", "minor", "info".'
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Supporting evidence or citation backing up the claim."
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Concrete suggestion for how the issue should be fixed."
    )

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v):
        """Normalise any casing or alias to a valid lowercase severity string."""
        if isinstance(v, Severity):
            return v.value
        if isinstance(v, str):
            norm = v.lower().strip()
            if norm in ("error", "high", "blocker"):
                return "critical"
            if norm in ("warn", "warning", "medium"):
                return "major"
            if norm in ("low", "note", "suggestion"):
                return "minor"
            if norm in ("critical", "major", "minor", "info"):
                return norm
        return "info"  # safe fallback

    @property
    def severity_value(self) -> str:
        """Severity as a lowercase string (e.g. 'critical')."""
        return self.severity if isinstance(self.severity, str) else self.severity.value

class Critique(BaseModel):
    """
    The complete structured output returned by one critic agent.

    Score ranges 0–100:
        0–39  = Poor      (multiple critical issues)
        40–59 = Fair      (significant problems)
        60–74 = Adequate  (minor issues)
        75–89 = Good      (mostly correct)
        90–100 = Excellent (near-perfect for this dimension)
    """

    dimension: CriticDimension = Field(
        description="Which evaluation dimension this critique covers."
    )
    score: int = Field(
        ge=0, le=100,
        description="Overall quality score for this dimension (0–100)."
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="How confident the critic is in its own assessment (0.0–1.0)."
    )

    @field_validator("dimension", mode="before")
    @classmethod
    def coerce_dimension(cls, v):
        if isinstance(v, CriticDimension):
            return v
        if isinstance(v, str):
            return CriticDimension(v.lower())
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        if v is None:
            return 0.5
        return v
    summary: str = Field(
        description="2–4 sentence human-readable summary of the critique."
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="List of specific issues found. Empty list means no problems detected."
    )
    passed: bool = Field(
        description="True if the response meets acceptable quality for this dimension "
                    "(i.e., no CRITICAL or MAJOR issues found)."
    )
    model_used: str = Field(
        description="Name/ID of the LLM that produced this critique."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Time in milliseconds the critic took to produce this output."
    )

    @field_validator("passed", mode="before")
    @classmethod
    def auto_derive_passed(cls, v, info):
        if v is None:
            issues = info.data.get("issues", [])
            blocking = {"critical", "major"}
            return not any(
                (i.severity if isinstance(i.severity, str) else i.severity.value) in blocking
                for i in issues
            )
        return v

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in (Severity.CRITICAL, "critical"))

    @property
    def major_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in (Severity.MAJOR, "major"))

    @property
    def grade(self) -> str:
        """Human-readable letter grade."""
        if self.score >= 90:
            return "A"
        elif self.score >= 75:
            return "B"
        elif self.score >= 60:
            return "C"
        elif self.score >= 40:
            return "D"
        return "F"


# ─────────────────────────────────────────────
#  Arbitration-level models
# ─────────────────────────────────────────────

class ArbitrationVerdict(BaseModel):
    """
    The final synthesized verdict produced by the Adjudicator after all
    individual critics have returned their Critique objects.
    """

    overall_score: int = Field(
        ge=0, le=100,
        description="Weighted aggregate score across all critic dimensions."
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Aggregate confidence, accounting for critic agreement/disagreement."
    )
    approved: bool = Field(
        description="True if the response clears the quality bar for all dimensions."
    )
    dimension_scores: dict[CriticDimension, int] = Field(
        description="Per-dimension scores for quick dashboard display."
    )
    critic_agreement: float = Field(
        ge=0.0, le=1.0,
        description="How much critics agreed with each other (0 = total disagreement, "
                    "1 = unanimous). High disagreement is itself a useful signal."
    )
    key_issues: list[Issue] = Field(
        default_factory=list,
        description="Top-priority issues surfaced across all critics, deduplicated."
    )
    recommendation: str = Field(
        description="Adjudicator's overall recommendation: APPROVE / REVISE / REJECT."
    )
    explanation: str = Field(
        description="3–6 sentence explanation synthesising all critic findings."
    )
    critiques: list[Critique] = Field(
        default_factory=list,
        description="The raw individual critiques that fed into this verdict."
    )


class ArbitrationRequest(BaseModel):
    """Input payload for the arbitration pipeline."""

    original_prompt: str = Field(
        description="The prompt that was sent to the LLM being evaluated."
    )
    llm_response: str = Field(
        description="The LLM-generated response to evaluate."
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional background context or ground truth to aid evaluation."
    )
    requested_critics: Optional[list[CriticDimension]] = Field(
        default=None,
        description="Which critics to run. Defaults to all five if not specified."
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied ID for tracking / storage."
    )
    critic_weights: Optional[dict[CriticDimension, float]] = Field(
        default=None,
        description=(
            "Optional per-dimension weights used when computing the overall score. "
            "Keys must be valid CriticDimension values; values must be floats that "
            "sum to 1.0 (±0.01 tolerance). Omit to use the system defaults: "
            "accuracy=0.30, logic=0.25, safety=0.20, completeness=0.15, style=0.10."
        ),
    )

    @model_validator(mode="after")
    def validate_critic_weights(self) -> "ArbitrationRequest":
        """Ensure provided weights are positive and sum to 1.0."""
        if self.critic_weights is None:
            return self
        weights = self.critic_weights
        if not weights:
            raise ValueError("critic_weights must not be empty if provided.")
        for dim, w in weights.items():
            if w < 0:
                raise ValueError(
                    f"Weight for '{dim.value}' is negative ({w}). All weights must be >= 0."
                )
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"critic_weights must sum to 1.0 (got {total:.4f}). "
                "Adjust your weights so they add up to exactly 1.0."
            )
        return self



# ─────────────────────────────────────────────
#  Phase 2 — Orchestration models
# ─────────────────────────────────────────────

class FallbackInfo(BaseModel):
    """Records when a critic had to use its backup model."""
    dimension: CriticDimension
    primary_model: str
    fallback_model: str
    reason: str


class DisagreementReport(BaseModel):
    """
    Captures meaningful disagreements between critics that the
    adjudicator should resolve before synthesizing a verdict.
    """
    has_disagreement: bool = Field(
        description="True if meaningful disagreement exists among critics."
    )
    score_variance: float = Field(
        ge=0.0,
        description="Standard deviation of the per-critic scores."
    )
    pass_fail_conflicts: list[str] = Field(
        default_factory=list,
        description="Dimensions where critics gave conflicting pass/fail signals."
    )
    severity_conflicts: list[str] = Field(
        default_factory=list,
        description="Specific issues where severity ratings differ significantly."
    )
    confidence_outliers: list[str] = Field(
        default_factory=list,
        description="Critics whose confidence score is an outlier (>1.5 stdev from mean)."
    )
    unique_issues: list[str] = Field(
        default_factory=list,
        description="Issues flagged by exactly one critic that others missed entirely."
    )
    summary: str = Field(
        description="Human-readable summary of the disagreements found."
    )


class DismissedIssue(BaseModel):
    """
    An issue raised by a critic that the adjudicator reviewed and dismissed.
    Imported here so FinalVerdict can carry dismissed_issues without a
    circular dependency on orchestration/adjudicator.py.
    """
    dimension: CriticDimension = Field(
        description="Which critic raised this issue."
    )
    original_issue: str = Field(
        description="Verbatim or paraphrased text of the original critic concern."
    )
    dismissal_reason: str = Field(
        description="Evidence-based reason why this issue does not affect quality."
    )


class FinalVerdict(BaseModel):
    """
    The complete, pipeline-level output delivered to the caller.

    Phase 3 additions:
      - quality_score     : 1-10 holistic score (adjudicator) or derived from overall_score
      - executive_summary : one-paragraph official arbitration report for the end user
      - strengths         : specific positive qualities confirmed by the adjudicator
      - dismissed_issues  : issues raised by critics but ruled non-problems by adjudicator
    """
    # ── Core verdict ─────────────────────────────────────────────────────
    overall_score: int = Field(ge=0, le=100)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    approved: bool
    recommendation: str = Field(
        description="APPROVE / REVISE / REJECT"
    )
    explanation: str = Field(
        description=(
            "3-5 sentence internal synthesis used by nodes.py. "
            "End users should read executive_summary instead."
        )
    )

    # ── Phase 3 enriched report fields ───────────────────────────────────
    quality_score: int = Field(
        default=0,
        ge=0, le=10,
        description=(
            "Holistic 1-10 quality score set by the adjudicator, or derived from "
            "overall_score (divide by 10, round) when no adjudication was used. "
            "0 = not computed."
        )
    )
    executive_summary: str = Field(
        default="",
        description=(
            "One paragraph (4-6 sentences) that IS the official arbitration report "
            "returned to the end user. States recommendation, key strengths, most "
            "important issues, and required actions. Empty when adjudicator was not used."
        )
    )
    strengths: list[str] = Field(
        default_factory=list,
        description=(
            "Specific positive qualities of the response identified by the adjudicator "
            "(e.g., 'Correctly explains TLS as asymmetric for key exchange'). "
            "Empty when adjudicator was not used."
        )
    )
    dismissed_issues: list[DismissedIssue] = Field(
        default_factory=list,
        description=(
            "Issues raised by individual critics that the adjudicator reviewed and "
            "determined do not materially affect response quality. Each entry includes "
            "the original concern and the evidence-based reason for dismissal."
        )
    )

    # ── Per-dimension breakdown ───────────────────────────────────────────
    dimension_scores: dict[str, int] = Field(
        description="Score per CriticDimension value string."
    )
    dimension_passed: dict[str, bool] = Field(
        description="Pass/fail per CriticDimension value string."
    )

    # ── Key issues surfaced across all critics ────────────────────────────
    key_issues: list[Issue] = Field(default_factory=list)

    # ── Agreement signal ──────────────────────────────────────────────────
    critic_agreement: float = Field(ge=0.0, le=1.0)
    disagreement_report: Optional[DisagreementReport] = None

    # ── Provenance ────────────────────────────────────────────────────────
    critiques: list[Critique] = Field(default_factory=list)
    fallbacks_used: list[FallbackInfo] = Field(default_factory=list)
    adjudication_used: bool = Field(
        default=False,
        description="True if the adjudicator LLM was invoked (disagreement path)."
    )
    fast_path: bool = Field(
        default=False,
        description="True if unanimous-pass fast path was taken (no adjudication needed)."
    )
    total_latency_ms: Optional[float] = None
    session_id: Optional[str] = None
