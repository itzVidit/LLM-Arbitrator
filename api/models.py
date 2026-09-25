"""
api/models.py
-------------
FastAPI request/response Pydantic models for the Arbitration API.

These are the wire-format models — separate from the internal pipeline
models in models/critique.py.  They are optimised for clean OpenAPI docs
and stable JSON serialisation, and translate to/from the internal types
in the route handlers.

Models
------
  Request
    ArbitrateRequest        POST /v1/arbitrate
    BatchArbitrateRequest   POST /v1/arbitrate/batch

  Response
    IssueOut                one issue in a response
    CritiqueOut             one critic's result in a response
    ArbitrateResponse       full verdict response (single + batch item)
    VerdictSummary          lightweight list-view row (GET /v1/arbitrations)
    BatchArbitrateResponse  wrapper for batch results

  Analytics
    CriticStats             per-critic aggregated stats
    AnalyticsReport         full GET /v1/analytics response
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
#  Request models
# ─────────────────────────────────────────────────────────────────────

class ArbitrateRequest(BaseModel):
    """
    Request body for POST /v1/arbitrate.
    Evaluates a single LLM response through the full five-critic pipeline.
    """

    llm_response: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="The LLM-generated text to evaluate.",
        examples=["Paris is the capital of France and has a population of about 2 million."],
    )
    original_prompt: str = Field(
        default="",
        max_length=8_000,
        description=(
            "The prompt that produced the response. "
            "Strongly recommended — improves completeness evaluation accuracy."
        ),
        examples=["What is the capital of France?"],
    )
    context: Optional[str] = Field(
        default=None,
        max_length=8_000,
        description="Optional ground truth or background context for the evaluators.",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Caller-supplied session/trace identifier stored with the verdict.",
    )
    critics: Optional[list[str]] = Field(
        default=None,
        description=(
            "Subset of critics to run: accuracy, logic, completeness, safety, style. "
            "Omit to run all five."
        ),
        examples=[["accuracy", "safety"]],
    )
    critic_weights: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Optional per-dimension weights for the overall score calculation. "
            "Keys: accuracy, logic, completeness, safety, style. "
            "Values must be floats >= 0 that sum to 1.0 (±0.01 tolerance). "
            "Omit to use system defaults: "
            "accuracy=0.30, logic=0.25, safety=0.20, completeness=0.15, style=0.10."
        ),
        examples=[{"accuracy": 0.40, "logic": 0.25, "safety": 0.20, "completeness": 0.10, "style": 0.05}],
    )

    model_config = {"json_schema_extra": {
        "example": {
            "original_prompt": "Explain how HTTPS works.",
            "llm_response": "HTTPS uses SSL to encrypt traffic...",
            "session_id": "demo-001",
        }
    }}


class BatchArbitrateRequest(BaseModel):
    """
    Request body for POST /v1/arbitrate/batch.
    Evaluates up to 10 LLM responses concurrently against the same prompt.
    """

    original_prompt: str = Field(
        default="",
        max_length=8_000,
        description="Shared prompt sent to all responses.",
    )
    responses: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of LLM responses to evaluate (1–10 items).",
    )
    context: Optional[str] = Field(
        default=None,
        max_length=8_000,
        description="Optional ground truth shared across all evaluations.",
    )
    critics: Optional[list[str]] = Field(
        default=None,
        description="Subset of critics to run.  Omit for all five.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "original_prompt": "What is 2+2?",
            "responses": ["4", "It depends on the context.", "22"],
        }
    }}


# ─────────────────────────────────────────────────────────────────────
#  Response sub-models
# ─────────────────────────────────────────────────────────────────────

class IssueOut(BaseModel):
    """A single issue found by a critic or confirmed by the adjudicator."""
    quote: str = Field(description="Verbatim excerpt from the evaluated response.")
    explanation: str = Field(description="Why this is a problem.")
    severity: str = Field(description="critical | major | minor | info")
    evidence: Optional[str] = Field(default=None, description="Supporting evidence.")
    recommendation: str = Field(description="Concrete fix suggestion.")


class DismissedIssueOut(BaseModel):
    """An issue raised by a critic but dismissed by the adjudicator."""
    dimension: str = Field(description="Which critic raised this.")
    original_issue: str = Field(description="The original concern.")
    dismissal_reason: str = Field(description="Why it was dismissed.")


class CritiqueOut(BaseModel):
    """One critic's structured result."""
    dimension: str = Field(description="accuracy | logic | completeness | safety | style")
    model_used: str = Field(description="LLM model that produced this critique.")
    score: int = Field(ge=0, le=100, description="Quality score 0–100.")
    grade: str = Field(description="Letter grade A–F.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence 0–1.")
    passed: bool = Field(description="True if no critical/major issues found.")
    summary: str = Field(description="2–4 sentence critique summary.")
    issues: list[IssueOut] = Field(default_factory=list)
    latency_ms: Optional[float] = Field(default=None)


class ArbitrateResponse(BaseModel):
    """
    Full arbitration verdict returned by POST /v1/arbitrate and
    as items in POST /v1/arbitrate/batch.
    """

    # Identity
    arbitration_id: str = Field(
        description="UUID of the stored arbitration record. Use with GET /v1/arbitrations/{id}."
    )
    session_id: Optional[str] = Field(default=None)
    created_at: str = Field(description="ISO-8601 UTC timestamp.")

    # Verdict
    recommendation: str = Field(description="APPROVE | REVISE | REJECT")
    overall_score: int = Field(ge=0, le=100, description="Weighted aggregate score 0–100.")
    quality_score: int = Field(ge=0, le=10, description="Holistic 1–10 quality score.")
    overall_confidence: float = Field(ge=0.0, le=1.0)
    critic_agreement: float = Field(ge=0.0, le=1.0)
    approved: bool

    # Report text
    executive_summary: str = Field(
        description=(
            "One-paragraph official arbitration report. "
            "Falls back to explanation when adjudicator was not used."
        )
    )
    explanation: str = Field(description="Internal synthesis used for fast-path cases.")
    strengths: list[str] = Field(default_factory=list)

    # Issues
    key_issues: list[IssueOut] = Field(default_factory=list)
    dismissed_issues: list[DismissedIssueOut] = Field(default_factory=list)

    # Per-dimension
    dimension_scores: dict[str, int]
    dimension_passed: dict[str, bool]

    # Critic detail
    critiques: list[CritiqueOut] = Field(default_factory=list)

    # Flags
    adjudication_used: bool
    fast_path: bool
    fallbacks_used: int = Field(description="Number of fallback models used.")
    total_latency_ms: Optional[float] = None


class VerdictSummary(BaseModel):
    """
    Lightweight row returned by GET /v1/arbitrations (list endpoint).
    Does not include the full verdict JSON — use GET /v1/arbitrations/{id}
    to retrieve the complete ArbitrateResponse.
    """
    arbitration_id: str
    session_id: Optional[str] = None
    created_at: str
    recommendation: str
    overall_score: int
    quality_score: int
    overall_confidence: float
    critic_agreement: float
    adjudication_used: bool
    fast_path: bool
    total_latency_ms: Optional[float] = None


class VerdictListResponse(BaseModel):
    """Paginated list of verdict summaries."""
    total: int = Field(description="Total number of arbitrations in the database.")
    limit: int
    offset: int
    items: list[VerdictSummary]


class BatchArbitrateResponse(BaseModel):
    """Response for POST /v1/arbitrate/batch."""
    total: int = Field(description="Number of responses submitted.")
    completed: int = Field(description="Number successfully arbitrated.")
    failed: int = Field(description="Number that errored.")
    results: list[ArbitrateResponse | BatchItemError]


class BatchItemError(BaseModel):
    """Returned for a batch item that failed arbitration."""
    index: int = Field(description="Zero-based index of the failed response.")
    error: str = Field(description="Error message.")


# ─────────────────────────────────────────────────────────────────────
#  Analytics models
# ─────────────────────────────────────────────────────────────────────

class CriticStats(BaseModel):
    """
    Per-critic aggregated statistics across all stored arbitrations.
    Shown in GET /v1/analytics.
    """
    dimension: str = Field(description="Critic dimension name.")
    model_used: str = Field(description="Primary model used by this critic.")

    # Volume
    total_evaluations: int = Field(description="Number of times this critic ran.")

    # Score distribution
    avg_score: float = Field(description="Mean score across all evaluations.")
    min_score: int
    max_score: int
    pass_rate: float = Field(description="Fraction of evaluations that passed (0–1).")

    # Confidence
    avg_confidence: float = Field(description="Mean confidence across all evaluations.")

    # Issue activity
    total_issues_raised: int = Field(description="Total issue count across all runs.")
    avg_issues_per_run: float
    critical_issues_raised: int
    major_issues_raised: int

    # Overruling
    overrule_count: int = Field(
        description="Times the adjudicator changed this critic's score/pass by >10pts."
    )
    overrule_rate: float = Field(description="overrule_count / total_evaluations (0–1).")
    raised_overrule_count: int = Field(
        description="Times adjudicator raised this critic's score."
    )
    lowered_overrule_count: int = Field(
        description="Times adjudicator lowered this critic's score."
    )

    # Latency
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None

    # Fallback
    fallback_count: int = Field(description="Times the fallback model was used.")
    fallback_rate: float = Field(description="fallback_count / total_evaluations (0–1).")

    # Common issue categories (severities)
    common_categories: list[str] = Field(
        default_factory=list,
        description="Most common issue severity categories seen from this critic.",
    )


class AnalyticsReport(BaseModel):
    """
    System-wide analytics returned by GET /v1/analytics.
    Aggregates all stored arbitration runs.
    """

    # ── Overview ──────────────────────────────────────────────────────
    total_arbitrations: int
    recommendation_counts: dict[str, int] = Field(
        description="Counts per recommendation: {APPROVE: N, REVISE: N, REJECT: N}"
    )
    approve_rate: float
    revise_rate: float
    reject_rate: float

    # ── Score health ──────────────────────────────────────────────────
    avg_overall_score: float
    avg_quality_score: float
    avg_confidence: float
    avg_critic_agreement: float

    # ── Pipeline behaviour ────────────────────────────────────────────
    adjudication_rate: float = Field(
        description="Fraction of runs that required the adjudicator."
    )
    fast_path_rate: float = Field(
        description="Fraction of runs that took the fast path (all critics agreed)."
    )
    avg_latency_ms: Optional[float] = None
    total_fallbacks_used: int

    # ── Per-critic detail ─────────────────────────────────────────────
    critic_stats: list[CriticStats] = Field(
        description="Aggregated statistics for each of the five critics."
    )

    # ── Top critics by activity ───────────────────────────────────────
    most_issues_raised_by: Optional[str] = Field(
        default=None,
        description="Dimension that raises the most issues on average.",
    )
    most_overruled_critic: Optional[str] = Field(
        default=None,
        description="Dimension most frequently overruled by the adjudicator.",
    )
    highest_confidence_critic: Optional[str] = Field(
        default=None,
        description="Dimension with the highest average confidence.",
    )
    lowest_confidence_critic: Optional[str] = Field(
        default=None,
        description="Dimension with the lowest average confidence.",
    )
    most_common_issue_category: Optional[str] = Field(
        default=None,
        description="Issue severity category that appears most across all critics.",
    )

    # ── Timestamp ─────────────────────────────────────────────────────
    generated_at: str = Field(description="ISO-8601 UTC timestamp of report generation.")
