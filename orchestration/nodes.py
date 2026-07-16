"""
orchestration/nodes.py
----------------------
All six node functions that make up the LangGraph arbitration pipeline.

Node execution order (see graph.py for the wiring):

  parse_input
      |
  [parallel fan-out]
      run_accuracy_critic
      run_logic_critic
      run_completeness_critic
      run_safety_critic
      run_style_critic
  [fan-in: all critiques collected]
      |
  detect_disagreement
      |-- YES --> adjudicate --> synthesize_verdict
      |-- NO  ---------------> synthesize_verdict

Each node function:
  - Receives the full GraphState
  - Returns a dict of fields to update (LangGraph merges these in)
  - Is synchronous (async work is run via asyncio.new_event_loop)
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from typing import Any

from models.critique import (
    ArbitrationRequest,
    Critique,
    CriticDimension,
    DismissedIssue,
    DisagreementReport,
    FallbackInfo,
    FinalVerdict,
    Issue,
    Severity,
)
from critics import (
    AccuracyCritic,
    LogicCritic,
    CompletenessCritic,
    SafetyCritic,
    StyleCritic,
)
from orchestration.state import GraphState
from orchestration.adjudicator import FinalAdjudicator, AdjudicationResult
from config import settings

log = logging.getLogger(__name__)

# Dimension weights for weighted overall score (must sum to 1.0)
DIMENSION_WEIGHTS: dict[CriticDimension, float] = {
    CriticDimension.ACCURACY:     0.30,
    CriticDimension.LOGIC:        0.25,
    CriticDimension.SAFETY:       0.20,
    CriticDimension.COMPLETENESS: 0.15,
    CriticDimension.STYLE:        0.10,
}


# ===========================================================
# Node 1 -- parse_input
# ===========================================================

def parse_input(state: GraphState) -> dict[str, Any]:
    """
    Validates and normalises the incoming ArbitrationRequest.
    Initialises all list fields that parallel nodes will append to.
    """
    request: ArbitrationRequest = state["request"]
    session_id = request.session_id or str(uuid.uuid4())[:8]

    log.info(
        "[parse_input] session=%s  prompt_len=%d  response_len=%d",
        session_id,
        len(request.original_prompt),
        len(request.llm_response),
    )

    # Default: run all critics
    if request.requested_critics is None:
        request = request.model_copy(
            update={"requested_critics": list(CriticDimension)}
        )

    return {
        "request": request,
        "session_id": session_id,
        "pipeline_start_ms": time.perf_counter() * 1000,
        "critiques": [],
        "fallbacks_used": [],
        "errors": [],
        "disagreement_report": None,
        "needs_adjudication": False,
        "adjudication_notes": None,
        "final_verdict": None,
    }


# ===========================================================
# Nodes 2a-2e -- parallel critic dispatchers
# ===========================================================

def _make_critic_node(critic_class):
    """
    Factory that returns a synchronous LangGraph node for a given critic class.
    Each returned node runs the critic's async evaluate() in a fresh event loop.
    """

    async def _async_run(request: ArbitrationRequest, critic_instance):
        critique = await critic_instance.evaluate(request)
        return critique, critic_instance.used_fallback

    def node(state: GraphState) -> dict[str, Any]:
        request: ArbitrationRequest = state["request"]
        critic_instance = critic_class()
        dimension = critic_instance.dimension

        # Respect requested_critics filter
        if (
            request.requested_critics is not None
            and dimension not in request.requested_critics
        ):
            log.debug("[%s] Skipped (not in requested_critics)", dimension.value)
            return {}

        log.info("[%s] Dispatched", dimension.value)
        try:
            loop = asyncio.new_event_loop()
            try:
                critique, used_fallback = loop.run_until_complete(
                    _async_run(request, critic_instance)
                )
            finally:
                loop.close()
        except Exception as exc:
            log.error("[%s] Node-level error: %s", dimension.value, exc, exc_info=True)
            critique = Critique(
                dimension=dimension,
                score=0,
                confidence=0.0,
                summary=f"Critic node failed: {exc}",
                issues=[
                    Issue(
                        quote="(node error)",
                        explanation=str(exc),
                        severity=Severity.INFO,
                        recommendation="Check logs for details.",
                    )
                ],
                passed=False,
                model_used=critic_instance.model_name,
            )
            used_fallback = False

        updates: dict[str, Any] = {"critiques": [critique]}

        if used_fallback:
            fb = FallbackInfo(
                dimension=dimension,
                primary_model=critic_instance.model_name,
                fallback_model=getattr(critic_instance, "fallback_model", "unknown"),
                reason=f"Primary model unavailable; used fallback for {dimension.value}",
            )
            updates["fallbacks_used"] = [fb]

        return updates

    node.__name__ = f"run_{critic_class.__name__.lower()}"
    return node


# One node per critic
run_accuracy_critic     = _make_critic_node(AccuracyCritic)
run_logic_critic        = _make_critic_node(LogicCritic)
run_completeness_critic = _make_critic_node(CompletenessCritic)
run_safety_critic       = _make_critic_node(SafetyCritic)
run_style_critic        = _make_critic_node(StyleCritic)


# ===========================================================
# Node 3 -- detect_disagreement
# ===========================================================

def detect_disagreement(state: GraphState) -> dict[str, Any]:
    """
    Analyses collected critiques for meaningful disagreements.

    Flags disagreement when:
      - Score std-dev >= settings.disagreement_variance_threshold
      - Any critics give opposite pass/fail
      - A critic's confidence is a statistical outlier (>1.5 sigma from mean)
      - Critical/major issues appear in only one critic (unique signals)
    """
    critiques: list[Critique] = state.get("critiques", [])

    if not critiques:
        log.warning("[detect_disagreement] No critiques to analyse")
        report = DisagreementReport(
            has_disagreement=False,
            score_variance=0.0,
            summary="No critiques were collected.",
        )
        return {"disagreement_report": report, "needs_adjudication": False}

    scores = [c.score for c in critiques]
    confidences = [c.confidence for c in critiques]

    # --- Score variance ---
    score_variance = statistics.stdev(scores) if len(scores) > 1 else 0.0

    # --- Pass/fail conflicts ---
    failed_dims = [c.dimension.value for c in critiques if not c.passed]
    pass_fail_conflicts = failed_dims  # Non-empty = at least one critic failed

    # --- Confidence outliers (>1.5 sigma below mean) ---
    confidence_outliers: list[str] = []
    if len(confidences) > 2:
        mean_conf = statistics.mean(confidences)
        stdev_conf = statistics.stdev(confidences)
        if stdev_conf > 0:
            confidence_outliers = [
                c.dimension.value
                for c in critiques
                if abs(c.confidence - mean_conf) > 1.5 * stdev_conf
            ]

    # --- Unique critical/major issues (only one critic caught them) ---
    unique_issues: list[str] = []
    if len(critiques) > 1:
        all_major = [
            (c.dimension.value, issue.explanation)
            for c in critiques
            for issue in c.issues
            if issue.severity in (Severity.CRITICAL, Severity.MAJOR, "critical", "major")
        ]
        for dim, explanation in all_major:
            keywords = (
                set(explanation.lower().split())
                - {"the", "a", "an", "is", "of", "in", "to", "that", "this"}
            )
            found_elsewhere = any(
                len(
                    keywords
                    & (
                        set(oi.explanation.lower().split())
                        - {"the", "a", "an", "is", "of", "in", "to", "that", "this"}
                    )
                )
                >= 3
                for c in critiques
                if c.dimension.value != dim
                for oi in c.issues
            )
            if not found_elsewhere:
                unique_issues.append(f"{dim}: {explanation[:60]}")

    # --- Final decision ---
    has_disagreement = (
        score_variance >= settings.disagreement_variance_threshold
        or len(confidence_outliers) > 0
    )

    parts = []
    if score_variance >= settings.disagreement_variance_threshold:
        parts.append(f"score variance sigma={score_variance:.1f}")
    if pass_fail_conflicts:
        parts.append(f"failed dimensions: {', '.join(pass_fail_conflicts)}")
    if confidence_outliers:
        parts.append(f"confidence outliers: {', '.join(confidence_outliers)}")
    if unique_issues:
        parts.append(f"{len(unique_issues)} unique issue(s) from single critics")

    summary = (
        ("Critics disagree: " + "; ".join(parts) + ".")
        if parts
        else (
            f"Critics broadly aligned "
            f"(sigma={score_variance:.1f}, scores={scores})."
        )
    )

    log.info(
        "[detect_disagreement] variance=%.1f  needs_adjudication=%s",
        score_variance,
        has_disagreement,
    )

    report = DisagreementReport(
        has_disagreement=has_disagreement,
        score_variance=round(score_variance, 2),
        pass_fail_conflicts=pass_fail_conflicts,
        severity_conflicts=[],
        confidence_outliers=confidence_outliers,
        unique_issues=unique_issues,
        summary=summary,
    )

    return {
        "disagreement_report": report,
        "needs_adjudication": has_disagreement,
    }


# ===========================================================
# Node 4 -- adjudicate  (disagreement path only)
# ===========================================================

def adjudicate(state: GraphState) -> dict[str, Any]:
    """
    Calls the Final Adjudicator LLM to resolve critic conflicts.
    Only reached when detect_disagreement sets needs_adjudication=True.
    """
    critiques: list[Critique] = state["critiques"]
    disagreement: DisagreementReport = state["disagreement_report"]
    request: ArbitrationRequest = state["request"]

    log.info("[adjudicate] Invoking Final Adjudicator on %d critiques", len(critiques))

    adjudicator = FinalAdjudicator()

    # Run the async adjudicator in a dedicated thread with its own event loop.
    # This avoids the "Cannot run event loop while another loop is running" error
    # that occurs when LangGraph's own event loop is already active.
    import concurrent.futures

    def _run_adjudicator():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                adjudicator.adjudicate(
                    critiques=critiques,
                    disagreement=disagreement,
                    original_prompt=request.original_prompt,
                    llm_response=request.llm_response,
                )
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_adjudicator)
        result: AdjudicationResult | None = future.result(timeout=120)

    if result is None:
        log.warning("[adjudicate] Adjudicator unavailable; proceeding without resolution")
        return {"adjudication_notes": "Adjudicator unavailable (no API keys configured)."}

    notes = (
        f"RECOMMENDATION: {result.final_recommendation}\n"
        f"CONFIDENCE: {result.confidence:.0%}\n\n"
        f"ASSESSMENT:\n{result.overall_assessment}\n\n"
        f"NOTES:\n{result.adjudication_notes}\n\n"
        f"DIMENSION RESOLUTIONS:\n"
        + "\n".join(
            f"  {r.dimension.value}: score={r.resolved_score}"
            f"  passed={r.resolved_passed}  reason={r.reasoning}"
            for r in result.dimension_resolutions
        )
    )

    return {
        "adjudication_notes": notes,
        "_adjudication_result": result,
    }


# ===========================================================
# Node 5 -- synthesize_verdict
# ===========================================================

def synthesize_verdict(state: GraphState) -> dict[str, Any]:
    """
    Builds the final FinalVerdict from all collected information.

    Fast path: all critics passed with high scores -> unanimous approval.
    Normal path: weighted scores + adjudicator overrides if present.
    """
    critiques: list[Critique]               = state.get("critiques", [])
    fallbacks: list[FallbackInfo]           = state.get("fallbacks_used", [])
    disagreement: DisagreementReport | None = state.get("disagreement_report")
    adj_result: AdjudicationResult | None   = state.get("_adjudication_result")
    request: ArbitrationRequest             = state["request"]
    start_ms: float | None                  = state.get("pipeline_start_ms")
    session_id: str | None                  = state.get("session_id")

    if not critiques:
        verdict = FinalVerdict(
            overall_score=0,
            overall_confidence=0.0,
            approved=False,
            recommendation="REJECT",
            explanation="No critics were able to evaluate the response.",
            dimension_scores={},
            dimension_passed={},
            key_issues=[],
            critic_agreement=0.0,
            fast_path=False,
            adjudication_used=False,
            total_latency_ms=_elapsed(start_ms),
            session_id=session_id,
        )
        return {"final_verdict": verdict}

    # Adjudicator score overrides
    adj_scores: dict[CriticDimension, int]  = {}
    adj_passed: dict[CriticDimension, bool] = {}
    if adj_result:
        for r in adj_result.dimension_resolutions:
            adj_scores[r.dimension]  = r.resolved_score
            adj_passed[r.dimension]  = r.resolved_passed

    # Per-dimension scores (adjudicator takes priority)
    dimension_scores: dict[str, int]  = {}
    dimension_passed: dict[str, bool] = {}
    for c in critiques:
        dimension_scores[c.dimension.value]  = adj_scores.get(c.dimension, c.score)
        dimension_passed[c.dimension.value]  = adj_passed.get(c.dimension, c.passed)

    # Weighted overall score
    total_weight = 0.0
    weighted_sum = 0.0
    for c in critiques:
        w = DIMENSION_WEIGHTS.get(c.dimension, 0.1)
        weighted_sum += dimension_scores[c.dimension.value] * w
        total_weight += w
    overall_score = round(weighted_sum / total_weight) if total_weight > 0 else 0

    # Confidence: mean minus fallback penalty
    mean_conf = (
        sum(c.confidence for c in critiques) / len(critiques) if critiques else 0.0
    )
    penalty = len(fallbacks) * settings.fallback_confidence_penalty
    overall_confidence = max(0.0, round(mean_conf - penalty, 3))

    # Critic agreement (1 - normalised stdev)
    effective_scores = list(dimension_scores.values())
    score_stdev = statistics.stdev(effective_scores) if len(effective_scores) > 1 else 0.0
    critic_agreement = max(0.0, min(1.0, 1.0 - score_stdev / 100.0))

    # Top key issues (CRITICAL then MAJOR, deduplicated, max 5)
    _blocking = (Severity.CRITICAL, Severity.MAJOR, "critical", "major")
    all_issues = [
        issue
        for c in critiques
        for issue in c.issues
        if issue.severity in _blocking
    ]
    if adj_result and adj_result.key_issues:
        all_issues = adj_result.key_issues + all_issues

    seen: set[str] = set()
    key_issues: list[Issue] = []
    sev_order = {"critical": 0, "major": 1, Severity.CRITICAL: 0, Severity.MAJOR: 1}
    for issue in sorted(all_issues, key=lambda i: sev_order.get(i.severity, 9)):
        key = issue.explanation[:40].lower()
        if key not in seen:
            seen.add(key)
            key_issues.append(issue)
        if len(key_issues) >= 5:
            break

    # Recommendation
    any_critical = any(
        i.severity in (Severity.CRITICAL, "critical") for c in critiques for i in c.issues
    )
    all_passed = all(dimension_passed.values()) if dimension_passed else False

    if adj_result:
        recommendation = adj_result.final_recommendation
    elif any_critical:
        recommendation = "REJECT"
    elif not all_passed:
        recommendation = "REVISE"
    else:
        recommendation = "APPROVE"

    approved = recommendation == "APPROVE"

    # Fast path flag
    fast_path = (
        all_passed
        and overall_score >= settings.fast_path_threshold
        and not (disagreement and disagreement.has_disagreement)
    )

    # Human-readable explanation
    if adj_result:
        explanation = adj_result.overall_assessment
    elif fast_path:
        explanation = (
            f"All {len(critiques)} critics passed with an aggregate score of "
            f"{overall_score}/100 (confidence {overall_confidence:.0%}). "
            "No significant issues found. Response is approved."
        )
    else:
        failed = [d for d, p in dimension_passed.items() if not p]
        explanation = (
            f"Aggregate score: {overall_score}/100 across {len(critiques)} critics "
            f"(confidence {overall_confidence:.0%}). "
        )
        if failed:
            explanation += f"Issues in: {', '.join(failed)}. "
        if key_issues:
            top = key_issues[0]
            explanation += (
                f"Top issue [{top.severity if isinstance(top.severity, str) else top.severity.value}]: "
                f"{top.explanation[:100]}. "
            )
        explanation += f"Recommendation: {recommendation}."

    # ── Phase 3 enriched fields from adjudicator ─────────────────────────
    # quality_score: use adjudicator's 1-10 if available; otherwise derive from
    # overall_score by mapping 0-100 → 1-10 (floor at 1 to match schema ge=1 when adj used)
    quality_score: int = 0
    executive_summary: str = ""
    strengths: list[str] = []
    dismissed_issues: list[DismissedIssue] = []

    if adj_result:
        quality_score = adj_result.quality_score
        executive_summary = adj_result.executive_summary
        strengths = list(adj_result.strengths)
        # Convert orchestration DismissedIssue -> models DismissedIssue
        # (they have the same fields; adj result uses the orchestration version
        # but FinalVerdict stores the models version imported above)
        dismissed_issues = [
            DismissedIssue(
                dimension=d.dimension,
                original_issue=d.original_issue,
                dismissal_reason=d.dismissal_reason,
            )
            for d in adj_result.dismissed_issues
        ]
    elif overall_score > 0:
        # Derive a 1-10 equivalent for display when adjudicator was not invoked
        quality_score = max(1, round(overall_score / 10))

    verdict = FinalVerdict(
        overall_score=overall_score,
        overall_confidence=overall_confidence,
        approved=approved,
        recommendation=recommendation,
        explanation=explanation,
        quality_score=quality_score,
        executive_summary=executive_summary,
        strengths=strengths,
        dismissed_issues=dismissed_issues,
        dimension_scores=dimension_scores,
        dimension_passed=dimension_passed,
        key_issues=key_issues,
        critic_agreement=round(critic_agreement, 3),
        disagreement_report=disagreement,
        critiques=critiques,
        fallbacks_used=fallbacks,
        adjudication_used=(adj_result is not None),
        fast_path=fast_path,
        total_latency_ms=_elapsed(start_ms),
        session_id=session_id,
    )

    log.info(
        "[synthesize_verdict] score=%d  rec=%s  fast_path=%s  adj=%s  latency=%.0fms",
        overall_score,
        recommendation,
        fast_path,
        adj_result is not None,
        verdict.total_latency_ms or 0,
    )

    return {"final_verdict": verdict}


# ===========================================================
# Routing function  (used by graph.py conditional edge)
# ===========================================================

def route_after_disagreement_check(state: GraphState) -> str:
    """
    Conditional edge: returns the name of the next node.

    "adjudicate"         -- critics disagree meaningfully
    "synthesize_verdict" -- critics agree; skip adjudicator (fast path)
    """
    if state.get("needs_adjudication", False):
        log.info("[router] Disagreement detected -> adjudicator")
        return "adjudicate"
    log.info("[router] Critics agree -> synthesis (fast path)")
    return "synthesize_verdict"


# ===========================================================
# Helper
# ===========================================================

def _elapsed(start_ms: float | None) -> float | None:
    if start_ms is None:
        return None
    return round((time.perf_counter() * 1000) - start_ms, 1)
