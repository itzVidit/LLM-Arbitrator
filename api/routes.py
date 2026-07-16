"""
api/routes.py
-------------
FastAPI router for the Arbitration API v1.

Endpoints
---------
  POST /v1/arbitrate
    Evaluate a single LLM response.  Runs the five-critic LangGraph pipeline
    in a thread pool, persists the verdict to SQLite, and returns the full
    ArbitrateResponse.

  POST /v1/arbitrate/batch
    Evaluate up to 10 LLM responses concurrently (asyncio.gather).  Each
    response is run in a separate thread-pool task so the critics inside each
    arbitration still fan out in parallel.  Results are persisted individually.

  GET /v1/arbitrations/{arbitration_id}
    Retrieve a previously generated verdict by its UUID.

  GET /v1/arbitrations
    Paginated list of verdict summaries (no full JSON), newest first.
    Optional query param: ?recommendation=APPROVE|REVISE|REJECT

  GET /v1/analytics
    Aggregated system-wide analytics across all stored verdicts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.analytics import compute_analytics
from api.models import (
    AnalyticsReport,
    ArbitrateRequest,
    ArbitrateResponse,
    BatchArbitrateRequest,
    BatchArbitrateResponse,
    BatchItemError,
    CritiqueOut,
    DismissedIssueOut,
    IssueOut,
    VerdictListResponse,
    VerdictSummary,
)
from models.critique import (
    ArbitrationRequest,
    CriticDimension,
    FinalVerdict,
)
from orchestration import run_arbitration_sync
from storage.database import Database, get_db

log = logging.getLogger(__name__)

router = APIRouter()

# Thread pool for running the synchronous LangGraph pipeline from async handlers
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="api-arbitrate")


# ─────────────────────────────────────────────────────────────────────
#  Conversion helpers
# ─────────────────────────────────────────────────────────────────────

def _parse_critics(critics: Optional[list[str]]) -> Optional[list[CriticDimension]]:
    """Convert a list of critic name strings to CriticDimension enums."""
    if critics is None:
        return None
    valid = {d.value for d in CriticDimension}
    result = []
    for name in critics:
        low = name.lower().strip()
        if low not in valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown critic '{name}'. Valid values: {sorted(valid)}",
            )
        result.append(CriticDimension(low))
    return result or None


def _verdict_to_response(verdict: FinalVerdict, arb_id: str) -> ArbitrateResponse:
    """Translate a FinalVerdict (internal model) → ArbitrateResponse (API model)."""
    critiques_out = [
        CritiqueOut(
            dimension=c.dimension.value,
            model_used=c.model_used,
            score=c.score,
            grade=c.grade,
            confidence=c.confidence,
            passed=c.passed,
            summary=c.summary,
            issues=[
                IssueOut(
                    quote=i.quote,
                    explanation=i.explanation,
                    severity=i.severity,
                    evidence=i.evidence,
                    recommendation=i.recommendation,
                )
                for i in c.issues
            ],
            latency_ms=c.latency_ms,
        )
        for c in verdict.critiques
    ]

    key_issues_out = [
        IssueOut(
            quote=i.quote,
            explanation=i.explanation,
            severity=i.severity,
            evidence=i.evidence,
            recommendation=i.recommendation,
        )
        for i in verdict.key_issues
    ]

    dismissed_out = [
        DismissedIssueOut(
            dimension=d.dimension.value,
            original_issue=d.original_issue,
            dismissal_reason=d.dismissal_reason,
        )
        for d in verdict.dismissed_issues
    ]

    exec_summary = verdict.executive_summary or verdict.explanation

    return ArbitrateResponse(
        arbitration_id=arb_id,
        session_id=verdict.session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        recommendation=verdict.recommendation,
        overall_score=verdict.overall_score,
        quality_score=verdict.quality_score,
        overall_confidence=verdict.overall_confidence,
        critic_agreement=verdict.critic_agreement,
        approved=verdict.approved,
        executive_summary=exec_summary,
        explanation=verdict.explanation,
        strengths=verdict.strengths,
        key_issues=key_issues_out,
        dismissed_issues=dismissed_out,
        dimension_scores=verdict.dimension_scores,
        dimension_passed=verdict.dimension_passed,
        critiques=critiques_out,
        adjudication_used=verdict.adjudication_used,
        fast_path=verdict.fast_path,
        fallbacks_used=len(verdict.fallbacks_used),
        total_latency_ms=verdict.total_latency_ms,
    )


def _row_to_summary(row: dict) -> VerdictSummary:
    return VerdictSummary(
        arbitration_id=row["id"],
        session_id=row.get("session_id"),
        created_at=row["created_at"],
        recommendation=row["recommendation"],
        overall_score=row["overall_score"],
        quality_score=row["quality_score"],
        overall_confidence=row["overall_confidence"],
        critic_agreement=row["critic_agreement"],
        adjudication_used=bool(row["adjudication_used"]),
        fast_path=bool(row["fast_path"]),
        total_latency_ms=row.get("total_latency_ms"),
    )


# ─────────────────────────────────────────────────────────────────────
#  POST /v1/arbitrate
# ─────────────────────────────────────────────────────────────────────

@router.post(
    "/arbitrate",
    response_model=ArbitrateResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a single LLM response",
    description=(
        "Runs the full five-critic LangGraph arbitration pipeline on a single "
        "LLM response and returns the complete structured verdict.\n\n"
        "The verdict is persisted to SQLite. Use the returned `arbitration_id` "
        "with `GET /v1/arbitrations/{id}` to retrieve it later.\n\n"
        "**Pipeline steps**\n"
        "1. Five specialist critics evaluate the response in parallel.\n"
        "2. If critics meaningfully disagree, the Adjudicator LLM resolves conflicts.\n"
        "3. A `FinalVerdict` is synthesised with scores, issues, strengths, and a "
        "one-paragraph executive summary.\n"
    ),
    tags=["Arbitration"],
    responses={
        200: {"description": "Arbitration completed successfully."},
        422: {"description": "Validation error in request body."},
        500: {"description": "Internal pipeline error."},
    },
)
async def arbitrate(
    body: ArbitrateRequest,
    db: Database = Depends(get_db),
) -> ArbitrateResponse:
    requested_critics = _parse_critics(body.critics)

    pipeline_request = ArbitrationRequest(
        original_prompt=body.original_prompt,
        llm_response=body.llm_response,
        context=body.context,
        session_id=body.session_id or str(uuid.uuid4())[:8],
        requested_critics=requested_critics,
    )

    log.info(
        "[POST /v1/arbitrate] session=%s  prompt_len=%d  response_len=%d",
        pipeline_request.session_id,
        len(body.original_prompt),
        len(body.llm_response),
    )

    try:
        loop = asyncio.get_event_loop()
        verdict: FinalVerdict = await loop.run_in_executor(
            _executor, run_arbitration_sync, pipeline_request
        )
    except Exception as exc:
        log.error("[POST /v1/arbitrate] pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Arbitration pipeline failed: {exc}",
        )

    arb_id = await db.save_verdict(verdict)
    log.info(
        "[POST /v1/arbitrate] saved %s  rec=%s  score=%d",
        arb_id,
        verdict.recommendation,
        verdict.overall_score,
    )

    return _verdict_to_response(verdict, arb_id)


# ─────────────────────────────────────────────────────────────────────
#  POST /v1/arbitrate/batch
# ─────────────────────────────────────────────────────────────────────

@router.post(
    "/arbitrate/batch",
    response_model=BatchArbitrateResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate multiple LLM responses concurrently",
    description=(
        "Accepts up to **10** LLM responses and evaluates them concurrently "
        "using `asyncio.gather`. Each response goes through the full pipeline "
        "independently. Results are persisted individually.\n\n"
        "Partial failures are captured in `BatchItemError` entries — the rest "
        "of the batch continues regardless.\n\n"
        "**Note:** Batch throughput is bounded by your LLM provider rate limits. "
        "All five critics inside each arbitration still run in parallel."
    ),
    tags=["Arbitration"],
    responses={
        200: {"description": "Batch completed (may include partial failures)."},
        422: {"description": "Validation error in request body."},
    },
)
async def arbitrate_batch(
    body: BatchArbitrateRequest,
    db: Database = Depends(get_db),
) -> BatchArbitrateResponse:
    requested_critics = _parse_critics(body.critics)

    log.info(
        "[POST /v1/arbitrate/batch] count=%d  prompt_len=%d",
        len(body.responses),
        len(body.original_prompt),
    )

    async def _run_one(idx: int, response_text: str):
        req = ArbitrationRequest(
            original_prompt=body.original_prompt,
            llm_response=response_text,
            context=body.context,
            session_id=f"batch-{uuid.uuid4().hex[:6]}",
            requested_critics=requested_critics,
        )
        loop = asyncio.get_event_loop()
        verdict: FinalVerdict = await loop.run_in_executor(
            _executor, run_arbitration_sync, req
        )
        arb_id = await db.save_verdict(verdict)
        return idx, _verdict_to_response(verdict, arb_id), None

    tasks = [_run_one(i, r) for i, r in enumerate(body.responses)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    completed = 0
    failed = 0

    for item in raw_results:
        if isinstance(item, Exception):
            # Can't determine index reliably from exception alone — mark as error
            results.append(BatchItemError(index=-1, error=str(item)))
            failed += 1
        else:
            idx, response, err = item
            if err:
                results.append(BatchItemError(index=idx, error=str(err)))
                failed += 1
            else:
                results.append(response)
                completed += 1

    log.info(
        "[POST /v1/arbitrate/batch] done  completed=%d  failed=%d",
        completed,
        failed,
    )

    return BatchArbitrateResponse(
        total=len(body.responses),
        completed=completed,
        failed=failed,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────
#  GET /v1/arbitrations/{arbitration_id}
# ─────────────────────────────────────────────────────────────────────

@router.get(
    "/arbitrations/{arbitration_id}",
    response_model=ArbitrateResponse,
    summary="Retrieve a stored arbitration verdict",
    description=(
        "Fetches a previously generated and stored arbitration verdict by its UUID.\n\n"
        "The `arbitration_id` is returned in the response of "
        "`POST /v1/arbitrate` and `POST /v1/arbitrate/batch`."
    ),
    tags=["Arbitration"],
    responses={
        200: {"description": "Arbitration record found."},
        404: {"description": "No record found with this ID."},
    },
)
async def get_arbitration(
    arbitration_id: str,
    db: Database = Depends(get_db),
) -> ArbitrateResponse:
    row = await db.get_verdict(arbitration_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Arbitration '{arbitration_id}' not found.",
        )

    try:
        verdict = FinalVerdict.model_validate_json(row["verdict_json"])
    except Exception as exc:
        log.error(
            "[GET /v1/arbitrations/%s] deserialise error: %s",
            arbitration_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored verdict could not be deserialised.",
        )

    return _verdict_to_response(verdict, arbitration_id)


# ─────────────────────────────────────────────────────────────────────
#  GET /v1/arbitrations  (list)
# ─────────────────────────────────────────────────────────────────────

@router.get(
    "/arbitrations",
    response_model=VerdictListResponse,
    summary="List stored arbitration summaries",
    description=(
        "Returns a paginated list of arbitration summaries, newest first.\n\n"
        "Use the `recommendation` query parameter to filter by outcome. "
        "Use `limit` and `offset` to paginate through results."
    ),
    tags=["Arbitration"],
)
async def list_arbitrations(
    limit: int = Query(default=20, ge=1, le=200, description="Max items to return."),
    offset: int = Query(default=0, ge=0, description="Pagination offset."),
    recommendation: Optional[str] = Query(
        default=None,
        description="Filter by recommendation: APPROVE, REVISE, or REJECT.",
        pattern="^(APPROVE|REVISE|REJECT)$",
    ),
    db: Database = Depends(get_db),
) -> VerdictListResponse:
    total = await db.count_verdicts()
    rows = await db.list_verdicts(
        limit=limit, offset=offset, recommendation=recommendation
    )
    return VerdictListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_row_to_summary(r) for r in rows],
    )


# ─────────────────────────────────────────────────────────────────────
#  GET /v1/analytics
# ─────────────────────────────────────────────────────────────────────

@router.get(
    "/analytics",
    response_model=AnalyticsReport,
    summary="Get system-wide analytics on critic behaviour",
    description=(
        "Computes and returns aggregated analytics across all stored arbitration runs.\n\n"
        "**Per-critic metrics**\n"
        "- Which critic raises the most issues on average\n"
        "- Which critic is overruled most frequently by the adjudicator\n"
        "- Most common issue severity categories per critic\n"
        "- Agreement / disagreement rates\n"
        "- Average confidence scores and p95 latency\n"
        "- Fallback model usage rates\n\n"
        "**System overview**\n"
        "- APPROVE / REVISE / REJECT rates\n"
        "- Adjudication rate (% of runs requiring conflict resolution)\n"
        "- Fast-path rate (% of runs where all critics agreed immediately)\n"
        "- Average pipeline latency\n\n"
        "Returns `total_arbitrations: 0` and empty stats if no runs are stored yet."
    ),
    tags=["Analytics"],
    responses={
        200: {"description": "Analytics report computed successfully."},
        500: {"description": "Analytics computation failed."},
    },
)
async def get_analytics(
    db: Database = Depends(get_db),
) -> AnalyticsReport:
    try:
        return await compute_analytics(db)
    except Exception as exc:
        log.error("[GET /v1/analytics] error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics computation failed: {exc}",
        )
