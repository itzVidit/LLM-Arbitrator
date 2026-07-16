"""
orchestration/__init__.py
--------------------------
Public API for the Phase 2 orchestration layer.

Primary entry point:

    from orchestration import run_arbitration
    from models import ArbitrationRequest, FinalVerdict

    verdict: FinalVerdict = await run_arbitration(
        ArbitrationRequest(
            original_prompt="Explain HTTPS",
            llm_response="HTTPS uses SSL...",
        )
    )
    print(verdict.recommendation, verdict.overall_score)

For synchronous callers:

    from orchestration import run_arbitration_sync
    verdict = run_arbitration_sync(request)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from models.critique import ArbitrationRequest, FinalVerdict
from orchestration.graph import get_graph
from orchestration.state import GraphState

log = logging.getLogger(__name__)

# Shared thread pool for running the graph from async contexts
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="arbitration")


def run_arbitration_sync(request: ArbitrationRequest) -> FinalVerdict:
    """
    Synchronous entry point. Runs the full LangGraph arbitration pipeline
    and returns the completed FinalVerdict.

    This is the recommended way to call the pipeline from:
      - Scripts / CLI tools
      - FastAPI endpoints (via run_in_executor)
      - Test code

    Args:
        request: The ArbitrationRequest containing the prompt and LLM response.

    Returns:
        FinalVerdict with scores, issues, recommendation, and provenance.
    """
    log.info(
        "[run_arbitration_sync] session=%s",
        request.session_id or "(auto)",
    )

    graph = get_graph()

    initial_state: GraphState = {
        "request": request,
    }

    # LangGraph's .invoke() is synchronous and thread-safe
    final_state: GraphState = graph.invoke(initial_state)

    verdict = final_state.get("final_verdict")
    if verdict is None:
        raise RuntimeError(
            "Arbitration pipeline completed but produced no FinalVerdict. "
            "Check logs for errors."
        )

    return verdict


async def run_arbitration(request: ArbitrationRequest) -> FinalVerdict:
    """
    Async entry point. Runs the synchronous LangGraph pipeline in a thread
    pool so it doesn't block the event loop.

    Use this from async FastAPI routes or other async callers.
    """
    loop = asyncio.get_event_loop()
    verdict = await loop.run_in_executor(
        _executor,
        run_arbitration_sync,
        request,
    )
    return verdict


__all__ = [
    "run_arbitration",
    "run_arbitration_sync",
]
