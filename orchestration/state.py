"""
orchestration/state.py
----------------------
Defines the LangGraph state object that flows through every node in the graph.

LangGraph requires state to be a TypedDict (or dataclass). Each node receives
the current state, does its work, and returns a dict of fields to update.
LangGraph merges those updates back into the shared state automatically.

Annotated fields with `operator.add` act as append-only lists — multiple
parallel nodes can each append their critique without overwriting each other.
This is the key mechanism that lets the fan-out/fan-in pattern work correctly
when all five critics run in parallel.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from models.critique import (
    ArbitrationRequest,
    Critique,
    DisagreementReport,
    FallbackInfo,
    FinalVerdict,
)


class GraphState(TypedDict, total=False):
    """
    The single shared state object that travels through the LangGraph pipeline.

    Field lifecycle:
        parse_input     →  request
        dispatch_critics → (fan-out, each critic appends to critiques)
        collect_critiques → critiques is now complete
        detect_disagreement → disagreement_report
        adjudicate       → adjudication_notes  (only on disagreement path)
        synthesize_verdict → final_verdict
    """

    # ── Set by parse_input node ──────────────────────────────────────────
    request: ArbitrationRequest
    """The validated input payload for this arbitration run."""

    session_id: Optional[str]
    """Caller-supplied or auto-generated session identifier."""

    # ── Populated by parallel critic nodes ──────────────────────────────
    # `Annotated[list, operator.add]` means LangGraph merges updates by
    # concatenation rather than overwrite — each parallel critic node
    # returns {"critiques": [its_critique]} and they all get appended.
    critiques: Annotated[list[Critique], operator.add]
    """One Critique per critic, collected via fan-in after parallel dispatch."""

    fallbacks_used: Annotated[list[FallbackInfo], operator.add]
    """Populated by any critic that fell back to its secondary model."""

    # ── Set by detect_disagreement node ─────────────────────────────────
    disagreement_report: Optional[DisagreementReport]
    """Structured report of conflicts between critics. None = full agreement."""

    needs_adjudication: bool
    """
    True  → critics disagreed meaningfully; route to adjudicator node.
    False → critics agree; take fast path directly to synthesize_verdict.
    """

    # ── Set by adjudicate node (only on disagreement path) ───────────────
    adjudication_notes: Optional[str]
    """
    Free-text resolution notes from the adjudicator LLM, injected into the
    synthesis prompt to resolve conflicting signals.
    """

    # ── Set by synthesize_verdict node ───────────────────────────────────
    final_verdict: Optional[FinalVerdict]
    """The complete, caller-facing result of the arbitration pipeline."""

    # ── Pipeline metadata ────────────────────────────────────────────────
    pipeline_start_ms: Optional[float]
    """Wall-clock timestamp (perf_counter * 1000) set at pipeline entry."""

    errors: Annotated[list[str], operator.add]
    """Non-fatal errors accumulated during the run (e.g., a critic that failed)."""
