"""
orchestration/graph.py
----------------------
Wires all nodes into a compiled LangGraph StateGraph.

Graph topology:

  [START]
     |
  parse_input
     |
  [parallel fan-out via Send API]
     |-- run_accuracy_critic     --|
     |-- run_logic_critic        --|
     |-- run_completeness_critic --|--> detect_disagreement
     |-- run_safety_critic       --|
     |-- run_style_critic        --|
     |
  detect_disagreement
     |
  [conditional edge]
     |-- needs_adjudication=True  --> adjudicate --> synthesize_verdict
     |-- needs_adjudication=False -----------------> synthesize_verdict
     |
  synthesize_verdict
     |
  [END]

Parallelism:
  LangGraph's Send API is used to fan out to all five critics simultaneously.
  Each critic node runs in its own thread (via asyncio.new_event_loop),
  and their outputs are merged back via the Annotated[list, operator.add]
  fields in GraphState.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from orchestration.state import GraphState
from orchestration.nodes import (
    parse_input,
    run_accuracy_critic,
    run_logic_critic,
    run_completeness_critic,
    run_safety_critic,
    run_style_critic,
    detect_disagreement,
    adjudicate,
    synthesize_verdict,
    route_after_disagreement_check,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fan-out edge function
# ---------------------------------------------------------------------------

# Map each critic node name to its function (used by the fan-out dispatcher)
_CRITIC_NODES = {
    "run_accuracy_critic":     run_accuracy_critic,
    "run_logic_critic":        run_logic_critic,
    "run_completeness_critic": run_completeness_critic,
    "run_safety_critic":       run_safety_critic,
    "run_style_critic":        run_style_critic,
}

_CRITIC_NODE_NAMES = list(_CRITIC_NODES.keys())


def dispatch_critics(state: GraphState) -> list[Send]:
    """
    Fan-out edge: after parse_input, dispatch all critic nodes in parallel.

    Returns a list of Send objects -- one per critic. LangGraph executes
    them concurrently in separate threads and merges their state updates
    via the operator.add annotations on `critiques` and `fallbacks_used`.
    """
    return [
        Send(node_name, state)
        for node_name in _CRITIC_NODE_NAMES
    ]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Build and compile the arbitration StateGraph.

    Returns:
        A compiled LangGraph graph that can be invoked via .invoke() or
        .stream() with a GraphState dict as input.
    """
    builder = StateGraph(GraphState)

    # --- Register all nodes ---
    builder.add_node("parse_input",             parse_input)
    builder.add_node("run_accuracy_critic",     run_accuracy_critic)
    builder.add_node("run_logic_critic",        run_logic_critic)
    builder.add_node("run_completeness_critic", run_completeness_critic)
    builder.add_node("run_safety_critic",       run_safety_critic)
    builder.add_node("run_style_critic",        run_style_critic)
    builder.add_node("detect_disagreement",     detect_disagreement)
    builder.add_node("adjudicate",              adjudicate)
    builder.add_node("synthesize_verdict",      synthesize_verdict)

    # --- Edges ---

    # Entry point
    builder.add_edge(START, "parse_input")

    # Fan-out: parse_input --> all critics in parallel (via Send)
    builder.add_conditional_edges(
        "parse_input",
        dispatch_critics,
        _CRITIC_NODE_NAMES,
    )

    # Fan-in: every critic node --> detect_disagreement
    for critic_name in _CRITIC_NODE_NAMES:
        builder.add_edge(critic_name, "detect_disagreement")

    # Conditional: detect_disagreement --> adjudicate OR synthesize_verdict
    builder.add_conditional_edges(
        "detect_disagreement",
        route_after_disagreement_check,
        {
            "adjudicate":        "adjudicate",
            "synthesize_verdict": "synthesize_verdict",
        },
    )

    # adjudicate always flows into synthesis
    builder.add_edge("adjudicate", "synthesize_verdict")

    # Terminal
    builder.add_edge("synthesize_verdict", END)

    compiled = builder.compile()
    log.info("Arbitration graph compiled successfully")
    return compiled


# Module-level singleton — compiled once and reused
_GRAPH = None


def get_graph() -> Any:
    """Return the (lazily compiled) arbitration graph singleton."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH
