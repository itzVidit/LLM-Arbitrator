"""
ui/batch_view.py
----------------
Batch Arbitration Mode — multi-response input, progress, sortable results table,
detail expander, summary stats.
"""

from __future__ import annotations

import html
import time
import uuid

import pandas as pd
import streamlit as st

from models.critique import ArbitrationRequest, FinalVerdict
from ui.styles import COLORS, badge, metric_card, rec_color, score_color


_MAX_RESPONSES = 10
_PREVIEW_LEN   = 80


def _esc(t: str) -> str:
    return html.escape(t)


# ─────────────────────────────────────────────────────────────────────
#  Session-state
# ─────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    if "batch_responses"     not in st.session_state:
        st.session_state.batch_responses = [""]
    if "batch_prompt"        not in st.session_state:
        st.session_state.batch_prompt = ""
    if "batch_results"       not in st.session_state:
        st.session_state.batch_results = []
    if "batch_selected_idx"  not in st.session_state:
        st.session_state.batch_selected_idx = None


# ─────────────────────────────────────────────────────────────────────
#  Batch runner
# ─────────────────────────────────────────────────────────────────────

def _run_batch(
    prompt: str,
    responses: list[str],
    progress_bar,
    status_text,
) -> list[dict]:
    from orchestration import run_arbitration_sync

    results: list[dict] = []
    n = len(responses)

    for i, response_text in enumerate(responses):
        if not response_text.strip():
            continue

        status_text.markdown(
            f'<span style="font-size:0.84rem;color:#94a3b8">Evaluating response {i+1} of {n}…</span>',
            unsafe_allow_html=True,
        )
        progress_bar.progress(i / n)
        t0 = time.perf_counter()

        try:
            request = ArbitrationRequest(
                original_prompt=prompt,
                llm_response=response_text,
                session_id=f"batch-{uuid.uuid4().hex[:6]}",
            )
            verdict: FinalVerdict = run_arbitration_sync(request)
            latency_ms = round((time.perf_counter() - t0) * 1000)
            preview = response_text.strip().replace("\n", " ")
            preview = (preview[:_PREVIEW_LEN] + "…") if len(preview) > _PREVIEW_LEN else preview

            results.append({
                "idx":          i + 1,
                "preview":      preview,
                "score":        verdict.overall_score,
                "quality":      verdict.quality_score if verdict.quality_score > 0 else None,
                "confidence":   round(verdict.overall_confidence * 100),
                "issues":       len(verdict.key_issues),
                "status":       verdict.recommendation,
                "latency_ms":   verdict.total_latency_ms or latency_ms,
                "session_id":   verdict.session_id,
                "verdict":      verdict,
                "response_text": response_text,
            })
        except Exception as exc:
            preview = response_text.strip().replace("\n", " ")[:_PREVIEW_LEN]
            results.append({
                "idx":          i + 1,
                "preview":      preview,
                "score":        0,
                "quality":      None,
                "confidence":   0,
                "issues":       0,
                "status":       "ERROR",
                "latency_ms":   round((time.perf_counter() - t0) * 1000),
                "session_id":   None,
                "verdict":      None,
                "error":        str(exc),
                "response_text": response_text,
            })

    progress_bar.progress(1.0)
    status_text.markdown(
        f'<span style="font-size:0.84rem;color:#86efac">Done — {len(results)} responses evaluated.</span>',
        unsafe_allow_html=True,
    )
    return results


# ─────────────────────────────────────────────────────────────────────
#  DataFrame builder
# ─────────────────────────────────────────────────────────────────────

def _build_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "#":             r["idx"],
            "Preview":       r["preview"],
            "Score /100":    r["score"],
            "Quality /10":   r["quality"] if r["quality"] is not None else 0,
            "Confidence %":  r["confidence"],
            "Issues":        r["issues"],
            "Status":        r["status"],
            "Latency ms":    int(r["latency_ms"] or 0),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────

def render_batch_view() -> None:
    _init_state()

    st.markdown(
        '<p style="color:#94a3b8;font-size:0.87rem;margin-bottom:1.2rem">'
        "Submit multiple LLM responses against the same prompt. "
        "The pipeline evaluates each independently and shows a sortable results table."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────
    with st.form("batch_form"):
        st.markdown(
            '<span style="font-size:0.78rem;font-weight:600;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:0.06em">Shared Prompt</span>',
            unsafe_allow_html=True,
        )
        prompt = st.text_area(
            "prompt",
            value=st.session_state.batch_prompt,
            height=90,
            placeholder="What is the capital of France? Explain why.",
            key="batch_prompt_input",
            label_visibility="collapsed",
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-size:0.78rem;font-weight:600;color:#94a3b8;'
            f'text-transform:uppercase;letter-spacing:0.06em">'
            f'LLM Responses (up to {_MAX_RESPONSES})</span>',
            unsafe_allow_html=True,
        )

        n = len(st.session_state.batch_responses)
        response_inputs: list[str] = []
        for i in range(n):
            val = st.text_area(
                f"Response #{i+1}",
                value=st.session_state.batch_responses[i],
                height=110,
                key=f"batch_resp_{i}",
                placeholder=f"Paste LLM response #{i+1} here…",
            )
            response_inputs.append(val)

        col_add, col_rm, col_run = st.columns([1, 1, 3])
        with col_add:
            add_btn = st.form_submit_button("+ Add Response",    disabled=(n >= _MAX_RESPONSES))
        with col_rm:
            rm_btn  = st.form_submit_button("− Remove Last",     disabled=(n <= 1))
        with col_run:
            run_btn = st.form_submit_button(
                "Run Batch Arbitration",
                type="primary",
                disabled=not prompt.strip(),
            )

    if add_btn:
        st.session_state.batch_prompt    = prompt
        st.session_state.batch_responses = response_inputs + [""]
        st.rerun()

    if rm_btn:
        st.session_state.batch_prompt    = prompt
        st.session_state.batch_responses = response_inputs[:-1] or [""]
        st.rerun()

    if run_btn:
        non_empty = [r for r in response_inputs if r.strip()]
        if not non_empty:
            st.warning("Add at least one non-empty response before running.")
        else:
            st.session_state.batch_prompt    = prompt
            st.session_state.batch_responses = response_inputs
            progress_bar = st.progress(0.0)
            status_text  = st.empty()
            with st.spinner("Running arbitration pipeline…"):
                results = _run_batch(
                    prompt=prompt,
                    responses=non_empty,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )
            st.session_state.batch_results       = results
            st.session_state.batch_selected_idx  = None
            st.rerun()

    # ── Results table ─────────────────────────────────────────────────
    results = st.session_state.batch_results
    if not results:
        st.markdown(
            '<div style="background:#1a1d27;border:1px solid #2d3148;border-radius:8px;'
            'padding:1.2rem;color:#64748b;font-size:0.86rem;text-align:center;margin-top:1rem">'
            "Results will appear here after you run the batch.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        f'<span style="font-size:0.95rem;font-weight:600;color:#f1f5f9">'
        f'Results — {len(results)} responses</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    df = _build_dataframe(results)

    col_config = {
        "#": st.column_config.NumberColumn("#", width="small"),
        "Preview": st.column_config.TextColumn("Preview", width="large"),
        "Score /100": st.column_config.ProgressColumn(
            "Score /100", min_value=0, max_value=100, format="%d", width="medium",
        ),
        "Quality /10": st.column_config.NumberColumn("Quality /10", format="%d", width="small"),
        "Confidence %": st.column_config.ProgressColumn(
            "Confidence %", min_value=0, max_value=100, format="%d%%", width="medium",
        ),
        "Issues": st.column_config.NumberColumn("Issues", width="small"),
        "Status": st.column_config.TextColumn("Status", width="small"),
        "Latency ms": st.column_config.NumberColumn("Latency ms", format="%d ms", width="small"),
    }

    sort_col, sort_dir_col = st.columns([3, 1])
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            options=["Score /100", "Quality /10", "Confidence %", "Issues", "Latency ms", "#"],
            index=0,
            key="batch_sort_col",
        )
    with sort_dir_col:
        sort_asc = st.checkbox("Ascending", value=False, key="batch_sort_asc")

    df_sorted = df.sort_values(sort_by, ascending=sort_asc).reset_index(drop=True)

    st.dataframe(
        df_sorted,
        column_config=col_config,
        width="stretch",
        hide_index=True,
        height=min(60 + len(df_sorted) * 38, 480),
    )

    # ── Row detail view ───────────────────────────────────────────────
    st.markdown(
        '<span style="font-size:0.78rem;color:#64748b">Select a response to view its full verdict:</span>',
        unsafe_allow_html=True,
    )
    row_options = {
        f"#{r['idx']} — {r['preview'][:50]}": i
        for i, r in enumerate(results)
    }
    selected_label = st.selectbox(
        "Select response",
        options=["(none)"] + list(row_options.keys()),
        key="batch_detail_select",
        label_visibility="collapsed",
    )

    if selected_label != "(none)":
        sel_idx = row_options[selected_label]
        sel = results[sel_idx]

        if sel.get("verdict"):
            verdict: FinalVerdict = sel["verdict"]
            rec = verdict.recommendation
            with st.expander(
                f"Full Verdict — Response #{sel['idx']}  ·  {rec}",
                expanded=True,
            ):
                st.session_state["_last_response"] = sel["response_text"]
                from ui.verdict_view import render_verdict_view
                render_verdict_view(verdict)
        elif sel.get("error"):
            st.markdown(
                f'<div style="background:#2d0f0f;border:1px solid #ef4444;border-radius:6px;'
                f'padding:0.7rem 1rem;color:#fca5a5;font-size:0.85rem;margin-top:0.5rem">'
                f'Arbitration failed for response #{sel["idx"]}: {_esc(sel["error"])}</div>',
                unsafe_allow_html=True,
            )

    # ── Summary statistics ────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.82rem;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.06em">Batch Summary</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    successful = [r for r in results if r["status"] != "ERROR"]
    if not successful:
        st.warning("All arbitrations failed.")
        return

    avg_score    = sum(r["score"]     for r in successful) / len(successful)
    pass_count   = sum(1 for r in successful if r["status"] == "APPROVE")
    revise_count = sum(1 for r in successful if r["status"] == "REVISE")
    reject_count = sum(1 for r in successful if r["status"] == "REJECT")
    avg_latency  = sum(r["latency_ms"] for r in successful) / len(successful)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card(f"{avg_score:.0f}/100", "Avg Score",    score_color(avg_score)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(str(pass_count),   "Approved",   COLORS["approve"]),             unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(str(revise_count), "Revise",     COLORS["revise"]),              unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(str(reject_count), "Rejected",   COLORS["reject"]),              unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card(f"{avg_latency:.0f} ms", "Avg Latency", COLORS["muted"]),        unsafe_allow_html=True)
