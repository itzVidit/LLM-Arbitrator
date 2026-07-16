"""
ui/critic_panel.py
------------------
Critic Comparison Panel — per-critic cards, radar chart, agreement matrix,
disagreement report.
"""

from __future__ import annotations

import html
import statistics

import plotly.graph_objects as go
import streamlit as st

from models.critique import Critique, FinalVerdict
from ui.styles import (
    COLORS,
    DIM_COLORS,
    badge,
    metric_card,
    score_color,
    sev_chip,
)


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _esc(t: str) -> str:
    return html.escape(t)


def _grade_color(grade: str) -> str:
    return {
        "A": "#22c55e",
        "B": "#4ade80",
        "C": "#f59e0b",
        "D": "#f97316",
        "F": "#ef4444",
    }.get(grade, "#64748b")


def _render_critic_card(critique: Critique, is_outlier: bool = False) -> None:
    """Render a single critic card."""
    passed = critique.passed
    accent = "#22c55e" if passed else "#ef4444"
    dim_color = DIM_COLORS.get(critique.dimension.value, "#6366f1")

    outlier_html = (
        '<div class="outlier-banner">Outlier Score</div><br>'
        if is_outlier else ""
    )

    pass_bg  = "#14532d" if passed else "#450a0a"
    pass_fg  = "#86efac" if passed else "#fca5a5"
    pass_lbl = "PASS"    if passed else "FAIL"

    # Card header
    st.markdown(
        f'<div style="background:#1a1d27;border:1px solid #2d3148;'
        f'border-left:3px solid {accent};border-radius:8px;padding:0.9rem 1rem;'
        f'margin-bottom:0.5rem">'
        f"{outlier_html}"
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        # Left: dimension + model
        f'<div>'
        f'<span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:{dim_color}">{critique.dimension.value}</span>'
        f'<div style="font-size:0.78rem;color:#64748b;margin-top:2px">'
        f'{_esc(critique.model_used)}</div>'
        f'</div>'
        # Right: score
        f'<div style="text-align:right;line-height:1">'
        f'<span style="font-size:2rem;font-weight:800;color:{score_color(critique.score)}">'
        f'{critique.score}</span>'
        f'<span style="font-size:0.8rem;color:#64748b">/100</span>'
        f'</div>'
        f'</div>'
        # Meta row
        f'<div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;margin-top:0.5rem">'
        f'<span style="background:{pass_bg};color:{pass_fg};border-radius:4px;'
        f'font-size:0.7rem;font-weight:700;padding:2px 8px;letter-spacing:0.05em">'
        f'{pass_lbl}</span>'
        f'<span style="font-size:0.78rem;color:#64748b">Conf: {critique.confidence:.0%}</span>'
        f'<span style="font-size:0.78rem;color:#64748b">Grade: '
        f'<strong style="color:{_grade_color(critique.grade)}">{critique.grade}</strong></span>'
        f'<span style="font-size:0.78rem;color:#64748b">'
        f'{len(critique.issues)} issue{"s" if len(critique.issues) != 1 else ""}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Summary
    st.markdown(
        f'<p style="font-size:0.83rem;color:#94a3b8;margin:0.4rem 0 0.5rem 0;'
        f'line-height:1.55">{_esc(critique.summary)}</p>',
        unsafe_allow_html=True,
    )

    # Issues
    if critique.issues:
        for issue in critique.issues:
            sev = issue.severity
            q = issue.quote
            q_trunc = (q[:65] + "…") if len(q) > 65 else q
            skip_q = q.strip() in ("(general)", "(node error)", "")
            quote_html = (
                f'<div style="font-size:0.76rem;color:#64748b;margin-top:3px;font-style:italic">'
                f'&ldquo;{_esc(q_trunc)}&rdquo;</div>'
                if not skip_q else ""
            )
            st.markdown(
                f'<div class="issue-card {sev}" style="margin-bottom:0.3rem">'
                f'{sev_chip(sev)} '
                f'<span style="font-size:0.82rem;color:#cbd5e1">'
                f'{_esc(issue.explanation[:100])}</span>'
                f'{quote_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="font-size:0.82rem;color:#22c55e;margin:0">No issues found.</p>',
            unsafe_allow_html=True,
        )


def _radar_chart(critiques: list[Critique]) -> go.Figure:
    fig = go.Figure(go.Barpolar(
        r=[c.score for c in critiques],
        theta=[c.dimension.value.capitalize() for c in critiques],
        marker=dict(
            color=[DIM_COLORS.get(c.dimension.value, "#6366f1") for c in critiques],
            opacity=0.82,
            line=dict(color="#0f1117", width=1),
        ),
        hovertemplate="%{theta}: %{r}/100<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1a1d27",
            radialaxis=dict(
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=9, color="#64748b"),
                gridcolor="#2d3148",
                linecolor="#2d3148",
            ),
            angularaxis=dict(
                direction="clockwise",
                tickfont=dict(size=11, color="#94a3b8"),
                gridcolor="#2d3148",
                linecolor="#2d3148",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=30, b=30),
        height=280,
        showlegend=False,
    )
    return fig


def _agreement_matrix(critiques: list[Critique]) -> dict[tuple[str, str], bool]:
    matrix: dict[tuple[str, str], bool] = {}
    for i, ca in enumerate(critiques):
        for cb in critiques[i + 1:]:
            key = (ca.dimension.value, cb.dimension.value)
            matrix[key] = abs(ca.score - cb.score) <= 15
    return matrix


def _find_outliers(critiques: list[Critique]) -> set[str]:
    if len(critiques) < 3:
        return set()
    scores = [c.score for c in critiques]
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    if stdev == 0:
        return set()
    return {c.dimension.value for c in critiques if abs(c.score - mean) > 1.5 * stdev}


# ─────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────

def render_critic_panel(verdict: FinalVerdict) -> None:
    """Render the Critic Comparison Panel."""
    critiques = verdict.critiques
    if not critiques:
        st.info("No critic data available. Run an arbitration first.")
        return

    # ── Header ────────────────────────────────────────────────────────
    agr_color = score_color(verdict.critic_agreement, 1.0)
    st.markdown(
        f'<p style="color:#94a3b8;font-size:0.87rem;margin-bottom:1rem">'
        f"{len(critiques)} critics evaluated this response. Agreement rate: "
        f'<strong style="color:{agr_color}">{verdict.critic_agreement:.0%}</strong></p>',
        unsafe_allow_html=True,
    )

    # ── Radar chart ───────────────────────────────────────────────────
    st.markdown(
        '<span style="font-size:0.82rem;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.06em">Score Overview</span>',
        unsafe_allow_html=True,
    )
    fig = _radar_chart(critiques)
    st.plotly_chart(fig, use_container_width=True)  # plotly_chart doesn't support width="stretch" yet

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Per-critic cards ──────────────────────────────────────────────
    st.markdown(
        '<span style="font-size:0.82rem;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.06em">Individual Critics</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    outliers = _find_outliers(critiques)
    n = len(critiques)
    cols_per_row = 3 if n >= 3 else n
    rows = [critiques[i:i + cols_per_row] for i in range(0, n, cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, c in zip(cols, row):
            with col:
                _render_critic_card(c, is_outlier=c.dimension.value in outliers)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Agreement matrix ──────────────────────────────────────────────
    st.markdown(
        '<span style="font-size:0.82rem;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.06em">Agreement Matrix</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.78rem;color:#64748b;margin:0.3rem 0 0.7rem 0">'
        "Score delta &le;15 pts = Agree &nbsp;·&nbsp; >15 pts = Differ</p>",
        unsafe_allow_html=True,
    )

    matrix = _agreement_matrix(critiques)
    dim_labels = [c.dimension.value.capitalize() for c in critiques]

    header_cells = '<th class="matrix-table" style="width:100px"></th>' + "".join(
        f'<th style="background:#1a1d27;color:#94a3b8;font-weight:600;padding:8px 12px;'
        f'text-align:center;border:1px solid #2d3148;font-size:0.76rem;'
        f'text-transform:uppercase;letter-spacing:0.04em">{d}</th>'
        for d in dim_labels
    )

    rows_html = ""
    for ca in critiques:
        row_cells = (
            f'<td style="background:#22263a;border:1px solid #2d3148;padding:7px 10px;'
            f'text-align:left;color:#94a3b8;font-weight:600;font-size:0.76rem;'
            f'text-transform:uppercase;letter-spacing:0.04em;white-space:nowrap">'
            f"{ca.dimension.value.capitalize()}</td>"
        )
        for cb in critiques:
            if ca == cb:
                row_cells += (
                    '<td style="background:#22263a;border:1px solid #2d3148;'
                    'text-align:center;color:#3d4266;padding:7px 10px">—</td>'
                )
            else:
                key = (ca.dimension.value, cb.dimension.value)
                rev_key = (cb.dimension.value, ca.dimension.value)
                agreed = matrix.get(key, matrix.get(rev_key, True))
                pill_cls = "agree-pill" if agreed else "disagree-pill"
                label = "Agree" if agreed else "Differ"
                row_cells += (
                    f'<td style="background:#1a1d27;border:1px solid #2d3148;'
                    f'text-align:center;padding:6px 8px">'
                    f'<span class="{pill_cls}">{label}</span></td>'
                )
        rows_html += f"<tr>{row_cells}</tr>"

    st.markdown(
        f'<table style="border-collapse:collapse;width:100%;font-size:0.82rem">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )

    # ── Disagreement Report ───────────────────────────────────────────
    if verdict.disagreement_report and verdict.disagreement_report.has_disagreement:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        dr = verdict.disagreement_report
        with st.expander("Disagreement Report", expanded=False):
            st.markdown(
                f'<p style="color:#cbd5e1;font-size:0.87rem">{_esc(dr.summary)}</p>',
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Score Std-Dev", f"{dr.score_variance:.1f} pts")
                if dr.pass_fail_conflicts:
                    st.markdown(
                        f'<p style="font-size:0.83rem;color:#94a3b8">'
                        f"<strong>Fail dimensions:</strong> {', '.join(dr.pass_fail_conflicts)}</p>",
                        unsafe_allow_html=True,
                    )
            with col_b:
                if dr.confidence_outliers:
                    st.markdown(
                        f'<p style="font-size:0.83rem;color:#94a3b8">'
                        f"<strong>Confidence outliers:</strong> {', '.join(dr.confidence_outliers)}</p>",
                        unsafe_allow_html=True,
                    )
                if dr.unique_issues:
                    st.markdown(
                        f'<p style="font-size:0.83rem;color:#94a3b8">'
                        f"Single-critic issues: {len(dr.unique_issues)}</p>",
                        unsafe_allow_html=True,
                    )
            if verdict.adjudication_used:
                st.markdown(
                    '<div style="background:#052e16;border:1px solid #166534;border-radius:6px;'
                    'padding:0.5rem 0.85rem;color:#86efac;font-size:0.85rem;margin-top:0.5rem">'
                    "Adjudicator was invoked to resolve this disagreement.</div>",
                    unsafe_allow_html=True,
                )
