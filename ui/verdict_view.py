"""
ui/verdict_view.py
------------------
Verdict View page — annotated response, metrics, issues, strengths, scores.
"""

from __future__ import annotations

import html
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from models.critique import FinalVerdict, Issue, Severity
from ui.styles import (
    COLORS,
    DIM_COLORS,
    SEV_BG,
    SEV_BORDER,
    SEV_COLORS,
    badge,
    inject_css,
    metric_card,
    rec_color,
    score_color,
    sev_chip,
)


# ─────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return html.escape(text)


def _annotate_response(response_text: str, issues: list[Issue]) -> str:
    """Wrap quoted text in colored <span> annotations."""
    sev_order = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    ranked = sorted(issues, key=lambda i: sev_order.get(i.severity, 9))

    intervals: list[tuple[int, int, str, str]] = []
    covered: list[tuple[int, int]] = []

    for issue in ranked:
        q = issue.quote.strip()
        if not q or q in ("(general)", "(node error)"):
            continue
        for needle in (q, q[:40]):
            if len(needle) < 8:
                continue
            idx = response_text.find(needle)
            if idx == -1:
                idx = response_text.lower().find(needle.lower())
            if idx == -1:
                continue
            end_idx = idx + len(needle)
            overlap = any(not (end_idx <= s or idx >= e) for s, e in covered)
            if not overlap:
                title = _esc(issue.explanation[:80])
                intervals.append((idx, end_idx, issue.severity, title))
                covered.append((idx, end_idx))
            break

    if not intervals:
        return _esc(response_text).replace("\n", "<br>")

    intervals.sort(key=lambda x: x[0])
    parts: list[str] = []
    cursor = 0
    for start, end, sev, title in intervals:
        if start > cursor:
            parts.append(_esc(response_text[cursor:start]).replace("\n", "<br>"))
        span_text = _esc(response_text[start:end])
        parts.append(f'<span class="ann-{sev}" title="{title}">{span_text}</span>')
        cursor = end
    if cursor < len(response_text):
        parts.append(_esc(response_text[cursor:]).replace("\n", "<br>"))
    return "".join(parts)


def _dimension_bar_chart(
    dimension_scores: dict[str, int],
    dimension_passed: dict[str, bool],
) -> go.Figure:
    dims = list(dimension_scores.keys())
    scores = [dimension_scores[d] for d in dims]
    colors = [DIM_COLORS.get(d, "#6366f1") for d in dims]

    fig = go.Figure(go.Bar(
        x=scores,
        y=[d.capitalize() for d in dims],
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color="#0f1117", width=1),
        ),
        text=[f"{s}/100" for s in scores],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#f1f5f9", size=12),
        hovertemplate="%{y}: %{x}/100<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="#2d3148",
            tickfont=dict(color="#94a3b8"),
            title=None,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#94a3b8"),
        ),
        plot_bgcolor="#1a1d27",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=210,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────

def render_verdict_view(verdict: FinalVerdict) -> None:
    """Render the full Verdict View page."""

    # ── 1. Top metrics row ────────────────────────────────────────────
    st.markdown("#### Arbitration Result")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        sc = score_color(verdict.overall_score)
        st.markdown(metric_card(f"{verdict.overall_score}/100", "Overall Score", sc), unsafe_allow_html=True)
    with c2:
        qs = verdict.quality_score
        qc = score_color(qs, out_of=10)
        st.markdown(metric_card(f"{qs}/10" if qs > 0 else "—", "Quality Score", qc), unsafe_allow_html=True)
    with c3:
        conf_pct = f"{verdict.overall_confidence:.0%}"
        cc = score_color(verdict.overall_confidence, out_of=1.0)
        st.markdown(metric_card(conf_pct, "Confidence", cc), unsafe_allow_html=True)
    with c4:
        agr_pct = f"{verdict.critic_agreement:.0%}"
        ac = score_color(verdict.critic_agreement, out_of=1.0)
        st.markdown(metric_card(agr_pct, "Critic Agreement", ac), unsafe_allow_html=True)
    with c5:
        rec = verdict.recommendation
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="margin-top:0.25rem">{badge(rec)}</div>'
            f'<div class="metric-label">Recommendation</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── 2. Summary ────────────────────────────────────────────────────
    summary_text = verdict.executive_summary or verdict.explanation
    if summary_text:
        st.markdown(
            f'<div style="background:#1a1d27;border:1px solid #2d3148;border-radius:8px;'
            f'padding:1rem 1.2rem;color:#cbd5e1;font-size:0.89rem;line-height:1.65">'
            f'<span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:#64748b">Summary</span>'
            f'<div style="margin-top:0.4rem">{_esc(summary_text)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── 3. Annotated Response ─────────────────────────────────────────
    st.markdown(
        '<span style="font-size:0.95rem;font-weight:600;color:#f1f5f9">Annotated Response</span>',
        unsafe_allow_html=True,
    )

    # Legend
    legend_parts = " &nbsp; ".join(
        f'<span class="ann-{sev}" style="padding:1px 6px;font-size:0.78rem">{sev.capitalize()}</span>'
        for sev in ("critical", "major", "minor", "strength")
    )
    st.markdown(
        f'<p style="font-size:0.78rem;color:#64748b;margin:0.3rem 0 0.6rem 0">'
        f"Highlights: {legend_parts}</p>",
        unsafe_allow_html=True,
    )

    # Gather all issues
    all_issues: list[Issue] = list(verdict.key_issues)
    for c in verdict.critiques:
        for issue in c.issues:
            if issue not in all_issues:
                all_issues.append(issue)

    original_response = st.session_state.get("_last_response", verdict.explanation)
    annotated_html = _annotate_response(original_response, all_issues)

    st.markdown(
        f'<div class="response-box">{annotated_html}</div>',
        unsafe_allow_html=True,
    )

    # ── 3a. Issue details ─────────────────────────────────────────────
    if all_issues:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<span style="font-size:0.82rem;font-weight:600;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:0.06em">Issue Details</span>',
            unsafe_allow_html=True,
        )
        for idx, issue in enumerate(all_issues, 1):
            sev = issue.severity
            label = (
                f"{sev.upper()}  ·  "
                f"{issue.explanation[:72]}{'…' if len(issue.explanation) > 72 else ''}"
            )
            with st.expander(label, expanded=False):
                col_a, col_b = st.columns([1, 5])
                with col_a:
                    st.markdown(sev_chip(sev), unsafe_allow_html=True)
                with col_b:
                    st.markdown(
                        f'<span style="color:#cbd5e1;font-size:0.88rem">{_esc(issue.explanation)}</span>',
                        unsafe_allow_html=True,
                    )

                if issue.quote and issue.quote not in ("(general)", "(node error)"):
                    st.markdown(
                        f'<div style="background:#22263a;border-left:3px solid {SEV_BORDER.get(sev, "#3d4266")};'
                        f'border-radius:4px;padding:0.5rem 0.85rem;font-style:italic;'
                        f'font-size:0.85rem;color:#94a3b8;margin:0.5rem 0">'
                        f'&ldquo;{_esc(issue.quote[:200])}&rdquo;</div>',
                        unsafe_allow_html=True,
                    )

                if issue.evidence:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#94a3b8;margin-top:0.3rem">'
                        f'<strong style="color:#64748b">Evidence:</strong> {_esc(issue.evidence)}</div>',
                        unsafe_allow_html=True,
                    )

                if issue.recommendation:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#86efac;margin-top:0.3rem">'
                        f'<strong>Fix:</strong> {_esc(issue.recommendation)}</div>',
                        unsafe_allow_html=True,
                    )

                # Which critics raised this
                raisers = [
                    c.dimension.value.capitalize()
                    for c in verdict.critiques
                    for ci in c.issues
                    if ci.explanation == issue.explanation
                ]
                if raisers:
                    st.markdown(
                        f'<div style="font-size:0.77rem;color:#64748b;margin-top:0.4rem">'
                        f"Raised by: {', '.join(raisers)}</div>",
                        unsafe_allow_html=True,
                    )

                # Adjudicator ruling
                if verdict.adjudication_used:
                    dismissed = any(
                        di.original_issue == issue.explanation
                        for di in verdict.dismissed_issues
                    )
                    if dismissed:
                        di = next(d for d in verdict.dismissed_issues if d.original_issue == issue.explanation)
                        st.markdown(
                            f'<div style="font-size:0.78rem;color:#f59e0b;margin-top:0.3rem">'
                            f"Adjudicator: Dismissed — {_esc(di.dismissal_reason)}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="font-size:0.78rem;color:#86efac;margin-top:0.3rem">'
                            "Adjudicator: Confirmed</div>",
                            unsafe_allow_html=True,
                        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── 4. Strengths ──────────────────────────────────────────────────
    if verdict.strengths:
        st.markdown(
            '<span style="font-size:0.95rem;font-weight:600;color:#f1f5f9">Strengths</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        for strength in verdict.strengths:
            st.markdown(
                f'<div class="strength-item">'
                f'<span class="strength-dot">+</span>'
                f'<span>{_esc(strength)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── 5. Dismissed Issues ───────────────────────────────────────────
    if verdict.dismissed_issues:
        with st.expander(
            f"Dismissed Issues ({len(verdict.dismissed_issues)})",
            expanded=False,
        ):
            for di in verdict.dismissed_issues:
                st.markdown(
                    f'<div class="issue-card info" style="margin-bottom:0.4rem">'
                    f'<span style="font-size:0.7rem;color:#64748b;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.05em">'
                    f"{di.dimension.value}</span>"
                    f'<div style="color:#94a3b8;margin-top:0.2rem">{_esc(di.original_issue[:120])}</div>'
                    f'<div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem">'
                    f"Dismissed: {_esc(di.dismissal_reason)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── 6. Dimension score chart ──────────────────────────────────────
    if verdict.dimension_scores:
        st.markdown(
            '<span style="font-size:0.95rem;font-weight:600;color:#f1f5f9">Dimension Scores</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        fig = _dimension_bar_chart(verdict.dimension_scores, verdict.dimension_passed)
        st.plotly_chart(fig, use_container_width=True)  # plotly_chart doesn't support width="stretch" yet

    # ── 7. Metadata ───────────────────────────────────────────────────
    meta_parts = []
    if verdict.session_id:
        meta_parts.append(f"Session `{verdict.session_id}`")
    if verdict.total_latency_ms:
        meta_parts.append(f"{verdict.total_latency_ms:.0f} ms")
    if verdict.adjudication_used:
        meta_parts.append("Adjudicated")
    if verdict.fast_path:
        meta_parts.append("Fast-path")
    if verdict.fallbacks_used:
        meta_parts.append(f"{len(verdict.fallbacks_used)} fallback(s)")
    if meta_parts:
        st.markdown(
            f'<div style="font-size:0.74rem;color:#3d4266;margin-top:0.5rem">'
            f"{'  ·  '.join(meta_parts)}</div>",
            unsafe_allow_html=True,
        )
