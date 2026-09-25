"""
ui/app.py
---------
Main Streamlit entry point for the LLM Output Arbitration Verdict Explorer.

Run with:
    cd "D:\\Desktop\\LLM Output Arbitration System"
    .venv\\Scripts\\streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from ui.styles import COLORS, badge, inject_css, score_color


# ─────────────────────────────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Arbitration",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
#  Session-state
# ─────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "verdict":        None,
        "current_page":   "Verdict Explorer",
        "_last_prompt":   "",
        "_last_response": "",
        "run_error":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────
#  Sample data
# ─────────────────────────────────────────────────────────────────────

SAMPLE_PROMPT = (
    "Explain how HTTPS works, including the TLS handshake process, "
    "and describe the difference between symmetric and asymmetric encryption "
    "with an example of each."
)

SAMPLE_RESPONSE = (
    "HTTPS works by using SSL to encrypt web traffic. When you visit a website, "
    "your browser and the server shake hands using a process invented by Google in 2010. "
    "During this handshake, they agree on an encryption key.\n\n"
    "Asymmetric encryption uses two different keys (public and private). "
    "RSA is an example -- it uses prime number magic to create unbreakable keys. "
    "The public key is used to encrypt, only the private key can decrypt.\n\n"
    "Symmetric encryption uses the same key for both -- AES is a good example. "
    "It's much slower than asymmetric encryption and therefore is only used "
    "for the initial handshake, after which both sides switch to asymmetric.\n\n"
    "HTTPS is 100% secure and no man-in-the-middle attack can ever succeed against it. "
    "You should always trust any website with HTTPS."
)


# ─────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    with st.sidebar:
        # Brand header
        st.markdown(
            '<div style="padding:0.75rem 0 1rem">'
            '<div style="display:flex;align-items:center;gap:0.6rem">'
            '<span style="font-size:1.4rem;line-height:1">⚖️</span>'
            '<div>'
            '<div style="font-size:1rem;font-weight:700;color:#f1f5f9;line-height:1.2">'
            'Verdict Explorer</div>'
            '<div style="font-size:0.73rem;color:#64748b;margin-top:1px">'
            'LLM Output Arbitration</div>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color:#2d3148;margin:0 0 0.8rem 0'>", unsafe_allow_html=True)

        # Navigation
        page = st.radio(
            "nav",
            options=["Verdict Explorer", "Critic Comparison", "Batch Mode"],
            index=["Verdict Explorer", "Critic Comparison", "Batch Mode"].index(
                st.session_state.current_page
            ),
            label_visibility="collapsed",
        )
        st.session_state.current_page = page

        st.markdown("<hr style='border-color:#2d3148;margin:0.8rem 0'>", unsafe_allow_html=True)

        # Input form (single-response pages only)
        if page in ("Verdict Explorer", "Critic Comparison"):
            st.markdown(
                '<div style="margin-bottom:0.75rem">'
                '<span style="font-size:0.68rem;font-weight:700;color:#6366f1;'
                'text-transform:uppercase;letter-spacing:0.1em">Evaluate</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            prompt = st.text_area(
                "PROMPT",
                value=st.session_state._last_prompt or SAMPLE_PROMPT,
                height=100,
                key="sidebar_prompt",
                placeholder="Enter the original prompt…",
            )
            response = st.text_area(
                "LLM RESPONSE",
                value=st.session_state._last_response or SAMPLE_RESPONSE,
                height=140,
                key="sidebar_response",
                placeholder="Paste the LLM response to evaluate…",
            )

            st.markdown("<div style='height:0.15rem'></div>", unsafe_allow_html=True)

            # ── Critic Weights ────────────────────────────────────────
            with st.expander("⚖️ Critic Weights", expanded=False):
                st.markdown(
                    '<div style="font-size:0.72rem;color:#64748b;margin-bottom:0.6rem">'
                    'Adjust how much each critic contributes to the overall score. '
                    'Weights are normalised automatically to sum to 1.0.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                w_accuracy     = st.slider("Accuracy",     0, 100, 30, step=5, key="w_accuracy")
                w_logic        = st.slider("Logic",        0, 100, 25, step=5, key="w_logic")
                w_safety       = st.slider("Safety",       0, 100, 20, step=5, key="w_safety")
                w_completeness = st.slider("Completeness", 0, 100, 15, step=5, key="w_completeness")
                w_style        = st.slider("Style",        0, 100, 10, step=5, key="w_style")

                _raw_total = w_accuracy + w_logic + w_safety + w_completeness + w_style
                if _raw_total == 0:
                    st.warning("All weights are 0 — defaults will be used.")
                    _use_custom_weights = False
                else:
                    # Show normalised preview
                    _norm = {
                        "accuracy":     round(w_accuracy     / _raw_total, 3),
                        "logic":        round(w_logic        / _raw_total, 3),
                        "safety":       round(w_safety       / _raw_total, 3),
                        "completeness": round(w_completeness / _raw_total, 3),
                        "style":        round(w_style        / _raw_total, 3),
                    }
                    st.markdown(
                        '<div style="font-size:0.71rem;color:#6366f1;margin-top:0.3rem">'
                        f'Normalised: acc={_norm["accuracy"]} · log={_norm["logic"]} · '
                        f'saf={_norm["safety"]} · com={_norm["completeness"]} · '
                        f'sty={_norm["style"]}'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    _use_custom_weights = True

            col_run, col_sample = st.columns([3, 2])
            with col_run:
                run_btn = st.button(
                    "▶  Run",
                    type="primary",
                    width="stretch",
                    disabled=not (prompt.strip() and response.strip()),
                )
            with col_sample:
                sample_btn = st.button("Sample", width="stretch")

            if sample_btn:
                st.session_state._last_prompt   = SAMPLE_PROMPT
                st.session_state._last_response = SAMPLE_RESPONSE
                st.session_state.verdict        = None
                st.session_state.run_error      = None
                st.rerun()

            if run_btn:
                st.session_state._last_prompt   = prompt
                st.session_state._last_response = response
                st.session_state.run_error      = None

                # Build critic_weights from sliders if customised
                from models.critique import CriticDimension
                if _use_custom_weights:
                    _total = w_accuracy + w_logic + w_safety + w_completeness + w_style
                    critic_weights = {
                        CriticDimension.ACCURACY:     round(w_accuracy     / _total, 4),
                        CriticDimension.LOGIC:        round(w_logic        / _total, 4),
                        CriticDimension.SAFETY:       round(w_safety       / _total, 4),
                        CriticDimension.COMPLETENESS: round(w_completeness / _total, 4),
                        CriticDimension.STYLE:        round(w_style        / _total, 4),
                    }
                    # Fix rounding drift so weights sum to exactly 1.0
                    _diff = 1.0 - sum(critic_weights.values())
                    critic_weights[CriticDimension.ACCURACY] = round(
                        critic_weights[CriticDimension.ACCURACY] + _diff, 4
                    )
                else:
                    critic_weights = None

                with st.spinner("Running…"):
                    try:
                        from models.critique import ArbitrationRequest
                        from orchestration import run_arbitration_sync
                        request = ArbitrationRequest(
                            original_prompt=prompt,
                            llm_response=response,
                            session_id=None,
                            critic_weights=critic_weights,
                        )
                        st.session_state.verdict = run_arbitration_sync(request)
                    except Exception as exc:
                        st.session_state.run_error = str(exc)
                        st.session_state.verdict   = None
                st.rerun()

        # Current verdict status
        verdict = st.session_state.get("verdict")
        if verdict:
            st.markdown("<hr style='border-color:#2d3148;margin:0.8rem 0'>", unsafe_allow_html=True)
            rc = score_color(verdict.overall_score)
            st.markdown(
                f'<div style="background:#1a1d27;border:1px solid #2d3148;border-radius:8px;'
                f'padding:0.75rem;text-align:center">'
                f'{badge(verdict.recommendation)}'
                f'<div style="font-size:1.5rem;font-weight:800;color:{rc};margin-top:0.35rem;line-height:1">'
                f"{verdict.overall_score}/100</div>"
                f'<div style="font-size:0.71rem;color:#64748b;margin-top:0.2rem">Current verdict</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    return page


# ─────────────────────────────────────────────────────────────────────
#  Page renderers
# ─────────────────────────────────────────────────────────────────────

def _page_verdict_explorer() -> None:
    inject_css()
    verdict = st.session_state.get("verdict")
    error   = st.session_state.get("run_error")

    st.markdown(
        '<h1 style="font-size:1.7rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem">'
        'Verdict Explorer</h1>',
        unsafe_allow_html=True,
    )

    if error:
        st.markdown(
            f'<div style="background:#2d0f0f;border:1px solid #ef4444;border-radius:6px;'
            f'padding:0.75rem 1rem;color:#fca5a5;font-size:0.87rem;margin-bottom:1rem">'
            f'Pipeline error: {error}</div>',
            unsafe_allow_html=True,
        )
        return

    if verdict is None:
        _render_empty_state()
        return

    from ui.verdict_view import render_verdict_view
    render_verdict_view(verdict)


def _page_critic_comparison() -> None:
    inject_css()
    verdict = st.session_state.get("verdict")
    error   = st.session_state.get("run_error")

    st.markdown(
        '<h1 style="font-size:1.7rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem">'
        'Critic Comparison</h1>',
        unsafe_allow_html=True,
    )

    if error:
        st.markdown(
            f'<div style="background:#2d0f0f;border:1px solid #ef4444;border-radius:6px;'
            f'padding:0.75rem 1rem;color:#fca5a5;font-size:0.87rem">'
            f'Pipeline error: {error}</div>',
            unsafe_allow_html=True,
        )
        return

    if verdict is None:
        st.markdown(
            '<div style="background:#1a1d27;border:1px solid #2d3148;border-radius:8px;'
            'padding:1.2rem;color:#64748b;font-size:0.87rem">'
            'Run an arbitration in the sidebar to see the critic comparison.</div>',
            unsafe_allow_html=True,
        )
        return

    from ui.critic_panel import render_critic_panel
    render_critic_panel(verdict)


def _page_batch_mode() -> None:
    inject_css()
    st.markdown(
        '<h1 style="font-size:1.7rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem">'
        'Batch Mode</h1>',
        unsafe_allow_html=True,
    )
    from ui.batch_view import render_batch_view
    render_batch_view()


def _render_empty_state() -> None:
    """Feature-preview tiles shown when no verdict is loaded."""
    st.markdown(
        '<div style="background:#1a1d27;border:1px solid #2d3148;border-radius:8px;'
        'padding:1rem 1.2rem;color:#94a3b8;font-size:0.87rem;margin-bottom:1.5rem">'
        'Enter a prompt and LLM response in the sidebar, then click <strong>Run</strong>. '
        'Or click <strong>Sample</strong> to load a demo with intentional errors.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span style="font-size:0.78rem;font-weight:600;color:#64748b;'
        'text-transform:uppercase;letter-spacing:0.07em">What you\'ll see</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    tiles = [
        ("📝", "Verdict Explorer",
         "Original response with color-coded inline annotations. "
         "Expand any highlight to see which critic flagged it and the adjudicator ruling."),
        ("🔬", "Critic Comparison",
         "All 5 critics side-by-side with a score radar chart, "
         "agreement matrix, and outlier detection."),
        ("📊", "Batch Mode",
         "Submit up to 10 responses at once. "
         "Sortable results table with score, confidence, and issue counts."),
    ]

    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, tiles):
        with col:
            st.markdown(
                f'<div class="feature-tile">'
                f'<div class="icon">{icon}</div>'
                f'<strong>{title}</strong>'
                f'<p>{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    page = _render_sidebar()

    if page == "Verdict Explorer":
        _page_verdict_explorer()
    elif page == "Critic Comparison":
        _page_critic_comparison()
    elif page == "Batch Mode":
        _page_batch_mode()


if __name__ == "__main__":
    main()
