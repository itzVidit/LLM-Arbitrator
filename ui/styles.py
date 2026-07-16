"""
ui/styles.py
------------
Single source of truth for all colors, CSS, and HTML helpers.

Design system: dark-mode, minimal, professional.
  - Background layers:  #0f1117 (page) → #1a1d27 (surface) → #22263a (raised)
  - Borders:            #2d3148 (subtle) / #3d4266 (prominent)
  - Text:               #f1f5f9 (primary) / #94a3b8 (muted) / #64748b (dimmed)
  - Accent (brand):     #6366f1 (indigo-500)
  - Status:             green #22c55e / amber #f59e0b / red #ef4444
  - Severity (4 levels): red / orange / yellow / slate — all on dark backgrounds

Usage:
    from ui.styles import COLORS, SEV_COLORS, inject_css
    inject_css()
    color = SEV_COLORS["critical"]
"""

from __future__ import annotations
import streamlit as st

# ─────────────────────────────────────────────────────────────────────
#  Palette tokens
# ─────────────────────────────────────────────────────────────────────

COLORS: dict[str, str] = {
    # Backgrounds
    "bg_page":       "#0f1117",
    "bg_surface":    "#1a1d27",
    "bg_raised":     "#22263a",
    "bg_input":      "#1e2235",
    # Borders
    "border":        "#2d3148",
    "border_strong": "#3d4266",
    # Text
    "text_primary":  "#f1f5f9",
    "text_muted":    "#94a3b8",
    "text_dimmed":   "#64748b",
    # Brand accent
    "accent":        "#6366f1",
    # Status
    "approve":       "#22c55e",
    "revise":        "#f59e0b",
    "reject":        "#ef4444",
    # Pass / fail (subtle fills)
    "pass_bg":       "#14532d",
    "pass_fg":       "#86efac",
    "fail_bg":       "#450a0a",
    "fail_fg":       "#fca5a5",
    # Agreement
    "agree":         "#166534",
    "agree_fg":      "#86efac",
    "disagree":      "#431407",
    "disagree_fg":   "#fdba74",
    # Score gradient
    "score_high":    "#22c55e",
    "score_mid":     "#f59e0b",
    "score_low":     "#ef4444",
    # Legacy alias
    "muted":         "#64748b",
    "surface":       "#1a1d27",
    "surface_dark":  "#0f1117",
}

SEV_COLORS: dict[str, str] = {
    "critical": "#ef4444",
    "major":    "#f97316",
    "minor":    "#eab308",
    "info":     "#64748b",
    "strength": "#22c55e",
}

# Dark-safe background tints (these sit ON the dark surface)
SEV_BG: dict[str, str] = {
    "critical": "#2d0f0f",
    "major":    "#2d1a0a",
    "minor":    "#2a2200",
    "info":     "#1a1d27",
    "strength": "#0f2d1a",
}

SEV_BORDER: dict[str, str] = {
    "critical": "#ef4444",
    "major":    "#f97316",
    "minor":    "#eab308",
    "info":     "#3d4266",
    "strength": "#22c55e",
}

DIM_COLORS: dict[str, str] = {
    "accuracy":     "#6366f1",
    "logic":        "#0ea5e9",
    "completeness": "#8b5cf6",
    "safety":       "#ef4444",
    "style":        "#f59e0b",
}


# ─────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────

def score_color(score: int | float, out_of: int | float = 100) -> str:
    pct = max(0.0, min(1.0, score / out_of))
    if pct >= 0.75:
        return COLORS["score_high"]
    elif pct >= 0.50:
        return COLORS["score_mid"]
    return COLORS["score_low"]


def rec_color(recommendation: str) -> str:
    return COLORS.get(recommendation.lower(), COLORS["muted"])


# ─────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Reset / layout ──────────────────────────────────────────────── */
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ── Verdict badge ──────────────────────────────────────────────── */
.verdict-badge {
    display: inline-block;
    padding: 0.25em 0.85em;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #fff;
}
.badge-approve { background: #16a34a; }
.badge-revise  { background: #d97706; }
.badge-reject  { background: #dc2626; }

/* ── Severity chips ─────────────────────────────────────────────── */
.sev-chip {
    display: inline-block;
    padding: 0.15em 0.55em;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.sev-critical { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }
.sev-major    { background: #431407; color: #fdba74; border: 1px solid #f97316; }
.sev-minor    { background: #2a2200; color: #fde68a; border: 1px solid #ca8a04; }
.sev-info     { background: #1e2235; color: #94a3b8; border: 1px solid #3d4266; }
.sev-strength { background: #052e16; color: #86efac; border: 1px solid #22c55e; }

/* ── Inline annotation spans ────────────────────────────────────── */
.ann-critical { background: rgba(239,68,68,0.18); border-bottom: 2px solid #ef4444; border-radius: 2px; padding: 0 2px; }
.ann-major    { background: rgba(249,115,22,0.18); border-bottom: 2px solid #f97316; border-radius: 2px; padding: 0 2px; }
.ann-minor    { background: rgba(234,179,8,0.15);  border-bottom: 2px solid #eab308; border-radius: 2px; padding: 0 2px; }
.ann-info     { background: rgba(100,116,139,0.12);border-bottom: 1px solid #64748b; border-radius: 2px; padding: 0 2px; }
.ann-strength { background: rgba(34,197,94,0.15);  border-bottom: 2px solid #22c55e; border-radius: 2px; padding: 0 2px; }

/* ── Metric card ────────────────────────────────────────────────── */
.metric-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    text-align: center;
    height: 100%;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.73rem;
    color: #64748b;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Critic card ────────────────────────────────────────────────── */
.critic-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.critic-card.pass { border-left: 3px solid #22c55e; }
.critic-card.fail { border-left: 3px solid #ef4444; }

/* ── Issue card ─────────────────────────────────────────────────── */
.issue-card {
    border-radius: 6px;
    padding: 0.55rem 0.75rem;
    margin: 0.35rem 0;
    font-size: 0.84rem;
    border-left: 3px solid #3d4266;
    background: #1a1d27;
    color: #cbd5e1;
}
.issue-card.critical { border-left-color: #ef4444; background: #1f0d0d; }
.issue-card.major    { border-left-color: #f97316; background: #1f1108; }
.issue-card.minor    { border-left-color: #eab308; background: #1a1900; }
.issue-card.info     { border-left-color: #3d4266; background: #1a1d27; }
.issue-card.strength { border-left-color: #22c55e; background: #0a1f10; }

/* ── Agreement / disagreement pills ────────────────────────────── */
.agree-pill {
    background: #14532d;
    color: #86efac;
    border: 1px solid #166534;
    padding: 0.12em 0.6em;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
}
.disagree-pill {
    background: #431407;
    color: #fdba74;
    border: 1px solid #7c2d12;
    padding: 0.12em 0.6em;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
}

/* ── Agreement matrix table ─────────────────────────────────────── */
.matrix-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.82rem;
}
.matrix-table th {
    background: #1a1d27;
    color: #94a3b8;
    font-weight: 600;
    padding: 8px 12px;
    text-align: center;
    border: 1px solid #2d3148;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.matrix-table td {
    background: #1a1d27;
    border: 1px solid #2d3148;
    padding: 7px 10px;
    text-align: center;
    color: #cbd5e1;
}
.matrix-table td.row-label {
    text-align: left;
    color: #94a3b8;
    font-weight: 600;
    background: #22263a;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.matrix-table td.diagonal {
    background: #22263a;
    color: #3d4266;
}

/* ── Annotated response box ─────────────────────────────────────── */
.response-box {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    font-size: 0.91rem;
    line-height: 1.8;
    color: #e2e8f0;
    font-family: 'Georgia', serif;
}

/* ── Section divider ────────────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1px solid #2d3148;
    margin: 1.5rem 0;
}

/* ── Batch status text ──────────────────────────────────────────── */
.batch-status-approve { color: #22c55e; font-weight: 700; }
.batch-status-revise  { color: #f59e0b; font-weight: 700; }
.batch-status-reject  { color: #ef4444; font-weight: 700; }

/* ── Sidebar cleanup ────────────────────────────────────────────── */
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}
section[data-testid="stSidebar"] hr {
    border-color: #2d3148;
    margin: 0.8rem 0;
}

/* ── Sidebar text area labels ───────────────────────────────────── */
section[data-testid="stSidebar"] .stTextArea label p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.3rem;
}

/* ── Sidebar text area inputs ───────────────────────────────────── */
section[data-testid="stSidebar"] .stTextArea textarea {
    background-color: #131620 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    padding: 0.6rem 0.75rem !important;
    resize: none !important;
    transition: border-color 0.15s ease;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
section[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
section[data-testid="stSidebar"] .stTextArea textarea::placeholder {
    color: #3d4266 !important;
}

/* ── Sidebar buttons ────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.45rem 0 !important;
    transition: all 0.15s ease !important;
    border: none !important;
}

/* Run button (primary) */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #6366f1 !important;
    color: #fff !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #4f52d3 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled {
    background: #2d3148 !important;
    color: #64748b !important;
    cursor: not-allowed !important;
}

/* Sample button (secondary) */
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: #1e2235 !important;
    color: #94a3b8 !important;
    border: 1px solid #2d3148 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: #2d3148 !important;
    color: #e2e8f0 !important;
    border-color: #3d4266 !important;
}

/* ── Sidebar radio nav ──────────────────────────────────────────── */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.1rem !important;
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.4rem 0.6rem !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    cursor: pointer !important;
    transition: background 0.12s ease !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #1e2235 !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
section[data-testid="stSidebar"] .stRadio input:checked ~ label {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}

/* ── Spinner text ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stSpinner p {
    font-size: 0.8rem !important;
    color: #64748b !important;
}

/* ── Feature preview tile ───────────────────────────────────────── */
.feature-tile {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 1.3rem 1.1rem;
    text-align: center;
    height: 100%;
}
.feature-tile .icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.feature-tile strong { color: #f1f5f9; font-size: 0.95rem; }
.feature-tile p {
    font-size: 0.81rem;
    color: #64748b;
    margin-top: 0.4rem;
    line-height: 1.55;
}

/* ── Outlier banner ─────────────────────────────────────────────── */
.outlier-banner {
    display: inline-block;
    background: #431407;
    color: #fdba74;
    border: 1px solid #7c2d12;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
    text-transform: uppercase;
}

/* ── Strength item ──────────────────────────────────────────────── */
.strength-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.4rem 0;
    color: #cbd5e1;
    font-size: 0.88rem;
}
.strength-dot {
    color: #22c55e;
    font-size: 1rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}
</style>
"""


def inject_css() -> None:
    """Inject shared CSS into the Streamlit page. Call once per page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  HTML component helpers
# ─────────────────────────────────────────────────────────────────────

def badge(recommendation: str) -> str:
    cls = f"badge-{recommendation.lower()}"
    return f'<span class="verdict-badge {cls}">{recommendation.upper()}</span>'


def sev_chip(severity: str) -> str:
    s = severity.lower()
    return f'<span class="sev-chip sev-{s}">{s.upper()}</span>'


def metric_card(value: str, label: str, color: str = "#f1f5f9") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f"</div>"
    )
