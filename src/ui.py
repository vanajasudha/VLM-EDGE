"""Small UI helpers: one palette, one stylesheet, a few reusable components."""
"""Industrial UI design system tokens, CSS stylesheet, and reusable components.

Engineered for VLM-Edge: High-density, dark industrial theme with Inter typography.
"""
from html import escape
import streamlit as st

# Official Visual Design Tokens
C = {
    "bg": "#0B1220",
    "sidebar": "#101B2D",
    "card": "#162235",
    "elevated": "#1B2B40",
    "line": "#29384B",
    "border": "#29384B",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "teal": "#22B8A8",
    "teal_hover": "#36D1BE",
    "blue": "#3B82F6",
    "amber": "#F2B84B",
    "red": "#F16D75",
    "green": "#34C78A",
}

CSS = f"""<style>
section[data-testid="stSidebar"] {{border-right:1px solid {C['line']};}}
.hdr h1 {{margin:0; font-size:2rem; letter-spacing:-0.5px;}}
.hdr p {{margin:.2rem 0 1rem; color:{C['muted']};}}
.card {{background:{C['card']}; border:1px solid {C['line']}; border-radius:14px; padding:16px 18px;}}
.kpi .lbl {{color:{C['muted']}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em;}}
.kpi .val {{font-size:1.9rem; font-weight:700; margin:.1rem 0;}}
.kpi .sub {{color:{C['muted']}; font-size:.8rem;}}
.pill {{display:inline-block; padding:2px 10px; border-radius:99px; font-size:.75rem; font-weight:600; margin:2px 4px 2px 0;}}
.pill.ok {{background:rgba(52,211,153,.15); color:{C['green']};}}
.pill.warn {{background:rgba(245,185,74,.15); color:{C['amber']};}}
.pill.off {{background:rgba(138,154,181,.15); color:{C['muted']};}}
.pill.info {{background:rgba(59,130,246,.18); color:#7FB0FF;}}
.callout {{border-left:4px solid; border-radius:8px; padding:10px 14px; margin:.5rem 0 1rem; background:{C['card']};}}
.callout.warn {{border-color:{C['amber']};}} .callout.info {{border-color:{C['blue']};}}
.callout.bad {{border-color:{C['red']};}} .callout.ok {{border-color:{C['green']};}}
.kicker {{color:#2DD4BF; font-size:.68rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase;}}
.status-strip {{display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; background:rgba(19,28,46,.8); border:1px solid #22304A; border-radius:9px; padding:.6rem .8rem; margin:.35rem 0 1.1rem;}}
.status-label {{color:#8A9AB5; font-size:.75rem; margin-right:.15rem;}}
.section-title {{color:#E6EDF7; font-size:.92rem; font-weight:720; margin:1.2rem 0 .55rem;}}
.camera-card {{min-height:242px; padding:10px; background:#131C2E; border:1px solid #22304A; border-radius:11px;}}
.camera-card img {{width:100%; aspect-ratio:16/10; object-fit:cover; border-radius:7px;}}
.camera-meta {{padding:.55rem .2rem .1rem;}}
.camera-name {{color:#E6EDF7; font-size:.84rem; font-weight:750; letter-spacing:.08em;}}
.answer-panel {{background:linear-gradient(145deg, rgba(45,212,191,.11), #131C2E); border:1px solid rgba(45,212,191,.38); border-radius:11px; padding:15px 17px; margin-top:.7rem;}}
.answer-label {{color:#2DD4BF; font-size:.69rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase;}}
.answer-text {{color:#E6EDF7; font-size:1rem; line-height:1.55; margin-top:.45rem;}}
.empty-state {{color:#8A9AB5; border:1px dashed #22304A; border-radius:10px; padding:1.15rem; text-align:center; background:rgba(19,28,46,.5);}}
[data-testid="stDataFrame"] {{border:1px solid #22304A; border-radius:9px;}}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], [class*="st-"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: {C['text']};
}}

.stApp {{
    background-color: {C['bg']};
}}

#MainMenu, footer, header, [data-testid="stToolbar"] {{ visibility: hidden; }}
.block-container {{ padding-top: 0.75rem; padding-bottom: 2rem; max-width: 1320px; }}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: {C['sidebar']};
    border-right: 1px solid {C['border']};
    overflow-x: hidden;
    max-width: 220px;
    min-width: 200px;
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 0.75rem;
    padding-bottom: 1rem;
    padding-left: 0.85rem;
    padding-right: 0.85rem;
    overflow-x: hidden;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.4rem 0 0.8rem;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 0.8rem;
}}

.brand-icon {{
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, {C['teal']}, #138E82);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0B1220;
    font-weight: 800;
    font-size: 1.05rem;
    box-shadow: 0 0 12px rgba(34, 184, 168, 0.35);
}}

.brand-title {{
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {C['text']};
    line-height: 1.15;
}}

.brand-subtitle {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {C['teal']};
    font-weight: 600;
}}

.sidebar-nav-section {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C['muted']};
    margin: 1rem 0 0.4rem;
    padding-left: 0.2rem;
}}

.sidebar-footer {{
    margin-top: 1.5rem;
    padding: 0.85rem;
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    font-size: 0.72rem;
    color: {C['muted']};
}}

.sidebar-footer-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
}}

.sidebar-footer-row:last-child {{
    margin-bottom: 0;
}}

.sidebar-footer-val {{
    color: {C['text']};
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

/* Top Header */
.hdr {{
    margin-bottom: 1.25rem;
    border-bottom: 1px solid {C['border']};
    padding-bottom: 0.85rem;
}}

.hdr .kicker {{
    color: {C['teal']};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.hdr h1 {{
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: {C['text']};
    line-height: 1.2;
}}

.hdr p {{
    margin: 0.3rem 0 0;
    color: {C['muted']};
    font-size: 0.9rem;
    line-height: 1.4;
}}

/* Cards & Surfaces */
.card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 0.85rem;
    transition: border-color 0.18s ease;
}}

.card-elevated {{
    background: {C['elevated']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 0.85rem;
}}

/* KPI Metrics */
.kpi {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 100px;
}}

.kpi .lbl {{
    color: {C['muted']};
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.kpi .val {{
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {C['text']};
    margin: 0.2rem 0;
    font-family: 'JetBrains Mono', monospace;
}}

.kpi .sub {{
    color: {C['muted']};
    font-size: 0.76rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}}

/* Pills & Badges */
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px 4px 2px 0;
    letter-spacing: 0.02em;
}}

.pill.ok {{
    background: rgba(52, 199, 138, 0.14);
    color: {C['green']};
    border: 1px solid rgba(52, 199, 138, 0.3);
}}

.pill.warn {{
    background: rgba(242, 184, 75, 0.14);
    color: {C['amber']};
    border: 1px solid rgba(242, 184, 75, 0.3);
}}

.pill.bad {{
    background: rgba(241, 109, 117, 0.14);
    color: {C['red']};
    border: 1px solid rgba(241, 109, 117, 0.3);
}}

.pill.off {{
    background: rgba(148, 163, 184, 0.12);
    color: {C['muted']};
    border: 1px solid rgba(148, 163, 184, 0.22);
}}

.pill.info {{
    background: rgba(34, 184, 168, 0.14);
    color: {C['teal']};
    border: 1px solid rgba(34, 184, 168, 0.3);
}}

/* Callouts & Banners */
.callout {{
    border-left: 3px solid;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 0.6rem 0 1rem;
    background: {C['card']};
    font-size: 0.85rem;
    line-height: 1.45;
    color: {C['text']};
}}

.callout.warn {{ border-color: {C['amber']}; }}
.callout.info {{ border-color: {C['teal']}; }}
.callout.bad {{ border-color: {C['red']}; }}
.callout.ok {{ border-color: {C['green']}; }}

/* Status strip */
.status-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
    margin: 0.4rem 0 1.1rem;
}}

.status-label {{
    color: {C['muted']};
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: 0.3rem;
}}

/* Section Headings */
.section-title {{
    color: {C['text']};
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 1.1rem 0 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

/* Camera Card & 2x2 Grid */
.camera-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}}

.camera-card:hover {{
    border-color: rgba(34, 184, 168, 0.5);
}}

.camera-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 12px;
    background: {C['elevated']};
    border-bottom: 1px solid {C['border']};
}}

.camera-name {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {C['text']};
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.camera-preview-container {{
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 10;
    background: #060B14;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.camera-preview-container img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}

.camera-card-footer {{
    padding: 9px 12px;
    font-size: 0.72rem;
    color: {C['muted']};
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid {C['border']};
    background: {C['card']};
}}

.camera-offline-placeholder {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: {C['muted']};
    padding: 1.5rem;
    width: 100%;
    height: 100%;
}}

.camera-offline-icon {{
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}}

.camera-offline-msg {{
    font-size: 0.76rem;
    color: {C['muted']};
    font-family: 'JetBrains Mono', monospace;
    max-width: 90%;
}}

/* AI Workspace Styles */
.answer-panel {{
    background: {C['elevated']};
    border: 1px solid {C['teal']};
    border-left: 4px solid {C['teal']};
    border-radius: 8px;
    padding: 16px 18px;
    margin-top: 0.8rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}}

.answer-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.55rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid {C['border']};
}}

.answer-label {{
    color: {C['teal']};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}

.answer-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {C['muted']};
}}

.answer-text {{
    color: {C['text']};
    font-size: 1.02rem;
    line-height: 1.6;
    font-weight: 450;
}}

.empty-state {{
    color: {C['muted']};
    border: 1px dashed {C['border']};
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    background: rgba(22, 34, 53, 0.4);
    font-size: 0.85rem;
}}

/* Streamlit Widget Overrides */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background-color: {C['card']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 6px !important;
}}

div[data-testid="stButton"] button[kind="primary"] {{
    background: linear-gradient(135deg, {C['teal']}, #179E8F) !important;
    color: #08111D !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.25rem !important;
    transition: all 0.15s ease !important;
}}

div[data-testid="stButton"] button[kind="primary"]:hover {{
    background: linear-gradient(135deg, {C['teal_hover']}, #1EB3A2) !important;
    box-shadow: 0 0 14px rgba(34, 184, 168, 0.35) !important;
}}

div[data-testid="stButton"] button[kind="secondary"] {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['text']} !important;
    border-radius: 6px !important;
}}

div[data-testid="stButton"] button[kind="secondary"]:hover {{
    border-color: {C['teal']} !important;
    color: {C['teal']} !important;
}}

/* Radio item active styling in sidebar */
div[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: transparent;
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 2px;
    transition: background 0.12s ease;
}}

div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: rgba(34, 184, 168, 0.08);
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    overflow: hidden;
}}

/* Tab Styling */
button[data-baseweb="tab"] {{
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: {C['muted']} !important;
    padding: 0.5rem 1rem !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {C['teal']} !important;
    border-bottom-color: {C['teal']} !important;
}}
</style>"""


def inject():
    """Inject global CSS rules."""
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "", kicker: str = "Industrial AI Operations"):
    """Render a clean industrial page header."""
    st.markdown(
        f'<div class="hdr">'
        f'  <div class="kicker"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{C["teal"]};"></span>{escape(kicker)}</div>'
        f'  <h1>{escape(title)}</h1>'
        f'  <p>{escape(subtitle)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "off") -> str:
    """Render a colored status pill."""
    return f'<span class="pill {kind}">{escape(text)}</span>'


def kpi(col, label: str, value, sub: str = "", delta_kind: str = "neutral"):
    """Render a high-density industrial KPI card."""
    col.markdown(
        f'<div class="card kpi">'
        f'  <div class="lbl">{escape(label)}</div>'
        f'  <div class="val">{escape(str(value))}</div>'
        f'  <div class="sub">{escape(sub)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info"):
    """Render an informative message block."""
    st.markdown(f'<div class="callout {kind}">{escape(text)}</div>', unsafe_allow_html=True)


def section_title(text: str, badge: str = ""):
    """Render a consistent section title."""
    badge_html = f' <span class="pill info">{escape(badge)}</span>' if badge else ""
    st.markdown(f'<div class="section-title">{escape(text)}{badge_html}</div>', unsafe_allow_html=True)


def status_strip(items):
    """Render a horizontal status ribbon with multiple status pills."""
    content = "".join(pill(text, kind) for text, kind in items)
    st.markdown(
        f'<div class="status-strip"><span class="status-label">System Scope</span>{content}</div>',
        unsafe_allow_html=True,
    )


def empty_state(text: str):
    """Render a dashed empty state container."""
    st.markdown(f'<div class="empty-state">{escape(text)}</div>', unsafe_allow_html=True)


def sidebar_brand():
    """Render the top branding section of the sidebar."""
    st.sidebar.markdown(
        f'<div class="sidebar-brand">'
        f'  <div class="brand-icon">V</div>'
        f'  <div>'
        f'    <div class="brand-title">VLM-Edge</div>'
        f'    <div class="brand-subtitle">Visual Intelligence Platform</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def sidebar_system_footer(gpu_name="RTX 3050 (4 GB)", model_name="Moondream2 NF4", version="v1.2.0"):
    """Render the bottom unobtrusive system status info in sidebar."""
    st.sidebar.markdown(
        f'<div class="sidebar-footer">'
        f'  <div class="sidebar-footer-row">'
        f'    <span>Model:</span><span class="sidebar-footer-val">{escape(model_name)}</span>'
        f'  </div>'
        f'  <div class="sidebar-footer-row">'
        f'    <span>Target GPU:</span><span class="sidebar-footer-val">{escape(gpu_name)}</span>'
        f'  </div>'
        f'  <div class="sidebar-footer-row">'
        f'    <span>Platform:</span><span class="sidebar-footer-val">WSL2 Ubuntu</span>'
        f'  </div>'
        f'  <div class="sidebar-footer-row">'
        f'    <span>Build:</span><span class="sidebar-footer-val">{escape(version)}</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def camera_offline_placeholder(camera_name: str, endpoint: str, message: str = "Stream disconnected"):
    """Render an SVG industrial placeholder when camera feed is offline."""
    return (
        f'<div class="camera-offline-placeholder">'
        f'  <div style="margin-bottom:0.4rem;color:{C["muted"]};">'
        f'    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        f'      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>'
        f'      <circle cx="12" cy="13" r="3"/>'
        f'      <line x1="2" y1="2" x2="22" y2="22" stroke="{C["red"]}" stroke-width="2"/>'
        f'    </svg>'
        f'  </div>'
        f'  <div style="font-weight:600;font-size:0.82rem;color:{C["text"]};margin-bottom:0.2rem;">{escape(camera_name)} Offline</div>'
        f'  <div class="camera-offline-msg">{escape(message)}</div>'
        f'  <div style="font-size:0.68rem;color:{C["muted"]};margin-top:0.35rem;font-family:\'JetBrains Mono\', monospace;">{escape(endpoint)}</div>'
        f'</div>'
    )
