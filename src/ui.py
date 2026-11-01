"""Central palette, responsive stylesheet and reusable presentation helpers."""
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

CSS = """<style>
.st-key-camera_grid_compact {max-width:960px;}
.camera-source-row {display:grid; grid-template-columns:4.5rem minmax(0,1fr) 9rem; gap:.5rem; padding:.55rem 0; border-bottom:1px solid #263346; align-items:start;}
.camera-source-row span {overflow-wrap:anywhere; font-size:.85rem;}
.camera-source-row .source-status {color:#AAB9CB;}
@media(max-width:700px) {
 .camera-source-row {grid-template-columns:4rem minmax(0,1fr);}
 .camera-source-row .source-status {grid-column:2;}
}
/* Native Streamlit owns geometry; only stable test IDs and our classes are used. */
[data-testid="stAppViewContainer"] {background:#0B1220; color:#E6EDF7;}
[data-testid="stSidebar"] {background:#111C2D; border-right:1px solid #263346;}
[data-testid="stMainBlockContainer"] {max-width:1280px; padding-top:2rem; padding-bottom:1.2rem;}
h1 {font-size:1.65rem !important; line-height:1.3 !important;}
h2 {font-size:1.2rem !important;}
h3 {font-size:1rem !important;}
p, label, .stMarkdown {overflow-wrap:anywhere;}
[data-testid="stVerticalBlock"], [data-testid="stColumn"] {min-width:0;}
[data-testid="stImage"] img {max-width:100%; height:auto;}
/* Native page links are focusable anchors, not styled radio inputs. */
.st-key-application_navigation [data-testid="stPageLink"] a {padding:.65rem .8rem; border-radius:.45rem; min-height:2.8rem; text-decoration:none;}
.st-key-application_navigation [data-testid="stPageLink"] a:hover {background:#172A3A;}
.st-key-application_navigation [data-testid="stPageLink"] a:focus-visible {outline:2px solid #68D6C8; outline-offset:2px;}
.st-key-active_navigation [data-testid="stPageLink"] a {background:#15353B; color:#8CE1D4; border-left:3px solid #2DB8A5;}
.sidebar-brand {padding:.5rem 0 1rem; margin-bottom:.8rem; border-bottom:1px solid #263346;}
.brand-title {font-size:1.3rem; font-weight:700; letter-spacing:-.02em;}
.brand-subtitle {color:#9BAABD; font-size:.8rem; margin-top:.2rem;}
.brand-icon {display:none;}
.hdr h1 {margin:0; font-size:1.65rem;}
.hdr p {color:#9BAABD; margin:.3rem 0 1rem;}
.kicker {display:none;}
.pill {display:inline-block; font-size:.8rem; padding:.15rem .5rem; border-radius:.35rem; background:#1D2B3D; color:#AAB9CB;}
.pill.ok {color:#79D8B0; background:#13372F;}
.pill.warn {color:#E9C77F; background:#362E1E;}
.camera-placeholder {aspect-ratio:16/9; background:#101A29; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#9BAABD; border-radius:.4rem; gap:.35rem;}
.camera-placeholder strong {color:#CFD8E5; font-size:.95rem;}
.empty-state {background:#111C2D; border-radius:.5rem; padding:1.2rem; color:#AAB9CB;}
.answer-panel {background:#122B30; border-left:3px solid #2DB8A5; border-radius:.4rem; padding:1rem; margin:.5rem 0;}
.answer-label {font-size:.8rem; color:#77D8CA; font-weight:600;}
.answer-text {white-space:pre-wrap; overflow-wrap:anywhere; margin-top:.4rem; line-height:1.6;}
.section-title {font-size:1rem; font-weight:600; margin:.8rem 0;}
.callout {padding:.8rem; background:#142437; border-radius:.4rem;}
.card {padding:.8rem; background:#142437; border-radius:.4rem;}
.lbl, .sub {color:#9BAABD; font-size:.8rem;} .val {font-size:1.4rem;}
@media(max-width:700px) {
 [data-testid="stMainBlockContainer"] {padding:4.5rem 1rem 1.2rem;}
 [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100% !important; width:100% !important;}
 h1 {font-size:1.4rem !important;}
}
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
