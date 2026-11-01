"""Central palette, responsive stylesheet and reusable presentation helpers."""
from html import escape
import streamlit as st

# Official Visual Design Tokens
C = {
    "bg": "#080F1D", "sidebar": "#101B2D", "card": "#111E31",
    "elevated": "#17263B", "line": "#28394F", "border": "#28394F",
    "text": "#F1F5FC", "muted": "#9AAAC0", "teal": "#20DCC3",
    "teal_hover": "#50E8D4", "blue": "#42C8FF", "amber": "#E9B95E",
    "red": "#E77C83", "green": "#54C99A",
}

CSS = """<style>
.st-key-camera_grid_compact {max-width:960px;}
.camera-source-row {display:grid; grid-template-columns:4.5rem minmax(0,1fr) 9rem; gap:.5rem; padding:.7rem 0; border-bottom:1px solid #28394F; align-items:start;}
.camera-source-row span {overflow-wrap:anywhere; font-size:.85rem;}
.camera-source-row .source-status {color:#9AAAC0;}
@media(max-width:700px) {
 .camera-source-row {grid-template-columns:4rem minmax(0,1fr);}
 .camera-source-row .source-status {grid-column:2;}
}
/* Native Streamlit owns layout, routing and focus behavior. */
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 78% -8%, rgba(32,220,195,.10) 0%, rgba(16,27,45,.42) 24%, #080F1D 62%); color:#F1F5FC;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:rgba(13,25,42,.96); border-right:1px solid rgba(140,168,197,.13);}
[data-testid="stMainBlockContainer"] {max-width:1420px; padding-top:2rem; padding-bottom:2rem;}
h1 {font-size:2rem !important; line-height:1.2 !important; letter-spacing:-.045em; font-weight:680 !important;}
h2 {font-size:1.25rem !important; letter-spacing:-.02em;}
h3 {font-size:1rem !important;}
p, label, .stMarkdown {overflow-wrap:anywhere;}
[data-testid="stCaptionContainer"] {color:#9AAAC0;}
[data-testid="stVerticalBlock"], [data-testid="stColumn"] {min-width:0;}
[data-testid="stImage"] img {max-width:100%; height:auto;}
.st-key-application_navigation [data-testid="stPageLink"] a {padding:.65rem .8rem; border-radius:.7rem; min-height:2.8rem; text-decoration:none; transition:background-color .16s ease,color .16s ease;}
.st-key-application_navigation [data-testid="stPageLink"] a:hover {background:#172A3A; color:#F1F5FC;}
.st-key-application_navigation [data-testid="stPageLink"] a:focus-visible {outline:2px solid #20DCC3; outline-offset:2px;}
.st-key-active_navigation [data-testid="stPageLink"] a {background:rgba(32,220,195,.11); color:#77EBDD; border-left:3px solid #20DCC3;}
.sidebar-brand {padding:.65rem .2rem 1.15rem; margin-bottom:.9rem; border-bottom:1px solid rgba(140,168,197,.14);}
.brand-title {font-size:1.35rem; font-weight:720; letter-spacing:-.035em;}
.brand-subtitle {color:#9AAAC0; font-size:.77rem; margin-top:.25rem; letter-spacing:.02em;}
.brand-icon {float:left; display:grid; place-items:center; width:2.25rem; height:2.25rem; margin:.1rem .7rem 0 0; border-radius:.7rem; background:rgba(32,220,195,.12); color:#20DCC3; font-weight:800;}
.hdr h1 {margin:0; font-size:1.85rem;}
.hdr p {color:#9AAAC0; margin:.35rem 0 1.25rem;}
.kicker {font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; color:#5DDBCB; margin:0 0 .45rem; font-weight:700;}
.page-subtitle {color:#9AAAC0; font-size:.98rem; margin:-.45rem 0 1.35rem; max-width:58rem; line-height:1.55;}
.pill {display:inline-block; font-size:.75rem; padding:.18rem .58rem; border:1px solid rgba(154,170,192,.16); border-radius:99px; background:rgba(154,170,192,.08); color:#AAB9CB; white-space:nowrap;}
.pill.ok {color:#75D8B3; background:rgba(37,151,108,.14); border-color:rgba(84,201,154,.25);}
.pill.warn {color:#E9C77F; background:rgba(180,133,42,.13); border-color:rgba(233,185,94,.22);}
.camera-placeholder {aspect-ratio:16/9; background:linear-gradient(145deg,#101B2D,#0A1423); display:flex; flex-direction:column; align-items:center; justify-content:center; color:#9AAAC0; border-radius:.75rem; gap:.35rem; border:1px solid rgba(140,168,197,.12);}
.empty-state {background:rgba(17,30,49,.8); border:1px solid rgba(140,168,197,.14); border-radius:1rem; padding:1.2rem; color:#9AAAC0;}
.answer-panel {background:rgba(15,37,48,.82); border:1px solid rgba(32,220,195,.2); border-left:3px solid #20DCC3; border-radius:.9rem; padding:1.1rem 1.2rem; margin:.7rem 0; box-shadow:0 10px 30px rgba(0,0,0,.14);}
.answer-label {font-size:.76rem; letter-spacing:.07em; text-transform:uppercase; color:#77EBDD; font-weight:650;}
.answer-text {white-space:pre-wrap; overflow-wrap:anywhere; word-break:normal; margin-top:.55rem; line-height:1.75; color:#E6EDF7;}
.section-title {font-size:1rem; font-weight:650; margin:.8rem 0;}
.callout,.card {padding:1rem 1.1rem; background:rgba(17,30,49,.72); border:1px solid rgba(140,168,197,.13); border-radius:.9rem; box-shadow:0 10px 30px rgba(0,0,0,.12);}
.lbl, .sub {color:#9AAAC0; font-size:.8rem;} .val {font-size:1.45rem; font-weight:650; letter-spacing:-.025em; margin:.35rem 0;}
[data-testid="stVerticalBlockBorderWrapper"] {background:linear-gradient(145deg,rgba(20,37,58,.88),rgba(12,24,40,.88)); border-color:rgba(140,168,197,.18); border-radius:1rem; box-shadow:0 14px 36px rgba(0,0,0,.22); transition:border-color .18s ease,transform .18s ease,box-shadow .18s ease;}
[data-testid="stVerticalBlockBorderWrapper"]:hover {border-color:rgba(32,220,195,.28); box-shadow:0 16px 38px rgba(0,0,0,.25),0 0 24px rgba(32,220,195,.035); transform:translateY(-1px);}
[data-testid="stImage"] img {border-radius:.72rem; border:1px solid rgba(140,168,197,.13);}
[data-testid="stMetric"] {background:linear-gradient(145deg,rgba(20,37,58,.9),rgba(12,24,40,.9)); padding:1rem 1.1rem; border:1px solid rgba(140,168,197,.16); border-radius:.9rem; box-shadow:0 12px 30px rgba(0,0,0,.16); min-height:7.2rem;}
[data-testid="stMetricLabel"] {color:#9AAAC0;}
[data-testid="stMetricValue"] {color:#F1F5FC; letter-spacing:-.035em;}
.stSelectbox [data-baseweb="select"] > div,.stTextInput [data-baseweb="input"] {background-color:rgba(8,15,29,.62); border-color:rgba(140,168,197,.2); border-radius:.7rem;}
.stButton > button[kind="primary"] {background:#20DCC3; color:#07151C; border:0; font-weight:650;}
.stButton > button[kind="primary"]:hover {background:#50E8D4; border:0; color:#07151C;}
.stButton > button {border-radius:.7rem; transition:background-color .16s ease,border-color .16s ease;}
[data-testid="stDataFrame"] {border:1px solid rgba(140,168,197,.15); border-radius:.8rem; overflow:hidden;}
@media(max-width:700px) {
 [data-testid="stMainBlockContainer"] {padding:1.2rem 1rem 1.2rem;}
 [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100% !important; width:100% !important;}
 h1 {font-size:1.5rem !important;}
}
</style>"""


def inject():
    """Inject global CSS rules."""
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "", kicker: str = "Industrial AI Operations"):
    """Render a clean industrial page header."""
    st.markdown(
        f'<div class="kicker">{escape(kicker)}</div>',
        unsafe_allow_html=True,
    )
    st.header(title)
    if subtitle:
        st.markdown(f'<p class="page-subtitle">{escape(subtitle)}</p>', unsafe_allow_html=True)


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
