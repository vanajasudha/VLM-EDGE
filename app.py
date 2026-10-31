"""One screen: live RTSP frames and on-demand local Moondream2 Q&A."""
import streamlit as st
from src.live_camera_ui import render_live_panel

st.set_page_config(page_title="VLM-Edge", layout="wide", initial_sidebar_state="collapsed")
st.title("VLM-Edge")
st.caption("Four live RTSP streams from looping sample videos · Moondream2 4-bit NF4 · AI runs only when asked")
render_live_panel()
