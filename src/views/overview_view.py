"""Compact fleet landing page."""
import streamlit as st
from src import ui
from src.live_camera_ui import render_camera_grid



def render_overview(*args):
    ui.header("Overview", "A live view of your visual intelligence network.", "VLM-Edge  /  Operations")
    left, right = st.columns([3, 1])
    with left:
        st.caption("Four RTSP sources · CPU frame capture · On-demand Moondream2 inference")
    with right:
        if st.button("Ask AI  →", key="open_assistant", type="primary", width="stretch"):
            st.switch_page("src/pages/assistant.py")
    render_camera_grid(compact=True)
