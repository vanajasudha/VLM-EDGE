"""Compact fleet landing page."""
import streamlit as st
from src.live_camera_ui import render_camera_grid



def render_overview(*args):
    st.header("Overview")
    if st.button("Open AI Assistant", key="open_assistant", type="primary"):
        st.switch_page("src/pages/assistant.py")
    render_camera_grid(compact=True)
