"""Camera-only monitoring; refreshes never invoke the model."""
import streamlit as st
from src import ui
from src.live_camera_ui import manager_or_error, capture_controls, render_camera_grid


def render_live_monitoring():
    ui.header("Live Monitoring", "Inspect camera feeds and manage the CPU capture workers.", "VLM-Edge  /  Camera network")
    st.caption("Four configured RTSP sources · CPU frame capture · Inference remains on demand in AI Assistant")
    manager = manager_or_error()
    if manager is not None:
        capture_controls(manager)
        render_camera_grid()
