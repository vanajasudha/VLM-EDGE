"""Camera-only monitoring; refreshes never invoke the model."""
import streamlit as st
from src.live_camera_ui import manager_or_error, capture_controls, render_camera_grid


def render_live_monitoring():
    st.header("Live Monitoring")
    st.caption("Four RTSP sources | CPU capture | On-demand AI in the Assistant")
    manager = manager_or_error()
    if manager is not None:
        capture_controls(manager)
        render_camera_grid()
