"""VLM-Edge native multi-page entrypoint; no models load during navigation."""
import streamlit as st
from src import ui

st.set_page_config(page_title="VLM-Edge", page_icon=":material/videocam:",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject()
pages = [
    st.Page("src/pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("src/pages/monitoring.py", title="Live Monitoring", icon=":material/videocam:", url_path="monitoring"),
    st.Page("src/pages/assistant.py", title="AI Assistant", icon=":material/chat:", url_path="assistant"),
    st.Page("src/pages/performance.py", title="Performance", icon=":material/speed:", url_path="performance"),
    st.Page("src/pages/settings.py", title="Settings", icon=":material/settings:", url_path="settings"),
]
# Native page registration plus native links lets the brand precede the menu.
# Page URLs, keyboard activation and routing remain owned by Streamlit.
selected_page = st.navigation(pages, position="hidden")
ui.sidebar_brand()
with st.sidebar.container(key="application_navigation", gap="xsmall"):
    for index, page in enumerate(pages):
        active = selected_page.title == page.title
        with st.container(key="active_navigation" if active else f"navigation_{index}"):
            st.page_link(page, label=page.title, icon=page.icon, width="stretch")
selected_page.run()
