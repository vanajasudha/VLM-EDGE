"""Current-session measurements only; no historical comparison claims."""
import streamlit as st


def render_performance(*args):
    st.header("Performance")
    st.caption("Measurements from this browser session. Recent history retains up to 100 successful requests.")
    history = st.session_state.get("request_history", [])
    errors = st.session_state.get("request_errors", 0)
    if not history and not errors:
        st.info("No requests measured yet. Ask a question in AI Assistant to collect measurements.")
        return
    completed, failures = st.columns(2)
    completed.metric("Completed requests", st.session_state.get("completed_requests", 0))
    failures.metric("Failed requests", errors)
    if not history:
        st.info("No completed inference measurements are available.")
        return
    last = history[-1]
    st.metric("Last inference latency", f"{last['Latency (ms)']:.1f} ms")
    if "process_peak_mib" in last:
        st.metric("Process peak allocated GPU memory", f"{last['process_peak_mib']:.1f} MiB")
        st.caption("PyTorch process-lifetime peak sampled after the last request; not a per-request peak.")
    st.caption("Inference latency excludes model loading, frame capture, decoding and queue wait.")
    if len(history) > 1:
        st.line_chart({"Inference latency (ms)": [r["Latency (ms)"] for r in history]})
    st.dataframe([{k: v for k, v in row.items() if k in ("Request", "Camera", "Received UTC", "Latency (ms)")}
                  for row in history], hide_index=True, width="stretch")
