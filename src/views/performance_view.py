"""Current-session measurements only; no historical comparison claims."""
import streamlit as st
from src import ui


def render_performance(*args):
    ui.header("Performance", "Measured inference activity for this session.", "VLM-Edge  /  Runtime analytics")
    st.caption("Recent history retains up to 100 successful requests. Latency excludes model loading, frame capture, decoding and queue wait.")
    history = st.session_state.get("request_history", [])
    errors = st.session_state.get("request_errors", 0)
    if not history and not errors:
        st.info("No requests measured yet. Ask a question in AI Assistant to collect measurements.")
        return
    if not history:
        completed, failures = st.columns(2)
        completed.metric("Completed requests", st.session_state.get("completed_requests", 0))
        failures.metric("Failed requests", errors)
        st.info("No completed inference measurements are available.")
        return
    last = history[-1]
    latencies = [float(row["Latency (ms)"]) for row in history]
    peaks = [float(row["process_peak_mib"]) for row in history if row.get("process_peak_mib") is not None]
    values = [
        ("Completed requests", str(st.session_state.get("completed_requests", len(history))), "Successful requests"),
        ("Failed requests", str(errors), "Unsuccessful submissions"),
        ("Latest latency", f"{last['Latency (ms)']:.1f} ms", "Most recent inference"),
        ("Average latency", f"{sum(latencies) / len(latencies):.1f} ms", f"Across {len(latencies)} measured requests"),
        ("Peak GPU memory", f"{max(peaks):.1f} MiB" if peaks else "Unavailable", "Observed process peak" if peaks else "No GPU sample recorded"),
    ]
    for start in range(0, len(values), 3):
        cols = st.columns(3, gap="small")
        for col, (label, value, help_text) in zip(cols, values[start:start + 3]):
            col.metric(label, value, help=help_text)
    if peaks:
        st.caption("GPU peak is PyTorch's process lifetime peak as sampled after requests, not a per-request peak.")
    if len(history) > 1:
        st.line_chart({"Inference latency (ms)": [r["Latency (ms)"] for r in history]})
    ui.section_title("Recent requests")
    st.dataframe([{k: v for k, v in row.items() if k in ("Request", "Camera", "Received UTC", "Latency (ms)")}
                  for row in reversed(history)], hide_index=True, width="stretch")
