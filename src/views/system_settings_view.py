"""Read-only effective configuration, without credentials or filesystem paths."""
from html import escape
import os
import streamlit as st
from src import model_inference
from src.live_camera_ui import manager_or_error, read_camera_config, safe_endpoint, connection_label, gpu_measurements


def render_system_settings():
    st.header("Settings")
    st.caption("Read-only runtime information. Edit local configuration outside the dashboard.")
    manager = manager_or_error()
    st.subheader("Camera sources")
    try:
        cameras, stale = read_camera_config()
        rows = []
        for key, config in cameras.items():
            status = connection_label(manager.snapshot(key))[0] if manager else "Unavailable"
            rows.append({"Camera": key, "RTSP source": safe_endpoint(config["url"]),
                         "Source type": config["source_kind"], "Status": status})
        for row in rows:
            st.markdown('<div class="camera-source-row">'
                        f"<strong>{escape(row['Camera'].upper())}</strong>"
                        f"<span>{escape(row['RTSP source'])}</span>"
                        f"<span class=\"source-status\">{escape(row['Status'])}</span></div>",
                        unsafe_allow_html=True)
        st.caption(f"Freshness threshold: {stale:g} seconds | Credentials and URL query strings are hidden.")
    except Exception:
        st.warning("Camera configuration could not be read. Check the local configuration.")
    st.subheader("Model")
    st.markdown(f"**Model:** `{model_inference.MODEL_ID}`")
    st.markdown("**Quantization:** selective 4-bit NF4 | FP16 quantized compute | BF16 model dtype")
    st.markdown("**Excluded modules:** vision, region | **Device placement:** auto")
    st.text("Model revision: " + (os.environ.get("VLM_EDGE_MODEL_REVISION") or "Cached main; not explicitly pinned"))
    st.caption("Cached local files only. Model resources are process-cached; inference is serialized.")
    st.subheader("GPU runtime")
    gpu = gpu_measurements()
    if gpu:
        st.text(gpu["device"])
        st.caption(f"Allocated: {gpu['allocated_mib']:.1f} MiB | Process peak: {gpu['process_peak_mib']:.1f} MiB")
    else:
        st.info("GPU measurements unavailable. This page does not initialize CUDA or load the model.")
    st.caption(f"Streamlit {st.__version__} | Preview refresh: 1 second | Inference lock: "
               + ("Busy" if model_inference._INFERENCE_LOCK.locked() else "Available"))
