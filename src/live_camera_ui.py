"""Shared camera presentation; capture/model resources remain backend-owned."""
from html import escape
import re
import sys
from time import monotonic
from src.camera_diagnostics import safe_rtsp_url

import streamlit as st
from PIL import Image, ImageOps
from src import ui
from src.live_camera import get_camera_manager, ask_latest, read_camera_config

SOURCE_LABELS = {
    "demo_rtsp": "Demo RTSP publisher | not a physical camera",
    "live_camera": "Configured RTSP camera",
    "unverified_rtsp": "RTSP stream | source unverified",
}


def safe_endpoint(url):
    """Never render userinfo, query strings or fragments."""
    return safe_rtsp_url(url)


def safe_error(exc):
    """Keep actionable failures while omitting arbitrary backend URLs/paths."""
    message = str(exc)
    if "out of memory" in message.lower():
        return "CUDA out of memory. Close other GPU workloads and retry."
    if "local cache" in message.lower() or "cached" in message.lower():
        return "Model cache unavailable. Check the local model files and environment."
    message = re.sub(r"(?:rtsps?|https?)://\S+", "[endpoint hidden]", message)
    message = re.sub(r"(?:[A-Za-z]:[\\/]|/)(?:[^\s]+)", "[local path hidden]", message)
    return message[:300] or "Request failed. Check the local runtime."


def manager_or_error():
    try:
        return get_camera_manager()
    except Exception:
        st.error("Camera configuration unavailable. Check the local camera configuration.")
        return None


def gpu_measurements():
    """Read an already initialized CUDA runtime; never initialize it for the UI."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return {"device": torch.cuda.get_device_name(0),
                    "allocated_mib": torch.cuda.memory_allocated(0) / 2**20,
                    "process_peak_mib": torch.cuda.max_memory_allocated(0) / 2**20}
    except Exception:
        pass
    return {}


def connection_label(snap):
    if snap.get("valid") and snap.get("status") == "connected":
        return "Online", "ok"
    return {"connecting": ("Connecting", "warn"),
            "reconnecting": ("Reconnecting", "warn"),
            "stale": ("Stale frame", "warn"),
            "stopping": ("Stopping", "warn"),
            "idle": ("Capture not started", "off"),
            "stopped": ("Capture stopped", "off")}.get(snap.get("status"), ("Offline", "off"))


def preview(snap, caption=None):
    if snap.get("valid") and snap.get("frame") is not None:
        # Letterbox consistently, without stretching or cropping visual evidence.
        image = Image.fromarray(snap["frame"][:, :, ::-1])
        image = ImageOps.pad(image, (960, 540), color="#0B1524")
        st.image(image, caption=caption, width="stretch")
    else:
        st.markdown('<div class="camera-placeholder"><span>◉</span><span>No recent frame available</span></div>',
                    unsafe_allow_html=True)


def camera_hint(snap):
    if snap.get("valid"):
        age = snap.get("age_seconds")
        return f"Frame #{snap.get('frame_sequence')} | Receipt age: {age:.2f} s" if age is not None else "Receiving decoded frames"
    if snap.get("status") in {"idle", "stopped"}:
        return "Start capture in Live Monitoring."
    if snap.get("status") == "stopping":
        return "Waiting for the decoder to release resources. Inference is blocked."
    if snap.get("status") == "stale":
        return "No recent frames; check the publisher. Inference is blocked."
    return {"server_unavailable": "MediaMTX unreachable. Start the server and publishers.",
            "decoder_unavailable": "OpenCV missing from the running Python environment.",
            "endpoint_unavailable": "Check the FFmpeg publisher, RTSP path and video codec.",
            "frame_read_failed": "Frames stopped. Capture will reconnect automatically."}.get(
                snap.get("failure_kind"), "Connecting/retrying. Check the server and FFmpeg publisher.")


@st.fragment(run_every=1.0)
def render_camera_grid(compact=False):
    manager = manager_or_error()
    if manager is None:
        return
    snapshots = [manager.snapshot(key) for key in manager.camera_ids]
    online = sum(connection_label(s)[0] == "Online" for s in snapshots)
    active = sum(s.get("status") not in {"idle", "stopped"} for s in snapshots)
    st.markdown(f'<div class="card" style="display:flex;justify-content:space-between;gap:1rem;align-items:center;margin:.3rem 0 1rem;">'
                f'<span><span style="color:#20DCC3;font-weight:700">{online} of {len(snapshots)} cameras connected</span></span>'
                f'<span>{active} capture workers active</span><span class="sub">Live refresh · 1 s</span></div>',
                unsafe_allow_html=True)
    ui.section_title("Camera network", "4 SOURCES" if len(snapshots) == 4 else f"{len(snapshots)} SOURCES")
    with st.container(key="camera_grid_compact" if compact else "camera_grid", gap="medium"):
        for offset in range(0, len(snapshots), 2):
            for col, snap in zip(st.columns(2, gap="small"), snapshots[offset:offset + 2]):
                with col, st.container(border=True, gap="small"):
                    label, kind = connection_label(snap)
                    st.markdown(f"**{escape(snap['camera_id'].upper())}** &nbsp; " + ui.pill(label, kind),
                                unsafe_allow_html=True)
                    preview(snap)
                    st.caption(camera_hint(snap))


def capture_controls(manager):
    with st.container(horizontal=True, gap="small"):
        if st.button("Start capture", key="start_streams", type="primary"):
            try:
                manager.start_many(manager.camera_ids)
            except Exception as exc:
                st.error(safe_error(exc))
        if st.button("Stop capture", key="stop_streams"):
            try:
                manager.stop_many(manager.camera_ids)
            except Exception as exc:
                st.error(safe_error(exc))
    st.caption("Capture reads existing RTSP feeds. Start MediaMTX and the four publishers first.")


def _record_request(result=None):
    if result is None:
        st.session_state["request_errors"] = st.session_state.get("request_errors", 0) + 1
        return
    st.session_state["completed_requests"] = st.session_state.get("completed_requests", 0) + 1
    history = st.session_state.get("request_history", [])
    metadata = result["frame_metadata"]
    history.append({"Request": st.session_state["completed_requests"],
                    "Camera": metadata["camera_id"], "Received UTC": metadata["received_at_utc"],
                    "Latency (ms)": result["duration_ms"],
                    **result.get("gpu", {})})
    st.session_state["request_history"] = history[-100:]


def remember_camera():
    st.session_state["assistant_camera"] = st.session_state["live_camera"]


def remember_question():
    st.session_state["assistant_question"] = st.session_state["live_question"]


@st.fragment(run_every=1.0)
def render_assistant():
    ui.header("AI Assistant", "Ask a question about a fresh frame from any connected camera.", "VLM-Edge  /  Visual query")
    manager = manager_or_error()
    if manager is None:
        return
    cameras = list(manager.camera_ids)
    previous_camera = st.session_state.get("assistant_camera")
    camera = st.selectbox("Camera", cameras, key="live_camera",
                          index=cameras.index(previous_camera) if previous_camera in cameras else 0,
                          on_change=remember_camera)
    snap = manager.snapshot(camera)
    evidence, query = st.columns([1.15, 1], gap="large")
    with evidence:
        preview(snap, caption=f"Recent preview | {camera}")
        st.caption(SOURCE_LABELS.get(snap.get("source_kind"), SOURCE_LABELS["unverified_rtsp"]))
        label, _ = connection_label(snap)
        st.caption(f"{label} | Received UTC: {snap.get('received_at_utc') or 'Unavailable'}")
        if not snap.get("valid"):
            st.warning(camera_hint(snap))
    with query:
        ui.section_title("Ask about this scene")
        question = st.text_input("Question", key="live_question",
                                 value=st.session_state.get("assistant_question", ""), on_change=remember_question, placeholder="What is visible in this frame?", help="The exact question is sent with a fresh frame from this camera.")
        inputs = (camera, question)
        st.session_state.setdefault("live_result", None)
        st.session_state.setdefault("live_error", None)
        if st.session_state.get("live_inputs") != inputs:
            st.session_state["live_inputs"] = inputs
            st.session_state["live_result"] = None
            st.session_state["live_error"] = None
        result = st.session_state["live_result"]
        if not snap.get("valid") or (result and result["frame_metadata"]["connection"] != snap.get("connection")):
            st.session_state["live_result"] = None
        if st.button("Ask AI", key="ask_live", type="primary", width="stretch",
                     disabled=not snap.get("valid") or not question.strip()):
            st.session_state["live_result"] = None
            st.session_state["live_error"] = None
            try:
                with st.spinner("Reading the latest frame and generating an answer..."):
                    result = ask_latest(manager, camera, question)
                result = {**result, "question": question, "inputs": inputs, "gpu": gpu_measurements()}
                st.session_state["live_result"] = result
                _record_request(result)
            except Exception as exc:
                st.session_state["live_error"] = safe_error(exc)
                _record_request()
        if st.session_state["live_error"]:
            st.error(st.session_state["live_error"])
        result = st.session_state["live_result"]
        if result and result["inputs"] == inputs:
            m = result["frame_metadata"]
            st.markdown('<div class="answer-panel"><div class="answer-label">Generated answer</div>'
                        f'<div class="answer-text">{escape(result["answer"])}</div></div>', unsafe_allow_html=True)
            st.caption(f"Camera: {m['camera_id']} | Question: {result['question']}")
            st.caption(f"Received UTC: {m['received_at_utc']} | Frame #{m['frame_sequence']} | "
                       f"Frame receipt age at inference: {m['frame_age_at_inference_seconds']:.3f} s")
            st.caption(f"Inference latency: {result['duration_ms']:.1f} ms | Excludes model loading and queue wait")
            if result["gpu"]:
                st.caption(f"PyTorch allocated: {result['gpu']['allocated_mib']:.1f} MiB | "
                           f"Process peak allocated: {result['gpu']['process_peak_mib']:.1f} MiB")
            with st.expander("Exact submitted frame"):
                st.image(m["submitted_rgb"], caption="This answer describes this frame only", width="stretch")
            st.caption(f"Submitted frame age now: {max(0, monotonic() - m['received_monotonic']):.1f} s")
            st.caption("Answers may be incorrect or incomplete. A single frame cannot establish safety.")



# Retained entrypoint for callers of the previously missing single-page UI.
def render_live_panel():
    manager = manager_or_error()
    if manager is not None:
        capture_controls(manager)
        render_camera_grid()
        render_assistant()
