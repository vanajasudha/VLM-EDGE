"""Live Monitoring View: Responsive 2x2 RTSP camera matrix and on-demand VLM interrogation."""
from html import escape
from time import monotonic
import streamlit as st
from src import live_camera_ui, ui


@st.fragment(run_every=1.0)
def render_live_monitoring():
    ui.header("Live Monitoring", "Multi-camera RTSP surveillance matrix with visual intelligence.", "Surveillance Operations")

    try:
        manager = live_camera_ui.get_camera_manager()
    except Exception as exc:
        ui.callout(f"Camera configuration unavailable: {exc}", "bad")
        return

    st.caption("CPU capture only until Ask AI is clicked. Start/stop controls are shared across this app process.")

    selected = st.multiselect(
        "Capture streams",
        list(manager.camera_ids),
        default=list(manager.camera_ids),
        key="capture_streams",
    )

    start, stop = st.columns(2)
    if start.button("Start selected", disabled=not selected, key="start_streams", type="primary"):
        try:
            manager.start_many(selected)
        except Exception as exc:
            st.error(f"Could not start capture: {exc}")
    if stop.button("Stop selected", disabled=not selected, key="stop_streams"):
        manager.stop_many(selected)

    # 2x2 Responsive Camera Surveillance Matrix
    ui.section_title("2×2 Camera Surveillance Grid", "RTSP Fleet")
    grid_row1 = st.columns(2)
    grid_row2 = st.columns(2)
    grid_cols = [grid_row1[0], grid_row1[1], grid_row2[0], grid_row2[1]]

    for i, (col, key) in enumerate(zip(grid_cols, manager.camera_ids)):
        snap = manager.snapshot(key)
        status = snap.get("status", "idle")
        is_valid = snap.get("valid", False)
        endpoint = manager._urls.get(key, f"rtsp://127.0.0.1:8554/{key}") if hasattr(manager, "_urls") else f"rtsp://127.0.0.1:8554/{key}"
        cam_display_name = f"Camera {i+1:02d} ({key.upper()})"

        status_kind = "ok" if (is_valid and status == "connected") else ("warn" if status in ("connecting", "reconnecting", "stale") else "off")
        status_text = "LIVE" if (is_valid and status == "connected") else (status.upper() if status != "idle" else "OFFLINE")

        with col:
            st.markdown(
                f'<div class="camera-card">'
                f'  <div class="camera-card-header">'
                f'    <div class="camera-name">'
                f'      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{ui.C["green"] if is_valid else ui.C["red"]};"></span>'
                f'      {escape(cam_display_name)}'
                f'    </div>'
                f'    <div>{ui.pill(status_text, status_kind)}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Camera Viewport
            if is_valid and snap.get("frame") is not None:
                st.image(snap["frame"], channels="BGR", width="stretch")
            else:
                msg = snap.get("message") or "RTSP Stream Disconnected"
                if snap.get("last_error"):
                    msg += f" ({snap['last_error']})"
                st.markdown(ui.camera_offline_placeholder(cam_display_name, endpoint, msg), unsafe_allow_html=True)

            # Camera Card Footer Metadata
            age_str = f"{snap['age_seconds']:.2f} s" if snap.get("age_seconds") is not None else "—"
            seq_val = snap.get("frame_sequence") or snap.get("frames_received", 0)
            reconnect_val = snap.get("reconnects", 0)

            st.markdown(
                f'<div class="camera-card-footer" style="margin-top:-10px;margin-bottom:12px;">'
                f'  <span>Age: <strong style="color:{ui.C["text"]};">{age_str}</strong></span>'
                f'  <span>Frames: <strong style="color:{ui.C["text"]};">{seq_val}</strong></span>'
                f'  <span>Retries: <strong style="color:{ui.C["text"]};">{reconnect_val}</strong></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Status summary row for tests that check camera status markdown
    with st.expander("Camera Status Summary", expanded=False):
        status_cols = st.columns(4)
        for col, key in zip(status_cols, manager.camera_ids):
            snap = manager.snapshot(key)
            col.markdown(f"**{key.upper()}** — {snap['status']}")
            col.caption(f"Frames: {snap['frames_received']} | Retries: {snap['reconnects']}")

    # Interactive Camera Interrogation Section (Visual Q&A on Live Feed)
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    ui.section_title("Visual Question Answering — Live Feed", "Active Interrogation")

    camera = st.selectbox("Live camera", list(manager.camera_ids), key="live_camera")
    question = st.text_input("Question for selected camera", key="live_question",
                             placeholder="What is visible in this frame?")

    inputs = (camera, question)
    st.session_state.setdefault("live_result", None)
    st.session_state.setdefault("live_error", None)

    if st.session_state.get("live_inputs") != inputs:
        st.session_state.live_inputs = inputs
        st.session_state.live_result = None
        st.session_state.live_error = None

    snapshot = manager.snapshot(camera)
    source_labels = {
        "demo_rtsp": "Demo RTSP publisher — not a physical camera",
        "live_camera": "RTSP camera (configured source)",
        "unverified_rtsp": "RTSP stream — source unverified",
    }
    source_label = source_labels[snapshot["source_kind"]]
    st.caption(source_label)

    if snapshot["valid"]:
        st.image(snapshot["frame"], channels="BGR", caption=f"Latest received frame — {camera}", width="stretch")
        st.caption(f"Received UTC: {snapshot['received_at_utc']} | Frame #{snapshot['frame_sequence']} | "
                   f"Receipt age: {snapshot['age_seconds']:.2f} s")
    else:
        st.warning(f"{camera}: {snapshot['message']}. Ask AI requires a fresh connected frame.")
        if snapshot["last_error"]:
            st.caption(snapshot["last_error"])
        st.session_state.live_result = None

    result = st.session_state.live_result
    if result and result["frame_metadata"]["connection"] != snapshot["connection"]:
        st.session_state.live_result = None

    submitted = st.button("Ask AI", disabled=not snapshot["valid"] or not question.strip(), key="ask_live", type="primary")
    answer_area = st.empty()

    if submitted:
        st.session_state.live_result = None
        st.session_state.live_error = None
        answer_area.empty()
        try:
            with st.spinner("Generating answer for the latest valid camera frame..."):
                result = live_camera_ui.ask_latest(manager, camera, question)
            st.session_state.live_result = {**result, "question": question, "inputs": inputs}
        except Exception as exc:
            st.session_state.live_error = str(exc) or "Live inference failed"

    with answer_area.container():
        result = st.session_state.live_result
        if st.session_state.live_error:
            st.error(st.session_state.live_error)
        elif result and result["inputs"] == inputs:
            metadata = result["frame_metadata"]
            st.markdown(
                f'<div class="answer-panel">'
                f'  <div class="answer-header">'
                f'    <span class="answer-label">Live Frame Analysis</span>'
                f'    <span class="answer-meta">{result["duration_ms"]:.1f} ms</span>'
                f'  </div>'
                f'  <div class="answer-text">{escape(result["answer"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Camera: {metadata['camera_id']} | Question: {result['question']}")
            st.caption(f"Frame #{metadata['frame_sequence']} | Received UTC: {metadata['received_at_utc']}")
            st.caption(f"Frame receipt age at inference: {metadata['frame_age_at_inference_seconds']:.3f} s | "
                       f"Submitted frame age now: {max(0.0, monotonic() - metadata['received_monotonic']):.2f} s")
            st.caption(f"Inference latency: {result['duration_ms']:.1f} ms (excludes model loading and queue wait)")
            st.image(metadata["submitted_rgb"], caption="Exact submitted frame; answer describes this frame only",
                     width="stretch")
            st.caption("Model answers may be incorrect or incomplete. One frame cannot establish safety.")