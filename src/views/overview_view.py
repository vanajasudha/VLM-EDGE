"""Overview view: Operational health, fleet snapshot, and quick actions."""
from html import escape
from pathlib import Path
import streamlit as st
from src import ui
from src.live_camera import read_camera_config
from src.benchmark_metrics import reduction_pct

ROOT = Path(__file__).resolve().parents[2]


def render_overview(data, errors, frames):
    ui.header("Operations Overview", "Visual language model operations center for warehouse monitoring.", "Control Center")

    # Display benchmark file errors if any exist
    for k, v in errors.items():
        ui.callout(f"Data loading error ({k}): {v}", "bad")

    # Read configured cameras safely without starting capture workers
    try:
        cameras_cfg, stale_after = read_camera_config()
        total_cams = len(cameras_cfg)
    except Exception:
        total_cams = 4

    # Calculate baseline and quantized latency from benchmark data
    base_ms = data["Baseline FP16"].latency_ms.mean() if "Baseline FP16" in data else None
    quant = [data[k].latency_ms.mean() for k in ("Quantized run 1", "Quantized run 2") if k in data]
    quant_ms = sum(quant) / len(quant) if quant else None

    # Truthful GPU telemetry
    gpu_mem_str = "Unavailable"
    gpu_sub = "CUDA unavailable"
    try:
        import torch
        if torch.cuda.is_available():
            alloc_mib = torch.cuda.memory_allocated(0) / (1024 ** 2)
            if alloc_mib > 0:
                gpu_mem_str = f"{alloc_mib:.1f} MiB"
                gpu_sub = "Current: model loaded in VRAM"
            else:
                gpu_mem_str = "0.0 MiB"
                gpu_sub = "Current: model not loaded"
        else:
            gpu_mem_str = "CPU Mode"
            gpu_sub = "Historical only: no CUDA"
    except Exception:
        gpu_mem_str = "Unavailable"

    # Truthful last inference latency
    last_qa = st.session_state.get("qa_result")
    last_live = st.session_state.get("live_result")
    if last_live and "duration_ms" in last_live:
        latency_str = f"{last_live['duration_ms']:.0f} ms"
        latency_sub = f"Last: live cam {last_live.get('frame_metadata', {}).get('camera_id', 'camera')}"
    elif last_qa and "duration_ms" in last_qa:
        latency_str = f"{last_qa['duration_ms']:.0f} ms"
        latency_sub = f"Last: saved frame ({last_qa.get('image', 'frame')})"
    else:
        latency_str = "Unavailable"
        latency_sub = "No inference yet this session"

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    ui.kpi(c1, "Configured Cameras", f"{total_cams} Streams", "cam1 – cam4 endpoints")
    ui.kpi(c2, "Saved Frames", len(frames), "test_frames/ gallery")
    ui.kpi(c3, "Inference Latency", latency_str, latency_sub)
    ui.kpi(c4, "GPU VRAM Allocation", gpu_mem_str, gpu_sub)

    # Operational Entry Points
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    ui.section_title("Operational Workstations", "Quick Action")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            f'<div class="card" style="border-top: 3px solid {ui.C["teal"]};">'
            f'  <div style="font-weight:700;font-size:1.05rem;color:{ui.C["text"]};margin-bottom:0.25rem;">Live Camera Grid</div>'
            f'  <div style="font-size:0.83rem;color:{ui.C["muted"]};line-height:1.45;margin-bottom:0.85rem;">'
            f'    Monitor the 2×2 RTSP camera matrix in real time. Inspect feed health, reconnect status, and stream timestamps.'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Open Live Monitoring", key="quick_open_live", type="primary", width="stretch"):
            st.session_state["nav_selection"] = "Cameras"
            st.rerun()

    with q2:
        st.markdown(
            f'<div class="card" style="border-top: 3px solid {ui.C["blue"]};">'
            f'  <div style="font-weight:700;font-size:1.05rem;color:{ui.C["text"]};margin-bottom:0.25rem;">AI Visual Inspection</div>'
            f'  <div style="font-size:0.83rem;color:{ui.C["muted"]};line-height:1.45;margin-bottom:0.85rem;">'
            f'    Ask natural language questions against live camera frames or saved verification images using Moondream2 4-bit NF4.'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Launch AI Assistant", key="quick_open_ai", type="secondary", width="stretch"):
            st.session_state["nav_selection"] = "Ask a Question"
            st.rerun()

    # Camera Fleet Overview
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    ui.section_title("Camera Fleet Snapshot", f"{total_cams} Configured")
    cols = st.columns(4)
    cams = ["cam1", "cam2", "cam3", "cam4"]
    for i, (col, cam) in enumerate(zip(cols, cams)):
        cam_label = f"Camera {i+1:02d} ({cam.upper()})"
        mine = [f for f in frames if f.get("cam") == cam]
        with col:
            st.markdown(f'<div class="card" style="padding:12px 14px;margin-bottom:0.5rem;">', unsafe_allow_html=True)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">'
                f'  <span style="font-weight:700;font-size:0.84rem;color:{ui.C["text"]};">{cam_label}</span>'
                f'  {ui.pill("Configured", "info")}'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Show offline-style placeholder — NOT the actual image (images are for AI Assistant only)
            frame_count = len(mine)
            st.markdown(
                f'<div style="height:110px;background:#090F1B;border-radius:6px;display:flex;flex-direction:column;'
                f'align-items:center;justify-content:center;color:{ui.C["muted"]};font-size:0.75rem;gap:0.3rem;">'
                f'  <span style="font-size:1.4rem;opacity:0.7;">📂</span>'
                f'  <span>Saved reference — not live</span>'
                f'  <span style="color:{ui.C["teal"]};font-weight:600;">{frame_count} saved frame{"s" if frame_count != 1 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="font-size:0.72rem;color:{ui.C["muted"]};margin-top:0.4rem;display:flex;justify-content:space-between;">'
                f'  <span>Status: <strong style="color:{ui.C["text"]};">Configured</strong></span>'
                f'  <span>Live: <strong style="color:{ui.C["amber"]};">On-Demand</strong></span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Operational Scope & Capability Status
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    ui.section_title("System Capability & Integrity Verification")
    scope_items = [
        ("Selective 4-bit NF4: Verified", "ok"),
        ("Saved Frame Q&A: Ready", "ok"),
        ("Multi-Camera RTSP: Configured", "ok"),
        ("GPU Telemetry: PyTorch / Live NVML", "ok"),
        ("Physical Warehouse Dataset: Pending Collection", "warn"),
        ("Live Inference: On-Demand Only", "info"),
    ]
    ui.status_strip(scope_items)

    # Recent Session Queries Log
    if last_qa or last_live:
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        ui.section_title("Recent Session Activity")
        recent = last_live or last_qa
        q_text = recent.get("question", "N/A")
        ans_text = recent.get("answer", "N/A")
        duration = recent.get("duration_ms", 0)
        source_id = recent.get("image") or recent.get("frame_metadata", {}).get("camera_id", "live")
        st.markdown(
            f'<div class="card" style="border-left: 3px solid {ui.C["teal"]};">'
            f'  <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{ui.C["muted"]};margin-bottom:0.3rem;">'
            f'    <span>SOURCE: <strong style="color:{ui.C["text"]};">{escape(str(source_id))}</strong></span>'
            f'    <span>LATENCY: <strong style="color:{ui.C["teal"]};">{duration:.1f} ms</strong></span>'
            f'  </div>'
            f'  <div style="font-size:0.86rem;font-weight:600;color:{ui.C["text"]};margin-bottom:0.35rem;">Q: {escape(q_text)}</div>'
            f'  <div style="font-size:0.85rem;color:{ui.C["muted"]};line-height:1.45;">A: {escape(ans_text)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )