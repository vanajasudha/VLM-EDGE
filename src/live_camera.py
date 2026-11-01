"""Process-cached capture and on-demand latest-frame Q&A; no disk snapshots."""
import json
import math
import os
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse

import streamlit as st

from src.camera_capture import CameraCaptureManager
from src import model_inference

ROOT = Path(__file__).resolve().parents[1]


def read_camera_config(path=None):
    path = Path(path or os.environ.get("VLM_EDGE_CAMERAS_CONFIG", ROOT / "config/cameras.json"))
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Camera configuration must be a JSON object")
    cameras = data.get("cameras", {})
    if not isinstance(cameras, dict) or set(cameras) != {"cam1", "cam2", "cam3", "cam4"}:
        raise ValueError("Configure exactly cam1, cam2, cam3, cam4")
    stale = data.get("stale_after_seconds", 3.0)
    if isinstance(stale, bool) or not isinstance(stale, (int, float)) or not math.isfinite(stale) or stale <= 0:
        raise ValueError("stale_after_seconds must be positive and finite")
    for key, entry in cameras.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise ValueError(f"{key}: configure a valid RTSP URL")
        try:
            url = urlparse(entry["url"])
            if url.scheme not in {"rtsp", "rtsps"} or not url.hostname:
                raise ValueError("invalid endpoint")
            if url.port is not None and not 1 <= url.port <= 65535:
                raise ValueError("invalid port")
        except ValueError:
            raise ValueError(f"{key}: configure a valid RTSP URL and port") from None
        if entry.get("source_kind") not in {"demo_rtsp", "live_camera", "unverified_rtsp"}:
            raise ValueError(f"{key}: source_kind must be demo_rtsp, live_camera or unverified_rtsp")
    return cameras, stale


@st.cache_resource
def get_camera_manager():
    cameras, stale = read_camera_config()
    return CameraCaptureManager({key: c["url"] for key, c in cameras.items()},
                                source_kinds={key: c["source_kind"] for key, c in cameras.items()},
                                stale_after_seconds=stale)


def ask_latest(manager, camera_id, question):
    """Load once, wait for the inference slot, THEN fetch a fresh frame."""
    model_inference.validate_question(question)
    manager.latest_valid(camera_id)  # Avoid loading a model when there is no stream.
    resource = model_inference.load_moondream()

    def provide_image():
        import cv2
        from PIL import Image
        snapshot = manager.latest_valid(camera_id)
        frame = snapshot["frame"]
        if getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3:
            raise model_inference.InferenceError("Camera did not return a valid BGR frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        age = max(0.0, monotonic() - snapshot["received_monotonic"])
        if age > manager.stale_after_seconds:
            image.close()
            raise model_inference.InferenceError("Camera frame became stale before inference")
        metadata = {k: snapshot[k] for k in ("camera_id", "frame_sequence", "connection",
                    "received_at_utc", "received_monotonic", "source_kind")}
        metadata["frame_age_at_inference_seconds"] = age
        # Freeze the actual image used, not a newer preview.
        metadata["submitted_rgb"] = rgb.copy()
        return image, metadata

    return model_inference.infer_from_provider(resource, provide_image, question)
