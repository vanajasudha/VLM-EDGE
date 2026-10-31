"""Mock-only live frame, dashboard and on-demand inference tests."""
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import Mock
import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import live_camera, live_camera_ui, model_inference

ROOT = Path(__file__).resolve().parent


class FakeManager:
    camera_ids = ("cam1", "cam2", "cam3", "cam4")
    stale_after_seconds = 3.0
    def __init__(self):
        self.valid = True
        self.frame = np.zeros((2, 3, 3), dtype=np.uint8)
        self.frame[:] = [3, 17, 29]  # OpenCV BGR, not RGB.
        self.sequence = 1
        self.started, self.stopped = [], []
    def snapshot(self, camera):
        return {"camera_id": camera, "frame": self.frame.copy(), "valid": self.valid,
                "status": "connected" if self.valid else "stale", "message": "Waiting for fresh frames",
                "received_monotonic": monotonic() - .02,
                "received_at_utc": "2026-10-29T12:00:00+00:00", "frame_sequence": self.sequence,
                "connection": 1, "source_kind": "demo_rtsp", "age_seconds": .02,
                "frames_received": self.sequence, "reconnects": 0, "last_error": None}
    def latest_valid(self, camera):
        result = self.snapshot(camera)
        if not result["valid"]: raise RuntimeError("No fresh frame")
        return result
    def start_many(self, cameras): self.started.extend(cameras)
    def stop_many(self, cameras): self.stopped.extend(cameras); self.valid = False


@pytest.fixture
def fake_model(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class Model:
        def encode_image(self, image):
            assert image.mode == "RGB" and image.getpixel((0, 0)) == (29, 17, 3)
            return "encoded"
        def answer_question(self, encoded, question, tokenizer):
            assert encoded == "encoded" and tokenizer == "tokenizer"
            return "Generated for " + question
    loader = Mock(return_value=(Model(), "tokenizer"))
    monkeypatch.setattr(model_inference, "load_moondream", loader)
    return loader


def test_latest_frame_is_selected_after_model_loading_and_keeps_exact_question(fake_model):
    manager = FakeManager()
    def load():
        manager.sequence = 99
        return fake_model.return_value
    fake_model.side_effect = load
    result = live_camera.ask_latest(manager, "cam3", " Exact question? ")
    assert result["answer"] == "Generated for  Exact question? "
    assert result["frame_metadata"]["camera_id"] == "cam3"
    assert result["frame_metadata"]["frame_sequence"] == 99
    assert result["frame_metadata"]["frame_age_at_inference_seconds"] < 3
    assert tuple(result["frame_metadata"]["submitted_rgb"][0, 0]) == (29, 17, 3)
    assert result["duration_ms"] >= 0
    fake_model.assert_called_once()


def test_disconnected_camera_does_not_load_model(fake_model):
    manager = FakeManager()
    manager.valid = False
    with pytest.raises(RuntimeError, match="fresh frame"):
        live_camera.ask_latest(manager, "cam1", "Question")
    fake_model.assert_not_called()


def test_disconnect_during_model_load_is_rechecked(fake_model):
    manager = FakeManager()
    def load():
        manager.valid = False
        return fake_model.return_value
    fake_model.side_effect = load
    with pytest.raises(model_inference.InferenceError, match="fresh frame"):
        live_camera.ask_latest(manager, "cam1", "Question")


def test_live_snapshot_is_acquired_after_waiting_for_shared_lock(fake_model):
    manager = FakeManager()
    entered = Event()
    answers = []
    model_inference._INFERENCE_LOCK.acquire()
    try:
        worker = Thread(target=lambda: (entered.set(), answers.append(
            live_camera.ask_latest(manager, "cam1", "Question"))))
        worker.start()
        assert entered.wait(1)
        sleep(.03)
        manager.sequence = 333
    finally:
        model_inference._INFERENCE_LOCK.release()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert answers[0]["frame_metadata"]["frame_sequence"] == 333


def test_stale_or_invalid_bgr_frame_cannot_reach_model(fake_model):
    manager = FakeManager()
    manager.frame = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(model_inference.InferenceError, match="valid BGR frame"):
        live_camera.ask_latest(manager, "cam1", "Question")
    assert not model_inference._INFERENCE_LOCK.locked()


@pytest.fixture
def dashboard(monkeypatch, fake_model):
    manager = FakeManager()
    monkeypatch.setattr(live_camera_ui, "get_camera_manager", lambda: manager)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    at.sidebar.radio[0].set_value("Cameras").run()
    assert not at.exception
    return at, manager, fake_model


def text(at):
    return "\n".join(str(item.value) for item in list(at.caption) + list(at.markdown))


def test_preview_never_calls_model_and_labels_demo(dashboard):
    at, manager, loader = dashboard
    at.run()
    loader.assert_not_called()
    assert "Demo RTSP publisher" in text(at)
    assert at.button(key="ask_live").disabled
    at.button(key="start_streams").click().run()
    assert manager.started == list(manager.camera_ids)


def test_live_submit_preview_answer_age_latency_and_no_automatic_inference(dashboard):
    at, manager, loader = dashboard
    at.selectbox(key="live_camera").set_value("cam2").run()
    at.text_input(key="live_question").set_value("Exact live question?").run()
    assert not at.button(key="ask_live").disabled
    at.button(key="ask_live").click().run()
    assert not at.exception
    assert "Generated for Exact live question?" in text(at)
    assert "Camera: cam2" in text(at) and "receipt age at inference" in text(at)
    assert "Inference latency:" in text(at)
    result = at.session_state["live_result"]
    assert result["frame_metadata"]["camera_id"] == "cam2"
    manager.sequence += 1
    at.run()
    assert at.session_state["live_result"] is not None
    loader.assert_called_once()


@pytest.mark.parametrize("change", ["camera", "question", "stale", "failure"])
def test_live_stale_result_clearing(dashboard, change):
    at, manager, loader = dashboard
    at.text_input(key="live_question").set_value("Original").run()
    at.button(key="ask_live").click().run()
    assert at.session_state["live_result"]
    if change == "camera": at.selectbox(key="live_camera").set_value("cam4").run()
    elif change == "question": at.text_input(key="live_question").set_value("Changed").run()
    elif change == "stale":
        manager.valid = False
        at.run()
    else:
        loader.side_effect = model_inference.InferenceError("CUDA out of memory")
        at.button(key="ask_live").click().run()
        assert "CUDA out of memory" in at.error[0].value
    assert not at.exception and at.session_state["live_result"] is None
    assert "Generated for Original" not in text(at)


def test_live_config_requires_four_streams_and_explicit_source_kind(tmp_path):
    path = ROOT / "config/cameras.json"
    config = json.loads(path.read_text())
    test_path = tmp_path / "cameras.json"
    config["cameras"]["cam1"]["source_kind"] = "invented"
    test_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source"):
        live_camera.read_camera_config(test_path)
    del config["cameras"]["cam4"]
    test_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exactly"):
        live_camera.read_camera_config(test_path)
