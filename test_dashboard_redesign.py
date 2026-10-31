"""Comprehensive UI & functional verification tests for the redesigned VLM-Edge dashboard."""
from html import unescape
from pathlib import Path
from unittest.mock import Mock
import pytest
from streamlit.testing.v1 import AppTest
from src import live_camera_ui, model_inference
from test_live_camera import FakeManager

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def app(monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(live_camera_ui, "get_camera_manager", lambda: manager)
    loader = Mock(return_value=("mock_model", "mock_tokenizer"))
    infer_mock = Mock(return_value={"answer": "Industrial inspection clear. No hazards detected.", "duration_ms": 845.2})
    monkeypatch.setattr(model_inference, "load_moondream", loader)
    monkeypatch.setattr(model_inference, "infer", infer_mock)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=25).run()
    assert not at.exception
    return at, manager, loader, infer_mock


def rendered_text(at):
    items = list(at.markdown) + list(at.caption) + list(at.title) + list(at.header)
    raw = "\n".join(str(item.value) for item in items)
    for df in at.dataframe:
        raw += "\n" + df.value.to_string()
    return unescape(raw)


def test_overview_page_renders_truthful_metrics_and_fleet(app):
    at, _, loader, infer_mock = app
    at.sidebar.radio[0].set_value("Overview").run()
    assert not at.exception
    loader.assert_not_called()
    infer_mock.assert_not_called()

    text = rendered_text(at)
    assert "Operations Overview" in text
    assert "Configured Cameras" in text
    assert "cam1" in text and "cam4" in text
    assert "System Capability & Integrity Verification" in text


def test_live_monitoring_page_2x2_grid_and_controls(app):
    at, manager, _, _ = app
    at.sidebar.radio[0].set_value("Cameras").run()
    assert not at.exception

    text = rendered_text(at)
    assert "Live Monitoring" in text
    assert "2×2 Camera Surveillance Grid" in text
    assert "Demo RTSP publisher" in text
    assert at.button(key="start_streams")
    assert at.button(key="stop_streams")
    assert at.button(key="ask_live")


def test_ai_assistant_page_workspace_interaction(app):
    at, _, loader, infer_mock = app
    at.sidebar.radio[0].set_value("Ask a Question").run()
    assert not at.exception

    text = rendered_text(at)
    assert "AI Assistant" in text
    assert "Visual Evidence Source" in text
    assert "AI Inspection Query" in text

    # Verify Ask button is at.button[0] and starts disabled when question is empty
    assert at.button[0].disabled

    at.selectbox(key="qa_image").set_value("cam1_0").run()
    at.text_input(key="qa_question").set_value("Is any forklift in motion?").run()
    assert not at.button[0].disabled

    at.button[0].click().run()
    assert not at.exception
    loader.assert_called_once()
    infer_mock.assert_called_once()

    text_after = rendered_text(at)
    assert "Industrial inspection clear. No hazards detected." in text_after
    assert "845.2 ms" in text_after


def test_performance_analytics_page(app):
    at, _, _, _ = app
    at.sidebar.radio[0].set_value("Performance").run()
    assert not at.exception

    text = rendered_text(at)
    assert "Performance Analytics" in text
    assert "FP16 Baseline Latency" in text
    assert "Quantized NF4 Latency" in text
    assert "Per-Frame Latency Distribution" in text
    assert "Measurement Incompatibility Notice" in text
    assert "Decimal MB" in text or "MiB" in text


def test_evaluation_page(app):
    at, _, _, _ = app
    at.sidebar.radio[0].set_value("Evaluation").run()
    assert not at.exception

    text = rendered_text(at)
    assert "Evaluation & Verification" in text
    assert "Quantitative Verification" in text
    assert "Dataset Manifest Validator" in text
    assert "Qualitative Description Review Protocol" in text


def test_system_settings_page(app):
    at, _, _, _ = app
    at.sidebar.radio[0].set_value("System Settings").run()
    assert not at.exception

    text = rendered_text(at)
    assert "System Settings & Diagnostics" in text
    assert "GPU & Accelerator Telemetry" in text
    assert "Moondream2 Quantization Parameters" in text
    assert "Camera Endpoints & Diagnostics" in text
    assert "NF4 (NormalFloat 4-bit)" in text
    assert "97 Linear4bit layers" in text