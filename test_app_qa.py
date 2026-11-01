"""Dashboard Q&A integration tests: no model weights or GPU required."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest
from src import model_inference

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def dashboard(monkeypatch):
    loader = Mock(return_value=("model", "tokenizer"))
    inference = Mock(return_value={"answer": "Fresh generated answer", "duration_ms": 123.45})
    monkeypatch.setattr(model_inference, "load_moondream", loader)
    monkeypatch.setattr(model_inference, "infer", inference)
    # Saved-image tools remain independently supported; the main Assistant is live-first.
    script = """
import streamlit as st
from pathlib import Path
from src import data_loader
from src.views.ai_assistant_view import render_ai_assistant
page = st.sidebar.radio("Page", ["Overview", "Ask a Question"])
if page == "Ask a Question":
    render_ai_assistant(data_loader.list_frames(Path("test_frames").resolve()))
"""
    at = AppTest.from_string(script, default_timeout=20).run()
    at.sidebar.radio[0].set_value("Ask a Question").run()
    assert not at.exception
    return at, loader, inference


def rendered(at):
    return "\n".join(str(item.value) for item in list(at.markdown) + list(at.caption))


def test_submit_actual_selection_and_question(dashboard):
    at, loader, inference = dashboard
    assert at.button[0].disabled
    at.text_input[0].set_value("   ").run()
    assert at.button[0].disabled
    at.selectbox[0].set_value("cam2_1").run()
    at.text_input[0].set_value("  What is in this selected image? ").run()
    assert not at.button[0].disabled
    at.button[0].click().run()
    assert not at.exception
    loader.assert_called_once_with()
    inference.assert_called_once_with(
        ("model", "tokenizer"), str(ROOT / "test_frames" / "cam2_1.jpg"),
        "  What is in this selected image? ")
    assert "Fresh generated answer" in rendered(at)
    assert "123.5 ms" in rendered(at)
    assert "incorrect or incomplete" in rendered(at)
    at.run()
    assert inference.call_count == 1
    assert "Fresh generated answer" in rendered(at)


@pytest.mark.parametrize("change", ["question", "image"])
def test_changed_inputs_clear_previous_answer(dashboard, change):
    at, _, inference = dashboard
    at.text_input[0].set_value("First question").run()
    at.button[0].click().run()
    assert "Fresh generated answer" in rendered(at)
    if change == "question":
        at.text_input[0].set_value("New question").run()
    else:
        at.selectbox[0].set_value("cam3_1").run()
    assert "Fresh generated answer" not in rendered(at)
    assert at.session_state["qa_result"] is None
    assert inference.call_count == 1


@pytest.mark.parametrize("stage,message", [
    ("load", "Could not load Moondream2 from the local cache"),
    ("infer", "CUDA ran out of memory during inference"),
    ("infer", "Inference failed: kernel error"),
])
def test_failed_retry_clears_answer_and_recovers(dashboard, stage, message):
    at, loader, inference = dashboard
    at.text_input[0].set_value("Question").run()
    at.button[0].click().run()
    target = loader if stage == "load" else inference
    target.side_effect = model_inference.InferenceError(message)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["qa_result"] is None
    assert "Fresh generated answer" not in rendered(at)
    assert message in at.error[0].value
    target.side_effect = None
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert "Fresh generated answer" in rendered(at)


def test_invalid_image_disables_request(dashboard, monkeypatch):
    at, loader, inference = dashboard
    at.text_input[0].set_value("Question").run()
    at.button[0].click().run()
    monkeypatch.setattr(model_inference, "image_identity",
                        Mock(side_effect=model_inference.InferenceError("Invalid image")))
    at.run()
    assert not at.exception
    assert at.button[0].disabled
    assert "Fresh generated answer" not in rendered(at)
    assert at.session_state["qa_result"] is None
    assert loader.call_count == inference.call_count == 1


def test_image_bytes_changed_at_same_path_clear_answer(dashboard, monkeypatch):
    at, _, _ = dashboard
    at.text_input[0].set_value("Question").run()
    at.button[0].click().run()
    monkeypatch.setattr(model_inference, "image_identity", lambda path: "different-image-bytes")
    at.run()
    assert at.session_state["qa_result"] is None
    assert "Fresh generated answer" not in rendered(at)


def test_overview_does_not_load_model(dashboard):
    at, loader, inference = dashboard
    at.sidebar.radio[0].set_value("Overview").run()
    loader.assert_not_called()
    inference.assert_not_called()


def test_repeat_submission_generates_a_new_answer(dashboard):
    at, loader, inference = dashboard
    at.text_input[0].set_value("Question").run()
    at.button[0].click().run()
    inference.return_value = {"answer": "New response to this request", "duration_ms": 456.0}
    at.button[0].click().run()
    assert not at.exception
    assert inference.call_count == 2
    assert "New response to this request" in rendered(at)
    assert "Fresh generated answer" not in rendered(at)


def test_missing_image_gallery_disables_ask(dashboard, monkeypatch):
    from src import data_loader
    at, loader, inference = dashboard
    monkeypatch.setattr(data_loader, "list_frames", lambda path: [])
    at.text_input[0].set_value("Question").run()
    assert not at.exception
    assert at.button[0].disabled
    loader.assert_not_called()
    inference.assert_not_called()
