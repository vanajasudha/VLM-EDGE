"""Five-page dashboard contract; CPU mocks never load weights or open streams."""
from pathlib import Path
from unittest.mock import Mock
import pytest
from streamlit.testing.v1 import AppTest
from src import live_camera_ui, model_inference
from test_live_camera import FakeManager

ROOT = Path(__file__).resolve().parent
PAGE_PATHS = {'Overview': 'src/pages/overview.py', 'Live Monitoring': 'src/pages/monitoring.py', 'AI Assistant': 'src/pages/assistant.py', 'Performance': 'src/pages/performance.py', 'Settings': 'src/pages/settings.py'}


@pytest.fixture
def app(monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(live_camera_ui, "get_camera_manager", lambda: manager)
    loader = Mock(side_effect=AssertionError("Preview must not load the model"))
    monkeypatch.setattr(model_inference, "load_moondream", loader)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=25).run()
    assert not at.exception
    return at, manager, loader


def text(at):
    return "\n".join(str(e.value) for e in list(at.markdown) + list(at.caption) + list(at.header) + list(at.info))


def test_every_navigation_item_and_roundtrip_do_not_load_model(app):
    at, manager, loader = app
    for page in ("Overview", "Live Monitoring", "AI Assistant", "Performance", "Settings", "Overview"):
        at.switch_page(PAGE_PATHS[page]).run()
        assert not at.exception
        assert page in [e.value for e in at.header]
    loader.assert_not_called()
    assert not manager.started


def test_overview_action_uses_callback_without_widget_state_error(app):
    at, _, _ = app
    at.button(key="open_assistant").click().run()
    assert not at.exception
    assert "AI Assistant" in [e.value for e in at.header]


def test_four_offline_cards_and_no_fake_metrics(app):
    at, manager, _ = app
    manager.valid = False
    at.run()
    assert not at.exception
    assert text(at).count("No recent frame available") == 4
    assert "0 of 4 cameras connected" in text(at)
    assert not at.metric
    at.switch_page("src/pages/assistant.py").run()
    assert at.button(key="ask_live").disabled


def test_capture_controls_are_explicit_and_shared(app):
    at, manager, _ = app
    at.switch_page("src/pages/monitoring.py").run()
    at.button(key="start_streams").click().run()
    at.run()
    assert manager.started == list(manager.camera_ids)
    at.button(key="stop_streams").click().run()
    assert manager.stopped == list(manager.camera_ids)
    assert not at.exception


def test_performance_empty_state_and_real_session_history(app, monkeypatch):
    at, _, _ = app
    at.switch_page("src/pages/performance.py").run()
    assert "No requests measured" in text(at)
    assert not at.metric
    assert not at.get("arrow_vega_lite_chart")
    result = {"answer": "Actual mock request", "duration_ms": 123.4,
              "frame_metadata": {**FakeManager().snapshot("cam2"),
                                 "frame_age_at_inference_seconds": .02,
                                 "submitted_rgb": FakeManager().frame[:, :, ::-1]}}
    infer = Mock(return_value=result)
    monkeypatch.setattr(live_camera_ui, "ask_latest", infer)
    at.switch_page("src/pages/assistant.py").run()
    at.text_input(key="live_question").set_value("Question?").run()
    at.button(key="ask_live").click().run()
    at.run()
    assert infer.call_count == 1
    at.switch_page("src/pages/performance.py").run()
    assert at.metric[0].value == "1"
    assert at.metric[2].value == "123.4 ms"
    assert not at.exception


def test_failed_request_has_no_stale_answer_and_is_counted(app, monkeypatch):
    at, _, _ = app
    monkeypatch.setattr(live_camera_ui, "ask_latest", Mock(side_effect=RuntimeError("Stream lost")))
    at.switch_page("src/pages/assistant.py").run()
    at.text_input(key="live_question").set_value("Question?").run()
    at.button(key="ask_live").click().run()
    assert "Stream lost" in at.error[0].value
    assert at.session_state["live_result"] is None
    at.switch_page("src/pages/performance.py").run()
    assert at.metric[0].value == "0" and at.metric[1].value == "1"


def test_settings_hide_endpoint_secrets_and_do_not_invent_gpu(app, monkeypatch):
    import src.views.system_settings_view as settings
    config = {key: {"url": "rtsp://user:PRIVATE@localhost:8554/stream?token=PRIVATE", "source_kind": "demo_rtsp"}
              for key in FakeManager.camera_ids}
    monkeypatch.setattr(settings, "read_camera_config", lambda: (config, 3))
    monkeypatch.setattr(settings, "gpu_measurements", lambda: {})
    at, _, loader = app
    at.switch_page("src/pages/settings.py").run()
    assert not at.exception
    assert "PRIVATE" not in "\n".join(e.value for e in at.markdown)
    assert "localhost:8554/stream" in "\n".join(e.value for e in at.markdown)
    assert "GPU measurements unavailable" in text(at)
    loader.assert_not_called()


def test_reconnection_invalidates_prior_live_result(app, monkeypatch):
    at, manager, _ = app
    snap = manager.snapshot("cam1")
    result = {"answer": "Prior connection answer", "duration_ms": 1,
              "frame_metadata": {**snap, "frame_age_at_inference_seconds": .02,
                                 "submitted_rgb": manager.frame[:, :, ::-1]}}
    monkeypatch.setattr(live_camera_ui, "ask_latest", lambda *args: result)
    at.switch_page("src/pages/assistant.py").run()
    at.text_input(key="live_question").set_value("Question?").run()
    at.button(key="ask_live").click().run()
    original = manager.snapshot
    manager.snapshot = lambda camera: {**original(camera), "connection": 2}
    at.run()
    assert at.session_state["live_result"] is None
    assert "Prior connection answer" not in text(at)


def test_sidebar_contains_native_links_and_no_radio_widgets(app):
    at, _, _ = app
    assert not at.sidebar.radio
    links = at.sidebar.get("page_link")
    assert len(links) == 5
    assert {element.label for element in links} == set(PAGE_PATHS)
    assert "Moondream2" not in "\n".join(e.value for e in at.sidebar.caption)


def test_camera_and_question_survive_navigation_without_reinference(app):
    at, _, loader = app
    at.switch_page(PAGE_PATHS["AI Assistant"]).run()
    at.selectbox(key="live_camera").set_value("cam3").run()
    at.text_input(key="live_question").set_value("Persistent question").run()
    at.switch_page(PAGE_PATHS["Settings"]).run()
    at.switch_page(PAGE_PATHS["AI Assistant"]).run()
    assert not at.exception
    assert at.selectbox(key="live_camera").value == "cam3"
    assert at.text_input(key="live_question").value == "Persistent question"
    loader.assert_not_called()


def test_configuration_failure_keeps_all_pages_navigable(app, monkeypatch):
    at, _, loader = app
    monkeypatch.setattr(live_camera_ui, "get_camera_manager", Mock(side_effect=ValueError("Invalid private config")))
    for page, path in PAGE_PATHS.items():
        at.switch_page(path).run()
        assert not at.exception
        assert len(at.sidebar.get("page_link")) == 5
    loader.assert_not_called()
