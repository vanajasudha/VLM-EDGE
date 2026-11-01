"""Diagnostic failure stages, privacy and capture preflight; no external services."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import pytest
from src import camera_diagnostics as diag
from src.camera_capture import CameraCaptureManager, CaptureError


@pytest.mark.parametrize("url,expected", [
    ("rtsp://user:SECRET@localhost:8554/cam1?token=SECRET#SECRET", "rtsp://localhost:8554/cam1"),
    ("rtsp://user:SECRET@[::1]:8554/cam2?key=SECRET", "rtsp://[::1]:8554/cam2"),
])
def test_safe_endpoint_has_no_userinfo_query_or_fragment(url, expected):
    assert diag.safe_rtsp_url(url) == expected


def test_server_failure_does_not_guess_publisher_or_try_stream_decoder(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "check_server", lambda url: {"reachable": False, "error_kind": "server_unavailable", "error": "No server"})
    probe, decode = Mock(), Mock()
    monkeypatch.setattr(diag, "probe_endpoint", probe)
    monkeypatch.setattr(diag, "decode_frames", decode)
    result = diag.inspect_camera("cam1", {"url": "rtsp://localhost:8554/cam1"}, tmp_path)
    assert result["error_kind"] == "server_unavailable"
    assert result["publisher_process"] == "NOT VERIFIED"
    assert result["latest_frame_timestamp"] is None
    probe.assert_not_called()
    decode.assert_not_called()


@pytest.mark.parametrize("stderr,kind", [
    ("rtsp://user:SECRET@host/cam: 404 Not Found", "publisher_unavailable"),
    ("401 Unauthorized SECRET", "authentication_failed"),
    ("Unrecognized decoder SECRET", "endpoint_failed"),
])
def test_probe_distinguishes_failures_without_echoing_stderr(monkeypatch, stderr, kind):
    monkeypatch.setattr(diag.shutil, "which", lambda name: name)
    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr=stderr))
    result = diag.probe_endpoint("rtsp://host/cam")
    assert result["error_kind"] == kind
    assert "SECRET" not in json.dumps(result)


def test_probe_success_is_not_claimed_as_frame_decoding(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "check_server", lambda url: {"reachable": True})
    monkeypatch.setattr(diag, "probe_endpoint", lambda url: {"status": "ok", "codec": "h264"})
    monkeypatch.setattr(diag, "decode_frames", lambda *a, **k: {"status": "failed", "error_kind": "decoder_read_failed"})
    result = diag.inspect_camera("cam1", {"url": "rtsp://host/cam"}, tmp_path)
    assert result["endpoint_reachable"] is True
    assert result["frame_decoding"]["status"] == "failed"
    assert result["latest_frame_timestamp"] is None


def test_decoder_always_releases_after_read_error(monkeypatch):
    import cv2
    capture = Mock()
    capture.open.return_value = True
    capture.read.side_effect = RuntimeError("decode broke")
    monkeypatch.setattr(cv2, "VideoCapture", lambda: capture)
    assert diag.decode_frames("rtsp://host/cam")["error_kind"] == "decoder_failure"
    capture.release.assert_called_once()


def test_unavailable_transport_is_classified_before_creating_decoder(monkeypatch):
    import src.camera_capture as capture_module
    import cv2
    decoder = Mock()
    monkeypatch.setattr(cv2, "VideoCapture", decoder)
    monkeypatch.setattr(capture_module, "check_server", lambda *a, **k: {"reachable": False})
    manager = CameraCaptureManager({"cam1": "rtsp://localhost/cam1"})
    try:
        with pytest.raises(CaptureError) as error:
            manager._open(manager._states["cam1"])
        assert error.value.kind == "server_unavailable"
        decoder.assert_not_called()
    finally:
        manager.close()


def test_probe_timeout_is_bounded_and_distinct(monkeypatch):
    monkeypatch.setattr(diag.shutil, "which", lambda name: name)
    monkeypatch.setattr(diag.subprocess, "run", Mock(side_effect=diag.subprocess.TimeoutExpired("ffprobe", 5)))
    assert diag.probe_endpoint("rtsp://host/cam")["error_kind"] == "endpoint_timeout"


def test_cli_rejects_empty_camera_configuration(monkeypatch, tmp_path):
    from tools.diagnose_cameras import main
    config = tmp_path / "cameras.json"
    config.write_text('{"cameras": {}}')
    monkeypatch.setattr("sys.argv", ["diagnose", "--config", str(config)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2


@pytest.mark.parametrize("url", ["rtsp://user:PRIVATE@host:bad/cam1", "rtsp://host:0/cam1", "http://host/cam1"])
def test_invalid_config_is_not_misreported_as_a_server_outage(monkeypatch, url):
    connect = Mock()
    monkeypatch.setattr(diag.socket, "create_connection", connect)
    result = diag.check_server(url)
    assert result["error_kind"] == "invalid_configuration"
    assert "PRIVATE" not in json.dumps(result)
    connect.assert_not_called()
