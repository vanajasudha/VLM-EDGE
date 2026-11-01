"""Linux lifecycle contracts tested with subprocess/network mocks."""
from unittest.mock import Mock, call
import pytest
from tools import demo_runtime as runtime


def test_h264_sources_use_stream_copy_and_keep_looping_pacing():
    args = runtime.publisher_args("ffmpeg", "cam1.mp4", "cam1", "h264")
    assert args[args.index("-c:v") + 1] == "copy"
    assert args[args.index("-stream_loop") + 1] == "-1"
    assert "-re" in args and "-an" in args
    assert args[-1] == "rtsp://127.0.0.1:8554/cam1"


def test_incompatible_codec_gets_existing_low_latency_fallback():
    args = runtime.publisher_args("ffmpeg", "cam2.mp4", "cam2", "mpeg4")
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-threads") + 1] == "1"


def test_native_windows_lifecycle_error_is_actionable(monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="WSL/Linux"):
        runtime.preflight()


def test_missing_dependency_starts_no_process(monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime, "require_executable", Mock(side_effect=RuntimeError("missing media tool")))
    monkeypatch.setattr(runtime, "spawn", spawn)
    with pytest.raises(RuntimeError, match="missing media tool"):
        runtime.start()
    spawn.assert_not_called()


def test_unrelated_listener_is_refused_not_killed(monkeypatch):
    stop, spawn = Mock(), Mock()
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime, "require_executable", lambda name, override=None: name)
    monkeypatch.setattr(runtime, "video_codec", lambda *args: "h264")
    monkeypatch.setattr(runtime, "alive", lambda entry: False)
    monkeypatch.setattr(runtime, "port_open", lambda port: True)
    monkeypatch.setattr(runtime, "stop", stop)
    monkeypatch.setattr(runtime, "spawn", spawn)
    with pytest.raises(RuntimeError, match="unowned"):
        runtime.start()
    stop.assert_not_called()
    spawn.assert_not_called()


def test_partial_start_rolls_back_new_processes_only(monkeypatch):
    monkeypatch.setattr(runtime, "preflight", lambda dashboard: ("mediamtx", "ffprobe", {"cam1": ["one"], "cam2": ["two"]}))
    # Server already existed; cam1 is newly created; cam2 fails.
    spawn = Mock(side_effect=[False, True, RuntimeError("cam2 exited")])
    stop = Mock()
    monkeypatch.setattr(runtime, "spawn", spawn)
    monkeypatch.setattr(runtime, "stop", stop)
    monkeypatch.setattr(runtime, "alive", lambda entry: True)
    monkeypatch.setattr(runtime, "port_open", lambda port: True)
    monkeypatch.setattr(runtime, "wait_rtsp", lambda *args: None)
    with pytest.raises(RuntimeError, match="cam2 exited"):
        runtime.start()
    stop.assert_called_once_with("cam1")


def test_spawn_reuses_matching_live_record_without_new_worker(monkeypatch):
    monkeypatch.setattr(runtime, "read_entry", lambda name: {"args": ["same"]})
    monkeypatch.setattr(runtime, "alive", lambda entry: True)
    process = Mock()
    monkeypatch.setattr(runtime.subprocess, "Popen", process)
    assert runtime.spawn("cam1", ["same"]) is False
    with pytest.raises(RuntimeError, match="different arguments"):
        runtime.spawn("cam1", ["different"])
    process.assert_not_called()


def test_managed_dashboard_rejects_remote_sources_without_spawning(monkeypatch, tmp_path):
    import json
    cameras = {f"cam{i}": {"url": f"rtsp://other-host:8554/cam{i}", "source_kind": "demo_rtsp"} for i in range(1, 5)}
    path = tmp_path / "private.json"
    path.write_text(json.dumps({"cameras": cameras}))
    monkeypatch.setenv("VLM_EDGE_CAMERAS_CONFIG", str(path))
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime, "require_executable", lambda name, override=None: name)
    spawn = Mock()
    monkeypatch.setattr(runtime, "spawn", spawn)
    with pytest.raises(RuntimeError, match="managed local demo"):
        runtime.start(dashboard=True)
    spawn.assert_not_called()
