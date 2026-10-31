import time

import pytest

from src.camera_capture import CameraCaptureManager


class FailingCapture:
    releases = 0

    def set(self, prop, value):
        return True

    def open(self, url, *args):
        return False

    def isOpened(self):
        return False

    def release(self):
        type(self).releases += 1


class OneFrameCapture:
    releases = 0

    def set(self, prop, value):
        return True

    def open(self, url, *args):
        return True

    def isOpened(self):
        return True

    def read(self):
        return True, "frame"

    def release(self):
        type(self).releases += 1


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_unreachable_stream_reports_error_and_releases_capture():
    FailingCapture.releases = 0
    manager = CameraCaptureManager({"cam1": "rtsp://invalid/cam1"},
                                   reconnect_delay=0.01, capture_factory=FailingCapture)
    manager.start("cam1")
    assert wait_for(lambda: manager.snapshot("cam1")["status"] in {"error", "reconnecting"})
    snapshot = manager.snapshot("cam1")
    assert snapshot["frame"] is None
    assert snapshot["last_error"]
    manager.close()
    assert FailingCapture.releases >= 1


def test_capture_buffer_is_bounded_and_shutdown_is_clean():
    OneFrameCapture.releases = 0
    manager = CameraCaptureManager({"cam1": "rtsp://valid/cam1"},
                                   buffer_size=2, capture_factory=OneFrameCapture)
    manager.start("cam1")
    assert wait_for(lambda: manager.snapshot("cam1")["frames_received"] > 0)
    manager.stop("cam1")
    snapshot = manager.snapshot("cam1")
    assert snapshot["status"] == "stopped"
    assert snapshot["frames_received"] > 0
    assert OneFrameCapture.releases >= 1
    manager.close()


def test_unknown_camera_and_start_after_close_fail():
    manager = CameraCaptureManager({"cam1": "rtsp://valid/cam1"}, capture_factory=FailingCapture)
    with pytest.raises(KeyError):
        manager.start("cam9")
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.start("cam1")



def test_receipt_timestamps_stale_detection_and_copy():
    from src.camera_capture import ReceivedFrame
    import numpy as np
    clock = [10.0]
    manager = CameraCaptureManager({"cam1": "rtsp://local/cam1"}, clock=lambda: clock[0],
                                   stale_after_seconds=2, capture_factory=FailingCapture)
    state = manager._states["cam1"]
    state.status = "connected"
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    state._buffer.append(ReceivedFrame(original, 9.5, "2026-10-29T12:00:00+00:00", 1, 1))
    snapshot = manager.latest_valid("cam1")
    assert snapshot["age_seconds"] == .5
    assert snapshot["received_at_utc"].endswith("+00:00")
    snapshot["frame"][0, 0] = 255
    assert not original.any()
    clock[0] = 12.0
    assert manager.snapshot("cam1")["status"] == "stale"
    with pytest.raises(RuntimeError, match="no valid fresh frame"):
        manager.latest_valid("cam1")
    manager.close()
    assert manager.snapshot("cam1")["frame"] is None


def test_open_passes_open_only_timeout_parameters_and_releases_on_exception():
    import cv2
    class Capture:
        released = False
        def open(self, url, backend, params):
            assert backend == cv2.CAP_FFMPEG
            assert params == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 123,
                              cv2.CAP_PROP_READ_TIMEOUT_MSEC, 456]
            raise RuntimeError("open exception")
        def release(self): self.released = True
    capture = Capture()
    manager = CameraCaptureManager({"cam1": "rtsp://local/cam1"}, open_timeout_ms=123,
                                   read_timeout_ms=456, capture_factory=lambda: capture)
    with pytest.raises(RuntimeError, match="open exception"):
        manager._open(manager._states["cam1"])
    assert capture.released
    manager.close()


def test_disconnect_clears_frame_and_successful_reconnect():
    from threading import Event
    gate = Event()
    class Capture:
        def __init__(self, failed): self.failed, self.count, self.released = failed, 0, False
        def open(self, *args): return True
        def isOpened(self): return True
        def read(self):
            self.count += 1
            if self.failed and self.count > 1:
                return False, None
            if self.failed:
                gate.wait(.2)
                return True, "first"
            time.sleep(.005)
            return True, "recovered"
        def release(self): self.released = True
    captures = []
    def factory():
        capture = Capture(not captures)
        captures.append(capture)
        return capture
    manager = CameraCaptureManager({"cam1": "rtsp://local/cam1"}, capture_factory=factory,
                                   reconnect_delay=.05)
    manager.start("cam1")
    gate.set()
    assert wait_for(lambda: manager.snapshot("cam1")["reconnects"] >= 1)
    assert not manager.snapshot("cam1")["valid"]
    assert manager.snapshot("cam1")["frame"] is None
    assert wait_for(lambda: manager.snapshot("cam1")["valid"])
    snapshot = manager.latest_valid("cam1")
    assert snapshot["frame"] == "recovered" and snapshot["connection"] == 2
    assert captures[0].released
    manager.close()
    assert all(c.released for c in captures)


def test_blocked_read_stop_is_truthful_and_cannot_restart_duplicate_worker():
    from threading import Event
    entered, release = Event(), Event()
    class Capture:
        def open(self, *args): return True
        def isOpened(self): return True
        def read(self):
            entered.set()
            release.wait(2)
            return True, "late"
        def release(self): pass
    manager = CameraCaptureManager({"cam1": "rtsp://local/cam1"}, capture_factory=Capture)
    manager.start("cam1")
    assert entered.wait(1)
    assert manager.stop("cam1", timeout=.01) is False
    assert manager.snapshot("cam1")["status"] == "stopping"
    assert not manager.snapshot("cam1")["valid"]
    with pytest.raises(RuntimeError, match="still stopping"):
        manager.start("cam1")
    release.set()
    assert wait_for(lambda: manager.snapshot("cam1")["status"] == "stopped")
    assert manager.snapshot("cam1")["frame"] is None
    manager.close()


def test_four_camera_workers_and_bounded_buffers():
    ids = {f"cam{i}": f"rtsp://local/cam{i}" for i in range(1, 5)}
    manager = CameraCaptureManager(ids, capture_factory=OneFrameCapture, buffer_size=2)
    manager.start_many(ids)
    assert wait_for(lambda: all(manager.snapshot(k)["frames_received"] >= 3 for k in ids))
    assert all(manager.snapshot(k)["buffered_frames"] == 2 for k in ids)
    manager.close()
    assert all(manager.snapshot(k)["status"] == "stopped" for k in ids)
