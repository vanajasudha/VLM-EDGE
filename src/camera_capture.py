"""Bounded RTSP workers with receipt timestamps, reconnects and freshness checks."""
from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from time import monotonic

from src.camera_diagnostics import check_server


class CaptureError(RuntimeError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ReceivedFrame:
    pixels: object
    received_monotonic: float
    received_at_utc: str
    sequence: int
    connection: int


@dataclass
class CameraState:
    camera_id: str
    url: str
    status: str = "idle"
    message: str = "Not started"
    frames_received: int = 0
    reconnects: int = 0
    connections: int = 0
    last_error: str | None = None
    failure_kind: str | None = None
    _buffer: deque = field(default_factory=lambda: deque(maxlen=2), repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _stop: Event = field(default_factory=Event, repr=False)
    _thread: Thread | None = field(default=None, repr=False)


class CameraCaptureManager:
    """One CPU decoder and bounded buffer per camera; never calls a VLM."""

    def __init__(self, urls, *, buffer_size=2, open_timeout_ms=3000,
                 read_timeout_ms=3000, reconnect_delay=1.0, stale_after_seconds=3.0,
                 source_kinds=None, capture_factory=None, clock=monotonic, wall_clock=None):
        if buffer_size < 1 or stale_after_seconds <= 0 or reconnect_delay < 0:
            raise ValueError("Invalid camera buffer/freshness/reconnect setting")
        if open_timeout_ms <= 0 or read_timeout_ms <= 0:
            raise ValueError("Capture timeouts must be positive")
        self._urls = dict(urls)
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._reconnect_delay = reconnect_delay
        self.stale_after_seconds = stale_after_seconds
        self._source_kinds = dict(source_kinds or {})
        self._capture_factory = capture_factory
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._states = {
            key: CameraState(key, url, _buffer=deque(maxlen=buffer_size))
            for key, url in self._urls.items()
        }
        self._manager_lock = Lock()
        self._closed = False
        atexit.register(self.close)

    @property
    def camera_ids(self):
        return tuple(self._states)

    def start(self, camera_id):
        state = self._states[camera_id]
        with self._manager_lock:
            if self._closed:
                raise RuntimeError("Camera capture manager is closed")
            if state._thread and state._thread.is_alive():
                if state._stop.is_set():
                    raise RuntimeError("Previous capture worker is still stopping")
                return
            with state._lock:
                state._buffer.clear()
                state.status, state.message, state.last_error = "connecting", "Connecting to RTSP stream", None
                state.failure_kind = None
            state._stop.clear()
            state._thread = Thread(target=self._run, args=(state,), name=f"vlm-edge-{camera_id}", daemon=True)
            state._thread.start()

    def start_many(self, camera_ids):
        for key in camera_ids:
            self.start(key)

    def stop(self, camera_id, timeout=4.0):
        state = self._states[camera_id]
        with self._manager_lock:
            state._stop.set()
            thread = state._thread
            with state._lock:
                state._buffer.clear()
                state.status, state.message = "stopping", "Stopping capture"
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        with state._lock:
            if not thread or not thread.is_alive():
                state.status, state.message = "stopped", "Capture stopped"
            else:
                state.message = "Worker still stopping; waiting for backend timeout"
                state.last_error = "Capture worker exceeded stop wait"
        return not thread or not thread.is_alive()

    def stop_many(self, camera_ids):
        return {key: self.stop(key) for key in camera_ids}

    def snapshot(self, camera_id):
        state = self._states[camera_id]
        with state._lock:
            packet = state._buffer[-1] if state._buffer else None
            age = max(0.0, self._clock() - packet.received_monotonic) if packet else None
            stale = packet is not None and age > self.stale_after_seconds
            status = "stale" if state.status == "connected" and stale else state.status
            valid = packet is not None and status == "connected" and not state._stop.is_set()
            pixels = packet.pixels if packet else None
            return {
                "camera_id": camera_id, "status": status,
                "message": "No fresh frames received" if status == "stale" else state.message,
                "frames_received": state.frames_received, "reconnects": state.reconnects,
                "connections": state.connections, "last_error": state.last_error,
                "failure_kind": "stale_frame" if stale else state.failure_kind,
                "frame": pixels.copy() if hasattr(pixels, "copy") else pixels,
                "received_monotonic": packet.received_monotonic if packet else None,
                "received_at_utc": packet.received_at_utc if packet else None,
                "frame_sequence": packet.sequence if packet else None,
                "connection": packet.connection if packet else None,
                "age_seconds": age, "valid": valid, "stale": stale,
                "buffered_frames": len(state._buffer),
                "source_kind": self._source_kinds.get(camera_id, "unverified_rtsp"),
            }

    def latest_valid(self, camera_id):
        snapshot = self.snapshot(camera_id)
        if not snapshot["valid"]:
            raise RuntimeError(f"{camera_id}: no valid fresh frame ({snapshot['status']})")
        return snapshot

    def _make_capture(self):
        if self._capture_factory is not None:
            return self._capture_factory()
        import os
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        # Four HD decoders must not each spawn one thread per logical CPU.
        os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
        import cv2
        return cv2.VideoCapture()

    def _open(self, state):
        # A bounded transport preflight distinguishes a missing server from decoder/open failures.
        # Injected capture factories remain network-free for unit tests.
        if self._capture_factory is None:
            server = check_server(state.url, timeout=min(.8, self._open_timeout_ms / 1000))
            if not server["reachable"]:
                raise CaptureError("server_unavailable", "RTSP server unavailable; start MediaMTX and check network routing")
        try:
            import cv2
        except ImportError:
            raise CaptureError("decoder_unavailable", "OpenCV decoder unavailable in this Python environment") from None
        capture = self._make_capture()
        try:
            # These are open-only FFmpeg properties; set() before open is ineffective.
            opened = capture.open(state.url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_timeout_ms,
            ])
            if not opened or (hasattr(capture, "isOpened") and not capture.isOpened()):
                raise CaptureError("endpoint_unavailable", "RTSP endpoint could not be opened; check publisher, path and codec")
            return capture
        except Exception:
            capture.release()
            raise

    def _run(self, state):
        try:
            while not state._stop.is_set():
                capture = None
                try:
                    with state._lock:
                        state.status, state.message = "connecting", "Connecting to RTSP stream"
                    capture = self._open(state)
                    with state._lock:
                        state.connections += 1
                    while not state._stop.is_set():
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            raise CaptureError("frame_read_failed", "Stream stopped delivering decoded frames; check publisher and decoder")
                        received, utc = self._clock(), self._wall_clock()
                        with state._lock:
                            if state._stop.is_set():
                                break
                            state.frames_received += 1
                            pixels = frame.copy() if hasattr(frame, "copy") else frame
                            state._buffer.append(ReceivedFrame(pixels, received, utc, state.frames_received,
                                                               state.connections))
                            state.status, state.message, state.last_error = "connected", "Receiving RTSP frames", None
                            state.failure_kind = None
                except Exception as exc:
                    if not state._stop.is_set():
                        with state._lock:
                            state._buffer.clear()
                            state.reconnects += 1
                            state.failure_kind = getattr(exc, "kind", "capture_failure")
                            state.status, state.message = "reconnecting", "Stream unavailable; retrying"
                            # Do not expose RTSP credentials if a backend includes its URL.
                            state.last_error = str(exc).replace(state.url, state.camera_id)
                finally:
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            pass
                if state._stop.wait(self._reconnect_delay):
                    break
        finally:
            with state._lock:
                state._buffer.clear()
                state.status, state.message = "stopped", "Capture stopped"

    def close(self):
        with self._manager_lock:
            if self._closed:
                return
            self._closed = True
            for state in self._states.values():
                state._stop.set()
        for key in self._states:
            self.stop(key)
        atexit.unregister(self.close)
