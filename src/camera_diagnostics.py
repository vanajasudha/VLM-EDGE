"""Bounded, model-free RTSP diagnostics. No Streamlit or torch imports."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


def safe_rtsp_url(url):
    try:
        p = urlsplit(url)
        host = p.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        return urlunsplit((p.scheme, host + (f":{p.port}" if p.port else ""), p.path, "", ""))
    except ValueError:
        return "Invalid endpoint"


def check_server(url, timeout=1.0):
    try:
        p = urlsplit(url)
        if p.scheme not in {"rtsp", "rtsps"} or not p.hostname:
            raise ValueError("invalid endpoint")
        port = p.port or 554
        if p.port == 0 or not 1 <= port <= 65535:
            raise ValueError("invalid port")
    except (ValueError, TypeError, AttributeError):
        return {"reachable": False, "error_kind": "invalid_configuration",
                "error": "Invalid RTSP URL or port. Check the private local camera configuration."}
    try:
        with socket.create_connection((p.hostname, port), timeout=timeout):
            return {"reachable": True, "error_kind": None, "error": None}
    except OSError as exc:
        return {"reachable": False, "error_kind": "server_unavailable",
                "error": f"RTSP server TCP connection failed ({type(exc).__name__}). Check MediaMTX and network routing."}


def probe_endpoint(url, timeout=5):
    ffprobe = shutil.which(os.environ.get("FFPROBE_BIN", "ffprobe"))
    if not ffprobe:
        return {"status": "NOT VERIFIED", "error_kind": None, "error": "ffprobe is not on PATH"}
    try:
        process = subprocess.run([ffprobe, "-v", "error", "-rtsp_transport", "tcp", "-i", url,
                                  "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                                  "-of", "json"], capture_output=True, text=True, timeout=timeout)
    except OSError:
        return {"status": "NOT VERIFIED", "error_kind": "probe_unavailable", "error": "ffprobe could not be executed"}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error_kind": "endpoint_timeout", "error": "RTSP probe timed out"}
    if process.returncode:
        # Never forward raw FFmpeg stderr: it may echo credentials and local paths.
        error = process.stderr.lower()
        if "404" in error or "not found" in error:
            kind, hint = "publisher_unavailable", "RTSP path is not published (404). Start the corresponding FFmpeg publisher."
        elif "401" in error or "unauthorized" in error:
            kind, hint = "authentication_failed", "RTSP authentication failed. Check the private local configuration."
        else:
            kind, hint = "endpoint_failed", "RTSP negotiation failed. Check publisher logs, path and codec."
        return {"status": "failed", "error_kind": kind, "error": hint}
    try:
        streams = json.loads(process.stdout).get("streams", [])
        codec = streams[0]["codec_name"] if streams else None
    except (ValueError, KeyError):
        codec = None
    return {"status": "ok" if codec else "failed", "codec": codec,
            "error_kind": None if codec else "no_video_track", "error": None if codec else "No video track advertised"}


def decode_frames(url, timeout_ms=2000, samples=12, rtsp=True):
    try:
        import cv2
    except ImportError:
        return {"status": "NOT VERIFIED", "error_kind": "decoder_unavailable", "error": "OpenCV is unavailable in this interpreter"}
    capture = cv2.VideoCapture()
    try:
        if rtsp:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
            opened = capture.open(url, cv2.CAP_FFMPEG,
                                  [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                                   cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
        else:
            opened = capture.open(str(url))
        if not opened:
            return {"status": "failed", "error_kind": "decoder_open_failed", "error": "OpenCV could not open the video source"}
        digests, count = set(), 0
        received = None
        deadline = time.monotonic() + 4
        for _ in range(samples):
            if time.monotonic() >= deadline:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            count += 1
            digests.add(hashlib.sha256(frame.tobytes()).hexdigest())
            received = datetime.now(timezone.utc).isoformat()
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).strip("\x00")
        return {"status": "ok" if count else "failed", "frames_decoded": count,
                "distinct_pixel_hashes": len(digests), "changing_pixels": len(digests) > 1,
                "last_decoded_utc": received, "codec_fourcc": codec,
                "error_kind": None if count else "decoder_read_failed",
                "error": None if count else "Source opened but no frame decoded"}
    except Exception:
        return {"status": "failed", "error_kind": "decoder_failure", "error": "Video decoder failed; inspect the local runtime"}
    finally:
        capture.release()


def inspect_camera(camera, entry, root, decode=True):
    source = Path(root) / "sample_videos" / f"{camera}.mp4"
    report = {"camera": camera, "safe_url": safe_rtsp_url(entry["url"]),
              "source_file_found": source.is_file(), "publisher_process": "NOT VERIFIED",
              "capture_state": "NOT VERIFIED (separate diagnostic, not the dashboard buffer)",
              "latest_frame_timestamp": None, "latest_frame_freshness": "NOT VERIFIED"}
    if source.is_file() and decode:
        report["source_decode"] = decode_frames(source, rtsp=False)
    server = check_server(entry["url"])
    report["server"] = server
    if not server["reachable"]:
        report.update(endpoint_reachable=False, frame_decoding="NOT VERIFIED (server unavailable)",
                      error_kind=server["error_kind"], error=server["error"])
        return report
    probe = probe_endpoint(entry["url"])
    report["rtsp_probe"] = probe
    if probe["status"] == "failed":
        report.update(endpoint_reachable=False, frame_decoding="NOT VERIFIED (RTSP probe failed)",
                      error_kind=probe["error_kind"], error=probe["error"])
        return report
    decoded = decode_frames(entry["url"]) if decode else {"status": "NOT VERIFIED"}
    report.update(endpoint_reachable=True if decoded["status"] == "ok" or probe["status"] == "ok" else "NOT VERIFIED",
                  frame_decoding=decoded, latest_frame_timestamp=decoded.get("last_decoded_utc"),
                  latest_frame_freshness="Decoded during this check; dashboard buffer NOT VERIFIED" if decoded["status"] == "ok" else "NOT VERIFIED",
                  error_kind=decoded.get("error_kind"), error=decoded.get("error"))
    return report
