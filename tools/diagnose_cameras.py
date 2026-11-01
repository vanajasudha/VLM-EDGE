"""Read-only camera preflight; never starts publishers, CUDA or a model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.camera_diagnostics import inspect_camera
from tools.demo_runtime import alive, read_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("VLM_EDGE_CAMERAS_CONFIG", ROOT / "config/cameras.json")))
    parser.add_argument("--no-decode", action="store_true", help="TCP/RTSP checks only; do not open video decoders")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if set(config.get("cameras", {})) != {f"cam{i}" for i in range(1, 5)}:
        parser.error("Configure exactly cam1, cam2, cam3 and cam4")
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "platform": sys.platform,
              "executables": {name: bool(shutil.which(os.environ.get(name.upper() + "_BIN", name))) for name in ("ffmpeg", "ffprobe", "mediamtx")},
              "cameras": [inspect_camera(camera, entry, ROOT, decode=not args.no_decode)
                          for camera, entry in config["cameras"].items()]}
    for camera in report["cameras"]:
        if sys.platform == "linux":
            camera["owned_publisher_running"] = alive(read_entry(camera["camera"]))
            camera["publisher_process"] = "Owned process running" if camera["owned_publisher_running"] else "No live owned record; external publishers NOT VERIFIED"
    print(json.dumps(report, indent=2))
    return 0 if all(c["endpoint_reachable"] is True and isinstance(c["frame_decoding"], dict)
                    and c["frame_decoding"]["status"] == "ok" for c in report["cameras"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
