# Camera integration report

31 October 2026. Local preparation only; no commit/push, package installation, weight download or GPU inference.

## Result and confirmed causes

**Four changing RTSP previews and a real Moondream2 answer: NOT VERIFIED.** The tested environment has no available MediaMTX/FFmpeg service. The MP4 files are healthy; starting the Windows dashboard alone cannot publish them.

- Windows: `ffmpeg`, `ffprobe` and `mediamtx` are not on PATH; no matching media processes were observed. `netstat` showed no TCP listener on 8554. Each configured endpoint timed out on a bounded TCP check, including outside the sandbox.
- WSL Ubuntu: those three executables are also absent from PATH; `ss` showed no listener on 8554 and process inspection showed no FFmpeg/MediaMTX process. All four TCP connections were refused. There is no original `~/vlm-edge` project, original venv interpreter or `~/mediamtx/mediamtx` at the inspected locations. Python 3.11, torch, OpenCV and Streamlit were unavailable in the inspected WSL environment.
- The Windows UI runs using the available Python 3.14 interpreter and copied `.venv/Lib/site-packages`. That does not restore the historical WSL CUDA environment. No WSL GPU execution capability was established.
- Capture is explicit: an idle worker is now labelled **Capture not started**. Even a correctly started worker cannot deliver frames without a server and publisher; it must not be labelled Online.
- Cross-environment routing is **not established as the cause**: there is no WSL server to reach. No guessed WSL IP or networking change was applied. No user `.wslconfig` was found at the inspected location; an effective networking mode was not inferred from that absence.

## Actual per-camera checks

| Camera | Local MP4 | Local decode | Publisher | RTSP transport | Live frame / freshness |
|---|---|---|---|---|---|
| cam1 | Found, 991017 bytes | H.264, 12 frames / 12 distinct pixel hashes | No FFmpeg process observed | Windows timeout; WSL refused | NOT VERIFIED; timestamp null |
| cam2 | Found, 991017 bytes | H.264, 12 frames / 12 distinct pixel hashes | No FFmpeg process observed | Windows timeout; WSL refused | NOT VERIFIED; timestamp null |
| cam3 | Found, 991017 bytes | H.264, 12 frames / 12 distinct pixel hashes | No FFmpeg process observed | Windows timeout; WSL refused | NOT VERIFIED; timestamp null |
| cam4 | Found, 991017 bytes | H.264, 12 frames / 12 distinct pixel hashes | No FFmpeg process observed | Windows timeout; WSL refused | NOT VERIFIED; timestamp null |

Local source decoding is not RTSP decoding. The four files have identical bytes, as established in the earlier audit; four looping publishers simulate four endpoints rather than four independent recorded scenes. The diagnostic command never invents a publisher PID or a dashboard-buffer timestamp. Its owned-process check cannot discover arbitrary external publishers.

An actual offline capture-manager check started all four workers twice: **4 workers**, all `reconnecting` with `failure_kind=server_unavailable`, zero received/valid frames and null receipt timestamps. Closing the manager left **0 workers**. No model was imported by that check.

## Changes in this task

| Files | Change |
|---|---|
| `src/live_camera_ui.py`, `src/ui.py`, `app.py`, `src/views/overview_view.py`, `src/views/system_settings_view.py` | Compact grid/spacing, short sidebar footer, content-width capture toolbar, distinct capture states, actionable hints, prominent Ask button and responsive Settings rows |
| `src/camera_capture.py` | Bounded TCP preflight for real decoders, classified transport/open/read/missing-decoder failures in snapshots; existing freshness/reconnect/worker lifecycle retained |
| `src/camera_diagnostics.py`, `tools/diagnose_cameras.py` | Model-free per-camera file/transport/probe/decode diagnosis, safe URLs, timeouts, exact four-camera validation and explicit NOT VERIFIED fields |
| `tools/demo_runtime.py` | Preflight, executable overrides, H.264 stream copy, RTSP-track readiness, loopback dashboard binding, occupied-port refusal, partial-start rollback and idempotent owned-process management |
| `start_demo.sh`, `stop_demo.sh`, `.gitattributes` | Literal `VLM_EDGE_PYTHON` override with a useful missing-interpreter error; preserve LF on checkout |
| `config/mediamtx-demo.yml` | Current documented `rtspTransports` key instead of legacy `protocols`; still loopback-only/TCP. Real MediaMTX startup with this configuration is NOT VERIFIED here. |
| `tools/verify_live_demo.py` | Preserve configured source provenance and redact credentials from recorded stream URLs |
| `.gitignore` | Ignore private `config/cameras.local.json` |
| `test_camera_diagnostics.py`, `test_demo_runtime.py`, `test_dashboard_redesign.py` | Failure-stage, privacy, startup ownership/rollback tests; adjust Settings presentation assertions without suppressing failures |
| `README.md`, this report | Updated workflow and evidence boundaries |

`src/model_inference.py`, `src/live_camera.py`, NF4 settings, model architecture/cache, inference locking, dependency manifests, videos and historical results were not changed. All 29 historical preservation hashes still match. No test/project file was deleted.

## UI and lifecycle verification

All five pages were browser-checked at **1440 x 1000** and **390 x 1000**. Each reached its expected heading, with no Streamlit exceptions or page/main horizontal overflow. Screenshots were inspected for desktop Overview/Live Monitoring and narrow Settings. The Overview fits its four compact previews in the desktop view; Settings uses wrapping rows instead of tall cards. Camera status is shown once per card, with an additional action hint only when needed.

Actual browser interaction with the unavailable server:

1. Start capture: 4 active workers, 0 connected, all four Reconnecting with MediaMTX-unreachable hints, no exception.
2. Navigate to AI Assistant and enter a nonempty question: Ask AI remains disabled because no fresh frame exists.
3. Performance remains the genuine no-measurements state.
4. Stop capture: 0 active workers, all cameras Capture stopped, no frames falsely labelled live.

The temporary verification browser/server were stopped after checking. No synthetic answer or performance data was injected into the running browser application. CPU mocks separately cover successful requests, camera changes, exact submitted frames, errors, freshness, cache/lock behavior and measurements.

Ignored local evidence is under `.runtime/`: `camera-check-windows.json`, `capture-offline-check.json`, `camera-polish-browser-check.json`, `camera-polish-browser-lifecycle.json`, `camera-polish-*.png`, and test logs. WSL TCP checks and Bash syntax checking also ran directly through `wsl`.

## Tests and startup verification

- Focused UI/capture/inference/diagnostics/startup run: **83 passed**; guard counters zero for model/tokenizer loads and CUDA initialization.
- Full suite: **150 passed, 3 failed, 35 setup errors**. The three evaluation failures and 35 annotation setup errors require Linux `fcntl` on native Windows. Existing Linux code/tests were not modified or skipped to hide them.
- Final targeted rerun after diagnostic validation and presentation changes: **26 passed**, again with zero model/CUDA guard attempts.
- `bash -n` accepted both launchers. WSL `python3 tools/demo_runtime.py check` refused startup with the expected missing-MediaMTX error, before spawning anything.
- Startup ownership/occupied-port protection, stream-copy selection and rollback were mock-tested. Real MediaMTX startup, publishers, decoded RTSP frames, reconnection to an actual publisher and GPU answer: **NOT VERIFIED**.
- Git diff/whitespace and Python syntax checked. No commit/push.

## Complete demo: prerequisites and exact commands

Use one WSL environment for MediaMTX, FFmpeg, capture and GPU inference. The working code is already accessible from WSL; copying it is unnecessary:

```bash
cd /mnt/c/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge
```

Before the commands below can succeed, restore/activate a working WSL Python 3.11 CUDA venv and the existing local Moondream2 cache; neither was found here. Do not upgrade/reinstall a working AI stack blindly. FFmpeg/ffprobe and MediaMTX must be installed separately in WSL. For FFmpeg, if installation is needed, the Ubuntu package command is `sudo apt-get install ffmpeg` after refreshing package metadata. Obtain MediaMTX using its [official installation instructions](https://mediamtx.org/docs/kickoff/install); its binary may be on PATH or at the existing helper's default `~/mediamtx/mediamtx`. Check the installed version/configuration rather than assuming an old binary accepts a newer schema. The current TCP key is documented in the [MediaMTX configuration reference](https://mediamtx.org/docs/references/configuration-file).

With the genuine WSL GPU venv activated:

```bash
# Refuse to use an unactivated or copied/broken environment.
: "${VIRTUAL_ENV:?Activate the working WSL CUDA venv first}"
export VLM_EDGE_PYTHON="$VIRTUAL_ENV/bin/python"
"$VLM_EDGE_PYTHON" --version
"$VLM_EDGE_PYTHON" -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
command -v ffmpeg
command -v ffprobe
# If MediaMTX is outside PATH, set MEDIAMTX_BIN to its actual executable.
# Example only when this installed file exists:
# export MEDIAMTX_BIN="$HOME/mediamtx/mediamtx"

# Validate sources/tools/ports without starting processes.
"$VLM_EDGE_PYTHON" tools/demo_runtime.py check --dashboard

# Start MediaMTX, then all four publishers, then Streamlit.
bash start_demo.sh
"$VLM_EDGE_PYTHON" tools/demo_runtime.py status

# Check each RTSP endpoint and decoded frames; no model load.
"$VLM_EDGE_PYTHON" tools/diagnose_cameras.py
# Require genuinely changing decoded frames over time; still no model without --ai.
"$VLM_EDGE_PYTHON" tools/verify_live_demo.py
```

Open **http://localhost:8501** from Windows. In Live Monitoring click **Start capture**, wait for **4 of 4 cameras connected**, then open AI Assistant, choose a camera, type a question and click Ask AI. Inspect the generated answer and its exact submitted-frame evidence. If CUDA/model-cache prerequisites are confirmed and no other model process is running, an explicit CLI check is also available: `"$VLM_EDGE_PYTHON" tools/verify_live_demo.py --ai`. That command performs real inference and writes new evidence; it was not run during this task.

To start streams and the dashboard separately:

```bash
"$VLM_EDGE_PYTHON" tools/demo_runtime.py start
"$VLM_EDGE_PYTHON" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

The foreground dashboard stops with Ctrl+C. Stop helper-owned streams/server/dashboard safely with:

```bash
bash stop_demo.sh
"$VLM_EDGE_PYTHON" tools/demo_runtime.py status
```

`start` reuses matching live process records, refuses mismatched arguments/foreign occupied ports and rolls back only processes created by a failed attempt. FFmpeg publishers use `-re -stream_loop -1`, the matching MP4/path, TCP and no audio. H.264 uses `-c:v copy`; other codecs use the existing single-thread libx264 fallback. Publisher crashes remain visible in `.runtime/camN.log`/status; rerun start after resolving the error. Output-side FFmpeg reconnection is not claimed. Capture workers retry automatically. No helper kills an unrelated server or publisher.

## Windows/WSL routing and code synchronization

For this host, the preferred reliable arrangement is the single shared checkout above with all demo runtime components inside WSL. The historical `~/vlm-edge` location does not exist in this inspected WSL instance. If a separate native Linux checkout is later restored, inspect `git status --short`, `git diff` and `git diff --cached` in both copies first. Review a patch/file diff and transfer only intended source changes into a clean destination. Stop on conflicting/uncommitted destination changes; do not use force reset, directory replacement or mirror-copy deletion. Do not copy a Windows virtual environment into Linux. Using the same mounted checkout avoids source divergence.

If Windows must consume a WSL RTSP server, test Windows-to-WSL localhost forwarding first. Microsoft documents Windows access to WSL services and the differences between NAT and mirrored networking; see [WSL networking](https://learn.microsoft.com/windows/wsl/networking). No firewall, `.wslconfig` or server bind was changed in this task. If localhost forwarding does not work, obtain the actual current WSL address rather than inventing one, and deliberately configure the server's listening interface/firewall. A server bound only to 127.0.0.1 cannot be assumed reachable via its VM interface address.

The application already supports a private endpoint configuration: copy `config/cameras.json` to ignored `config/cameras.local.json`, put the genuinely reachable endpoints there, and set `VLM_EDGE_CAMERAS_CONFIG` to its absolute path **before starting Streamlit**. On PowerShell:

```powershell
$env:VLM_EDGE_CAMERAS_CONFIG = "$PWD\config\cameras.local.json"
$env:PYTHONPATH = "$PWD\.venv\Lib\site-packages"
python -m streamlit run app.py --server.address 127.0.0.1
```

Restart Streamlit after changing source configuration, because the manager is process-cached. This override changes capture endpoints; it does not move MediaMTX or publishers. The managed helper intentionally publishes to the four local WSL endpoints; use separately managed publishers/server for a deliberate remote arrangement. Keep credentials in ignored local configuration; Settings/diagnostics strip URL userinfo/query/fragment. `FFMPEG_BIN` and `FFPROBE_BIN` are optional executable overrides for the helper; `MEDIAMTX_BIN` selects MediaMTX. None installs software or guesses network addresses.

Remaining blockers are the missing media executables/service and the missing working WSL GPU environment/cache. Resolve them before claiming a four-camera real inference demo.

Cleanup note: disposable browser profiles, screenshots, temporary helper scripts and pytest directories were subsequently removed. The verification findings and retained JSON/test-log records remain valid.
