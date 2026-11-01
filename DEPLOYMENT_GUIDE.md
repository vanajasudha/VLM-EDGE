# VLM-Edge deployment guide

Target: a separate NVIDIA GPU laptop. No driver repair, model download or real GPU inference was performed on the current laptop. Commands below are Bash/WSL unless labelled PowerShell.

## What is verified and what is pending

| Scope | Evidence/status |
|---|---|
| Current Windows UI | Native five-page sidebar, offline workflows and CPU/mock tests; browser results are recorded at the end of this guide |
| Historical inference target | WSL2 Ubuntu, Python 3.11, RTX 3050 Laptop GPU with about 4 GiB, torch 2.6.0+cu124 and selective NF4 Moondream2; see [Task 1 evidence](TASK1_VERIFICATION.md) |
| Dependency pins | Read from copied WSL package metadata; not a fully resolved lockfile or a clean-install validation |
| New laptop | Clean install, driver/runtime compatibility, four real RTSP feeds, model-cache provisioning and real inference require verification there |

This working copy has uncommitted changes. A GitHub clone will not contain them until the owner reviews and publishes them later. No commit or push is part of this preparation. Alternatively, transfer the reviewed source changes into a clean checkout without copying environments, weights or secrets.

## 1. Prerequisites and compatibility

Use WSL2 Ubuntu with **Python 3.11**, a supported NVIDIA GPU and a working Windows/WSL GPU driver on the target laptop. Keep MediaMTX, FFmpeg, Streamlit and capture in the same WSL environment to avoid cross-environment RTSP routing. Exact Ubuntu/driver/MediaMTX versions were not recorded in historical GPU evidence; a newer laptop or GPU is not automatically a verified configuration. The historical target supports the BF16/FP16 operations used by this backend.

The recorded Python dependency set is:

| Packages | Pins |
|---|---|
| torch / transformers / bitsandbytes | 2.6.0+cu124 / 4.51.3 / 0.50.2 |
| accelerate / einops | 1.15.0 / 0.8.2 |
| streamlit / pandas / numpy | 1.66.1 / 3.0.6 / 2.4.6 |
| Pillow / opencv-python / altair | 12.3.0 / 5.0.0.93 / 6.3.0 |
| nvidia-ml-py / pytest | 13.615.71 / 9.2.0 |

These pins are in [requirements.txt](requirements.txt). Copied metadata for the direct dependencies is consistent with Python 3.11; that does not prove index availability or transitive compatibility on a new installation. Python 3.14 on the current Windows machine is used for UI verification only, not as a validated Moondream2 deployment target.

The CUDA-enabled PyTorch build is **cu124**. Do not equate the CUDA compatibility number printed by `nvidia-smi` with `torch.version.cuda`. The target host driver must support the selected runtime. The commands use the wheel installation documented for PyTorch 2.6.0/CUDA 12.4 in [official previous-version instructions](https://docs.pytorch.org/get-started/previous-versions/); no broader GPU or CUDA compatibility is claimed. A standalone CUDA compiler/toolkit is not required by the normal Python dashboard workflow. The separate `zero_copy_bench.cu` experiment is outside this deployment path.

Provide sufficient disk/host memory for packages and model files; a minimum was not measured. Close other GPU model processes before inference on a 4 GiB device. Do not install NVIDIA drivers on this current non-GPU laptop as part of this task.

## 2. Clone the reviewed source

On the new laptop, after the final source is available at the configured remote:

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/vanajasudha/VLM-EDGE.git
cd VLM-EDGE
git status --short
git log -1 --oneline
```

Record the reviewed commit ID used for the demo. Authenticate with the normal Git credential flow if needed; never put access tokens in URLs or scripts. Do not overwrite an existing clone with uncommitted changes. Media and environments are intentionally absent from the clone.

## 3. Create a clean Python 3.11 environment

Run from the repository root. If `python3.11` is unavailable, provision that interpreter with the target OS's approved package/environment manager first. No command here assumes every Ubuntu release ships it by default.

```bash
python3.11 --version
# For a fresh clone only: do not overwrite an existing venv.
python3.11 -m venv venv
source venv/bin/activate
python --version
python -m pip --version
export VLM_EDGE_PYTHON="$VIRTUAL_ENV/bin/python"
```

Do not copy `venv/` or `.venv/` from Windows or another laptop. Their interpreter paths and native binaries are not portable. If a destination venv already exists, inspect it and use a separate new environment rather than reinstalling into it blindly.

## 4. Install the pinned dependencies safely

**New laptop only; not run during this preparation.** Install PyTorch from its CUDA wheel index first, then the direct dependencies:

```bash
python -m pip install 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124
python -m pip install --dry-run -r requirements.txt
python -m pip install -r requirements.txt
python -m pip check
python -m pip show torch transformers bitsandbytes accelerate streamlit opencv-python
```

Inspect the dry-run output before installing. Stop on unavailable pins or resolver conflicts; do not substitute latest torch/Transformers/bitsandbytes or a CPU torch build to make installation appear successful. No application feature needs torchvision, torchaudio or AutoAWQ. `quantize_model.py` is a separate legacy Qwen/AWQ experiment and is not a setup command for Moondream2. `nvidia-ml-py` supplies the legacy `pynvml` import.

After a successful tested installation, retain `pip check` and package-version evidence outside source control. The repository intentionally does not claim that a full transitive freeze has been tested.

## 5. Install FFmpeg and MediaMTX

In WSL on the target laptop:

```bash
sudo apt-get update
sudo apt-get install --yes ffmpeg
ffmpeg -version
ffprobe -version
```

For MediaMTX, obtain the official standalone Linux binary for the actual architecture (`uname -m`) from [MediaMTX installation instructions](https://mediamtx.org/docs/kickoff/install). Extract it to a separate installation directory such as `~/mediamtx`, preserving any existing installation. Set its actual executable path:

```bash
export MEDIAMTX_BIN="$HOME/mediamtx/mediamtx"
test -x "$MEDIAMTX_BIN"
"$MEDIAMTX_BIN" --version
```

This path is an example matching the helper's default; change it if installed elsewhere. The bundled YAML uses `rtspTransports`, documented in the [MediaMTX configuration reference](https://mediamtx.org/docs/references/configuration-file). Version [1.21.3](https://github.com/bluenviron/mediamtx/releases/tag/v1.21.3) is a documented configuration reference, not a version this project has run on the current host. Record and validate the installed version before declaring it compatible. No MediaMTX or FFmpeg binary is committed to the repository.

Optional helper overrides are `FFMPEG_BIN` and `FFPROBE_BIN` when those executables are outside PATH.

## 6. Supply four local video sources

Obtain permitted local MP4s and copy them into the fresh clone as:

```text
sample_videos/cam1.mp4
sample_videos/cam2.mp4
sample_videos/cam3.mp4
sample_videos/cam4.mp4
```

The empty directory may not exist in a clone because videos are Git-ignored; create it with `mkdir -p sample_videos`. Check each source:

```bash
for camera in cam1 cam2 cam3 cam4; do
  test -f "sample_videos/$camera.mp4" || break
  ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height \
    -of default=nw=1 "sample_videos/$camera.mp4"
done
```

Confirm all four files, not just the loop's final exit code. Startup preflight also refuses missing/unreadable sources. The existing demo files decode as H.264 and are byte-identical copies of the same animated scene; they simulate endpoints, not four independent warehouse scenes. The helper uses `-re -stream_loop -1` and stream copy for H.264; incompatible codecs retain its single-thread libx264 fallback. Ensure `ffmpeg -encoders` lists libx264 before using that fallback.

[config/cameras.json](config/cameras.json) expects `cam1` through `cam4` at `rtsp://127.0.0.1:8554/cam1` through `/cam4`, with `source_kind=demo_rtsp` and a three-second freshness threshold. Keep these endpoints for the managed local demo. For deliberate remote sources, use ignored `config/cameras.local.json` and set `VLM_EDGE_CAMERAS_CONFIG` to its absolute path before launch. Restart Streamlit after changing configuration. Do not hardcode a guessed WSL IP or commit credentials.

## 7. Verify CUDA on the new laptop

**GPU laptop only.** These checks initialize/query that laptop's GPU; they were not executed on the current laptop:

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('wheel CUDA:', torch.version.cuda)
assert torch.__version__ == '2.6.0+cu124', 'Unexpected PyTorch build'
assert torch.cuda.is_available(), 'CUDA is unavailable in this environment'
print('GPU:', torch.cuda.get_device_name(0))
print('total MiB:', torch.cuda.get_device_properties(0).total_memory / 2**20)
assert torch.cuda.is_bf16_supported(), 'Required BF16 support is not available'
PY
```

Stop if these fail. Check that the intended WSL distro/interpreter is active; driver maintenance belongs to the target laptop's setup process, not this application's UI. A working driver alone does not establish bitsandbytes/model compatibility.

## 8. Provision the exact model snapshot

The historical evidence and evaluation runner use commit:

```bash
export VLM_EDGE_MODEL_REVISION=5d6c926f44e26b07957b0dd315bbedcb4c17a5fe
```

The dashboard loader now accepts this optional environment default; explicit loader revisions still take precedence. It changes only snapshot selection: NF4, FP16 quantized compute, BF16 model dtype, skipped vision/region modules, local-files-only loading and caching/serialization are unchanged. Set the variable before launching; changing it requires restarting the process. Without it, legacy behavior uses cached `main`, which may not be the historically exercised snapshot.

Use an existing permitted Hugging Face cache containing this exact snapshot, or provision it deliberately **on the new laptop only**. Review the snapshot's model code and terms; the loader uses `trust_remote_code=True`. The optional command below downloads model files and is not an inference test:

```bash
# Run only when download is intended. Do not use offline flags during provisioning.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='vikhyatk/moondream2',
                         revision=os.environ['VLM_EDGE_MODEL_REVISION'])
print('Provisioned snapshot:', path)
PY
```

This follows the [Hub snapshot-download API](https://huggingface.co/docs/huggingface_hub/guides/download). Exact snapshot availability and provisioning on the new laptop remain unverified. Do not silently replace an unavailable snapshot with latest main. If using a custom cache location, set the same `HF_HOME` before provisioning and before every launch. Keep caches outside Git and do not copy virtual environments with them.

After provisioning, keep execution offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The application itself never downloads missing model files. A missing cache produces an actual error instead of a substitute answer.

## 9. Run non-GPU verification and preflight

From the project root in the activated target venv:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider -p tools.pytest_cpu_guard
python tools/demo_runtime.py check --dashboard
```

The guard rejects actual model/tokenizer loads and CUDA initialization during tests. Linux-only `fcntl` tests remain in the suite; Windows failures are reported, not hidden. Preflight starts no service and checks sources/tools/port ownership; it does not prove GPU readiness.

## 10. Launch MediaMTX, four publishers and Streamlit

Retain the exports for `VLM_EDGE_PYTHON`, `MEDIAMTX_BIN`, `VLM_EDGE_MODEL_REVISION` and offline cache settings in the launch terminal:

```bash
bash start_demo.sh
python tools/demo_runtime.py status
```

The helper starts MediaMTX using `config/mediamtx-demo.yml`, publishes four looping videos, waits for each RTSP video track and starts Streamlit on loopback port 8501. Open **http://localhost:8501** in Windows. Native links lead to `/monitoring`, `/assistant`, `/performance` and `/settings`; Overview is the root page.

Startup refuses unowned listeners on 8554/8501, reuses matching owned processes, and rolls back newly created processes on failure. Logs and process identity records are under ignored `.runtime/`. Status tracks owned processes, not the presence of changing pixels.

For a separate foreground dashboard, start only the streams first:

```bash
python tools/demo_runtime.py start
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Do not start both dashboard modes simultaneously. Keep one model-owning dashboard process on the 4 GiB target.

## 11. Verify feeds and submit real inference

With services running:

```bash
# TCP/RTSP/OpenCV diagnostics, no model load.
python tools/diagnose_cameras.py
# Requires changing decoded pixels across all four streams, no model without --ai.
python tools/verify_live_demo.py
```

In the browser:

1. Open Live Monitoring and click Start capture. Expect **4 of 4 cameras connected** only after fresh frames decode, with sequences/receipt ages updating.
2. Open AI Assistant, select a camera and enter a nonempty question. Ask AI must remain disabled if its frame is absent/stale.
3. Click Ask AI. First loading may be slow. Inspect the genuine answer, synchronized inference latency, receipt timestamp/age and exact submitted-frame evidence.
4. Change the camera/question and confirm an old answer is not presented for the new inputs. Navigate away/back and confirm camera/question selection persists without automatic inference.
5. Open Performance. Only completed/failed requests from the current browser session are measured; no accuracy/confidence score is implied.

Optional explicit real-GPU CLI check, only after the checks above and with other model processes stopped:

```bash
python tools/verify_live_demo.py --ai
```

It verifies streams before one actual latest-frame question and writes new timestamped evidence in `verification/`. Do not run it concurrently with a dashboard model on a 4 GiB GPU. The process-local inference lock does not coordinate separate programs. Older `tools/smoke_saved_qa.py` targets an earlier UI and is not the deployment acceptance command.

Record versions, reviewed source commit, model revision, four-feed evidence, exact submitted question/image, generated answer and memory/latency definitions. Historical 2420.71 MiB/latency results are not acceptance criteria or guaranteed new-laptop measurements. No warehouse accuracy has been established.

## 12. Stop services safely

Click Stop capture in Live Monitoring, then run:

```bash
bash stop_demo.sh
python tools/demo_runtime.py status
```

A separately launched foreground Streamlit process stops with Ctrl+C in its own terminal. The helper stops only processes it owns; it does not find/kill arbitrary external servers. Keep process records until stopping those services. There is no broad `pkill`/taskkill step.

## Troubleshooting

| Symptom | Check/action |
|---|---|
| Clone lacks the new menu | Confirm final source was published/transferred; compare the reviewed commit and `git status` without overwriting local work |
| Python/venv executable missing | Activate the clean Linux venv and export `VLM_EDGE_PYTHON`; do not reuse the copied Windows environment |
| Resolver cannot find a pinned package | Stop and investigate that exact pin/index; no automatic upgrades or CPU torch substitution |
| MediaMTX executable/config error | Check `MEDIAMTX_BIN`, executable permission, version and `.runtime/mediamtx.log` |
| 8554/8501 already used | Inspect/stop the actual owner yourself or use external services; the helper refuses to replace it |
| Capture not started | Publishers and capture are separate stages; click Start capture |
| Server unavailable | Start MediaMTX; verify port/routing from the same environment as capture |
| Publisher unavailable/404 | Inspect the matching FFmpeg log and path; restart only after fixing its source/server |
| Decoder/stale frame | Check codec, OpenCV environment, RTSP path and changing decoded frames; stale data is never accepted for inference |
| No CUDA / unsupported GPU | Validate the target laptop's driver/runtime/hardware separately; current-laptop driver repair is out of scope |
| Model cache unavailable | Set the intended `HF_HOME`/revision before startup; provision the exact snapshot on the new laptop |
| CUDA OOM | Close other model programs and use one dashboard process; do not alter NF4 settings blindly |
| Empty Performance page | Expected until this browser session completes or fails a real request |
| Linux-only tests fail on Windows | Use the intended WSL target; failures are retained and must not be presented as a passing full suite |

Keep `venv/`, `.venv/`, caches, large videos, runtime logs and credentials out of Git. They are covered by `.gitignore`, including the private camera override file. Review generated verification images/answers and media permissions before public staging.

## Current preparation results

- Focused sidebar/dashboard/camera/diagnostic/inference-mock suite: **94 passed**. Additional diagnostic/preflight tests: **22 passed**.
- Final full suite on native Windows: **165 passed, 3 failed, 35 setup errors**. All remaining failures/setup errors require Linux-only `fcntl` in existing evaluation/annotation paths. They were not skipped or suppressed.
- Every guarded test run recorded **zero real model/tokenizer loads and zero CUDA initialization attempts**.
- Actual browser: all five pages at 1440 x 1000 and 390 x 1000, zero sidebar radios/radiogroups, five native links, correct teal active item, no page exceptions or horizontal overflow. Native mobile expand/collapse and the expanded menu were inspected. Native navigation can collapse the menu after a mobile page selection; it remains available through the header toggle. Tab moved focus between links with a visible outline; Enter opened the focused page.
- Actual offline capture: four workers remained active across page changes, a nonempty question stayed blocked without a fresh frame, and Stop capture returned to zero active workers. Model-free diagnostics reported all four servers unavailable with null frame timestamps. The native Windows startup helper produced its expected WSL/Linux requirement message.
- No dependency installation, driver change, model download or GPU inference. No commit/push.

Changed for this preparation: `app.py`, five thin `src/pages/` entrypoints, `src/ui.py`, `src/views/overview_view.py`, `src/live_camera_ui.py`, `src/live_camera.py`, `src/camera_diagnostics.py`, `src/model_inference.py` (optional revision default only), `src/views/system_settings_view.py`, `tools/demo_runtime.py`, `test_dashboard_redesign.py`, `test_live_camera.py`, `test_model_inference.py`, `test_camera_diagnostics.py`, `test_demo_runtime.py`, `README.md` and this guide. `.gitignore` and dependency pins were inspected and retained.

Native navigation uses `st.navigation`/`st.Page` registration and keyboard-focusable `st.page_link` items placed after the brand. It does not create a radio widget or hide radio inputs with CSS. Thin entrypoints call existing views, preserving backend resource caching. The Overview shortcut uses `st.switch_page` after page registration.

Ignored local evidence is under `.runtime/`: `portable-focused.txt`, `portable-preflight.txt`, `portable-full-final.txt`, `native-sidebar-browser.json`, `native-sidebar-lifecycle.json`, `native-sidebar-*.png`, and `portable-diagnostics.json`.

Still required on the new laptop: clean pinned installation and `pip check`; working CUDA/BF16/bitsandbytes kernels; exact local snapshot availability; actual MediaMTX/FFmpeg startup and four changing RTSP feeds; real fresh-frame Moondream2 answers, model cache reuse, memory/latency measurements and stop/reconnect behavior. No new-laptop compatibility or end-to-end GPU success is claimed here.
