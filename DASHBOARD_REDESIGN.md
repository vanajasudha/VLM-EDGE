# Dashboard redesign verification

Date: 31 October 2026. Scope: local Streamlit UI; no deployment, commit or push.

## Implemented pages

- **Overview:** four camera cards, actual connected count and a callback-based Assistant shortcut.
- **Live Monitoring:** two-column camera grid, explicit shared capture start/stop, accurate connection states, offline placeholders and 1-second fragment refresh.
- **AI Assistant:** live camera selection, letterboxed recent preview, question/Ask AI, actual latest-frame backend, loading/error states, exact submitted visual evidence and source/timestamp/latency metadata. A changed camera/question, stale/disconnected source, new connection or failed retry clears the active result.
- **Performance:** successful/failed session request counts, last inference latency, up to 100 recent measured requests and a chart only with multiple measurements. GPU allocation is shown only when read from an already initialized runtime; the process-lifetime peak is explicitly distinguished from per-request memory.
- **Settings:** read-only camera sources/status, freshness threshold, actual configured model/quantization and available GPU information. URL userinfo/query/fragment and arbitrary backend paths are omitted from displayed diagnostics. No invented GPU model, capacity, CUDA version, confidence or accuracy.

Navigation uses a persistent native sidebar radio. The Overview action updates its state in a callback, before widget construction on the next rerun. The small centralized stylesheet keeps the native header/sidebar toggle and uses native containers/columns rather than positioned cards, negative margins, external fonts or generated class selectors. At narrow widths columns stack; previews use a consistent 16:9 letterbox without stretching/cropping.

## Files changed

`app.py`; `src/live_camera_ui.py`; `src/ui.py`; `src/views/overview_view.py`; `src/views/live_monitoring_view.py`; `src/views/performance_view.py`; `src/views/system_settings_view.py`; `pytest.ini`; `test_app_qa.py`; `test_live_camera.py`; `test_dashboard_redesign.py`; `README.md`; this record.

The previously empty live UI module is implemented. Existing inference, capture, camera configuration, model settings, cache, locking, startup scripts, datasets and dependency manifests are unchanged. Saved-image Q&A and evaluation implementations remain available in source; they are not extra pages in the requested five-page navigation. Saved-image regression tests use a standalone view harness. Navigation assertions were updated to the current five-page workflow. No test/project files were deleted and no dependencies were added.

## Automated checks

The copied Windows `.venv` launcher references another machine, but its Python 3.14 site-packages can be imported by the available Windows Python. Tests used process-local `PYTHONPATH=.venv/Lib/site-packages`, offline Hugging Face flags, disabled pytest plugin auto-discovery, and the existing `tools.pytest_cpu_guard`. Nothing was installed or upgraded during this redesign.

A first collection entered ignored audit-package tests under `.runtime/`; pytest now excludes that runtime directory. A subsequent run hit inherited temp-directory permissions; later runs used a new workspace-local `--basetemp` without deleting previous temp files.

Final focused command (PowerShell):

```powershell
$env:PYTHONPATH = "$PWD\.venv\Lib\site-packages"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest test_dashboard_redesign.py test_live_camera.py test_app_qa.py test_model_inference.py test_camera_capture.py test_core.py test_discovery_safety.py -q -p no:cacheprovider -p tools.pytest_cpu_guard --basetemp=.runtime/ui-test-temp-2
```

**66 passed.** After replacing deprecated width arguments in the new views, the eight dashboard regression tests were rerun: **8 passed**, again with zero guard attempts. Coverage includes every navigation route and roundtrip, Overview callback, offline camera grid, explicit controls, real-session measurement bookkeeping, failure handling, reconnection invalidation, secret redaction, live selection/submission with mocked model, selected image bytes/question, stale-frame rejection, locking/cache behavior and existing capture tests. Guard counters: zero real model/tokenizer loads and zero CUDA initialization attempts.

**Full suite: 133 passed, 3 failed, 35 setup errors.** All three failures are evaluation-backend tests importing Linux-only `fcntl`; all 35 setup errors are annotation tests requiring the same module. Native Windows does not provide it. These existing Linux paths were not changed or skipped to accommodate this verification host. Full-suite guard counters remained zero for model/tokenizer loads and CUDA initialization. All 29 historical preservation hashes also matched.

## Browser and application verification

Started the actual application with the available interpreter and copied site-packages on loopback port 8502. `/_stcore/health` returned HTTP 200 `ok`. A hidden headless Chrome instance was controlled over its local DevTools socket, without adding application packages.

Inspected Overview, Live Monitoring, AI Assistant, Performance and Settings at **1440 x 1000** and **390 x 1000**. All ten navigations reached the expected heading, with no Streamlit exceptions. DOM checks found document/main scroll width equal to viewport/main width and no content extending past the viewport. Reviewed screenshots of desktop Overview/Assistant and narrow Overview/Settings. Corrected the initial mobile heading collision, oversized Assistant preview, broken icon separators and narrow Settings table clipping; the final Settings view uses wrapping cards instead of a table.

Ignored local evidence:

- `.runtime/dashboard-browser-check.json`: ten route/width checks.
- `.runtime/dashboard-1440-*.png`, `.runtime/dashboard-390-*.png`: screenshots of each page.
- `.runtime/dashboard-ui-tests.txt`: focused test output.
- `.runtime/dashboard-full-final.txt`: full suite output.

Real browser checks used offline camera sources, so no fabricated preview/answer/performance data was injected. Mock tests separately exercised successful and failed submissions. No RTSP publisher, model download, GPU benchmark or real model inference was started. The temporary verification browser and Streamlit server were stopped afterward. Preview/navigation do not load the model; capture starts only on explicit action and retains the process-cached manager. Existing manager tests cover worker reuse rather than spawning new workers on reruns.

## Remaining limits

- Verify real four-stream operation and question submission on the intended WSL Python 3.11/CUDA host. Historical GPU/live records are not current redesign verification.
- No current GPU-memory measurement was obtained on this verification host. The Settings empty state is intentional.
- Browser verification covered offline layouts at two widths, not every browser/device or long real model output. Answer text wraps and exact submitted evidence is expandable.
- The inference lock and capture/model caches remain process-local. One dashboard process is recommended on the 4 GiB target GPU.
- Performance history is session-local and capped at 100 rows; completed/error counters persist for the current session. It is not durable experiment telemetry.
- Existing media-rights, dataset and environment reproducibility concerns in the earlier repository audit remain.

Cleanup note: disposable browser profiles, screenshots, temporary helper scripts and pytest directories were subsequently removed. The verification findings and retained JSON/test-log records remain valid.
