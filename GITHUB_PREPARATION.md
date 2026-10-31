# GitHub preparation audit - 31 October 2026

## Scope and files changed

Prepared the actual Windows checkout supplied to this session. No original WSL working tree was modified. Modified `README.md` and `requirements.txt`; created `.gitignore`, this audit and an empty Git repository on branch `main`. Application source, model settings, datasets and historical results were not edited. No files were deleted, staged, committed or pushed. No remote was added.

The README covers overview/motivation, problem/objectives, implemented components, Mermaid architecture, stack, Moondream2, selective NF4, actual structure, prerequisites, setup/dependencies, startup/stream commands, Q&A flow, evaluation, historical evidence, current status, limitations, future work and academic attribution.

## Current application and data blockers

- `src/live_camera_ui.py` is zero bytes. `app.py` imports its nonexistent `render_live_panel`, so the current entrypoint cannot launch successfully even with dependencies repaired. This is documented rather than repaired because application development is out of scope.
- `src/views/` retains six older dashboard views; `app.py` no longer uses that routing. Older documentation/test counts do not prove this version works.
- WSL `venv/` is a copied Python 3.11 environment with an original-machine interpreter path. Windows `.venv/` references an unavailable original-machine Python executable. Neither is portable or usable as-is here.
- Available WSL reports Python 3.14.4 and no original `/home/*/vlm-edge/venv/bin/python` was found. Host Windows Python is 3.14. The original GPU environment could not be executed.
- All 12 `test_frames/` JPEGs have identical SHA-256 bytes. All four videos also have identical bytes. These are not 12 independent images or four distinct camera scenes.
- Pilot sources/labels are empty. No real warehouse accuracy/counting experiment or training/fine-tuning is established.

## Dependencies

The old manifest listed only Streamlit, pandas, pytest and OpenCV without versions. The replacement lists direct runtime/test dependencies from imports, pinned to metadata read from copied `venv/lib/python3.11/site-packages`:

| Distribution | Copied version |
|---|---|
| torch | 2.6.0+cu124 |
| transformers | 4.51.3 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |
| einops | 0.8.2 |
| streamlit | 1.66.1 |
| pandas | 3.0.6 |
| numpy | 2.4.6 |
| Pillow | 12.3.0 |
| opencv-python | 5.0.0.93 |
| altair | 6.3.0 |
| nvidia-ml-py | 13.615.71 |
| pytest | 9.2.0 |

These are metadata observations, not fresh installation validation or a transitive lockfile. The copied environment additionally has deprecated `pynvml==13.0.1`; the manifest uses `nvidia-ml-py` for the same import. This distribution choice was not verified by running historical NVML scripts. Do not uninstall/upgrade anything in a working AI environment based on this audit. The baseline script's older Transformers 4.46.3 comment differs from the later cached inference stack; do not interpret the current requirements as a validated environment for every legacy experiment.

The separate `quantize_model.py` targets Qwen2-VL/AWQ, not Moondream2. Its `awq` dependency is intentionally excluded from the primary environment. MediaMTX/FFmpeg, GPU drivers and local weights are external requirements. Clean install, package-index availability and exact system-tool versions remain unverified.

No critical AI packages were installed or changed. To run CPU checks, pytest 9.2.0 and its small dependencies were installed only under ignored `.runtime/audit-deps/`, then exposed via process-local PYTHONPATH. pip reported existing host Streamlit incompatibilities with Altair/pyarrow and the audit-only packaging version. This is not an application environment and must not be committed or used for a GPU reproducibility claim.

## Verification

- All eligible Python source files parsed successfully with `ast.parse`; this is syntax checking, not execution.
- 29 existing historical artifact hashes in `verification/live_preservation_baseline.json` matched; no protected historical artifacts were changed.
- README relative links and repository paths checked; startup commands traced against scripts/configuration. No FFmpeg publisher, MediaMTX server or model benchmark was launched.
- Full existing pytest collection with offline flags and `tools.pytest_cpu_guard` stopped on four collection errors in `test_app_qa.py`, `test_dashboard_redesign.py`, `test_live_camera.py`, `test_model_inference.py`. Host Streamlit/protobuf raised `TypeError: Metaclasses with custom tp_new are not supported` under Python 3.14. CPU guard: zero real model/tokenizer load and CUDA initialization attempts.
- Focused run of `test_core.py test_camera_capture.py test_discovery_safety.py test_evaluation.py test_warehouse_annotation.py`: **84 passed, 3 failed, 35 errors**. Three evaluation tests hit the same host Streamlit/protobuf error; 35 annotation setups failed because Linux-only `fcntl` is unavailable on native Windows. CPU guard again recorded zero real model/tokenizer loads and CUDA initialization attempts. These results do not establish failure on the intended WSL Python 3.11 stack or a clean test suite for this checkout.

## Security, size and redistribution review

A heuristic scan of Git-eligible text found no recognized provider tokens, private-key headers or literal password/API-key assignments. This is not a guarantee against all secret formats. The annotation tool generates local CSRF tokens dynamically; no token value is reported here.

`quantization_diagnostic.txt` contains an original user's absolute home path on lines 1 and 3. Leave it out of a public first commit pending deliberate review/redaction. Copied environment metadata also contains personal paths but is ignored. No source-level hardcoded personal filesystem path was identified by this scan.

No Git-eligible file exceeds 10 MiB. Each excluded local video is 991,017 bytes: videos are excluded by policy even though individually below GitHub's large-file threshold. No model weights were identified outside the ignored environments. The local CUDA keyring `.deb` is ignored and retained. Virtual environments, caches and generated logs are retained locally and excluded.

No LICENSE or authoritative redistribution evidence for the sample imagery was found. Do not invent a licence grant. `test_frames/`, the submitted verification PNG and any future dataset require rights review. Historical source/quality records may reveal filenames or identifiers; review their content before publication. Excluding videos does not resolve rights for still images extracted from them. Model weights/code and third-party executables have their own terms.

## Git state and ignore rules

Git was absent; `git init -b main` created it safely. There are no tracked files, commits, staged changes or remotes. All eligible files are untracked; `git diff` and `git diff --cached` are therefore empty and cannot represent the new file contents. Review untracked content and use `git add --dry-run` for the proposed first staging set. `git diff --check` alone cannot validate untracked contents.

`.gitignore` excludes both environments, Python/test/type/lint caches, secrets/.env/credentials/private keys, downloaded weights/model caches, videos, runtime state/logs, experiments/outputs, backups, installer packages and IDE/OS files. Private pilot image/permission/annotation/export contents are ignored while guides and `.gitkeep` scaffolds remain eligible. Existing benchmark CSVs and blank evaluation fixtures are intentionally eligible. Historical evidence JSON/TXT is not blanket-ignored. No tracked files were removed (none existed).

## Proposed first commit and exact next commands

The following conservative first commit includes source, tests, configuration, documentation and empty dataset scaffolding. It includes historical text/JSON evidence while omitting demo imagery and `quantization_diagnostic.txt` pending rights/privacy review; consequently it is not a complete working demo or passing-test release. Tests depending on omitted images cannot pass in that source-only clone. Add permitted test media separately after review. Resolve the missing UI and review the candidate contents before calling the repository demo-ready.

Run from this checkout in PowerShell. The sandbox created `.git` under a different Windows identity; elevated Git detected ownership mismatch. Commands below use a one-command `safe.directory` exception for this known checkout without changing global Git configuration. The normal sandbox cannot obtain the index lock for a dry run, so the 88-file preview was run with elevated access and left the index empty. This dry run was checked during preparation; the actual staging/commit/push commands below have NOT been run:

```powershell
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge add --dry-run -- .gitignore '*.md' requirements.txt pytest.ini app.py src tools config .streamlit data evaluation start_demo.sh stop_demo.sh 'test_*.py' benchmark_baseline.py benchmark_quant.py capture_frames.py telemetry.py quantize_model.py zero_copy_bench.cu '*results*.csv' '*report*.txt' final_comparison.txt 'qa_gpu_smoke_*.json' 'verification/*.json'
```

Only after owner approval and resolution/review of the blockers:

```powershell
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge add -- .gitignore '*.md' requirements.txt pytest.ini app.py src tools config .streamlit data evaluation start_demo.sh stop_demo.sh 'test_*.py' benchmark_baseline.py benchmark_quant.py capture_frames.py telemetry.py quantize_model.py zero_copy_bench.cu '*results*.csv' '*report*.txt' final_comparison.txt 'qa_gpu_smoke_*.json' 'verification/*.json'
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge diff --cached --stat
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge diff --cached --check
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge diff --cached
# Stop and review the exact staged contents before committing.
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge commit -m "Prepare VLM-Edge source and documentation"
# Replace YOUR_ACCOUNT with your account and create the empty GitHub repository first.
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge remote add origin https://github.com/YOUR_ACCOUNT/vlm-edge.git
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge remote -v
git -c safe.directory=C:/Users/pujar/OneDrive/Documents/vlm-edge/vlm-edge push -u origin main
```

Do not run `git remote add` if an origin was subsequently configured; inspect it instead. Do not use `git add .` until all remaining evidence/media/privacy issues have been reviewed. No commit or push authorization is implied by this document.
