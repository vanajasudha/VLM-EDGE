# Task 1 verification - saved-image question answering

Status: COMPLETE. Verified on 29 October 2026, WSL2 Ubuntu, existing Python 3.11 venv and RTX 3050 Laptop GPU (4 GiB). No UI redesign, RTSP implementation or telemetry implementation.

## Behavior

The Ask page validates the saved image, enables Ask for a valid image and nonempty question, and submits the selected path and exact question text. A new generation shows its answer, image/question and synchronized inference latency, excluding model load and lock queue wait. The uncertainty warning remains visible.

Results are session-specific and bound to image path, image byte fingerprint and question. Input changes, corrupt/missing images, and failed retries clear previous results. Normal reruns do not generate answers automatically; repeated Ask clicks make new requests. Model load/inference errors and CUDA OOM display without crashing the page. Empty model responses are treated as failures.

The existing st.cache_resource loader and process-local inference lock are retained. Quantization remains NF4 4-bit, FP16 quantized compute, BF16 model loading, device_map=auto, with vision/region skipped. Loading now requires existing local cached files, preventing implicit downloads.

## Changed files

| File | Change |
|---|---|
| app.py | Connected Ask page, input/result association, measured latency, errors/uncertainty, truthful overview Q&A status. |
| src/model_inference.py | Image validation/fingerprint, verbatim questions, cached-files-only loading, explicit loading OOM and empty-answer failures. Existing cache/lock/configuration retained. |
| pytest.ini | Excludes both virtual environments from discovery. |
| test_moondream.py | Preserved legacy script body under an explicit main entrypoint. |
| test_quantized_moondream.py | Preserved legacy script body under an explicit main entrypoint. |
| test_quantized_moondream_bf16.py | Preserved legacy script body under an explicit main entrypoint. |
| test_model_inference.py | Image identity, question freshness, validation, errors/OOM, empty replies, cache/configuration/retry and concurrent serialization tests. |
| test_app_qa.py | Actual page submission, button validity, stale-answer clearing, failed retry/recovery, changed image bytes, repeat submission and empty-gallery tests. |
| test_discovery_safety.py | Isolated import check rejects any attempted model-dependency imports by legacy GPU scripts. |
| tools/smoke_saved_qa.py | Explicit offline, single-model GPU check through dashboard AppTest with unique JSON output. |
| README.md | Q&A behavior, safe test/GPU commands, local-cache requirement; corrected deferred live-camera claims. |
| TASK1_VERIFICATION.md | This verification record. |
| qa_gpu_smoke_20261029T162400437910Z.json | New real GPU evidence; existing reports were not overwritten. |

## Automated results

After guarding the legacy GPU scripts, normal discovery was executed:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest -q -p no:cacheprovider

39 passed, 1 warning in 10.65 seconds. Warning: the installed pynvml package is deprecated; packages were not replaced.

Mock tests validate one cached load across concurrent callers, exact quantization arguments, serialized inference, selected image pixels, verbatim questions and new calls, invalid images, loading failure/retry, inference exceptions/OOM, empty output and stale-answer removal.

All five dashboard routes executed in Streamlit AppTest without exceptions. Python syntax checks passed. A temporary headless Streamlit server returned HTTP 200 ok and was stopped afterward.

## Real GPU smoke

Command: venv/bin/python tools/smoke_saved_qa.py

The tool forces HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Existing model snapshot: 5d6c926f44e26b07957b0dd315bbedcb4c17a5fe. A second model load is guarded against. CUDA preflight reported 3303.4 MiB free after importing/initializing the model stack.

Two sequential submissions through the actual app completed:

| Image | Submitted question | Genuine generated answer | Inference latency |
|---|---|---|---:|
| cam1_0 | What is happening in this scene? | The scene features a large, majestic tree situated on a grassy hill or mound filled with green grass and moss. It has a hollow in its center, serving as a natural shelter or den for the tree, its roots spread out and anchoring it to the ground. Around the tree, there is a variety of rocks and stones scattered out the grassy area, adding to the natural and picturesque setting. | 10179.8 ms |
| cam2_1 | Is there a tree in this image? | Yes, there is a large tree in the image. | 1007.6 ms |

Confirmed: exactly one model load, cache reuse, 97 four-bit layers, zero quantized vision/region layers, device map {'model': 0}, actual CUDA parameters, peak PyTorch allocated memory 2420.71 MiB. The first request was not warmed up; these values are smoke-test evidence, not new comparative benchmarks or accuracy scores.

The smoke process exited. A subsequent nvidia-smi reading showed 0 MiB used and 3964 MiB free; no smoke model/server remained running.

## Preservation and remaining limits

SHA-256 checks verified all 22 protected pre-existing CSVs, reports, saved images and the original audit unchanged. No packages installed or weights downloaded. Historical benchmark scripts/results, model architecture/settings and images were preserved. Legacy GPU script bodies were only guarded for safe import.

- Real model behavior was exercised through Streamlit AppTest; a human browser interaction/rendering review was not performed.
- OOM and failure recovery were tested with mocks, not by exhausting the 4 GiB GPU.
- The lock coordinates requests within one process; separate dashboard processes can still allocate separate models. Run one dashboard process.
- Local-cache loading deliberately fails if required files are absent. It does not download replacements.
- Inference remains synchronous, without cancellation or queue deadlines.
- Accuracy dataset/metric defects, benchmark comparability, camera integration and telemetry remain deferred to Tasks 2 and 3.
