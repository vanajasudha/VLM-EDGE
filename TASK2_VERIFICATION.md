# Task 2 verification - synthetic fixtures only

Date: 29 October 2026. Status: Phases A-C and reusable Phase D implemented and CPU/mock-validated. Phase E protocol prepared, not experimentally executed. No genuine warehouse accuracy claim; real GPU evaluation remains pending genuine labels and explicit dataset/experiment approval.

## Final tests

Executed in the existing WSL Python venv:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest test_evaluation.py test_core.py test_model_inference.py test_app_qa.py test_camera_capture.py test_discovery_safety.py -q -p no:cacheprovider -p tools.pytest_cpu_guard

Result: 111 passed in 10.20 seconds.

CPU guard counters: model_load=0, tokenizer_load=0, cuda_initialization=0. The session-start import of installed PyTorch printed the existing pynvml deprecation warning. No model weights, real model/tokenizer factories or CUDA initialization were used. Backend contract/preflight tests replace the whole GPU stack with synthetic Python objects.

Verified coverage:

- Binary all-wrong and zero-denominator F1; invalid/ambiguous responses and three-column confusion matrix.
- Count exact match/MAE, all-invalid predictions, negative/fractional/ambiguous responses; equal-length and empty-data checks.
- Duplicate/missing/unknown CSV headers, malformed rows, required values, corrupt/missing images, duplicate IDs, invalid references and verification flags.
- Identical image bytes across different splits, including renamed copies; changed bytes after validation.
- Per-request inference/OOM/empty-output failures, continuation with errors, warm-up/load failure and cleanup failure reporting.
- Wrong/missing dataset/experiment hashes, empty reviewer, invalid/naive timestamp and false/nonboolean human-label attestations: refused before constructing a real backend.
- Sequential sample order, one mock load, verbatim question/path delivery, explicit native-query settings, pinned model/tokenizer revision and provenance outputs.
- Exclusive new run directories, unchanged sentinel artifacts, rejected overwrite of existing plan/summary and repeated-run preservation.
- Every mock JSON/JSONL object carries execution=mock and not_a_real_accuracy_result=true; mock run directories start mock_eval_. Extracted raw/prediction rows remain explicitly labelled.
- CLI validation and planning run with model/GPU imports forbidden.
- Existing saved-image dashboard, camera helper, loading/cache/concurrency and discovery-safety focused tests still pass.

Additional checks: Python source syntax passes; four historical benchmark CSVs retain eight rows each and their audited mean latency values; all 12 existing test images verify. No project experiments directory or real evaluation outputs were created. Original benchmark scripts, reports and saved images were not modified or executed. CPU/mock artifacts live only in pytest temporary directories.

## Files modified

| File | Change |
|---|---|
| src/benchmark_metrics.py | Fixed F1 and unequal-length defects; strict reference validation, conservative normalizers and explicit invalid-response policies/count denominators. |
| src/model_inference.py | Optional immutable revision on the existing cached loader; optional native query settings for experiments. Existing dashboard path/quantization/lock retained. |
| app.py | Existing upload scorer counts and discloses invalid binary predictions; appearance/layout/styles unchanged. |
| test_model_inference.py | Native generation-setting forwarding and pinned revision tests. |
| README.md | Links/commands for manifests, CPU tests and prepared benchmark protocol. |

## Files created

| File | Purpose |
|---|---|
| src/evaluation_manifest.py | CSV/image/reference validation, split-leakage checks and dataset identities. |
| src/evaluation_runner.py | Approval-bound, sequential quantized runner and injectable explicitly marked mock backend contract. |
| tools/evaluate.py | validate, plan and approved evaluate commands. |
| tools/pytest_cpu_guard.py | Reject real model/tokenizer loading and CUDA initialization during focused tests. |
| test_evaluation.py | Synthetic fixture/metric/manifest/runner/approval/output-preservation tests. |
| evaluation/manifest_template.csv | Blank schema; no invented labels. |
| evaluation/description_review_template.csv | Blank human-review worksheet. |
| EVALUATION.md | Schema, normalization/denominator policies, rubric, commands and operational limits. |
| BENCHMARK_PROTOCOL.md | Historical confounds and prepared matched-input/measurement procedure. |
| TASK2_VERIFICATION.md | This verification record. |

## Exact schema and validation

CSV header:

    sample_id,image_path,source_id,question,question_type,reference_answer,label_provenance,reference_verified,split,annotation_notes

Question types: binary/count/description. References: yes/no, nonnegative integer or human-verified prose respectively. reference_verified must be true; provenance must identify the genuine annotation process. source_id and annotation_notes are optional. Relative images resolve against the CSV parent unless --image-root is supplied. Full field rules are in EVALUATION.md.

Future human-created dataset validation:

    venv/bin/python tools/evaluate.py validate evaluation/warehouse_manifest.csv --image-root . --split test

Plan only (no model):

    venv/bin/python tools/evaluate.py plan evaluation/warehouse_manifest.csv --image-root . --split test --max-tokens 256 --temperature 0 --top-p 1 --seed 0 --warmup 1 --output evaluation/warehouse_plan.json

The warehouse manifest and approval file do not exist yet. The blank template is intentionally rejected as having no labelled samples. Do not turn synthetic fixture answers or historical model outputs into labels.

After explicit user approval only:

    venv/bin/python tools/evaluate.py evaluate evaluation/warehouse_manifest.csv --image-root . --split test --max-tokens 256 --temperature 0 --top-p 1 --seed 0 --warmup 1 --approval evaluation/warehouse_approval.json --output-root experiments

The approval JSON must bind dataset_sha256 and experiment_sha256 from the reviewed plan and include actual approved_by, timezone-aware approved_at and human_labels_confirmed=true. No approval record was manufactured or experiment run.

## Benchmark preparation and limitations

The historical baseline used fresh RTSP frames, unpinned revision, a different timing/inference context, decimal MB NVML readings and post-request power; quantized runs used saved frames, PyTorch cumulative peak MiB and before/after power. Only prompt wording matches by inspection. Historical speedup cannot be attributed solely to quantization.

New plans bind image bytes/questions/order, immutable revision, generation controls, seed, warm-up, quantization and source/version provenance. Native query settings avoid the cached wrapper's silently ignored generation-token argument. Latency uses the shared CUDA-synchronized helper with decode/load/queue excluded. Future paired measurement definitions and repeats are documented.

No FP16 backend/load was added. No paired GPU experiments, memory sampler or power integration were executed. The optional native-query experiment path is mock-tested and inspected against cached model code, not GPU-validated in Task 2. Stop other model processes before future execution; the runner's lock does not control unrelated applications. Numeric reproducibility is not guaranteed across hardware/CUDA versions.

Description predictions/references are preserved for human rubric review with no fabricated semantic score. Normalization is intentionally conservative and can reject conversational but semantically correct answers; invalid rate is always visible. Human verification/provenance are attestations, and exact-hash split checking does not detect near-duplicate scenes.

Next step: select genuine warehouse images, prepare/adjudicate human reference answers and freeze the manifest/splits, then review the dataset and experiment plan with the user. Stop before real GPU evaluation.
