# Evaluation manifests and execution

Implementation is CPU/mock-validated. No warehouse accuracy has been measured. The only supplied dataset is a blank template. Synthetic unit-test fixtures are illustrative code tests, never human-verified project labels.

## Manifest format

UTF-8 CSV, one image/question/reference per row. Copy evaluation/manifest_template.csv to a new manifest before annotating. Its exact header is:

    sample_id,image_path,source_id,question,question_type,reference_answer,label_provenance,reference_verified,split,annotation_notes

| Field | Required value |
|---|---|
| sample_id | Unique nonempty identifier in this manifest. |
| image_path | Existing decodable image. Relative to the CSV directory by default, or to --image-root when supplied. Absolute paths are accepted. |
| source_id | Optional camera/source ID; empty if unavailable. |
| question | Nonempty original question; passed verbatim, without automatic prompt additions. Ask binary/count questions to request only yes/no or a nonnegative integer. Freeze wording before evaluation. |
| question_type | binary, count, or description. |
| reference_answer | Human-verified answer: yes/no for binary; a nonnegative decimal integer for count; nonempty reference prose for description. |
| label_provenance | Nonempty audit trail: actual annotator/reviewer IDs, annotation date, method and adjudication information. Historical model answers are not labels. |
| reference_verified | true only after a human has inspected the image/question and verified the reference. Other values are rejected. |
| split | train, validation, or test. |
| annotation_notes | Optional ambiguity, occlusion, counting rules, exclusions and adjudication notes. |

Required columns must exist. source_id and annotation_notes columns may be omitted; empty cells are supported for those two only. Unknown/duplicate headers, malformed rows, duplicate IDs, missing/invalid images, empty required fields and invalid reference types are rejected. An empty template is intentionally unevaluable.

A verification flag and provenance text are attestations, not machine proof that a label is correct. Do not mark placeholders or synthetic examples as genuine human labels. No illustrative labelled CSV is shipped.

Validation hashes actual image bytes and decodes images. Identical image bytes cannot appear in different splits, even with different names. This does not detect near-duplicate video frames; group related frames, scenes, cameras and capture sessions during dataset design to prevent leakage. Multiple questions on the same image within one split are permitted and must not be treated as independent scenes.

Dataset identity includes ordered sample metadata and image hashes. Exact manifest bytes have a separate hash. Moving the dataset does not change identity when relative image paths and bytes remain unchanged. Reordering samples, changing references/provenance or replacing an image changes identity. Each image is checked again before inference.

## Annotation and description review

Choose genuine warehouse images with suitable permissions, representative conditions and clear question definitions. Freeze inclusion criteria, sources, label rules and splits before looking at model predictions. Review labels independently where possible, adjudicate disagreements and document ambiguous/unanswerable images rather than inventing a reference.

For descriptions, preserve the raw answer and reference. Use evaluation/description_review_template.csv for a separately saved human review; review is not automated. Reviewers should be blind to model variant when comparing variants.

| Review dimension | Anchors |
|---|---|
| factual_correctness | 0: material contradiction; 1: mixture of supported and incorrect/unverifiable statements; 2: all factual claims supported by visible evidence. |
| coverage | 0: misses essential content requested; 1: partially addresses it; 2: covers the predefined key entities/actions/relationships. |
| hallucination_severity | none; minor (noncritical unsupported detail); major (invented object/action/relationship affecting the answer). Cite evidence in notes. |
| uncertainty_appropriate | yes, no, or not_applicable; assess whether visibility/occlusion warrants qualification. |
| review_notes | Identify claims, image evidence, disagreements and exclusions; do not silently edit model outputs. |

Record reviewer identity/time and adjudicator. Keep dimension ratings separate. This rubric supplies no semantic accuracy score; future human-review aggregation requires an agreed protocol and genuine reviews.

## Metric policies (version 1)

Binary normalization accepts only an entire yes/no token (case-insensitive, whitespace trimmed), optionally followed by one final ., ! or ?. Extra prose, ambiguity, both labels, missing answers and inference errors are invalid. For example, a conversational sentence starting with Yes is conservatively invalid; this is deliberate, not a semantic judgement. Report the invalid rate alongside accuracy.

Binary accuracy uses all samples: invalid responses count as incorrect. Confusion matrix rows are truth yes/no and columns predicted yes/no/invalid. Precision = TP/(TP+FP). Recall includes invalid positive references as missed positives. F1 = 2TP/(2TP+FP+FN+invalid_positive). Zero denominators use 0 and are explicitly recorded. Thus all-wrong binary predictions produce F1=0, never a misleading NaN. Invalid negatives remain separately visible; they do not become false positives.

Counting accepts only nonnegative integer digits; it rejects floats, ranges, number words, negative numbers and prose. Exact-match accuracy uses all samples; invalid answers count as wrong. MAE uses valid predictions only, reports its valid denominator, and is null if none are valid. Always interpret MAE with invalid-response rate.

Both metric helpers reject empty data and mismatched reference/prediction lengths. Invalid references raise errors. Strict helper mode rejects invalid predictions; runner mode counts them explicitly. Description outputs have no automated semantic score, and aggregate semantic_accuracy is null.

## Commands (validation does not load a model)

Run from the project root in the existing WSL venv:

    venv/bin/python tools/evaluate.py validate evaluation/warehouse_manifest.csv --image-root . --split test

Here evaluation/warehouse_manifest.csv is a future human-created file, not an existing labelled dataset. --image-root . means CSV image paths are relative to the project root. Omit this argument to resolve them relative to the CSV directory.

The blank template is expected to fail:

    venv/bin/python tools/evaluate.py validate evaluation/manifest_template.csv

Prepare a reproducible plan without loading a model:

    venv/bin/python tools/evaluate.py plan evaluation/warehouse_manifest.csv --image-root . --split test --max-tokens 256 --temperature 0 --top-p 1 --seed 0 --warmup 1 --output evaluation/warehouse_plan.json

Existing plan files are never overwritten. Default immutable model revision: 5d6c926f44e26b07957b0dd315bbedcb4c17a5fe, the cached Task 1 revision. Other revisions must be explicit 40-character commits and locally available.

Stop here until the user approves the genuine dataset and experiment. No approval record has been created by this implementation.

After actual approval, a separately created approval JSON must contain dataset_sha256 and experiment_sha256 copied from the reviewed plan, approved_by, approved_at (ISO-8601 timestamp with timezone), and human_labels_confirmed set to JSON true. A changed dataset, split, order, revision or generation setting invalidates approval. This is an auditable local approval record, not a cryptographic signature or identity authentication mechanism.

Future approved execution only:

    venv/bin/python tools/evaluate.py evaluate evaluation/warehouse_manifest.csv --image-root . --split test --max-tokens 256 --temperature 0 --top-p 1 --seed 0 --warmup 1 --approval evaluation/warehouse_approval.json --output-root experiments

No command was run against a real labelled dataset or GPU in Task 2.

## Runner behavior and outputs

Validation/planning import no torch, transformers, bitsandbytes or Streamlit. Real execution validates approval before constructing the GPU backend. The backend forces offline mode, requires at least 3 GiB free CUDA memory, takes an exclusive evaluation-process lock, loads the quantized model once, and processes requests sequentially. Stop other dashboard/model processes before execution: the evaluation lock cannot control unrelated programs.

Quantization is unchanged: NF4 four-bit, FP16 quantized compute, BF16 model dtype, vision/region excluded, device_map=auto. The optional experiment path uses encode_image then the cached revision's native query(settings=...), because its legacy answer_question wrapper silently ignores generation limits. Dashboard Q&A retains its existing default path. Explicit runner settings are token cap, temperature, top-p, seed and warm-up count; they are verified by mocks, not a new real-GPU run.

One warm-up on the first selected sample is the default and excluded from metrics. Warm-up/load failure marks the run failed rather than claiming accuracy. Per-request failures are recorded and count as invalid predictions, with completed_with_errors status. CLI exit codes: 0 complete, 1 failed/completed_with_errors, 2 validation/approval/input refusal.

New timestamp/UUID run directories contain exclusively created files:

- metadata.json: immutable initialization record, plan, approval, versions and source hashes.
- metadata_final.json: final status, hardware/loaded revision/tokenizer fingerprint and failures.
- manifest_snapshot.json: immutable sample/reference/provenance/image-hash snapshot.
- warmup.jsonl: explicitly excluded warm-up outputs.
- raw_answers.jsonl: exact answer/reference, sample metadata, latency, attempt elapsed time and errors.
- predictions.jsonl: normalized predictions separately, with normalization status and references.
- summary.json: task-appropriate aggregates and denominators.

Mock directories start mock_eval_; every JSON/JSONL record identifies execution=mock and not_a_real_accuracy_result=true. Copying an individual row preserves this distinction. Unit tests store these artifacts only in temporary directories. An interrupted run may retain only initial metadata and flushed rows; incomplete runs must not be interpreted as complete experiments.

Inference latency is CUDA-synchronized image encoding plus generation, excluding model loading, image decoding and lock wait. Attempt elapsed time also records failed requests but is not substituted for inference latency. No memory/power/energy values are fabricated. Evaluation memory/power are explicitly not measured.

Software, exact pinned revision, quantization, generation settings, dataset content identity, source hashes and observed backend properties are recorded. Seeding and greedy default decoding improve repeatability but do not guarantee bitwise identity across CUDA/hardware/library changes. Raw outputs and errors remain the evidence.

## CPU-only verification

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest test_evaluation.py test_core.py test_model_inference.py test_app_qa.py test_camera_capture.py test_discovery_safety.py -q -p no:cacheprovider -p tools.pytest_cpu_guard

The guard rejects actual model/tokenizer loading and CUDA initialization; fake factories remain usable. The synthetic fixture explicitly states it is not human project ground truth.
