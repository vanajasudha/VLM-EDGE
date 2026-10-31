# Task 3 - warehouse dataset preparation

Date: 29 October 2026. Annotation workflow and CPU validation are ready for human review. The genuine dataset remains empty; image collection and human labelling are outstanding.

## Actual available sources and counts

| Item | Available / verified |
|---|---:|
| Permitted genuine warehouse images registered | 0 |
| Warehouse images with genuinely human-verified labels | 0 |
| Verified binary/count/description pairs | 0 / 0 / 0 |
| Draft annotations in the actual pilot | 0 |
| Exported manifests | 0 |
| Target pilot, subject to quality/availability | about 40 distinct images / 120 pairs |

The repository has 12 saved JPEGs, all with one identical SHA-256. Visual inspection shows a stylized forest scene, not warehouse imagery. Four sample videos also share one identical SHA-256. No warehouse source/permission records were found. This demo media remains untouched and was not registered as genuine data.

No arbitrary download, synthetic warehouse generation, fabricated source/reviewer/permission record, model-generated ground truth, experiment approval or real GPU evaluation was created.

## Directory structure

    data/warehouse_pilot/
      images/                 empty; permitted registered image copies will go here
      sources.csv             exact source-ledger header only, no invented rows
      annotations/            local lock; events.jsonl starts after a human save
      permissions/            instructions only; no permission evidence yet
      exports/                empty; future validated verified-only manifests
      quality/
        initial_quality_2026-10-29.json
        source_inventory_2026-10-29.json
        historical_artifact_hashes_baseline.json
      ANNOTATION_GUIDE.md
      README.md

Directory placeholders contain no images or labels.

## Workflow

Read data/warehouse_pilot/ANNOTATION_GUIDE.md. The guide defines visible binary questions, visible_instances_v1 counting scope, descriptions and a human rubric, ambiguity/drafts/exclusions, actual identities/provenance and independent review preferences.

Source registration requires a human-supplied image, actual source/rights/evidence/reviewer information, globally meaningful scene/session IDs and explicit permission/genuine-warehouse confirmations. Originals are not modified. No import was performed during preparation.

The annotation helper is separate from the main dashboard:

    venv/bin/python tools/annotate_warehouse.py serve --port 8766

Open http://127.0.0.1:8766 manually. A registered image can be previewed, then a human enters question/type/reference/notes and genuine identities. Verification is unchecked by default and requires explicit confirmation. Counts require a defined rule and target/scope notes. Save creates a draft unless verified. The local server is not left running after tests.

Revisions append events rather than replacing history. Latest drafts revoke earlier verification for future exports. Image/source changes require reverification. Exclusion retains files/history. Independent review is preferred; self-review is disclosed, not concealed with fictitious identities.

All questions on an image inherit its registered split. Scene/session groups remain together. A roughly 60/20/20 image allocation is guidance only when enough independent groups exist; otherwise a test-only pilot is honest. No artificial grouping or split assignment was invented.

Quality and export:

    venv/bin/python tools/annotate_warehouse.py quality
    venv/bin/python tools/annotate_warehouse.py export

The actual empty pilot is NOT_READY. Export correctly refuses: no permitted warehouse images and no human-verified labels. No empty dataset was represented as validated.

After actual human registration/labelling, resolve QC errors and review warnings, then export to a fresh filename. Export invokes the existing Task 2 validator without changing its schema/rules:

    sample_id,image_path,source_id,question,question_type,reference_answer,label_provenance,reference_verified,split,annotation_notes

Validate a future exported file:

    venv/bin/python tools/evaluate.py validate data/warehouse_pilot/exports/ACTUAL_EXPORTED_FILE.csv --split test

Export paths are manifest-relative; no --image-root is needed. Choose the actual populated split if not test.

## Quality checks

- Existing/matching image paths, image byte hashes, source/permission attestations and reviewer provenance.
- Exact duplicate rejection and perceptual dHash near-duplicate nomination (distance <=5).
- Human duplicate-pair decisions, bound to current hashes; duplicate exclusion and same-scene cross-split blocking.
- Session/scene split leakage, including manually inconsistent ledger entries.
- Invalid/missing references, duplicate questions that would inflate sample counts, count-rule omissions and stale verification.
- Per-type counts, binary yes/no balance, declared scene/task coverage, unlabelled images and pilot-size warnings.

Perceptual hashing is a candidate detector, not proof. Human review is mandatory for candidates; unrelated uniform images may collide and transformed/nearby scenes may be missed. Source grouping and manual review remain necessary. Rights/provenance are human attestations, not automated authentication. Smaller valid exports are allowed with disclosed warnings; exporting never authorizes a model experiment.

## CPU-only verification

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest test_warehouse_annotation.py test_evaluation.py test_core.py test_model_inference.py test_app_qa.py test_camera_capture.py test_discovery_safety.py -q -p no:cacheprovider -p tools.pytest_cpu_guard

Final result: 146 passed in 11.22 seconds (35 annotation/workflow checks plus 111 existing focused tests).

CPU guard counters: model_load=0, tokenizer_load=0, cuda_initialization=0. Installed PyTorch emits the existing pynvml deprecation warning on import; packages were not changed.

Synthetic temporary fixtures test source-registration requirements, original-image preservation, duplicate handling, explicit verification/draft exclusion, count rules, exact manifest schema/validator integration, immutable export/history, stale provenance, manual corruption/leakage, self-review disclosure, unregistered media/path checks and loopback HTTP preview/form submission/token validation. Synthetic identity/source strings are explicitly test-only and never entered into the actual pilot.

The actual pilot has zero image files, events and exports. SHA-256 checks confirm all 34 tracked original CSV/report/document/image/video artifacts unchanged. Python syntax checks pass. Main dashboard, inference, evaluation validator, model architecture, quantization and historical artifacts were not changed.

## Files created

- src/warehouse_annotation.py: human annotation/source registry, event history, duplicate/coverage/leakage QC and validator-backed export.
- tools/annotate_warehouse.py: source registration, loopback image/form server, QC, duplicate review, exclusion and export CLI.
- test_warehouse_annotation.py: CPU-only synthetic workflow tests.
- data/warehouse_pilot/: empty data/material directories, header-only sources.csv, guide, usage/permission instructions and inventory/preservation/QC evidence.
- TASK3_PREPARATION.md: this report.

## Remaining work and stopping point

Obtain approximately 40 distinct genuinely permitted warehouse images and actual source/session/scene/evidence records. Register them, assign honest groups/splits, prepare up to about 120 answerable questions and human references, independently review where possible, resolve QC findings and validate a real export.

Stop for human review. No experiment approval record or GPU evaluation has been created. Dataset acquisition and verification remain pending rather than falsely complete.
