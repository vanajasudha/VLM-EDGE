# Warehouse pilot annotation guide

Preparation only: no genuine warehouse sources or verified answers have been supplied. Do not fill gaps with model answers, demo scenes, synthetic footage or invented reviewer identities.

## Collection goals

Aim for about 40 distinct permitted images and 120 answerable pairs, subject to availability and annotation quality. A starting allocation is one binary, count and description question per image (about 40 of each type). Skip unsuitable questions rather than force three onto an ambiguous image. Seek roughly balanced binary yes/no answers without selecting scenes to favor a model.

Cover forklifts/pallet trucks, workers, boxes/pallets, loading docks, storage aisles, clear/obstructed marked walkways, empty/occupied scenes, lighting challenges and occlusion. Categories overlap. These are collection goals, not measured coverage. Distinct filenames do not establish distinct images.

## Source and permissions

Use genuine images you own and are authorized to use, or images supplied with explicit compatible permission/licensing. Record actual origin, owner, rights scope, evidence, reviewer, session and scene. The helper cannot authenticate permission or determine genuine footage from a flag.

Keep retainable evidence in permissions/ or reference a genuine permission ticket/licence record in sources.csv. Do not download arbitrary images or assume public availability grants permission. Handle identifiable people only within the actual permission scope. If authorized redaction/cropping is needed, preserve the source separately and document the transformation before registering final fixed image bytes.

Registration copies JPEG/PNG/WEBP bytes without modifying originals. Check orientation: the existing inference helper converts RGB without automatically rotating EXIF orientation. Prefer correctly oriented pixels or an approved, documented orientation-normalized copy. Do not change registered pixels after verification; hashes and source metadata bind the labels.

## Grouping and splits

Assign genuine, globally meaningful session_id and scene_id before annotation, not arbitrary per-frame IDs to evade checks. All images from a session or scene stay in one split. Group related camera views, adjacent frames and repeated views of the same aisle/action conservatively.

With enough independent groups, target roughly 60/20/20 train/validation/test by images (about 24/8/8), moving whole groups. A test-only pilot is acceptable if independent groups are unavailable; make no held-out generalization claim. Every question inherits its image's registered split. Do not revise selection/splits after seeing model outputs.

## Binary questions

Ask one visible, objectively checkable fact and request only yes/no. References are exactly yes or no. Generic question patterns, without labels:

- Is a forklift visible in this image? Answer only yes or no.
- Is a box visibly inside the marked walkway? Answer only yes or no.

Do not infer safety, training, intentions or hidden objects. Do not ask about walkway obstruction when the relevant walkway boundaries are not visible. Phrase absence questions about visible evidence, not objects possibly hidden behind occlusion.

## Counting rule: visible_instances_v1

Name the object class and region, include visible in the question, and request only a nonnegative integer. Count distinct, clearly identifiable visible instances once:

- Workers: partially visible people count only when evidence distinguishes a separate person. Exclude reflections, posters, screens and ambiguous isolated body parts.
- Forklifts/pallet trucks: distinguish classes visually; exclude a class-specific question if the vehicle cannot be identified reliably.
- Boxes: count separate outer boxes with visible boundaries; do not infer boxes hidden in stacks or containers.
- Pallets: count distinguishable physical pallets, not loads presumed to contain pallets.
- Zero is valid only when no target instance is visibly present under the stated scope.

Ranges/uncertain boundaries remain unverified. The form requires selecting visible_instances_v1 and entering nonempty human notes identifying target, scope/region and partial-visibility decisions before verifying a count. Freeze the rules; record future revisions rather than silently relabelling.

## Descriptions and ambiguity

Ask about visible entities/actions/relationships. Write a short human reference and note expected key facts and acceptable uncertainty. Avoid unsupported hazards or intent. Later human assessment uses EVALUATION.md and evaluation/description_review_template.csv, with separate correctness, coverage, hallucination and uncertainty dimensions. No automatic semantic accuracy is generated.

For glare, occlusion, unclear classes or undefined regions, save a draft with the reason. Do not verify maybe, unknown, ranges or guessed answers. Rephrase only when it creates an answerable task; preserve history. Exclude unsuitable images/questions with an actual identity and reason, without deleting images or earlier records.

## Human verification

Enter actual annotator and verifier IDs. The verifier personally opens the image and checks this exact question/reference. Check confirmation only after that review. It is unchecked by default.

Prefer independent review. If one person annotates and verifies, record that same genuine identity; do not invent a second reviewer. QC warns about self-review. Record disagreements/adjudication in notes. Model-generated answers are never ground truth.

Drafts are never exported. Revising an existing sample appends a new event. A latest draft revokes prior verification for future exports. Changed source metadata or image bytes requires reverification.

## Review workflow

After source registration (command pattern in README.md), run:

    venv/bin/python tools/annotate_warehouse.py serve --port 8766

Open http://127.0.0.1:8766 in a local browser. Select an image, inspect it, enter the annotation and actual identities, then save a draft or explicitly verified label. To edit, copy the sample ID from event history and re-enter the complete revised annotation. Every save appends history.

Run QC:

    venv/bin/python tools/annotate_warehouse.py quality

Inspect both images in every perceptual-duplicate candidate. Record only a genuine visual decision:

    venv/bin/python tools/annotate_warehouse.py review-pair IMAGE_A IMAGE_B --verdict distinct --reviewer REAL_ID --notes "Actual evidence supporting the decision"

Verdicts are distinct, same_scene or duplicate. Exact SHA-256 duplicates are rejected at import. Perceptual dHash distance <=5 nominates candidates only: uniform/brightness-only changes can collide; crops or large transformations can be missed. Human source grouping remains essential. Never declare distinct simply to silence QC.

same_scene across splits blocks export. A duplicate verdict requires excluding one image:

    venv/bin/python tools/annotate_warehouse.py exclude IMAGE_ID --reviewer REAL_ID --reason "Actual exclusion reason"

Files and all past events remain intact. After resolving errors:

    venv/bin/python tools/annotate_warehouse.py export

Export uses a new filename, exports only latest verified labels, and invokes the unchanged Task 2 validator before writing the final file. Pilot-size, balance, coverage, unlabelled-image and self-review warnings require human review; a smaller valid pilot can be exported and must be reported honestly.

Validate the printed path:

    venv/bin/python tools/evaluate.py validate data/warehouse_pilot/exports/ACTUAL_EXPORTED_FILE.csv --split test

Paths are relative to the manifest (../images/...), so no --image-root is needed. Select the actual split if there are no test samples.

## Exact manifest schema

    sample_id,image_path,source_id,question,question_type,reference_answer,label_provenance,reference_verified,split,annotation_notes

Provenance JSON contains actual annotator/verifier IDs, timestamp, event ID, source/permission evidence and hashes. Notes JSON contains human notes, counting rule, session/scene and coverage. No extra columns weaken or replace the existing schema.

Stop for review of sources, permissions, selection, annotations and QC. Do not create an experiment approval or run GPU evaluation.
