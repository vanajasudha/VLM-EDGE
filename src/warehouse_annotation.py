"""Local, human-operated annotation and dataset QC. No model/GPU dependencies."""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path
from uuid import uuid4
import csv
import json
import os
import re
import tempfile

from src.evaluation_manifest import identity, load_manifest, validated_image_hash

MANIFEST_FIELDS = ["sample_id", "image_path", "source_id", "question", "question_type",
                   "reference_answer", "label_provenance", "reference_verified", "split", "annotation_notes"]
SOURCE_FIELDS = ["image_id", "image_path", "image_sha256", "source_id", "source_origin",
                 "rights_basis", "permission_evidence", "permission_confirmed",
                 "rights_reviewer", "rights_verified_at", "warehouse_confirmed",
                 "session_id", "scene_id", "split", "coverage_tags", "notes"]
COVERAGE = {"forklift", "pallet_truck", "worker", "boxes", "pallets", "loading_dock",
            "storage_aisle", "walkway_clear", "walkway_obstructed", "empty", "occupied",
            "challenging_lighting", "occlusion"}
COUNT_RULE = "visible_instances_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def required(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be entered by the human reviewer")
    return value.strip()


@contextmanager
def dataset_lock(root):
    import fcntl
    root = Path(root).resolve()
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    with (root / "annotations" / ".annotation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def initialize(root):
    root = Path(root).resolve()
    for folder in ("images", "annotations", "exports", "quality", "permissions"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    with dataset_lock(root):
        registry = root / "sources.csv"
        if not registry.exists():
            with registry.open("x", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=SOURCE_FIELDS).writeheader()
    return root


def image_path(root, source):
    root = Path(root).resolve()
    path = (root / source["image_path"]).resolve()
    if not path.is_relative_to(root / "images"):
        raise ValueError("Source image must remain inside the dataset images directory")
    return path


def read_sources(root):
    registry = Path(root) / "sources.csv"
    with registry.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SOURCE_FIELDS:
            raise ValueError("Source registry schema differs from the documented template")
        rows = list(reader)
    ids = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed source registry row")
        if row["image_id"] in ids:
            raise ValueError("Duplicate source image_id")
        ids.add(row["image_id"])
    return rows


def read_events(root):
    path = Path(root) / "annotations/events.jsonl"
    if not path.exists():
        return []
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
            if event.get("event_type") not in {"annotation", "duplicate_review", "image_exclusion"}:
                raise ValueError("Unknown event_type")
            common = {"event_type", "event_id", "recorded_at"}
            needed = {
                "annotation": {"sample_id", "image_id", "image_sha256", "source_record_sha256",
                               "question", "question_type", "reference_answer", "verified",
                               "annotator_id", "reviewer_id", "verified_at", "annotation_notes", "counting_rule"},
                "duplicate_review": {"image_ids", "image_hashes", "verdict", "reviewer_id", "notes"},
                "image_exclusion": {"image_id", "reviewer_id", "reason"},
            }[event["event_type"]]
            if not common | needed <= set(event):
                raise ValueError("Missing event fields")
            if event["event_type"] == "annotation" and type(event["verified"]) is not bool:
                raise ValueError("Verification is not a boolean")
            events.append(event)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Malformed annotation event on line {number}") from exc
    return events


def append_event(root, event):
    with (Path(root) / "annotations/events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def active_sources(root, events=None):
    events = read_events(root) if events is None else events
    excluded = {e["image_id"] for e in events if e["event_type"] == "image_exclusion"}
    return [s for s in read_sources(root) if s["image_id"] not in excluded]


def source_check(root, source):
    for field in ("image_id", "source_origin", "rights_basis", "permission_evidence", "rights_reviewer",
                  "rights_verified_at", "session_id", "scene_id"):
        required(source[field], field)
    if source["permission_confirmed"] != "true" or source["warehouse_confirmed"] != "true":
        raise ValueError("Human permission and genuine-warehouse confirmations are required")
    if source["split"] not in {"train", "validation", "test"}:
        raise ValueError("Invalid split")
    if set(filter(None, source["coverage_tags"].split("|"))) - COVERAGE:
        raise ValueError("Unknown coverage tag")
    if validated_image_hash(image_path(root, source)) != source["image_sha256"]:
        raise ValueError("Image bytes differ from their registered source")
    # Evidence can be a local document, permission ticket or licence URL; its truth is human-attested.
    try:
        stamp = datetime.fromisoformat(source["rights_verified_at"])
        if stamp.tzinfo is None:
            raise ValueError()
    except ValueError as exc:
        raise ValueError("rights_verified_at needs a timezone-aware timestamp") from exc


def register_image(root, source_file, *, image_id, source_origin, rights_basis, permission_evidence,
                   rights_reviewer, session_id, scene_id, split, source_id="", coverage_tags=(),
                   notes="", permission_confirmed=False, warehouse_confirmed=False):
    """Copy one human-supplied permitted genuine image; never download or guess metadata."""
    root = initialize(root)
    if permission_confirmed is not True or warehouse_confirmed is not True:
        raise ValueError("Explicit human permission and genuine-warehouse confirmations required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
        raise ValueError("image_id must contain letters, digits, underscore or hyphen")
    fields = {k: required(v, k) for k, v in {
        "source_origin": source_origin, "rights_basis": rights_basis,
        "permission_evidence": permission_evidence, "rights_reviewer": rights_reviewer,
        "session_id": session_id, "scene_id": scene_id}.items()}
    if split not in {"train", "validation", "test"}:
        raise ValueError("Invalid split")
    tags = sorted(set(coverage_tags))
    if set(tags) - COVERAGE:
        raise ValueError("Unknown coverage tag")
    source_file = Path(source_file).resolve()
    digest = validated_image_hash(source_file)
    from PIL import Image
    with Image.open(source_file) as im:
        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(im.format)
    if extension is None:
        raise ValueError("Import JPEG, PNG or WEBP images only")
    row = {"image_id": image_id, "image_path": f"images/{image_id}{extension}",
           "image_sha256": digest, "source_id": source_id, **fields,
           "permission_confirmed": "true", "rights_verified_at": now(), "warehouse_confirmed": "true",
           "split": split, "coverage_tags": "|".join(tags), "notes": notes}
    with dataset_lock(root):
        sources = read_sources(root)
        if any(s["image_id"] == image_id for s in sources):
            raise ValueError("image_id already registered")
        if any(s["image_sha256"] == digest for s in sources):
            raise ValueError("Exact duplicate image already registered")
        for s in active_sources(root):
            if (s["session_id"] == fields["session_id"] or s["scene_id"] == fields["scene_id"]) and s["split"] != split:
                raise ValueError("Scene/session leakage: group must stay in one split")
        # Read once more and check before copying in case the original was replaced.
        data = source_file.read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError("Source changed during registration")
        with image_path(root, row).open("xb") as f:
            f.write(data)
        with (root / "sources.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SOURCE_FIELDS).writerow(row)
    return row


def save_annotation(root, *, image_id, question, question_type, reference_answer="",
                    annotation_notes="", annotator_id="", reviewer_id="", verified=False,
                    counting_rule="", sample_id=None):
    root = initialize(root)
    if type(verified) is not bool:
        raise ValueError("verified must be an explicit boolean")
    question = required(question, "question")
    annotator_id = required(annotator_id, "annotator_id")
    if question_type not in {"binary", "count", "description"}:
        raise ValueError("Invalid question_type")
    with dataset_lock(root):
        sources = {s["image_id"]: s for s in active_sources(root)}
        if image_id not in sources:
            raise ValueError("Image not registered or excluded")
        source = sources[image_id]
        source_check(root, source)
        if verified:
            reviewer_id = required(reviewer_id, "reviewer_id")
            required(reference_answer, "reference_answer")
            from src.benchmark_metrics import normalize_count
            if question_type == "binary" and reference_answer.strip().lower() not in {"yes", "no"}:
                raise ValueError("Binary reference must be yes/no")
            if question_type == "count":
                if normalize_count(reference_answer) is None:
                    raise ValueError("Count reference must be a nonnegative integer")
                if counting_rule != COUNT_RULE:
                    raise ValueError("A defined counting rule is required")
                required(annotation_notes, "count target/scope notes")
        if sample_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError("Invalid sample_id")
        sample_id = sample_id or "sample_" + uuid4().hex
        prior = [e for e in read_events(root) if e["event_type"] == "annotation" and e["sample_id"] == sample_id]
        if prior and prior[-1]["image_id"] != image_id:
            raise ValueError("Cannot move an existing sample_id to another image")
        event = {
            "event_type": "annotation", "event_id": uuid4().hex, "recorded_at": now(),
            "sample_id": sample_id, "image_id": image_id, "image_sha256": source["image_sha256"],
            "source_record_sha256": identity(source), "question": question, "question_type": question_type,
            "reference_answer": reference_answer, "verified": verified,
            "annotator_id": annotator_id, "reviewer_id": reviewer_id if verified else "",
            "verified_at": now() if verified else None, "annotation_notes": annotation_notes,
            "counting_rule": counting_rule,
        }
        append_event(root, event)
    return event


def review_pair(root, a, b, *, verdict, reviewer_id, notes):
    root = initialize(root)
    if a == b or verdict not in {"distinct", "same_scene", "duplicate"}:
        raise ValueError("Choose two different images and a valid verdict")
    with dataset_lock(root):
        sources = {s["image_id"]: s for s in active_sources(root)}
        if a not in sources or b not in sources:
            raise ValueError("Unknown active image")
        for key in (a, b):
            source_check(root, sources[key])
        event = {"event_type": "duplicate_review", "event_id": uuid4().hex,
                 "image_ids": sorted([a, b]), "image_hashes": {i: sources[i]["image_sha256"] for i in (a, b)},
                 "verdict": verdict, "reviewer_id": required(reviewer_id, "reviewer_id"),
                 "notes": required(notes, "notes"), "recorded_at": now()}
        append_event(root, event)
    return event


def exclude_image(root, image_id, *, reviewer_id, reason):
    root = initialize(root)
    with dataset_lock(root):
        if image_id not in {s["image_id"] for s in active_sources(root)}:
            raise ValueError("Unknown active image")
        append_event(root, {"event_type": "image_exclusion", "event_id": uuid4().hex,
                            "image_id": image_id, "reviewer_id": required(reviewer_id, "reviewer_id"),
                            "reason": required(reason, "reason"), "recorded_at": now()})
    # Files and past events remain untouched.


def dhash(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = im.tobytes()
    return sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


def quality_report(root, *, near_threshold=5):
    root = Path(root).resolve()
    if not 0 <= near_threshold <= 64:
        raise ValueError("near_threshold must be 0..64")
    events = read_events(root)
    sources = active_sources(root, events)
    by_id = {s["image_id"]: s for s in sources}
    errors, warnings, near = [], [], []
    hashes, perceptual = {}, {}
    for s in sources:
        try:
            source_check(root, s)
            if s["image_sha256"] in hashes:
                errors.append("Exact duplicate images: " + hashes[s["image_sha256"]] + ", " + s["image_id"])
            hashes[s["image_sha256"]] = s["image_id"]
            perceptual[s["image_id"]] = dhash(image_path(root, s))
        except (ValueError, OSError) as exc:
            errors.append(s["image_id"] + ": " + str(exc))
    for field in ("session_id", "scene_id"):
        groups = {}
        for s in sources:
            groups.setdefault(s[field], set()).add(s["split"])
        for group, splits in groups.items():
            if len(splits) > 1:
                errors.append(f"{field} leakage: {group}")
    reviews = {}
    for e in events:
        if e["event_type"] == "duplicate_review":
            reviews[tuple(e["image_ids"])] = e
    ids = sorted(perceptual)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            distance = (perceptual[a] ^ perceptual[b]).bit_count()
            review = reviews.get((a, b))
            valid_review = (review is not None and review.get("reviewer_id", "").strip() and
                            review.get("notes", "").strip() and
                            review["image_hashes"] == {k: by_id[k]["image_sha256"] for k in (a, b)})
            if distance <= near_threshold or valid_review:
                verdict = review["verdict"] if valid_review else "unreviewed"
                near.append({"image_a": a, "image_b": b, "dhash_distance": distance, "verdict": verdict})
                if verdict == "unreviewed":
                    errors.append(f"Near-duplicate candidate needs human review: {a}, {b}")
                elif verdict == "duplicate":
                    errors.append(f"Exclude one duplicate image before export: {a}, {b}")
                elif verdict == "same_scene" and by_id[a]["split"] != by_id[b]["split"]:
                    errors.append(f"Human-reviewed same-scene leakage: {a}, {b}")
    latest = {}
    for e in events:
        if e["event_type"] == "annotation":
            latest[e["sample_id"]] = e
    verified, drafts, task_counts = [], 0, {"binary": 0, "count": 0, "description": 0}
    balance = {"yes": 0, "no": 0}
    question_keys = {}
    for e in latest.values():
        if e["image_id"] not in by_id:
            continue
        if e.get("verified") is not True:
            if e.get("verified") is not False:
                errors.append(e["sample_id"] + ": verification must be an explicit boolean")
            drafts += 1
            continue
        s = by_id[e["image_id"]]
        try:
            required(e["annotator_id"], "annotator_id")
            required(e["reviewer_id"], "reviewer_id")
            required(e["verified_at"], "verified_at")
            verified_at = datetime.fromisoformat(e["verified_at"])
            if verified_at.tzinfo is None:
                raise ValueError("Verification timestamp needs a timezone")
            required(e["question"], "question")
            required(e["reference_answer"], "reference_answer")
            if e["source_record_sha256"] != identity(s) or e["image_sha256"] != s["image_sha256"]:
                raise ValueError("Reverification needed after source/image change")
            if e["question_type"] not in task_counts:
                raise ValueError("Unknown question type")
            from src.benchmark_metrics import normalize_count
            if e["question_type"] == "binary" and e["reference_answer"].strip().lower() not in {"yes", "no"}:
                raise ValueError("Binary reference must be yes/no")
            if e["question_type"] == "count" and normalize_count(e["reference_answer"]) is None:
                raise ValueError("Count reference must be a nonnegative integer")
            if e["question_type"] == "count" and (e["counting_rule"] != COUNT_RULE or not e["annotation_notes"].strip()):
                raise ValueError("Missing count rule/target notes")
            key = (e["image_id"], " ".join(e["question"].lower().split()))
            if key in question_keys:
                raise ValueError("Duplicate question on one image; do not inflate sample count")
            question_keys[key] = e["sample_id"]
            task_counts[e["question_type"]] += 1
            if e["question_type"] == "binary":
                label = e["reference_answer"].strip().lower()
                if label in balance:
                    balance[label] += 1
            verified.append(e)
        except (ValueError, KeyError) as exc:
            errors.append(e["sample_id"] + ": " + str(exc))
    used_images = {e["image_id"] for e in verified}
    registered_paths = {str(image_path(root, s)) for s in read_sources(root)}
    unregistered = [str(p.relative_to(root)) for p in (root / "images").iterdir()
                    if p.is_file() and not p.name.startswith(".") and str(p.resolve()) not in registered_paths]
    if unregistered:
        warnings.append("Unregistered images are excluded from annotation/export: " + ", ".join(unregistered))
    if not sources:
        errors.append("No permitted genuine warehouse images registered")
    if not verified:
        errors.append("No human-verified labels available")
    missing_labels = sorted(set(by_id) - used_images)
    if missing_labels:
        warnings.append("Images without verified labels: " + ", ".join(missing_labels))
    coverage = {tag: sum(tag in s["coverage_tags"].split("|") for s in sources) for tag in sorted(COVERAGE)}
    missing_coverage = [tag for tag, n in coverage.items() if not n]
    if missing_coverage:
        warnings.append("Missing declared coverage: " + ", ".join(missing_coverage))
    self_reviewed = sum(e["annotator_id"].strip() == e["reviewer_id"].strip() for e in verified)
    if self_reviewed:
        warnings.append(f"{self_reviewed} labels were self-verified; seek an independent review where possible")
    if verified and (not balance["yes"] or not balance["no"]):
        warnings.append("Binary yes/no coverage is one-sided or absent")
    if verified and any(n == 0 for n in task_counts.values()):
        warnings.append("Some question types are absent")
    if len(sources) < 40 or len(verified) < 120:
        warnings.append("Pilot target (~40 distinct images / ~120 pairs) not reached")
    return {"status": "NOT_READY" if errors else "READY_FOR_HUMAN_REVIEW",
            "registered_active_images": len(sources), "distinct_byte_hashes": len(hashes),
            "images_with_verified_labels": len(used_images), "verified_labels": len(verified),
            "draft_labels": drafts, "question_types": task_counts, "binary_balance": balance,
            "coverage": coverage, "near_duplicate_candidates": near,
            "split_images": {split: sum(s["split"] == split for s in sources) for split in ("train", "validation", "test")},
            "duplicate_detection": {"algorithm": "EXIF-transposed grayscale dHash 9x8",
                                    "hamming_threshold": near_threshold, "candidate_only": True},
            "self_verified_labels": self_reviewed,
            "errors": errors, "warnings": warnings, "gpu_evaluation_authorized": False}


def manifest_rows(root, output_path):
    root = Path(root).resolve()
    sources = {s["image_id"]: s for s in active_sources(root)}
    latest = {}
    for event in read_events(root):
        if event["event_type"] == "annotation":
            latest[event["sample_id"]] = event
    rows = []
    for e in latest.values():
        if not e["verified"] or e["image_id"] not in sources:
            continue
        s = sources[e["image_id"]]
        provenance = {"method": "human_image_review", "annotator_id": e["annotator_id"],
                      "reviewer_id": e["reviewer_id"], "verified_at": e["verified_at"],
                      "event_id": e["event_id"], "image_sha256": e["image_sha256"],
                      "source_origin": s["source_origin"], "rights_basis": s["rights_basis"],
                      "permission_evidence": s["permission_evidence"], "rights_reviewer": s["rights_reviewer"],
                      "source_record_sha256": e["source_record_sha256"]}
        notes = {"human_notes": e["annotation_notes"], "counting_rule": e["counting_rule"],
                 "session_id": s["session_id"], "scene_id": s["scene_id"], "coverage_tags": s["coverage_tags"]}
        rows.append({"sample_id": e["sample_id"],
                     "image_path": os.path.relpath(image_path(root, s), Path(output_path).parent),
                     "source_id": s["source_id"], "question": e["question"], "question_type": e["question_type"],
                     "reference_answer": e["reference_answer"], "label_provenance": json.dumps(provenance),
                     "reference_verified": "true", "split": s["split"], "annotation_notes": json.dumps(notes)})
    return rows


def export_manifest(root, output=None):
    root = initialize(root)
    with dataset_lock(root):
        report = quality_report(root)
        if report["errors"]:
            raise ValueError("Dataset export blocked: " + "; ".join(report["errors"]))
        output = Path(output).resolve() if output is not None else root / "exports" / ("manifest_" + uuid4().hex + ".csv")
        if not output.parent.is_dir():
            raise ValueError("Export directory does not exist")
        rows = manifest_rows(root, output)
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        contents = stream.getvalue()
        with tempfile.TemporaryDirectory() as temp:
            candidate = Path(temp) / "candidate.csv"
            candidate.write_text(contents, encoding="utf-8")
            # Existing validator is unchanged, including references and split-leakage rules.
            load_manifest(candidate, image_root=output.parent)
        with output.open("x", encoding="utf-8", newline="") as f:
            f.write(contents)
    return output
