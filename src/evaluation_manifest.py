"""Validated CSV manifests and content identities; no model/GPU imports."""
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
import csv
import json

SCHEMA_VERSION = 1
REQUIRED = {"sample_id", "image_path", "question", "question_type", "reference_answer",
            "label_provenance", "reference_verified", "split"}
OPTIONAL = {"source_id", "annotation_notes"}
SPLITS = {"train", "validation", "test"}


class ManifestError(ValueError):
    pass


def identity(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    resolved_image_path: str
    image_sha256: str
    source_id: str
    question: str
    question_type: str
    reference_answer: str
    label_provenance: str
    reference_verified: bool
    split: str
    annotation_notes: str

    def content(self):
        result = asdict(self)
        result.pop("resolved_image_path")
        return result


@dataclass(frozen=True)
class Dataset:
    manifest_path: str
    manifest_sha256: str
    dataset_sha256: str
    samples: tuple

    def select(self, split=None):
        selected = tuple(s for s in self.samples if split is None or s.split == split)
        if not selected:
            raise ManifestError(f"No labelled samples for split {split!r}")
        return selected


def validated_image_hash(path):
    from PIL import Image
    try:
        data = Path(path).read_bytes()
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
        return sha256(data).hexdigest()
    except Exception as exc:
        raise ManifestError(f"Missing or unreadable image {path}: {exc}") from exc


def load_manifest(path, *, image_root=None):
    """Paths resolve against the CSV parent unless an explicit image root is supplied.

    reference_verified is a human attestation, not proof of annotation correctness.
    Exact image bytes cannot occur across different splits.
    """
    from src.benchmark_metrics import normalize_count
    path = Path(path).resolve()
    try:
        data = path.read_bytes()
        reader = csv.DictReader(StringIO(data.decode("utf-8-sig")))
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ManifestError("Duplicate manifest column names")
        if not REQUIRED <= set(headers):
            raise ManifestError(f"Missing columns: {sorted(REQUIRED - set(headers))}")
        if set(headers) - REQUIRED - OPTIONAL:
            raise ManifestError(f"Unknown columns: {sorted(set(headers) - REQUIRED - OPTIONAL)}")
        rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"Cannot read manifest: {exc}") from exc
    if not rows:
        raise ManifestError("Manifest has no labelled samples; a blank template is not evaluable")
    base = Path(image_root).resolve() if image_root is not None else path.parent
    samples, ids, image_splits = [], set(), {}
    for number, row in enumerate(rows, start=2):
        try:
            if None in row or any(v is None for v in row.values()):
                raise ManifestError("Malformed CSV row")
            for field in REQUIRED:
                if not row[field].strip():
                    raise ManifestError(f"{field} must be nonempty")
            sid = row["sample_id"].strip()
            if sid in ids:
                raise ManifestError(f"Duplicate sample_id: {sid}")
            ids.add(sid)
            kind = row["question_type"].strip()
            ref = row["reference_answer"]
            split = row["split"].strip()
            if kind not in {"binary", "count", "description"}:
                raise ManifestError("question_type must be binary, count or description")
            if split not in SPLITS:
                raise ManifestError("split must be train, validation or test")
            if row["reference_verified"].strip().lower() != "true":
                raise ManifestError("reference_verified must be true after actual human verification")
            if kind == "binary" and ref.strip().lower() not in {"yes", "no"}:
                raise ManifestError("Binary reference_answer must be yes/no")
            if kind == "count" and normalize_count(ref) is None:
                raise ManifestError("Count reference_answer must be a nonnegative integer")
            image = (base / row["image_path"]).resolve()
            image_hash = validated_image_hash(image)
            if image_hash in image_splits and image_splits[image_hash] != split:
                raise ManifestError("Identical image bytes occur across splits (data leakage)")
            image_splits[image_hash] = split
            samples.append(Sample(sid, row["image_path"], str(image), image_hash,
                                  row.get("source_id", "").strip(), row["question"], kind, ref,
                                  row["label_provenance"], True, split,
                                  row.get("annotation_notes", "")))
        except (ValueError, OSError) as exc:
            raise ManifestError(f"Row {number}: {exc}") from exc
    samples = tuple(samples)
    return Dataset(str(path), sha256(data).hexdigest(),
                   identity({"schema_version": SCHEMA_VERSION,
                             "ordered_samples": [s.content() for s in samples]}), samples)


def verify_sample_image(sample):
    if validated_image_hash(sample.resolved_image_path) != sample.image_sha256:
        raise ManifestError(f"Image changed after manifest validation: {sample.sample_id}")
