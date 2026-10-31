"""Only synthetic temporary images/identities; no genuine warehouse labels are created."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request

from PIL import Image
import pytest

from src.evaluation_manifest import load_manifest
from src.warehouse_annotation import (
    COUNT_RULE, SOURCE_FIELDS, active_sources, exclude_image, export_manifest, initialize,
    quality_report, read_events, read_sources, register_image, review_pair, save_annotation,
)
from tools.annotate_warehouse import create_server, render

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def pilot(tmp_path):
    root = initialize(tmp_path / "synthetic_pilot")
    source = tmp_path / "SYNTHETIC_NOT_WAREHOUSE.png"
    Image.new("RGB", (32, 32), "white").save(source)
    return root, source


def register(pilot, name="fixture1", color=None, split="test", session=None, scene=None, **extra):
    root, source = pilot
    if color is not None:
        source = source.parent / (name + ".png")
        Image.new("RGB", (32, 32), color).save(source)
    fields = dict(
        image_id=name, source_origin="SYNTHETIC UNIT TEST: not a genuine warehouse source",
        rights_basis="SYNTHETIC TEST FIXTURE", permission_evidence="SYNTHETIC TEST ONLY",
        rights_reviewer="TEST PERSON NOT REAL", session_id=session or name + "_session",
        scene_id=scene or name + "_scene", split=split, permission_confirmed=True,
        warehouse_confirmed=True, coverage_tags=["empty"], notes="Synthetic fixture only")
    fields.update(extra)
    return register_image(root, source, **fields)


def label(root, image="fixture1", kind="binary", answer="yes", verified=False, **extra):
    fields = dict(image_id=image, question="SYNTHETIC QUESTION: Is the test square visible?",
                  question_type=kind, reference_answer=answer,
                  annotator_id="SYNTHETIC TEST ANNOTATOR", reviewer_id="SYNTHETIC TEST REVIEWER",
                  verified=verified)
    fields.update(extra)
    return save_annotation(root, **fields)


def test_empty_pilot_is_not_ready_and_has_no_fake_labels(pilot):
    root, _ = pilot
    assert read_sources(root) == [] and read_events(root) == []
    qc = quality_report(root)
    assert qc["registered_active_images"] == qc["verified_labels"] == 0
    assert qc["status"] == "NOT_READY" and qc["gpu_evaluation_authorized"] is False
    with pytest.raises(ValueError, match="No permitted genuine"):
        export_manifest(root)


@pytest.mark.parametrize("override", [
    {"permission_confirmed": False}, {"warehouse_confirmed": False},
    {"rights_reviewer": ""}, {"permission_evidence": ""}, {"source_origin": ""},
    {"session_id": ""}, {"scene_id": ""}, {"coverage_tags": ["unknown"]},
])
def test_registration_requires_human_permission_source_and_grouping(pilot, override):
    with pytest.raises(ValueError):
        register(pilot, **override)
    assert read_sources(pilot[0]) == []


def test_import_preserves_original_and_rejects_duplicates(pilot):
    root, source = pilot
    before = source.read_bytes()
    row = register(pilot)
    assert source.read_bytes() == before
    assert (root / row["image_path"]).read_bytes() == before
    with pytest.raises(ValueError, match="Exact duplicate"):
        register(pilot, name="renamed")
    with pytest.raises(ValueError, match="already registered"):
        register(pilot)
    assert len(read_sources(root)) == 1


@pytest.mark.parametrize("field", ["session_id", "scene_id"])
def test_registration_blocks_group_leakage(pilot, field):
    register(pilot, session="shared-session", scene="shared-scene")
    kwargs = {field: "shared-session" if field == "session_id" else "shared-scene"}
    with pytest.raises(ValueError, match="leakage"):
        register(pilot, name="fixture2", color="black", split="train", **kwargs)


def test_draft_never_becomes_exported_or_auto_verified(pilot):
    root, _ = pilot
    register(pilot)
    event = label(root, answer="", reviewer_id="")
    assert event["verified"] is False and event["verified_at"] is None
    assert event["reviewer_id"] == ""
    qc = quality_report(root)
    assert qc["draft_labels"] == 1 and qc["verified_labels"] == 0
    with pytest.raises(ValueError, match="No human-verified"):
        export_manifest(root)


@pytest.mark.parametrize("kwargs", [
    {"reviewer_id": ""}, {"annotator_id": ""}, {"answer": "maybe"},
    {"kind": "count", "answer": "-1", "counting_rule": COUNT_RULE, "annotation_notes": "test"},
    {"kind": "count", "answer": "2", "counting_rule": "", "annotation_notes": "test"},
    {"kind": "count", "answer": "2", "counting_rule": COUNT_RULE, "annotation_notes": ""},
])
def test_explicit_verification_validates_identity_reference_and_count_rules(pilot, kwargs):
    register(pilot)
    with pytest.raises(ValueError):
        label(pilot[0], verified=True, **kwargs)
    assert read_events(pilot[0]) == []


def test_export_exact_schema_and_existing_validator(pilot):
    root, _ = pilot
    register(pilot)
    first = label(root, verified=True, sample_id="synthetic_binary")
    label(root, verified=True, kind="count", answer="0", counting_rule=COUNT_RULE,
          annotation_notes="SYNTHETIC target: workers; visible instances only",
          question="SYNTHETIC: How many workers are visible?")
    label(root, verified=True, kind="description", answer="SYNTHETIC white square",
          question="SYNTHETIC: Describe the test image.")
    qc = quality_report(root)
    assert qc["verified_labels"] == 3 and not qc["errors"]
    output = export_manifest(root)
    dataset = load_manifest(output)
    assert len(dataset.samples) == 3
    assert list(csv.DictReader(output.open()).fieldnames) == list(csv.DictReader((ROOT / "evaluation/manifest_template.csv").open()).fieldnames)
    provenance = json.loads(dataset.samples[0].label_provenance)
    assert provenance["reviewer_id"] == first["reviewer_id"]
    assert provenance["event_id"] == first["event_id"]
    assert dataset.samples[0].image_path.startswith("../images/")
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_manifest(root, output)
    assert output.read_bytes() == original


def test_latest_draft_revokes_old_verified_label_and_preserves_history(pilot):
    root, _ = pilot
    register(pilot)
    old = label(root, verified=True, sample_id="same")
    label(root, sample_id="same", answer="")
    assert len(read_events(root)) == 2
    assert quality_report(root)["verified_labels"] == 0
    with pytest.raises(ValueError):
        export_manifest(root)


def test_duplicate_questions_do_not_inflate_label_count(pilot):
    register(pilot)
    label(pilot[0], verified=True)
    label(pilot[0], verified=True, answer="no")
    assert any("Duplicate question" in e for e in quality_report(pilot[0])["errors"])
    with pytest.raises(ValueError, match="Duplicate question"):
        export_manifest(pilot[0])


def test_near_duplicate_requires_human_review(pilot):
    root, _ = pilot
    register(pilot)
    register(pilot, name="fixture2", color="black")
    label(root, verified=True)
    qc = quality_report(root)
    assert qc["near_duplicate_candidates"][0]["dhash_distance"] == 0
    assert any("needs human review" in e for e in qc["errors"])
    with pytest.raises(ValueError, match="Near-duplicate"):
        export_manifest(root)
    review_pair(root, "fixture1", "fixture2", verdict="distinct",
                reviewer_id="SYNTHETIC TEST REVIEWER", notes="Synthetic color squares are intentionally distinct")
    assert not quality_report(root)["errors"]
    assert load_manifest(export_manifest(root)).samples


def test_human_same_scene_review_prevents_cross_split_leakage(pilot):
    root, _ = pilot
    register(pilot)
    register(pilot, name="fixture2", color="black", split="train")
    review_pair(root, "fixture1", "fixture2", verdict="same_scene",
                reviewer_id="SYNTHETIC TEST REVIEWER", notes="Synthetic same-scene assertion")
    assert any("same-scene leakage" in e for e in quality_report(root)["errors"])


def test_exclusion_preserves_files_events_and_removes_from_export(pilot):
    root, _ = pilot
    row1 = register(pilot)
    row2 = register(pilot, name="fixture2", color="black")
    label(root, verified=True)
    review_pair(root, "fixture1", "fixture2", verdict="duplicate",
                reviewer_id="SYNTHETIC TEST REVIEWER", notes="Synthetic duplicate decision")
    assert any("Exclude one duplicate" in e for e in quality_report(root)["errors"])
    exclude_image(root, "fixture2", reviewer_id="SYNTHETIC TEST REVIEWER", reason="Synthetic exclusion")
    assert (root / row2["image_path"]).exists()
    assert len(read_sources(root)) == 2 and len(active_sources(root)) == 1
    assert len(load_manifest(export_manifest(root)).samples) == 1


def test_image_or_provenance_mutation_requires_reverification(pilot):
    root, _ = pilot
    source = register(pilot)
    label(root, verified=True)
    rows = read_sources(root)
    rows[0]["rights_basis"] = "CHANGED SYNTHETIC RIGHTS"
    with (root / "sources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    assert any("Reverification" in e for e in quality_report(root)["errors"])
    Image.new("RGB", (32, 32), "red").save(root / source["image_path"])
    assert any("Image bytes differ" in e for e in quality_report(root)["errors"])


def test_unregistered_media_are_not_silently_used(pilot):
    root, source = pilot
    (root / "images/unregistered.png").write_bytes(source.read_bytes())
    qc = quality_report(root)
    assert qc["registered_active_images"] == 0
    assert any("Unregistered images" in w for w in qc["warnings"])


def test_source_path_traversal_rejected(pilot):
    root, _ = pilot
    source = register(pilot)
    from src.warehouse_annotation import image_path
    with pytest.raises(ValueError, match="inside"):
        image_path(root, {**source, "image_path": "../../outside.png"})


def test_annotation_form_unchecked_and_escaped(pilot):
    root, _ = pilot
    register(pilot)
    page = render(root, "TOKEN", values={"question": "<script>bad</script>"})
    checkbox = re.search(r'<input[^>]*name="verified"[^>]*>', page).group(0)
    assert 'value="yes"' in checkbox and " checked" not in checkbox
    assert "&lt;script&gt;bad&lt;/script&gt;" in page
    assert "No model predictions" in page


def test_local_http_preview_draft_verification_and_csrf(pilot):
    root, _ = pilot
    row = register(pilot)
    server = create_server(root, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            page = response.read().decode()
        token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
        with urlopen(base + "/image?image=fixture1") as response:
            assert response.read() == (root / row["image_path"]).read_bytes()
        form = dict(csrf=token, image_id="fixture1", question="SYNTHETIC visible square?",
                    question_type="binary", reference_answer="yes",
                    annotator_id="SYNTHETIC TEST", reviewer_id="")
        req = Request(base + "/annotation", data=urlencode({**form, "csrf": "WRONG"}).encode())
        with pytest.raises(HTTPError) as bad:
            urlopen(req)
        assert bad.value.code == 403 and not read_events(root)
        with urlopen(Request(base + "/annotation", data=urlencode(form).encode())) as response:
            assert "Saved draft" in response.read().decode()
        assert read_events(root)[0]["verified"] is False
        form.update(verified="yes", reviewer_id="SYNTHETIC TEST REVIEWER", sample_id=read_events(root)[0]["sample_id"])
        with urlopen(Request(base + "/annotation", data=urlencode(form).encode())) as response:
            assert "Saved human-verified" in response.read().decode()
        assert len(read_events(root)) == 2
        assert quality_report(root)["verified_labels"] == 1
        assert load_manifest(export_manifest(root)).samples
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_no_model_imports_for_annotation_cli(pilot):
    code = """
import builtins, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "transformers", "bitsandbytes", "streamlit"}:
        raise AssertionError("Model dependency imported: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from tools.annotate_warehouse import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable, "-c", code, "--root", str(pilot[0]), "quality"],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and '"verified_labels": 0' in result.stdout


def test_self_verification_is_disclosed_not_hidden(pilot):
    register(pilot)
    label(pilot[0], verified=True, annotator_id="SYNTHETIC SAME PERSON",
          reviewer_id="SYNTHETIC SAME PERSON")
    report = quality_report(pilot[0])
    assert report["self_verified_labels"] == 1
    assert any("self-verified" in warning for warning in report["warnings"])


def test_malformed_event_history_is_rejected(pilot):
    root, _ = pilot
    (root / "annotations/events.jsonl").write_text('{"event_type":"annotation"}\n')
    with pytest.raises(ValueError, match="Malformed annotation event"):
        read_events(root)


def test_qc_detects_manual_registry_session_leakage(pilot):
    root, _ = pilot
    register(pilot)
    register(pilot, name="fixture2", color="black", split="train")
    sources = read_sources(root)
    sources[1]["session_id"] = sources[0]["session_id"]
    with (root / "sources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(sources)
    assert any("session_id leakage" in error for error in quality_report(root)["errors"])


def test_qc_does_not_trust_invalid_reference_in_manual_event(pilot):
    root, _ = pilot
    register(pilot)
    event = label(root, verified=True)
    event["reference_answer"] = "maybe"
    from src.warehouse_annotation import append_event
    append_event(root, event)
    report = quality_report(root)
    assert any("Binary reference" in error for error in report["errors"])
    assert report["verified_labels"] == 0
