"""CPU/mock verification only. Synthetic fixtures are not project ground truth."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from src.benchmark_metrics import binary_metrics, count_metrics, normalize_binary, normalize_count
from src.evaluation_manifest import ManifestError, load_manifest
from src.evaluation_runner import Experiment, build_plan, run_evaluation, validate_approval

ROOT = Path(__file__).resolve().parent
FIELDS = ["sample_id", "image_path", "source_id", "question", "question_type", "reference_answer",
          "label_provenance", "reference_verified", "split", "annotation_notes"]


@pytest.fixture
def manifest(tmp_path):
    # Deliberately synthetic unit-test data, never a genuine project evaluation dataset.
    Image.new("RGB", (4, 4), "white").save(tmp_path / "fixture.png")
    rows = []
    for i, (kind, ref) in enumerate([("binary", "yes"), ("binary", "no"),
                                      ("count", "2"), ("description", "White test square.")]):
        rows.append(dict(zip(FIELDS, [
            f"synthetic-{i}", "fixture.png", "synthetic-fixture",
            f"SYNTHETIC TEST QUESTION {i}?", kind, ref,
            "SYNTHETIC UNIT TEST ONLY; not human project annotations", "true", "test",
            "Illustrative code fixture; never report as model accuracy"])))
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows)
    return path, rows


def write_manifest(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("truth,pred", [
    (["yes", "no"], ["no", "yes"]), (["no", "no"], ["no", "no"]),
    (["yes"], ["no"]), (["no"], ["yes"]),
])
def test_binary_zero_division_is_finite(truth, pred):
    m = binary_metrics(truth, pred)
    assert m["f1"] == 0
    assert all(0 <= m[k] <= 1 for k in ("precision", "recall", "accuracy", "f1"))


def test_binary_invalid_denominators_and_confusion():
    m = binary_metrics(["yes", "yes", "no", "no"], ["yes", "maybe", "no", None],
                       invalid_policy="count")
    assert m["accuracy"] == .5
    assert m["precision"] == 1
    assert m["recall"] == .5
    assert m["f1"] == pytest.approx(2/3)
    assert m["invalid_response_rate"] == .5
    assert m["confusion_matrix"]["values"] == [[1, 0, 1], [0, 1, 1]]
    assert binary_metrics(["yes"], [None], invalid_policy="count")["f1"] == 0


@pytest.mark.parametrize("value,expected", [(" Yes! ", "yes"), ("no.", "no"),
    ("Yes, but no", None), ("not sure", None), ("yes/no", None), (None, None), ("", None),
    ("Yes, there is a tree.", None)])
def test_binary_normalization_is_conservative(value, expected):
    assert normalize_binary(value) == expected


@pytest.mark.parametrize("value,expected", [("02", 2), (" 0 ", 0), ("2.0", None), ("two", None),
    ("2 or 3", None), ("-1", None), (1.7, None), (True, None), (None, None)])
def test_count_normalization(value, expected):
    assert normalize_count(value) == expected


@pytest.mark.parametrize("metric", [binary_metrics, count_metrics])
def test_metrics_reject_mismatch_and_empty(metric):
    with pytest.raises(ValueError, match="equal lengths"):
        metric([1], [1, 2])
    with pytest.raises(ValueError, match="empty"):
        metric([], [])


def test_count_partial_invalid_and_all_invalid():
    m = count_metrics([2, 3, 1], ["2", "5", "unknown"], invalid_policy="count")
    assert m["exact_match"] == pytest.approx(1/3)
    assert m["mean_abs_error"] == 1
    assert m["n_valid"] == 2 and m["invalid_response_rate"] == pytest.approx(1/3)
    m = count_metrics([2], [None], invalid_policy="count")
    assert m["mean_abs_error"] is None and m["exact_match"] == 0
    with pytest.raises(ValueError, match="nonnegative integers"):
        count_metrics([1.5], [1])
    with pytest.raises(ValueError, match="nonnegative integers"):
        count_metrics([1], ["one"])


def test_manifest_identity_and_split(manifest):
    path, rows = manifest
    dataset = load_manifest(path)
    assert len(dataset.select("test")) == 4
    assert dataset.dataset_sha256 == load_manifest(path).dataset_sha256
    rows[0]["question"] = "A changed question"
    write_manifest(path, rows)
    assert dataset.dataset_sha256 != load_manifest(path).dataset_sha256


@pytest.mark.parametrize("field,value,error", [
    ("sample_id", "synthetic-1", "Duplicate sample_id"),
    ("image_path", "missing.png", "unreadable image"),
    ("question", " ", "nonempty"),
    ("question_type", "free", "question_type"),
    ("reference_answer", "maybe", "yes/no"),
    ("label_provenance", "", "nonempty"),
    ("reference_verified", "false", "human verification"),
    ("split", "demo", "split must"),
])
def test_manifest_rejects_invalid_rows(manifest, field, value, error):
    path, rows = manifest
    rows[0][field] = value
    write_manifest(path, rows)
    with pytest.raises(ManifestError, match=error):
        load_manifest(path)


def test_manifest_rejects_count_float_and_leakage(manifest):
    path, rows = manifest
    rows[2]["reference_answer"] = "2.1"
    write_manifest(path, rows)
    with pytest.raises(ManifestError, match="nonnegative integer"):
        load_manifest(path)
    rows[2]["reference_answer"] = "2"
    rows[2]["split"] = "train"
    write_manifest(path, rows)
    with pytest.raises(ManifestError, match="data leakage"):
        load_manifest(path)


def test_template_is_unevaluated():
    with pytest.raises(ManifestError, match="no labelled samples"):
        load_manifest(ROOT / "evaluation/manifest_template.csv")


class MockBackend:
    kind = "mock"
    def __init__(self, responses=None):
        self.responses = responses or ["yes", "uncertain", "2", "Synthetic description"]
        self.loads = 0
        self.calls = []
        self.closed = False

    def load(self, experiment):
        self.loads += 1

    def metadata(self):
        return {"gpu": None, "fixture": "SYNTHETIC MOCK"}

    def predict(self, sample, generation):
        self.calls.append((sample.sample_id, sample.resolved_image_path, sample.question, generation))
        # Warm-up and scored calls can return the same fixture response.
        result = self.responses[int(sample.sample_id.rsplit("-", 1)[1])]
        if isinstance(result, Exception):
            raise result
        return {"answer": result, "duration_ms": 2.5}

    def close(self):
        self.closed = True


def test_sequential_runner_writes_separate_raw_and_normalized(manifest, tmp_path):
    dataset = load_manifest(manifest[0])
    backend = MockBackend()
    run, summary = run_evaluation(dataset, Experiment(), tmp_path / "runs", backend=backend)
    assert summary["status"] == "complete"
    assert summary["not_a_real_accuracy_result"] is True
    assert backend.loads == 1 and backend.closed
    assert [c[0] for c in backend.calls] == ["synthetic-0", "synthetic-0", "synthetic-1",
                                           "synthetic-2", "synthetic-3"]
    raw = [json.loads(line) for line in (run / "raw_answers.jsonl").read_text().splitlines()]
    pred = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    assert raw[1]["raw_answer"] == "uncertain"
    assert pred[1]["normalized_prediction"] is None
    assert pred[2]["normalized_prediction"] == 2
    assert raw[3]["reference_answer"] == "White test square."
    assert summary["metrics"]["description"]["semantic_accuracy"] is None
    assert summary["metrics"]["binary"]["invalid_response_rate"] == .5
    assert summary["metrics"]["count"]["exact_match"] == 1
    assert all(c[3] == Experiment().generation() for c in backend.calls)
    assert backend.calls[1][2] == dataset.samples[0].question
    assert Path(backend.calls[1][1]) == Path(dataset.samples[0].resolved_image_path)
    assert json.loads((run / "metadata_final.json").read_text())["plan"]["experiment"]["revision"]
    # Strict JSON (not NaN) and fresh run directories; prior artifacts survive.
    first_bytes = (run / "raw_answers.jsonl").read_bytes()
    second, _ = run_evaluation(dataset, Experiment(warmup=0), tmp_path / "runs", backend=MockBackend())
    assert second != run and (run / "raw_answers.jsonl").read_bytes() == first_bytes


def test_runner_records_errors_and_continues(manifest, tmp_path):
    backend = MockBackend(["yes", RuntimeError("CUDA out of memory"), "invalid count", ""])
    run, summary = run_evaluation(load_manifest(manifest[0]), Experiment(warmup=0),
                                  tmp_path / "runs", backend=backend)
    assert summary["status"] == "completed_with_errors"
    assert summary["processed_samples"] == 4
    assert summary["metrics"]["n_errors"] == 2
    assert summary["metrics"]["count"]["mean_abs_error"] is None
    records = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    assert records[1]["error"]["message"] == "CUDA out of memory"
    assert records[1]["inference_latency_ms"] is None
    assert records[1]["attempt_elapsed_ms"] >= 0


def test_load_failure_keeps_failure_metadata(manifest, tmp_path):
    backend = MockBackend()
    def fail(experiment): raise RuntimeError("Failed to load")
    backend.load = fail
    run, summary = run_evaluation(load_manifest(manifest[0]), Experiment(), tmp_path / "runs",
                                  backend=backend)
    assert summary["status"] == "failed" and backend.closed
    assert summary["processed_samples"] == 0
    assert json.loads((run / "metadata_final.json").read_text())["failure"]["message"] == "Failed to load"


def test_image_mutation_after_validation_is_recorded_error(manifest, tmp_path):
    dataset = load_manifest(manifest[0])
    Image.new("RGB", (4, 4), "red").save(manifest[0].parent / "fixture.png")
    backend = MockBackend()
    _, summary = run_evaluation(dataset, Experiment(warmup=0), tmp_path / "runs", backend=backend)
    assert summary["metrics"]["n_errors"] == 4 and not backend.calls


def test_approval_binds_dataset_and_settings(manifest, tmp_path):
    dataset = load_manifest(manifest[0])
    config = Experiment()
    plan = build_plan(dataset, config)
    approval = {k: plan[k] for k in ("dataset_sha256", "experiment_sha256")}
    approval.update(approved_by="TEST ONLY", approved_at="2026-10-29T12:00:00+05:30",
                    human_labels_confirmed=True)
    validate_approval(approval, plan)
    with pytest.raises(ValueError, match="experiment_sha256"):
        validate_approval(approval, build_plan(dataset, replace(config, max_tokens=12)))
    with pytest.raises(ValueError, match="approval record"):
        run_evaluation(dataset, config, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_cli_validation_and_planning_do_not_import_model(manifest):
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "transformers", "bitsandbytes", "streamlit"}:
        raise AssertionError("Unexpected model/GPU import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.evaluate import main
raise SystemExit(main(sys.argv[1:]))
"""
    for command in ("validate", "plan"):
        result = subprocess.run([sys.executable, "-c", code, command, str(manifest[0])],
                                capture_output=True, text=True, timeout=20, cwd=ROOT)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"revision": "main"}, {"warmup": -1},
                                    {"temperature": float("nan")}, {"top_p": 0}, {"seed": -1}])
def test_invalid_generation_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Experiment(**kwargs)


@pytest.mark.parametrize("contents,expected", [
    ("sample_id,sample_id\nx,x\n", "Duplicate manifest column"),
    ("sample_id,image_path\nx,p.png\n", "Missing columns"),
    (",".join(FIELDS) + ",unexpected\n", "Unknown columns"),
    (",".join(FIELDS) + "\nx,missing-only-two-fields\n", "Malformed CSV row"),
])
def test_malformed_csv_headers_and_rows(tmp_path, contents, expected):
    path = tmp_path / "bad.csv"
    path.write_text(contents)
    with pytest.raises(ManifestError, match=expected):
        load_manifest(path)


def test_corrupt_image_and_duplicate_bytes_under_different_names(manifest):
    path, rows = manifest
    (path.parent / "fixture.png").write_bytes(b"not an image")
    with pytest.raises(ManifestError, match="unreadable image"):
        load_manifest(path)
    Image.new("RGB", (4, 4), "white").save(path.parent / "fixture.png")
    (path.parent / "another-name.png").write_bytes((path.parent / "fixture.png").read_bytes())
    rows[1].update(image_path="another-name.png", split="validation")
    write_manifest(path, rows)
    with pytest.raises(ManifestError, match="data leakage"):
        load_manifest(path)


def test_missing_selected_split_rejected(manifest):
    dataset = load_manifest(manifest[0])
    with pytest.raises(ManifestError, match="No labelled samples"):
        dataset.select("validation")


@pytest.mark.parametrize("mutation", [
    {"dataset_sha256": "wrong"}, {"experiment_sha256": "wrong"}, {"approved_by": ""},
    {"approved_at": "not-a-date"}, {"approved_at": "2026-10-29T12:00:00"},
    {"human_labels_confirmed": False}, {"human_labels_confirmed": "true"},
])
def test_invalid_approvals_refused_before_loading(manifest, tmp_path, mutation, monkeypatch):
    from src import evaluation_runner
    dataset = load_manifest(manifest[0])
    plan = build_plan(dataset, Experiment())
    approval = {k: plan[k] for k in ("dataset_sha256", "experiment_sha256")}
    approval.update(approved_by="SYNTHETIC TEST", approved_at="2026-10-29T12:00:00Z",
                    human_labels_confirmed=True)
    approval.update(mutation)
    def forbidden():
        pytest.fail("Backend constructed before approval validation")
    monkeypatch.setattr(evaluation_runner, "QuantizedBackend", forbidden)
    with pytest.raises(ValueError):
        run_evaluation(dataset, Experiment(), tmp_path / "runs", approval=approval)
    assert not (tmp_path / "runs").exists()


def test_every_mock_result_file_is_explicitly_labelled(manifest, tmp_path):
    run, summary = run_evaluation(load_manifest(manifest[0]), Experiment(), tmp_path,
                                  backend=MockBackend())
    assert run.name.startswith("mock_eval_")
    for path in run.iterdir():
        if path.suffix == ".json":
            values = [json.loads(path.read_text())]
        elif path.suffix == ".jsonl":
            values = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            pytest.fail(f"Unexpected output: {path}")
        for value in values:
            assert value["execution"] == "mock", path
            assert value["not_a_real_accuracy_result"] is True, path


def test_existing_output_files_are_never_overwritten(manifest, tmp_path):
    dataset = load_manifest(manifest[0])
    output = tmp_path / "experiments"
    output.mkdir()
    protected = output / "summary.json"
    protected.write_text("HISTORICAL SENTINEL")
    runs = [run_evaluation(dataset, Experiment(warmup=0), output,
                           backend=MockBackend())[0] for _ in range(2)]
    assert runs[0] != runs[1]
    assert protected.read_text() == "HISTORICAL SENTINEL"
    from src.evaluation_runner import write_json
    with pytest.raises(FileExistsError):
        write_json(runs[0] / "summary.json", {})
    assert json.loads((runs[0] / "summary.json").read_text())["status"] == "complete"


def test_existing_plan_file_is_never_overwritten(manifest, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("KEEP")
    result = subprocess.run([sys.executable, "tools/evaluate.py", "plan", str(manifest[0]),
                             "--output", str(plan)], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and plan.read_text() == "KEEP"


def test_warmup_failure_not_reported_as_accuracy(manifest, tmp_path):
    backend = MockBackend([RuntimeError("warmup failed"), "no", "2", "description"])
    run, summary = run_evaluation(load_manifest(manifest[0]), Experiment(), tmp_path, backend=backend)
    assert summary["status"] == "failed" and summary["processed_samples"] == 0
    assert summary["metrics"]["binary"] is None
    assert summary["failure"]["message"] == "warmup failed"


def test_backend_cleanup_failure_still_writes_failure_record(manifest, tmp_path):
    backend = MockBackend()
    def fail(): raise RuntimeError("cleanup error")
    backend.close = fail
    run, summary = run_evaluation(load_manifest(manifest[0]), Experiment(warmup=0), tmp_path,
                                  backend=backend)
    assert summary["status"] == "failed"
    assert "cleanup error" in summary["failure"]["message"]
    assert (run / "metadata_final.json").is_file()


def test_all_invalid_responses_are_not_dropped_from_accuracy():
    m = binary_metrics(["yes", "no"], ["maybe", "unclear"], invalid_policy="count")
    assert m["accuracy"] == 0 and m["n_invalid"] == 2 and m["n"] == 2
    m = count_metrics([0, 1], ["none", "-1"], invalid_policy="count")
    assert m["exact_match"] == 0 and m["mean_abs_error"] is None
    assert m["invalid_response_rate"] == 1


def test_real_backend_contract_with_entire_gpu_stack_mocked(monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src import model_inference
    from src.evaluation_runner import QuantizedBackend
    cuda = SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (4 * 2**30, 4 * 2**30),
                           manual_seed_all=Mock(), get_device_name=lambda index: "MOCK GPU")
    fake_torch = SimpleNamespace(cuda=cuda, manual_seed=Mock(),
                                backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True)),
                                version=SimpleNamespace(cuda="MOCK CUDA"),
                                are_deterministic_algorithms_enabled=lambda: False)
    linear = type("SyntheticLinear4bit", (), {})
    model = SimpleNamespace(
        query=lambda *args, **kwargs: {"answer": "yes"},
        config=SimpleNamespace(_commit_hash=Experiment().revision),
        named_modules=lambda: [("model.text.synthetic", linear())],
        hf_device_map={"model": 0},
        model=SimpleNamespace(tokenizer=SimpleNamespace(to_str=lambda: "SYNTHETIC TOKENIZER")))
    loader = Mock(return_value=(model, "MOCK TOKENIZER"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", SimpleNamespace(Linear4bit=linear))
    monkeypatch.setattr(model_inference, "load_moondream", loader)
    backend = QuantizedBackend()
    try:
        backend.load(Experiment())
        info = backend.metadata()
        assert info["gpu"] == "MOCK GPU"
        assert info["loaded_commit"] == Experiment().revision
        assert info["native_tokenizer_sha256"] == hashlib.sha256(b"SYNTHETIC TOKENIZER").hexdigest()
        assert info["quantized_layers"] == 1 and not info["quantized_vision_region"]
        loader.assert_called_once_with(revision=Experiment().revision)
    finally:
        backend.close()
    loader.clear.assert_called_once()


@pytest.mark.parametrize("available,free,message", [
    (False, 4 * 2**30, "CUDA unavailable"), (True, 2 * 2**30, "Less than 3 GiB"),
])
def test_gpu_preflight_rejected_without_loading_model(monkeypatch, available, free, message):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src import model_inference
    from src.evaluation_runner import QuantizedBackend
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available, mem_get_info=lambda: (free, 4 * 2**30)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    loader = Mock()
    monkeypatch.setattr(model_inference, "load_moondream", loader)
    backend = QuantizedBackend()
    try:
        with pytest.raises(RuntimeError, match=message):
            backend.load(Experiment())
        loader.assert_not_called()
    finally:
        backend.close()
