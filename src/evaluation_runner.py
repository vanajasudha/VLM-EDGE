"""Sequential, append-only experiment execution with injectable CPU/mock backend."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import sys
from time import perf_counter
from uuid import uuid4

from src.benchmark_metrics import binary_metrics, count_metrics, normalize_binary, normalize_count
from src.evaluation_manifest import identity, verify_sample_image

MODEL_ID = "vikhyatk/moondream2"
# Same cached revision verified in Task 1; never resolve a moving 'main' for experiments.
MODEL_REVISION = "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe"
QUANTIZATION = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16", "model_dtype": "bfloat16",
                "llm_int8_skip_modules": ["vision", "region"], "device_map": "auto"}


@dataclass(frozen=True)
class Experiment:
    split: str = "test"
    revision: str = MODEL_REVISION
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    warmup: int = 1

    def __post_init__(self):
        import re
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be an immutable 40-character commit")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 768:
            raise ValueError("max_tokens must be 1..768")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.warmup) is not int or not 0 <= self.warmup <= 10:
            raise ValueError("warmup must be 0..10")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def generation(self):
        return {"max_tokens": self.max_tokens, "temperature": self.temperature, "top_p": self.top_p}


def build_plan(dataset, experiment):
    samples = dataset.select(experiment.split)
    content = {
        "schema_version": 1, "dataset_sha256": dataset.dataset_sha256,
        "manifest_sha256": dataset.manifest_sha256,
        "selected_samples_sha256": identity([s.content() for s in samples]),
        "sample_ids": [s.sample_id for s in samples],
        "experiment": asdict(experiment), "model_id": MODEL_ID,
        "quantization": QUANTIZATION, "generation_api": "encode_image + query(settings, stream=False)",
        "metric_policy_version": 1,
        "timing": "CUDA-synchronized encode_image + query; excludes decode/load/queue; warmups excluded",
        "memory": "not measured by this evaluation runner",
        "power": "not measured; no energy or power savings claims",
    }
    return {**content, "experiment_sha256": identity(content)}


def validate_approval(approval, plan):
    if not isinstance(approval, dict):
        raise ValueError("A dataset/experiment approval record is required before GPU execution")
    for field in ("dataset_sha256", "experiment_sha256"):
        if approval.get(field) != plan[field]:
            raise ValueError(f"Approval {field} does not match the current plan")
    for field in ("approved_by", "approved_at"):
        if not isinstance(approval.get(field), str) or not approval[field].strip():
            raise ValueError(f"Approval needs {field}")
    try:
        stamp = datetime.fromisoformat(approval["approved_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError()
    except ValueError as exc:
        raise ValueError("approved_at must be an ISO-8601 timestamp with timezone") from exc
    if approval.get("human_labels_confirmed") is not True:
        raise ValueError("Approval must confirm genuine human reference labels")


def software_versions():
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in ("torch", "transformers", "bitsandbytes", "accelerate", "Pillow", "streamlit",
                 "pandas", "numpy", "tokenizers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_hashes():
    from hashlib import sha256
    root = Path(__file__).resolve().parents[1]
    paths = ["src/evaluation_manifest.py", "src/benchmark_metrics.py", "src/model_inference.py",
             "src/evaluation_runner.py", "tools/evaluate.py"]
    return {p: sha256((root / p).read_bytes()).hexdigest() for p in paths}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def normalized_prediction(kind, raw):
    if kind == "binary":
        value = normalize_binary(raw)
    elif kind == "count":
        value = normalize_count(raw)
    else:
        return None, "human_review_required" if isinstance(raw, str) and raw.strip() else "invalid"
    return value, "valid" if value is not None else "invalid"


def aggregate(records):
    result = {"n": len(records), "n_errors": sum(r["status"] == "error" for r in records)}
    for kind, metric in (("binary", binary_metrics), ("count", count_metrics)):
        group = [r for r in records if r["question_type"] == kind]
        result[kind] = metric([r["reference_answer"] for r in group],
                              [r["normalized_prediction"] for r in group],
                              invalid_policy="count") if group else None
    descriptions = [r for r in records if r["question_type"] == "description"]
    result["description"] = {"n": len(descriptions), "status": "human_review_required",
                             "semantic_accuracy": None,
                             "n_invalid_or_errors": sum(r["normalization_status"] != "human_review_required"
                                                        for r in descriptions)}
    durations = [r["inference_latency_ms"] for r in records if r["inference_latency_ms"] is not None]
    if durations:
        import pandas as pd
        from src.benchmark_metrics import latency_summary
        result["latency_successful_requests_ms"] = latency_summary(pd.DataFrame({"latency_ms": durations}))
        if not math.isfinite(result["latency_successful_requests_ms"]["std_ms"]):
            result["latency_successful_requests_ms"]["std_ms"] = None
    else:
        result["latency_successful_requests_ms"] = None
    return result


def run_evaluation(dataset, experiment, output_root, *, backend=None, approval=None):
    """One load, sequential requests, fresh exclusive directory and flushed JSONL records.

    An injected backend must explicitly declare kind='mock'; mock summaries cannot
    be mistaken for real model accuracy. Real execution always checks approval.
    """
    plan = build_plan(dataset, experiment)
    if backend is None:
        validate_approval(approval, plan)
        backend = QuantizedBackend()
    elif backend.kind != "mock":
        raise ValueError("Only explicit mock backends can be injected")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "mock_eval_" if backend.kind == "mock" else "eval_"
    run = root / (prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid4().hex)
    run.mkdir(exist_ok=False)
    metadata = {"status": "initializing", "execution": backend.kind,
                "not_a_real_accuracy_result": backend.kind == "mock",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "plan": plan, "approval": approval, "software": software_versions(),
                "source_sha256": source_hashes(), "hardware": None}
    write_json(run / "metadata.json", metadata)
    write_json(run / "manifest_snapshot.json", {
        "execution": backend.kind, "not_a_real_accuracy_result": backend.kind == "mock",
        "samples": [s.content() for s in dataset.samples]})
    selected = dataset.select(experiment.split)
    records = []
    state = "complete"
    failure = None
    try:
        backend.load(experiment)
        metadata["hardware"] = backend.metadata()
        with (run / "warmup.jsonl").open("x", encoding="utf-8") as warm:
            for index in range(experiment.warmup):
                verify_sample_image(selected[0])
                response = backend.predict(selected[0], experiment.generation())
                warm.write(json.dumps({"execution": backend.kind,
                                       "not_a_real_accuracy_result": backend.kind == "mock",
                                       "index": index, "sample_id": selected[0].sample_id,
                                       "excluded_from_metrics": True, "response": response},
                                      allow_nan=False) + "\n")
                warm.flush()
        with (run / "raw_answers.jsonl").open("x", encoding="utf-8") as raw_file, (
                run / "predictions.jsonl").open("x", encoding="utf-8") as pred_file:
            for sample in selected:
                started = perf_counter()
                raw, latency, error = None, None, None
                try:
                    verify_sample_image(sample)
                    response = backend.predict(sample, experiment.generation())
                    raw = response["answer"]
                    measured = response["duration_ms"]
                    if not isinstance(raw, str) or not raw.strip():
                        raise ValueError("Empty or non-text model answer")
                    if isinstance(measured, bool) or not isinstance(measured, (float, int)) or not math.isfinite(measured) or measured < 0:
                        raise ValueError("Invalid measured inference latency")
                    latency = float(measured)
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                elapsed = (perf_counter() - started) * 1000
                normalized, status = normalized_prediction(sample.question_type, raw) if error is None else (None, "error")
                common = {**sample.content(), "execution": backend.kind,
                          "not_a_real_accuracy_result": backend.kind == "mock",
                          "status": "error" if error else "ok",
                          "inference_latency_ms": latency, "attempt_elapsed_ms": elapsed, "error": error}
                raw_record = {**common, "raw_answer": raw}
                record = {**common, "normalized_prediction": normalized, "normalization_status": status}
                raw_file.write(json.dumps(raw_record, ensure_ascii=False, allow_nan=False) + "\n")
                pred_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                raw_file.flush()
                pred_file.flush()
                records.append(record)
        if any(r["status"] == "error" for r in records):
            state = "completed_with_errors"
    except Exception as exc:
        state = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            backend.close()
        except Exception as exc:
            state = "failed"
            failure = {"type": type(exc).__name__, "message": f"Backend cleanup failed: {exc}"}
    summary = {"status": state, "execution": backend.kind,
               "not_a_real_accuracy_result": backend.kind == "mock",
               "planned_samples": len(selected), "processed_samples": len(records),
               "failure": failure, "metrics": aggregate(records)}
    metadata.update(status=state, finished_at=datetime.now(timezone.utc).isoformat(), failure=failure)
    write_json(run / "summary.json", summary)
    # Immutable initial metadata and separate final record preserve interrupted-run evidence.
    write_json(run / "metadata_final.json", metadata)
    return run, summary


class QuantizedBackend:
    kind = "gpu"

    def load(self, experiment):
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import fcntl
        self.lock = open("/tmp/vlm-edge-evaluation-gpu.lock", "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluation process owns the GPU lock") from exc
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free, total = torch.cuda.mem_get_info()
        if free < 3 * 2**30:
            raise RuntimeError("Less than 3 GiB free CUDA memory; stop other GPU workloads")
        random.seed(experiment.seed)
        import numpy as np
        np.random.seed(experiment.seed)
        torch.manual_seed(experiment.seed)
        torch.cuda.manual_seed_all(experiment.seed)
        torch.backends.cudnn.benchmark = False
        from src.model_inference import load_moondream
        self.resource = load_moondream(revision=experiment.revision)
        if not callable(getattr(self.resource[0], "query", None)):
            raise RuntimeError("Pinned revision does not support native query settings")
        from bitsandbytes.nn import Linear4bit
        model = self.resource[0]
        quantized = [name for name, module in model.named_modules() if isinstance(module, Linear4bit)]
        skipped_quantized = [name for name in quantized
                             if any(part in name.split(".") for part in ("vision", "region"))]
        if not quantized or skipped_quantized:
            raise RuntimeError("Loaded model does not match selective quantization requirements")
        from hashlib import sha256
        native_tokenizer = getattr(getattr(model, "model", None), "tokenizer", None)
        tokenizer_hash = (sha256(native_tokenizer.to_str().encode()).hexdigest()
                          if native_tokenizer is not None and hasattr(native_tokenizer, "to_str") else None)
        self.info = {"native_tokenizer_sha256": tokenizer_hash,
                     "loaded_commit": getattr(model.config, "_commit_hash", None),
                     "quantized_layers": len(quantized), "quantized_vision_region": skipped_quantized,
                     "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
                     "free_before_load_mib": free / 2**20, "total_mib": total / 2**20,
                     "device_map": str(getattr(self.resource[0], "hf_device_map", None)),
                     "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "bitwise_reproducibility_guaranteed": False}

    def metadata(self):
        return self.info

    def predict(self, sample, generation):
        from src.model_inference import infer
        return infer(self.resource, sample.resolved_image_path, sample.question,
                     generation_settings=generation)

    def close(self):
        if hasattr(self, "resource"):
            from src.model_inference import load_moondream
            load_moondream.clear()
            del self.resource
        if hasattr(self, "lock"):
            self.lock.close()
