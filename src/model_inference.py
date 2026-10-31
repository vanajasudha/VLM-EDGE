"""Local Moondream2 inference for the Streamlit dashboard."""

from pathlib import Path
from hashlib import sha256
from io import BytesIO
from threading import Lock
import streamlit as st
from time import perf_counter

MODEL_ID = "vikhyatk/moondream2"
_INFERENCE_LOCK = Lock()


class InferenceError(RuntimeError):
    """A user-facing model loading or inference failure."""


def validate_question(question: str) -> str:
    if not question.strip():
        raise ValueError("Please enter a question before asking the model.")
    return question


def image_identity(image_path: str | Path) -> str:
    """Validate a saved image without loading the model; fingerprint its actual bytes."""
    from PIL import Image
    try:
        data = Path(image_path).read_bytes()
        with Image.open(BytesIO(data)) as image:
            image.verify()
        return sha256(data).hexdigest()
    except Exception as exc:
        raise InferenceError(f"Cannot read selected image: {exc}") from exc


@st.cache_resource
def load_moondream(revision=None):
    """Load the verified selective-4-bit Moondream2 configuration."""
    try:
        import torch
        from bitsandbytes.nn import Linear4bit
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        revision_args = {"revision": revision} if revision is not None else {}
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["vision", "region"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, local_files_only=True, quantization_config=config,
            torch_dtype=torch.bfloat16, device_map="auto", **revision_args,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True, **revision_args)
        vision_quantized = [
            name for name, module in model.named_modules()
            if isinstance(module, Linear4bit) and ".vision." in "." + name + "."
        ]
        if vision_quantized:
            raise InferenceError("Vision layers were quantized unexpectedly; model loading was stopped.")
        return model, tokenizer
    except InferenceError:
        raise
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise InferenceError("CUDA ran out of memory while loading Moondream2. Close other GPU workloads and try again.") from exc
        raise InferenceError(f"Could not load Moondream2 from the local cache: {exc}") from exc


def _run_model_locked(model_and_tokenizer, image, question, generation_settings=None):
    """Caller holds the one process-wide inference lock."""
    import torch
    model, tokenizer = model_and_tokenizer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        encoded = model.encode_image(image)
        if generation_settings is None:
            answer = model.answer_question(encoded, question, tokenizer)
        else:
            answer = model.query(encoded, question, stream=False,
                                 settings=dict(generation_settings))["answer"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration_ms = (perf_counter() - started) * 1000
    if answer is None or not str(answer).strip():
        raise InferenceError("The model returned an empty answer. Please try again.")
    return {"answer": str(answer), "duration_ms": duration_ms}


def _raise_inference_error(exc):
    if isinstance(exc, InferenceError):
        raise exc
    if isinstance(exc, RuntimeError):
        if "out of memory" in str(exc).lower():
            raise InferenceError("CUDA ran out of memory during inference. Close other GPU workloads and try again.") from exc
        raise InferenceError(f"Inference failed: {exc}") from exc
    raise InferenceError(f"Could not read the image or complete inference: {exc}") from exc


def infer(model_and_tokenizer, image_path: str | Path, question: str, *, generation_settings=None) -> dict:
    question = validate_question(question)
    path = Path(image_path)
    if not path.is_file():
        raise InferenceError(f"Selected image does not exist: {path.name}")
    try:
        from PIL import Image
        with Image.open(path) as source:
            image = source.convert("RGB")
        try:
            with _INFERENCE_LOCK:
                return _run_model_locked(model_and_tokenizer, image, question, generation_settings)
        finally:
            image.close()
    except Exception as exc:
        _raise_inference_error(exc)


def infer_from_provider(model_and_tokenizer, image_provider, question: str) -> dict:
    """Get a fresh in-memory RGB image AFTER acquiring the shared GPU lock.

    The provider returns (owned PIL image, request metadata); the image is closed
    after inference. Capture/decode/file writes are not part of inference latency.
    """
    question = validate_question(question)
    try:
        with _INFERENCE_LOCK:
            image, metadata = image_provider()
            try:
                return {**_run_model_locked(model_and_tokenizer, image, question),
                        "frame_metadata": metadata}
            finally:
                image.close()
    except Exception as exc:
        _raise_inference_error(exc)
