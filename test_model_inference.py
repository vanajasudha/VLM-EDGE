import pytest

from src.model_inference import InferenceError, infer, validate_question


def test_validate_question_preserves_submitted_text():
    assert validate_question("  Is the path clear? ") == "  Is the path clear? "


def test_validate_question_rejects_empty_text():
    with pytest.raises(ValueError, match="enter a question"):
        validate_question("  ")


def test_infer_rejects_missing_image(tmp_path):
    with pytest.raises(InferenceError, match="does not exist"):
        infer((object(), object()), tmp_path / "missing.jpg", "What is visible?")


def test_infer_calls_native_model_api(tmp_path):
    from PIL import Image

    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (2, 2), "white").save(image_path)

    class FakeModel:
        def encode_image(self, image):
            return "encoded"

        def answer_question(self, encoded, question, tokenizer):
            assert encoded == "encoded"
            assert question == " What is visible? "
            assert tokenizer == "tokenizer"
            return "Generated answer"

    result = infer((FakeModel(), "tokenizer"), image_path, " What is visible? ")
    assert result["answer"] == "Generated answer"



@pytest.fixture(autouse=True)
def cpu_only_mocks(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def saved_image(tmp_path):
    from PIL import Image
    path = tmp_path / "selected.png"
    Image.new("RGB", (3, 2), (17, 29, 43)).save(path)
    return path


def test_selected_image_and_fresh_question_reach_model(saved_image):
    class Model:
        calls = 0
        def encode_image(self, image):
            assert image.mode == "RGB"
            assert image.size == (3, 2)
            assert image.getpixel((0, 0)) == (17, 29, 43)
            return image.getpixel((0, 0))
        def answer_question(self, encoded, question, tokenizer):
            self.calls += 1
            return f"{question}:{self.calls}"
    model = Model()
    first = infer((model, None), saved_image, "  First question? ")
    second = infer((model, None), saved_image, "Second question?")
    assert first["answer"] == "  First question? :1"
    assert second["answer"] == "Second question?:2"
    assert first["duration_ms"] >= 0 and second["duration_ms"] >= 0


def test_image_validation_and_fingerprint(saved_image):
    from PIL import Image
    from src.model_inference import image_identity
    before = image_identity(saved_image)
    Image.new("RGB", (3, 2), "red").save(saved_image)
    assert image_identity(saved_image) != before
    saved_image.write_bytes(b"not an image")
    with pytest.raises(InferenceError, match="selected image"):
        image_identity(saved_image)
    with pytest.raises(InferenceError, match="read the image"):
        infer((object(), None), saved_image, "Question")


@pytest.mark.parametrize("error, expected", [
    (RuntimeError("CUDA out of memory"), "CUDA ran out of memory"),
    (RuntimeError("kernel failed"), "Inference failed"),
    (ValueError("bad encoding"), "complete inference"),
])
def test_inference_errors_release_lock(saved_image, error, expected):
    from src.model_inference import _INFERENCE_LOCK
    class Model:
        def encode_image(self, image):
            raise error
    with pytest.raises(InferenceError, match=expected):
        infer((Model(), None), saved_image, "Question")
    assert _INFERENCE_LOCK.acquire(blocking=False)
    _INFERENCE_LOCK.release()


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_empty_model_answer_is_failure(saved_image, answer):
    class Model:
        def encode_image(self, image): return "encoded"
        def answer_question(self, *args): return answer
    with pytest.raises(InferenceError, match="empty answer"):
        infer((Model(), None), saved_image, "Question")


def test_concurrent_inference_is_serialized(saved_image):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock, Barrier
    import time
    guard = Lock()
    gate = Barrier(3)
    class Model:
        active = 0
        max_active = 0
        def encode_image(self, image):
            with guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            return "encoded"
        def answer_question(self, encoded, question, tokenizer):
            with guard:
                self.active -= 1
            return question
    model = Model()
    def request(question):
        gate.wait(timeout=5)
        return infer((model, None), saved_image, question)["answer"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(request, ["one", "two", "three"]))
    assert results == ["one", "two", "three"]
    assert model.max_active == 1


@pytest.fixture
def fake_loader(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src.model_inference import load_moondream
    load_moondream.clear()
    model = Mock()
    model.named_modules.return_value = []
    model_factory = Mock(return_value=model)
    tokenizer_factory = Mock(return_value=object())
    config_factory = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_factory),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_factory),
        BitsAndBytesConfig=config_factory))
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", SimpleNamespace(Linear4bit=type("Linear4bit", (), {})))
    yield load_moondream, model_factory, tokenizer_factory, config_factory
    load_moondream.clear()


def test_cached_load_shared_across_concurrent_callers(fake_loader):
    from concurrent.futures import ThreadPoolExecutor
    import torch
    loader, model_factory, tokenizer_factory, config_factory = fake_loader
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: loader(), range(4)))
    assert all(value is values[0] for value in values)
    model_factory.assert_called_once()
    tokenizer_factory.assert_called_once()
    config_factory.assert_called_once_with(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["vision", "region"])
    assert model_factory.call_args.kwargs["torch_dtype"] == torch.bfloat16
    assert model_factory.call_args.kwargs["device_map"] == "auto"
    assert model_factory.call_args.kwargs["local_files_only"] is True


@pytest.mark.parametrize("message, expected", [
    ("missing cached weights", "local cache"), ("CUDA out of memory", "CUDA ran out of memory"),
])
def test_load_failure_is_not_cached_and_can_retry(fake_loader, message, expected):
    loader, model_factory, _, _ = fake_loader
    model = model_factory.return_value
    model_factory.side_effect = [RuntimeError(message), model]
    with pytest.raises(InferenceError, match=expected):
        loader()
    assert loader()[0] is model
    assert model_factory.call_count == 2


def test_experiment_native_query_receives_generation_settings(saved_image):
    class Model:
        def encode_image(self, image):
            return "encoded"
        def query(self, encoded, question, *, stream, settings):
            assert encoded == "encoded" and question == "Exact question?"
            assert stream is False
            assert settings == {"max_tokens": 12, "temperature": 0.0, "top_p": 1.0}
            return {"answer": "yes"}
        def answer_question(self, *args):
            raise AssertionError("Legacy wrapper must not silently ignore generation limits")
    result = infer((Model(), None), saved_image, "Exact question?",
                   generation_settings={"max_tokens": 12, "temperature": 0.0, "top_p": 1.0})
    assert result["answer"] == "yes"


def test_experiment_revision_pins_model_and_tokenizer(fake_loader):
    loader, model_factory, tokenizer_factory, _ = fake_loader
    revision = "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe"
    loader(revision=revision)
    assert model_factory.call_args.kwargs["revision"] == revision
    assert tokenizer_factory.call_args.kwargs["revision"] == revision
