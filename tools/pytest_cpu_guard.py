"""Optional pytest plugin that forbids real model loading and CUDA initialization.

Use: python -m pytest -p tools.pytest_cpu_guard ...
Mock from_pretrained factories can still be injected by tests.
"""
import pytest

_guard = None
_attempts = {"model_load": 0, "tokenizer_load": 0, "cuda_initialization": 0}


def pytest_sessionstart(session):
    global _guard
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _guard = pytest.MonkeyPatch()

    def reject(kind):
        def blocked(*args, **kwargs):
            _attempts[kind] += 1
            raise AssertionError(f"CPU-only suite blocked real {kind}")
        return blocked

    _guard.setattr(AutoModelForCausalLM, "from_pretrained", reject("model_load"))
    _guard.setattr(AutoTokenizer, "from_pretrained", reject("tokenizer_load"))
    _guard.setattr(torch.cuda, "_lazy_init", reject("cuda_initialization"))


def pytest_sessionfinish(session, exitstatus):
    if _guard is not None:
        _guard.undo()


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"CPU guard: real model/CUDA calls attempted: {_attempts}")
