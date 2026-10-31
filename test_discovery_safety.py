"""Legacy opt-in GPU scripts must remain inert during normal pytest collection."""
import subprocess
import sys

def test_importing_legacy_gpu_scripts_never_imports_model_dependencies():
    code = """
import builtins, importlib
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "transformers", "bitsandbytes", "PIL"}:
        raise AssertionError("GPU script imported model dependency: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
for name in ("test_moondream", "test_quantized_moondream", "test_quantized_moondream_bf16"):
    module = importlib.import_module(name)
    assert callable(module.main)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
