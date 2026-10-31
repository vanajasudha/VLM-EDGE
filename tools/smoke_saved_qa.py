"""Explicit, offline, single-model GPU check through the actual dashboard.
Run from the project root: venv/bin/python tools/smoke_saved_qa.py
Creates a new timestamped JSON record; never rewrites historical artifacts.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from bitsandbytes.nn import Linear4bit
    from transformers import AutoModelForCausalLM
    from streamlit.testing.v1 import AppTest
    from src.model_inference import load_moondream

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = {"status": "PARTIAL", "utc": stamp, "offline": True, "requests": []}
    output = ROOT / f"qa_gpu_smoke_{stamp}.json"
    model_loads = []
    original = AutoModelForCausalLM.from_pretrained

    def tracked_load(*args, **kwargs):
        if model_loads:
            raise RuntimeError("Second model load prevented by smoke-test guard")
        model_loads.append(None)
        model = original(*args, **kwargs)
        model_loads[0] = model
        return model

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free, total = torch.cuda.mem_get_info()
        record.update(gpu=torch.cuda.get_device_name(0), free_before_mib=free / 2**20,
                      total_mib=total / 2**20, torch=torch.__version__)
        if free < 3 * 2**30:
            raise RuntimeError("Less than 3 GiB free VRAM; smoke test stopped before model loading")
        print("GPU preflight:", record, flush=True)
        load_moondream.clear()
        with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=tracked_load):
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()
            at.sidebar.radio[0].set_value("Ask a Question").run()
            for image, question in [
                ("cam1_0", "What is happening in this scene?"),
                ("cam2_1", "Is there a tree in this image?"),
            ]:
                at.selectbox[0].set_value(image).run()
                at.text_input[0].set_value(question).run()
                assert not at.button[0].disabled
                print("Submitting:", image, question, flush=True)
                at.button[0].click().run(timeout=240)
                assert not at.exception, [e.message for e in at.exception]
                result = at.session_state["qa_result"]
                if result is None:
                    raise RuntimeError(at.session_state["qa_error"])
                assert result["answer"].strip() and result["duration_ms"] > 0
                assert result["image"] == image and result["question"] == question
                assert any("Inference latency:" in c.value for c in at.caption)
                record["requests"].append({k: result[k] for k in ("image", "question", "answer", "duration_ms")})
                print("Generated:", json.dumps(record["requests"][-1]), flush=True)
            cached_a = load_moondream()
            cached_b = load_moondream()
            assert cached_a is cached_b and cached_a[0] is model_loads[0]
        model = model_loads[0]
        quantized = [n for n, m in model.named_modules() if isinstance(m, Linear4bit)]
        skipped_quantized = [n for n in quantized if any(part in n.split(".") for part in ("vision", "region"))]
        assert quantized and not skipped_quantized
        assert any(p.device.type == "cuda" for p in model.parameters())
        record.update(status="PASS", model_loads=len(model_loads), cache_reused=True,
                      quantized_layers=len(quantized), quantized_vision_region=skipped_quantized,
                      device_map=str(getattr(model, "hf_device_map", None)),
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    except Exception as exc:
        record["error"] = str(exc)
        traceback.print_exc()
    finally:
        record["model_loads"] = len(model_loads)
        with output.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print("RECORD:", output, flush=True)
        print(json.dumps(record, indent=2), flush=True)
        load_moondream.clear()
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
