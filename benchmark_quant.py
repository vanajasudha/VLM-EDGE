import csv
import time
import torch
import pynvml
from PIL import Image
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from bitsandbytes.nn import Linear4bit

MODEL_ID = "vikhyatk/moondream2"
QUESTION = "What is happening in this scene?"

# Match the original benchmark: 2 frames from each of 4 cameras.
frames = [
    ("cam1", "test_frames/cam1_0.jpg"),
    ("cam1", "test_frames/cam1_1.jpg"),
    ("cam2", "test_frames/cam2_0.jpg"),
    ("cam2", "test_frames/cam2_1.jpg"),
    ("cam3", "test_frames/cam3_0.jpg"),
    ("cam3", "test_frames/cam3_1.jpg"),
    ("cam4", "test_frames/cam4_0.jpg"),
    ("cam4", "test_frames/cam4_1.jpg"),
]

for camera, path in frames:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing frame: {path}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Benchmark stopped.")

torch.cuda.empty_cache()
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

def power_watts():
    try:
        return pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000.0
    except Exception:
        return None

print("Loading Moondream2 with selective 4-bit quantization...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["vision", "region"],
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

quantized_layers = [
    name for name, module in model.named_modules()
    if isinstance(module, Linear4bit)
]
vision_quantized = [
    name for name in quantized_layers
    if name.startswith("model.vision.")
]

print("4-bit layers:", len(quantized_layers))
print("4-bit vision layers:", len(vision_quantized))

if vision_quantized:
    raise RuntimeError(
        "Vision layers were quantized unexpectedly. "
        "Benchmark stopped; no results report was written."
    )

def infer(image):
    with torch.inference_mode():
        encoded = model.encode_image(image)
        return model.answer_question(encoded, QUESTION, tokenizer)

# One warm-up run, excluded from results.
print("Running warm-up...")
warm_image = Image.open(frames[0][1]).convert("RGB")
infer(warm_image)
torch.cuda.synchronize()
del warm_image

torch.cuda.reset_peak_memory_stats()
latencies = []
powers = []
results = []

print("Starting benchmark for 8 frames...")

for index, (camera, path) in enumerate(frames, start=1):
    image = Image.open(path).convert("RGB")

    power_before = power_watts()
    torch.cuda.synchronize()
    start = time.perf_counter()

    answer = infer(image)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    power_after = power_watts()

    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    measured_power = None
    if power_before is not None and power_after is not None:
        measured_power = (power_before + power_after) / 2

    latencies.append(latency_ms)
    if measured_power is not None:
        powers.append(measured_power)

    results.append({
        "camera": camera,
        "frame": Path(path).name,
        "latency_ms": round(latency_ms, 1),
        "peak_gpu_memory_mb": round(memory_mb, 1),
        "sampled_power_w": (
            round(measured_power, 1)
            if measured_power is not None else ""
        ),
        "answer": answer,
    })

    print(
        f"[{index}/8] {camera} {Path(path).name}: "
        f"{latency_ms:.1f} ms, peak allocated GPU memory "
        f"{memory_mb:.0f} MB"
    )
    print("Answer:", answer)

    image.close()

with open("quantized_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

avg_latency = sum(latencies) / len(latencies)
avg_memory = sum(r["peak_gpu_memory_mb"] for r in results) / len(results)
avg_power = sum(powers) / len(powers) if powers else None

with open("quantized_report.txt", "w", encoding="utf-8") as f:
    f.write("VLM-Edge Quantized Benchmark Report\n")
    f.write("Model: Moondream2, selective 4-bit with BFloat16 compute\n")
    f.write(f"Question: {QUESTION}\n")
    f.write("Warm-up runs (excluded): 1\n")
    f.write(f"Total frames: {len(results)}\n")
    f.write(f"Average latency: {avg_latency:.1f} ms\n")
    f.write(
        f"Average peak allocated GPU memory: {avg_memory:.1f} MB\n"
    )
    if avg_power is not None:
        f.write(
            f"Average sampled GPU power: {avg_power:.1f} W\n"
        )
    else:
        f.write("Average sampled GPU power: unavailable\n")
    f.write(
        "\nNote: Memory is PyTorch peak allocated memory. "
        "Power is sampled before and after each inference, not "
        "continuous power monitoring.\n"
    )

print("\nBenchmark completed.")
print(f"Average latency: {avg_latency:.1f} ms")
print(f"Average peak allocated GPU memory: {avg_memory:.1f} MB")
print("Saved: quantized_report.txt")
print("Saved: quantized_results.csv")

pynvml.nvmlShutdown()
