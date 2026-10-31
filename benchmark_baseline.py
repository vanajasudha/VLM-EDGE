"""
benchmark_baseline.py
---------------------
Baseline benchmark for VLM-Edge.

What it does:
  1. Loads the Moondream2 vision-language model onto the GPU (FP16).
  2. Grabs frames from 4 simulated RTSP camera streams.
  3. Asks the model one question about each frame.
  4. Measures latency, GPU memory and GPU power for every answer.
  5. Saves a text report (baseline_report.txt) and a CSV (baseline_results.csv).

Requirements before running:
  - MediaMTX and the 4 ffmpeg stream windows must be running.
  - transformers must be version 4.46.3 (what Moondream2 worked with).
  - No other program (e.g. quantization) should be using the GPU.
"""

import csv
import time

import cv2                      # OpenCV: reads video streams
import torch                    # PyTorch: runs the AI model on the GPU
import pynvml                   # NVIDIA monitoring: GPU memory and power
from PIL import Image           # Picture format the model expects
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------
# SETTINGS: change these to run a different test
# ---------------------------------------------------------------
MODEL_ID = "vikhyatk/moondream2"          # model downloaded from Hugging Face
STREAMS = [                                # the 4 simulated cameras

    "rtsp://localhost:8554/cam1",
    "rtsp://localhost:8554/cam2",
    "rtsp://localhost:8554/cam3",
    "rtsp://localhost:8554/cam4",
]
QUESTION = "What is happening in this scene?"   # same question for every frame
NUM_FRAMES_PER_STREAM = 2                 # frames measured per camera
WARMUP_RUNS = 1                  # throwaway runs before measuring

# ---------------------------------------------------------------
# GPU MONITORING SETUP
# ---------------------------------------------------------------
pynvml.nvmlInit()                                    # start the monitor
handle = pynvml.nvmlDeviceGetHandleByIndex(0)        # GPU number 0 (RTX 3050)


def gpu_stats():
    """Return (memory used in MB, power draw in watts) right now."""
    mem_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6
    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000   # milliwatts -> watts
    return mem_mb, power_w


# ---------------------------------------------------------------
# LOAD THE MODEL
# ---------------------------------------------------------------
print("Loading Moondream2...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,       # Moondream2 ships its own custom code
    torch_dtype=torch.float16,    # store numbers in 16 bits (FP16) to fit 4 GB VRAM
).to("cuda")                      # move the model onto the GPU

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)   # converts text <-> numbers
print("Model loaded.\n")


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def grab_frame(url):
    """Connect to a camera stream, grab one current frame, disconnect.

    Reconnecting each time avoids reading old frames that pile up in the
    stream buffer while the model is busy. Returns a PIL image, or None.
    """
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()        # ok = did it work, frame = the picture
    cap.release()
    if not ok:
        return None
    # OpenCV uses blue-green-red colour order; the model expects red-green-blue
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def run_inference(image):
    """Ask the model the question about one image.

    Returns (answer text, latency in milliseconds).
    """
    start = time.time()                                      # stopwatch on
    encoded = model.encode_image(image)                      # model "looks" at the image
    answer = model.answer_question(encoded, QUESTION, tokenizer)   # generates the answer
    torch.cuda.synchronize()      # wait until the GPU is truly finished (GPU runs in background)
    latency_ms = (time.time() - start) * 1000                # stopwatch off

    return answer, latency_ms


# ---------------------------------------------------------------
# WARM-UP (not recorded)
# The first runs include one-time startup work, so we run a few
# throwaway inferences to keep that out of the measurements.
# ---------------------------------------------------------------
print(f"Warming up ({WARMUP_RUNS} runs, not recorded)...")
for _ in range(WARMUP_RUNS):
    img = grab_frame(STREAMS[0])
    if img is not None:
        run_inference(img)
print("Warm-up done.\n")

# ---------------------------------------------------------------
# MAIN BENCHMARK LOOP
# ---------------------------------------------------------------
results = []   # one dictionary per measured frame

for stream_url in STREAMS:
    print(f"Benchmarking {stream_url}...")

    for frame_number in range(NUM_FRAMES_PER_STREAM):
        image = grab_frame(stream_url)
        if image is None:
            print(f"  frame {frame_number}: could not read stream, skipping")
            continue

        answer, latency_ms = run_inference(image)
        mem_mb, power_w = gpu_stats()     # read GPU state right after the answer


        results.append({
            "stream": stream_url,
            "frame": frame_number,
            "latency_ms": latency_ms,
            "vram_mb": mem_mb,
            "power_w": power_w,
            "answer": answer,
        })
        # Print the FULL answer (the old version cut it at 60 characters)
        print(f"  frame {frame_number}: {latency_ms:.0f} ms | "
              f"{mem_mb:.0f} MB | {power_w:.1f} W")
        print(f"     Answer: {answer}")

pynvml.nvmlShutdown()   # stop the GPU monitor

# ---------------------------------------------------------------
# SUMMARY AND REPORT
# ---------------------------------------------------------------
if not results:
    print("No results collected. Check that the RTSP streams are running.")
else:
    def average(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    avg_latency = average(results, "latency_ms")
    avg_vram = average(results, "vram_mb")
    avg_power = average(results, "power_w")

    print("\n=== Baseline Summary (Stock PyTorch FP16) ===")
    print(f"Total frames processed: {len(results)}")
    print(f"Average latency: {avg_latency:.1f} ms")
    print(f"Average VRAM used: {avg_vram:.0f} MB")

    print(f"Average power draw: {avg_power:.1f} W")

    # ----- Text report (easy to read and to show) -----
    with open("baseline_report.txt", "w") as f:
        f.write("VLM-Edge Baseline Benchmark Report\n")
        f.write("Model: Moondream2 (stock PyTorch, FP16)\n")
        f.write(f"Question asked: {QUESTION}\n")
        f.write(f"Warm-up runs (excluded): {WARMUP_RUNS}\n")
        f.write(f"Total frames: {len(results)}\n")
        f.write(f"Average latency: {avg_latency:.1f} ms\n")
        f.write(f"Average VRAM used: {avg_vram:.0f} MB\n")
        f.write(f"Average power draw: {avg_power:.1f} W\n\n")

        f.write("Per-camera averages:\n")
        for url in STREAMS:
            rows = [r for r in results if r["stream"] == url]
            if rows:
                f.write(f"{url} | {average(rows, 'latency_ms'):.1f} ms | "
                        f"{average(rows, 'vram_mb'):.0f} MB | "
                        f"{average(rows, 'power_w'):.1f} W\n")

        f.write("\nPer-frame results (with the model's full answer):\n")
        for r in results:
            f.write(f"{r['stream']} | frame {r['frame']} | "
                    f"{r['latency_ms']:.1f} ms | {r['vram_mb']:.0f} MB | "
                    f"{r['power_w']:.1f} W\n")
            f.write(f"    Answer: {r['answer']}\n")

    # ----- CSV file (for charts in Excel or Python) -----
    with open("baseline_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stream", "frame", "latency_ms", "vram_mb", "power_w", "answer"]

        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved: baseline_report.txt and baseline_results.csv")