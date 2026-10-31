import time, csv, pynvml

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("time_s | temp_C | gpu_% | mem_MB | power_W    (Ctrl+C to stop)")
t0 = time.time()

with open("telemetry_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "temp_c", "gpu_util", "mem_mb", "power_w"])
    try:
        while True:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e6
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
            t = time.time() - t0
            print(f"{t:6.1f} | {temp:5d} | {util:5d} | {mem:6.0f} | {power:6.1f}")
            w.writerow([round(t, 1), temp, util, round(mem), round(power, 1)])
            time.sleep(1)
    except KeyboardInterrupt:
        pass

pynvml.nvmlShutdown()
