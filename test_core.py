import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_loader import load_benchmark_csv, list_frames
from src.benchmark_metrics import latency_summary, reduction_pct, speedup, binary_metrics, count_metrics


def test_baseline_schema(tmp_path):
    p = tmp_path / "b.csv"
    pd.DataFrame({"stream": ["rtsp://localhost:8554/cam1"], "frame": [0], "latency_ms": [8000.0],
                  "vram_mb": [4280.8], "power_w": [29.8], "answer": ["x"]}).to_csv(p, index=False)
    df, err = load_benchmark_csv(p)
    assert err is None and df.loc[0, "cam"] == "cam1" and df.loc[0, "memory_mb"] == 4280.8
    assert "System-level" in df.loc[0, "memory_method"]


def test_quant_schema(tmp_path):
    p = tmp_path / "q.csv"
    pd.DataFrame({"camera": ["cam2"], "frame": [1], "latency_ms": [2000.0],
                  "peak_gpu_memory_mb": [2420.7], "sampled_power_w": [47.6], "answer": ["y"]}).to_csv(p, index=False)
    df, err = load_benchmark_csv(p)
    assert err is None and df.loc[0, "cam"] == "cam2" and "PyTorch" in df.loc[0, "memory_method"]


def test_missing_and_malformed(tmp_path):
    assert load_benchmark_csv(tmp_path / "nope.csv")[0] is None
    p = tmp_path / "bad.csv"; p.write_text("a,b\n1,2\n")
    assert "latency_ms" in load_benchmark_csv(p)[1]


def test_metrics():
    df = pd.DataFrame({"latency_ms": [100.0, 300.0]})
    assert latency_summary(df)["mean_ms"] == 200.0
    assert round(reduction_pct(8635.8, 2149.5), 1) == 75.1
    assert round(speedup(8635.8, 2149.5), 2) == 4.02
    m = binary_metrics(["yes", "no", "yes", "no"], ["yes", "yes", "no", "no"])
    assert m["accuracy"] == 0.5 and m["tp"] == 1 and m["fp"] == 1
    assert count_metrics([2, 3], [2, 5])["mean_abs_error"] == 1.0


def test_frames(tmp_path):
    (tmp_path / "cam1_0.jpg").write_bytes(b"x"); (tmp_path / "notes.txt").write_text("x")
    f = list_frames(tmp_path)
    assert len(f) == 1 and f[0]["frame_id"] == "cam1_0"
    assert list_frames(tmp_path / "missing") == []


def test_csv_retains_valid_rows_and_reports_schema_warnings(tmp_path):
    p = tmp_path / "mixed.csv"
    pd.DataFrame({"camera": ["cam1", "cam1"], "frame": ["cam1_0.jpg", "cam1_1.jpg"],
                  "latency_ms": [100.0, "bad"], "peak_gpu_memory_mb": [10.0, 10.0]}).to_csv(p, index=False)
    df, err = load_benchmark_csv(p)
    assert err is None and len(df) == 1
    assert any("non-numeric" in warning for warning in df.attrs["validation_warnings"])
    assert any("power" in warning.lower() for warning in df.attrs["validation_warnings"])


def test_latency_summary_has_distribution_statistics():
    summary = latency_summary(pd.DataFrame({"latency_ms": [100.0, 200.0, 300.0, 400.0]}))
    assert summary["n"] == 4
    assert summary["mean_ms"] == 250.0
    assert summary["median_ms"] == 250.0
    assert summary["p95_ms"] == pytest.approx(385.0)
    assert summary["min_ms"] == 100.0 and summary["max_ms"] == 400.0


def test_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError, match="valid latency"):
        latency_summary(pd.DataFrame({"latency_ms": ["bad"]}))
    with pytest.raises(ValueError, match="equal lengths"):
        binary_metrics(["yes"], ["yes", "no"])
    with pytest.raises(ValueError, match="yes/no"):
        binary_metrics(["maybe"], ["yes"])


