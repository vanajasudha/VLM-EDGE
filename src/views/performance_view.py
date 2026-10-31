"""Performance Analytics View: Benchmark metrics, latency distribution, and methodology provenance."""
from html import escape
import altair as alt
import pandas as pd
import streamlit as st
from src import ui
from src.benchmark_metrics import latency_summary, reduction_pct, speedup


def render_performance(data, errors):
    ui.header("Performance Analytics", "Rigorous benchmark analysis: FP16 baseline vs selective 4-bit NF4.", "Inference Benchmarks")

    for k, v in errors.items():
        ui.callout(f"Data loading issue ({k}): {v}", "bad")

    if "Baseline FP16" not in data:
        st.warning("Baseline benchmark data unavailable.")
        return

    base_ms = data["Baseline FP16"].latency_ms.mean()
    quant_runs = [data[k].latency_ms.mean() for k in ("Quantized run 1", "Quantized run 2") if k in data]
    quant_ms = sum(quant_runs) / len(quant_runs) if quant_runs else None

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    ui.kpi(c1, "FP16 Baseline Latency", f"{base_ms/1000:.2f} s", "Mean, 8 frames")
    ui.kpi(c2, "Quantized NF4 Latency", f"{quant_ms/1000:.2f} s" if quant_ms else "n/a", "Mean across 2 test runs")
    ui.kpi(c3, "Latency Reduction", f"{reduction_pct(base_ms, quant_ms):.1f}%" if base_ms and quant_ms else "n/a", "Observed difference")
    ui.kpi(c4, "Speedup Ratio", f"{speedup(base_ms, quant_ms):.2f}×" if base_ms and quant_ms else "n/a", "Historical benchmark ratio")

    t1, t2, t3 = st.tabs(["Latency Distribution", "Memory & Power Analysis", "Methodology & Provenance"])

    with t1:
        ui.section_title("Per-Frame Latency Distribution", "Comparative Breakdown")
        
        # Summary row
        cols = st.columns(len(data))
        for col, (name, df) in zip(cols, data.items()):
            s = latency_summary(df)
            sub = "Baseline reference" if name.startswith("Base") else f"{reduction_pct(base_ms, s['mean_ms']):.1f}% lower, {speedup(base_ms, s['mean_ms']):.2f}×"
            ui.kpi(col, name, f"{s['mean_ms']/1000:.2f} s", f"Median: {s['median_ms']/1000:.2f} s | {sub}")

        # Altair grouped bar chart
        long_df = pd.concat([
            pd.DataFrame({"Frame": [f"Frame {i+1}" for i in range(len(d))], "Run": n, "Latency_ms": d.latency_ms})
            for n, d in data.items()
        ])

        chart_colors = {
            "Baseline FP16": ui.C["amber"],
            "Quantized run 1": ui.C["teal"],
            "Quantized run 2": ui.C["blue"],
        }

        chart = alt.Chart(long_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Frame:O", title="Test Frame Index"),
            y=alt.Y("Latency_ms:Q", title="Latency (ms)", scale=alt.Scale(zero=True)),
            xOffset=alt.XOffset("Run:N"),
            color=alt.Color(
                "Run:N",
                scale=alt.Scale(domain=list(chart_colors.keys()), range=list(chart_colors.values())),
                legend=alt.Legend(title="Execution Run", orient="top")
            ),
            tooltip=[
                alt.Tooltip("Run:N", title="Benchmark Run"),
                alt.Tooltip("Frame:O", title="Frame"),
                alt.Tooltip("Latency_ms:Q", title="Latency (ms)", format=".1f"),
            ]
        ).properties(
            height=340
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            gridColor=ui.C["border"],
            labelColor=ui.C["muted"],
            titleColor=ui.C["muted"],
        )

        st.altair_chart(chart, width="stretch")

        with st.expander("Detailed Per-Frame Measurements Table"):
            st.dataframe(long_df, width="stretch", hide_index=True)

    with t2:
        ui.section_title("Memory Allocation & Telemetry", "Hardware Metrics")
        ui.callout(
            "Measurement Incompatibility Notice: Baseline VRAM and quantized peak memory use fundamentally different metrics. "
            "Baseline recorded NVML device-wide memory in decimal MB (4,280.8 MB ≈ 4,082.5 MiB), whereas quantized runs measured "
            "PyTorch cumulative peak allocated memory (2,420.7 MiB). Because these metrics measure different scopes and use different units, "
            "no comparative memory reduction is claimed.",
            "warn"
        )

        mem_records = []
        for n, d in data.items():
            mem_records.append({
                "Benchmark Run": n,
                "Mean Memory Value": round(d.memory_mb.mean(), 1),
                "Reported Unit": "Decimal MB" if n.startswith("Base") else "MiB (labelled MB)",
                "Measurement Method": d.memory_method.iloc[0],
                "Scope": "System-wide GPU memory" if n.startswith("Base") else "PyTorch process allocated",
            })
        st.dataframe(pd.DataFrame(mem_records), width="stretch", hide_index=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        ui.section_title("Sampled Power Telemetry", "Unverified")
        ui.callout(
            "Power Sampling Caveat: Power was sampled differently across runs (post-inference instantaneous reading vs average). "
            "Neither constitutes integrated electrical energy (Joules). Data is presented for provenance reference; no energy efficiency claims are made.",
            "warn"
        )
        pwr_records = [{"Benchmark Run": n, "Mean Sampled Power (W)": round(d.power_w.mean(), 1)} for n, d in data.items()]
        st.dataframe(pd.DataFrame(pwr_records), width="stretch", hide_index=True)

    with t3:
        ui.section_title("Benchmark Methodology & Scientific Limitations", "Provenance")
        st.markdown(
            f"""
            <div class="card">
                <div style="font-weight:700;font-size:0.95rem;color:{ui.C['text']};margin-bottom:0.5rem;">
                    Key Audit Findings & Experimental Protocol
                </div>
                <ul style="color:{ui.C['muted']};font-size:0.85rem;line-height:1.65;padding-left:1.2rem;margin:0;">
                    <li><strong>Input Divergence:</strong> Baseline FP16 benchmark utilized live RTSP frame grabs, while quantized runs used saved static images from <code>test_frames/</code>. Exact scene identity cannot be verified.</li>
                    <li><strong>Warm-up Conditions:</strong> Baseline attempted one warm-up, whereas quantized runs verified one complete saved-image pass before timing.</li>
                    <li><strong>Timing Semantics:</strong> Baseline used <code>time.time</code> with post-inference synchronization. Quantized runs utilized high-precision <code>time.perf_counter</code> with pre- and post-inference CUDA synchronization.</li>
                    <li><strong>Unit Nuances:</strong> Baseline memory of 4,280.8 decimal MB equals 4,082.5 MiB, which fits inside the physical 4,096 MiB VRAM buffer. The apparent overflow in older reports was an artifact of comparing decimal MB against binary MiB.</li>
                    <li><strong>Causality:</strong> While the 4.02× latency reduction is mathematically genuine for the recorded logs, quantization cannot be established as the sole cause due to these methodological variances.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )