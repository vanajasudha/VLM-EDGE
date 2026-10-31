"""Read-only loaders for benchmark CSVs and saved frames. Never writes files."""
import re
from pathlib import Path

import pandas as pd

# Baseline and quantized CSVs use different column names; map both to one schema.
ALIASES = {
    "camera": ["camera", "stream"],
    "frame": ["frame"],
    "latency_ms": ["latency_ms"],
    "memory_mb": ["peak_gpu_memory_mb", "vram_mb"],
    "power_w": ["sampled_power_w", "power_w"],
    "answer": ["answer"],
}
MEMORY_METHOD = {
    "peak_gpu_memory_mb": "PyTorch peak allocated memory",
    "vram_mb": "System-level VRAM reading",
}


def load_benchmark_csv(path):
    """Return ``(DataFrame, None)`` or ``(None, error_message)``.

    Non-fatal schema issues are retained in ``DataFrame.attrs['validation_warnings']``
    so the dashboard can show usable rows without hiding data-quality limitations.
    """
    path = Path(path)
    if not path.exists():
        return None, f"File not found: {path.name}"
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        return None, f"Could not read {path.name}: {exc}"

    latency_col = next((n for n in ALIASES["latency_ms"] if n in raw.columns), None)
    if latency_col is None:
        return None, f"{path.name} has no latency_ms column"

    warnings = []
    out = pd.DataFrame(index=raw.index)
    selected_columns = {}
    for standard, names in ALIASES.items():
        col = next((n for n in names if n in raw.columns), None)
        selected_columns[standard] = col
        out[standard] = raw[col] if col else pd.NA
        if standard == "memory_mb":
            out["memory_method"] = MEMORY_METHOD.get(col, "Not recorded")
            if col is None:
                warnings.append("Memory measurement column is not recorded.")
        elif standard == "power_w" and col is None:
            warnings.append("Sampled power column is not recorded.")

    for standard in ("latency_ms", "memory_mb", "power_w"):
        col = selected_columns[standard]
        if col is None:
            out[standard] = pd.NA
            continue
        converted = pd.to_numeric(raw[col], errors="coerce")
        invalid = raw[col].notna() & converted.isna()
        if invalid.any():
            warnings.append(f"{col} contains {int(invalid.sum())} non-numeric value(s); those values were ignored.")
        out[standard] = converted

    out["cam"] = out["camera"].astype("string").str.extract(r"(cam\d+)", expand=False)
    if out["cam"].isna().any():
        warnings.append(f"{int(out['cam'].isna().sum())} row(s) have no recognizable camera ID.")
    out = out.dropna(subset=["latency_ms"]).reset_index(drop=True)
    if out.empty:
        return None, f"{path.name} contains no valid numeric latency rows"
    out.attrs["validation_warnings"] = warnings
    out.attrs["source_path"] = str(path)
    return out, None


FRAME_RE = re.compile(r"^(cam\d+)_(\d+)\.(jpe?g|png)$", re.IGNORECASE)


def list_frames(frames_dir):
    """List saved frames. No timestamps are invented: none are recorded."""
    d = Path(frames_dir)
    if not d.is_dir():
        return []
    frames = []
    for p in sorted(d.iterdir()):
        m = FRAME_RE.match(p.name)
        if m:
            frames.append({"cam": m.group(1).lower(), "frame": int(m.group(2)),
                           "frame_id": f"{m.group(1).lower()}_{m.group(2)}",
                           "filename": p.name, "path": str(p)})
    return frames
