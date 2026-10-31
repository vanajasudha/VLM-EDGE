"""Pure calculation helpers (no file or UI code) so they are easy to test."""
import math

import pandas as pd


def _latency_series(df):
    if "latency_ms" not in df.columns:
        raise ValueError("Data has no latency_ms column")
    values = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("Data has no valid latency measurements")
    return values


def latency_summary(df):
    """Return validated descriptive statistics for a latency column."""
    s = _latency_series(df)
    return {
        "n": int(s.count()),
        "mean_ms": float(s.mean()),
        "median_ms": float(s.median()),
        "p95_ms": float(s.quantile(0.95)),
        "min_ms": float(s.min()),
        "max_ms": float(s.max()),
        "std_ms": float(s.std(ddof=1)) if s.count() > 1 else float("nan"),
    }


def reduction_pct(baseline_ms, other_ms):
    if baseline_ms <= 0:
        raise ValueError("Baseline latency must be positive")
    return (baseline_ms - other_ms) / baseline_ms * 100.0


def speedup(baseline_ms, other_ms):
    if baseline_ms <= 0 or other_ms <= 0:
        raise ValueError("Latencies must be positive")
    return baseline_ms / other_ms


def _paired(truth, pred):
    truth, pred = list(truth), list(pred)
    if len(truth) != len(pred):
        raise ValueError("Ground truth and prediction columns must have equal lengths")
    if not truth:
        raise ValueError("Labelled data is empty")
    return truth, pred


def normalize_binary(value):
    """Conservative normalization: a whole yes/no token, optional final punctuation."""
    import re
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(yes|no)[.!?]?", value.strip().lower())
    return match.group(1) if match else None


def normalize_count(value):
    """Nonnegative decimal integers only; no truncation, ranges or number-word guessing."""
    import re
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    return None


def binary_metrics(truth, pred, *, invalid_policy="error"):
    """Score yes/no, optionally count invalid predictions as errors/abstentions.

    Zero denominators use zero, including F1=0 when no positive label is predicted.
    With invalid_policy='count', invalid positive predictions enter the recall/F1
    denominator as false negatives; invalid negatives are separately counted.
    """
    if invalid_policy not in {"error", "count"}:
        raise ValueError("invalid_policy must be error or count")
    truth, pred = _paired(truth, pred)
    t = [str(x).strip().lower() for x in truth]
    if any(x not in {"yes", "no"} for x in t):
        raise ValueError("Ground truth labels must be yes/no")
    p = [normalize_binary(x) for x in pred]
    if invalid_policy == "error" and any(x is None for x in p):
        raise ValueError("Prediction labels must be yes/no")
    tp = sum(a == "yes" and b == "yes" for a, b in zip(t, p))
    tn = sum(a == "no" and b == "no" for a, b in zip(t, p))
    fp = sum(a == "no" and b == "yes" for a, b in zip(t, p))
    fn = sum(a == "yes" and b == "no" for a, b in zip(t, p))
    invalid_yes = sum(a == "yes" and b is None for a, b in zip(t, p))
    invalid_no = sum(a == "no" and b is None for a, b in zip(t, p))
    invalid = invalid_yes + invalid_no
    def div(a, b): return a / b if b else 0.0
    return {
        "n": len(t), "n_valid": len(t) - invalid, "n_invalid": invalid,
        "invalid_response_rate": invalid / len(t),
        "accuracy": (tp + tn) / len(t),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn + invalid_yes),
        "f1": div(2 * tp, 2 * tp + fp + fn + invalid_yes),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "invalid_yes": invalid_yes, "invalid_no": invalid_no,
        "confusion_matrix": {"rows": ["yes", "no"], "columns": ["yes", "no", "invalid"],
                             "values": [[tp, fn, invalid_yes], [fp, tn, invalid_no]]},
        "zero_division": 0,
    }


def count_metrics(truth, pred, *, invalid_policy="error"):
    """Exact match uses all samples; MAE uses valid responses only (None if none)."""
    if invalid_policy not in {"error", "count"}:
        raise ValueError("invalid_policy must be error or count")
    truth, pred = _paired(truth, pred)
    t = [normalize_count(x) for x in truth]
    p = [normalize_count(x) for x in pred]
    if any(x is None for x in t):
        raise ValueError("Count references must be nonnegative integers")
    if invalid_policy == "error" and any(x is None for x in p):
        raise ValueError("Count predictions must be nonnegative integers")
    err = [abs(a - b) for a, b in zip(t, p) if b is not None]
    invalid = len(t) - len(err)
    return {"n": len(t), "n_valid": len(err), "n_invalid": invalid,
            "invalid_response_rate": invalid / len(t),
            "mean_abs_error": sum(err) / len(err) if err else None,
            "exact_match": sum(e == 0 for e in err) / len(t),
            "mae_denominator": "valid_responses_only"}
