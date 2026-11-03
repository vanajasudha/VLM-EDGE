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


# --------------------------------------------------------------------------------------
# Additive metrics (LOCO integration). Nothing above this line was changed.
# --------------------------------------------------------------------------------------
def wilson_interval(successes, n, z=1.959964):
    """95% Wilson score interval for a proportion; None when n == 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def parse_binary_lenient(value):
    """Secondary parser: answer must START with yes/no and not contain the opposite token.

    'Yes, there is a large tree.' -> yes; 'Yes, there is no obstruction.' -> None (ambiguous).
    Reported next to the strict parser; never replaces it.
    """
    import re
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    match = re.match(r"(yes|no)\b", text)
    if not match:
        return None
    opposite = "no" if match.group(1) == "yes" else "yes"
    return None if re.search(rf"\b{opposite}\b", text) else match.group(1)


_HEDGES = ("about", "approx", "around", "roughly", "nearly", "almost", "over ", "more than",
           "at least", "less than", "up to", "between", "~", "several", "many", "few")


def parse_count_lenient(value):
    """Secondary parser: exactly one distinct decimal integer, no ranges or hedging words."""
    import re
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    text = value.strip().lower()
    if any(h in text for h in _HEDGES) or re.search(r"\d\s*(-|–|to)\s*\d", text):
        return None
    numbers = {int(n) for n in re.findall(r"(?<![\d.])\d+(?![\d.])", text)}
    return numbers.pop() if len(numbers) == 1 else None


def extended_binary_metrics(truth, pred, *, parser=normalize_binary):
    """binary_metrics plus specificity, FPR, FNR and Wilson intervals.

    Invalid answers are never correct: they stay in the recall/specificity denominators.
    Note 1 - specificity = FPR + (invalid answers on negatives) / negatives.
    """
    truth = [str(x).strip().lower() for x in truth]
    parsed = [parser(x) for x in pred]
    base = binary_metrics(truth, ["yes" if p == "yes" else "no" if p == "no" else None for p in parsed],
                          invalid_policy="count")
    positives = base["tp"] + base["fn"] + base["invalid_yes"]
    negatives = base["tn"] + base["fp"] + base["invalid_no"]
    flagged = base["tp"] + base["fp"]
    base.update(
        positives=positives, negatives=negatives, prevalence=positives / base["n"] if base["n"] else None,
        specificity=base["tn"] / negatives if negatives else None,
        false_positive_rate=base["fp"] / negatives if negatives else None,
        false_negative_rate=(base["fn"] + base["invalid_yes"]) / positives if positives else None,
        ci95={"accuracy": wilson_interval(base["tp"] + base["tn"], base["n"]),
              "recall": wilson_interval(base["tp"], positives),
              "specificity": wilson_interval(base["tn"], negatives),
              "precision": wilson_interval(base["tp"], flagged)},
        small_sample=base["n"] < 100 or positives < 30 or negatives < 30,
        parser=getattr(parser, "__name__", "custom"),
        undefined_notes="rates are None (not 0) when their denominator is empty")
    # The legacy function reports 0 for undefined ratios; here undefined is explicit.
    if not positives:
        base["recall"] = None
    if not flagged:
        base["precision"] = None
    if not (2 * base["tp"] + base["fp"] + base["fn"] + base["invalid_yes"]):
        base["f1"] = None
    base["zero_division"] = None
    return base


def extended_count_metrics(truth, pred, *, parser=normalize_count):
    """count_metrics plus RMSE, within-one, over/undercount rates, bias and CI for exact match."""
    t = [normalize_count(x) for x in truth]
    if any(x is None for x in t):
        raise ValueError("Count references must be nonnegative integers")
    p = [parser(x) for x in pred]
    if len(t) != len(p):
        raise ValueError("truth and pred must have equal length")
    n = len(t)
    pairs = [(a, b) for a, b in zip(t, p) if b is not None]
    errors = [b - a for a, b in pairs]
    valid = len(pairs)
    exact = sum(e == 0 for e in errors)
    return {
        "n": n, "n_valid": valid, "n_invalid": n - valid,
        "invalid_response_rate": (n - valid) / n if n else None,
        "exact_match": exact / n if n else None,                   # invalid counts as wrong
        "exact_match_ci95": wilson_interval(exact, n),
        "within_one": sum(abs(e) <= 1 for e in errors) / n if n else None,
        "mean_abs_error": sum(abs(e) for e in errors) / valid if valid else None,
        "rmse": math.sqrt(sum(e * e for e in errors) / valid) if valid else None,
        "mean_signed_error": sum(errors) / valid if valid else None,
        "overcount_rate": sum(e > 0 for e in errors) / n if n else None,
        "undercount_rate": sum(e < 0 for e in errors) / n if n else None,
        "mae_denominator": "valid_responses_only", "small_sample": n < 100,
        "parser": getattr(parser, "__name__", "custom")}
