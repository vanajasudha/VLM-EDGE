"""Join a finished Moondream2 evaluation run with the LOCO sidecar: stratified metrics + error review list.

Reads the artifacts written by ``src.evaluation_runner.run_evaluation`` (``predictions.jsonl``,
``raw_answers.jsonl``) and a sidecar from ``loco_dataset.export_vlm_manifest``. Never modifies them.
Strict parsing (bare yes/no, bare integer) is the primary result; lenient parsing is reported
beside it so prompt-format failures are distinguishable from perception failures.
"""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.benchmark_metrics import (extended_binary_metrics, extended_count_metrics, normalize_binary,
                                   normalize_count, parse_binary_lenient, parse_count_lenient)

STRATA = ("subset", "sensor", "group_id")


def _read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_run(run_dir, sidecar_csv):
    run = Path(run_dir)
    preds = {r["sample_id"]: r for r in _read_jsonl(run / "predictions.jsonl")}
    raws = {r["sample_id"]: r for r in _read_jsonl(run / "raw_answers.jsonl")}
    with open(sidecar_csv, newline="", encoding="utf-8") as f:
        side = {r["sample_id"]: r for r in csv.DictReader(f)}
    joined, unmatched = [], []
    for sid, p in preds.items():
        if sid not in side:
            unmatched.append(sid)
            continue
        joined.append({**side[sid], "status": p["status"], "raw_answer": raws.get(sid, {}).get("raw_answer"),
                       "error": p.get("error"), "latency_ms": p.get("inference_latency_ms"),
                       "image_path": p["image_path"], "execution": p["execution"]})
    return joined, unmatched, json.loads((run / "summary.json").read_text(encoding="utf-8"))


def _score(task, items, lenient):
    truth = [i["reference_answer"] for i in items]
    # Failed requests have no raw answer: they stay in the denominator as invalid, never dropped.
    answers = [i["raw_answer"] if i["status"] == "ok" else None for i in items]
    if task == "pallet_count":
        return extended_count_metrics(truth, answers, parser=parse_count_lenient if lenient else normalize_count)
    return extended_binary_metrics(truth, answers, parser=parse_binary_lenient if lenient else normalize_binary)


def analyze(run_dir, sidecar_csv, out_root, *, min_stratum=10):
    items, unmatched, summary = load_run(run_dir, sidecar_csv)
    if unmatched:
        raise ValueError(f"{len(unmatched)} predictions have no sidecar row (wrong sidecar for this run?)")
    by_task = defaultdict(list)
    for i in items:
        by_task[i["task"]].append(i)
    result = {"run_dir": str(run_dir), "execution": summary.get("execution"),
              "not_a_real_accuracy_result": summary.get("not_a_real_accuracy_result"),
              "run_status": summary.get("status"), "memory": summary.get("memory"), "tasks": {}}
    errors = []
    for task, group in sorted(by_task.items()):
        entry = {"n": len(group), "failed_requests": sum(i["status"] != "ok" for i in group),
                 "strict": _score(task, group, False), "lenient": _score(task, group, True), "strata": {}}
        keys = list(STRATA) + (["count_bucket"] if task == "pallet_count" else [])
        if task == "forklift_presence":
            for i in group:
                i["pallet_truck_in_image"] = "yes" if int(i["n_pallet_truck"]) > 0 else "no"
            keys.append("pallet_truck_in_image")
        for key in keys:
            buckets = defaultdict(list)
            for i in group:
                buckets[i[key] or "(none)"].append(i)
            entry["strata"][key] = {
                value: {"n": len(v), "below_min_stratum": len(v) < min_stratum, "strict": _score(task, v, False),
                        "lenient": _score(task, v, True)} for value, v in sorted(buckets.items())}
        for i in group:
            if task == "pallet_count":
                ref, got = int(i["reference_answer"]), normalize_count(i["raw_answer"]) if i["status"] == "ok" else None
                kind = ("failed_request" if i["status"] != "ok" else "invalid_format" if got is None else
                        "overcount" if got > ref else "undercount" if got < ref else None)
            else:
                got = normalize_binary(i["raw_answer"]) if i["status"] == "ok" else None
                ref = i["reference_answer"]
                kind = ("failed_request" if i["status"] != "ok" else "invalid_format" if got is None else
                        "false_positive" if (ref, got) == ("no", "yes") else "false_negative" if (ref, got) == ("yes", "no") else None)
            if kind:
                errors.append({"task": task, "error_type": kind, "reference": i["reference_answer"], "raw_answer": i["raw_answer"],
                               "image_key": i["image_key"], "image_path": i["image_path"], "subset": i["subset"],
                               "sensor": i["sensor"], "group_id": i["group_id"], "n_pallet_truck": i["n_pallet_truck"],
                               "pallet_count_min_area": i["pallet_count_min_area"], "sample_id": i["sample_id"]})
        result["tasks"][task] = entry
    out = Path(out_root) / f"loco_vlm_analysis_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "analysis.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    if errors:
        with (out / "errors.csv").open("x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(errors[0]))
            w.writeheader()
            w.writerows(errors)
    (out / "analysis.md").write_text(render(result, len(errors)), encoding="utf-8")
    return out, result


def _f(v):
    return "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def render(result, n_errors):
    flag = "MOCK RUN - NOT A REAL ACCURACY RESULT" if result.get("not_a_real_accuracy_result") else "real execution"
    lines = [f"# LOCO VLM baseline analysis ({flag})", "", f"Run status: {result['run_status']}. Error review list: errors.csv ({n_errors} rows).",
             f"GPU memory: {result['memory']}", ""]
    for task, e in result["tasks"].items():
        lines += [f"## {task} (n={e['n']}, failed requests {e['failed_requests']})", ""]
        for name in ("strict", "lenient"):
            m = e[name]
            if task == "pallet_count":
                lines.append(f"- **{name}**: exact {_f(m['exact_match'])} (95% CI {m['exact_match_ci95']}), MAE {_f(m['mean_abs_error'])}, "
                             f"within-one {_f(m['within_one'])}, over {_f(m['overcount_rate'])}, under {_f(m['undercount_rate'])}, invalid {_f(m['invalid_response_rate'])}")
            else:
                lines.append(f"- **{name}**: acc {_f(m['accuracy'])}, precision {_f(m['precision'])}, recall {_f(m['recall'])}, F1 {_f(m['f1'])}, "
                             f"FNR {_f(m['false_negative_rate'])}, spec {_f(m['specificity'])}, invalid {_f(m['invalid_response_rate'])}, "
                             f"TP/FP/FN/TN {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}" + ("  _(small sample)_" if m["small_sample"] else ""))
        lines.append("")
        for key, buckets in e["strata"].items():
            lines += [f"**by {key}** (strict)", "", "| value | n | " + ("exact | MAE |" if task == "pallet_count" else "recall | specificity | invalid |"),
                      "|---|---|" + ("---|---|" if task == "pallet_count" else "---|---|---|")]
            for value, b in buckets.items():
                s = b["strict"]
                cells = (f"{_f(s['exact_match'])} | {_f(s['mean_abs_error'])}" if task == "pallet_count" else
                         f"{_f(s['recall'])} | {_f(s['specificity'])} | {_f(s['invalid_response_rate'])}")
                lines.append(f"| {value}{' (n<min)' if b['below_min_stratum'] else ''} | {b['n']} | {cells} |")
            lines.append("")
    lines.append("_Causes are not asserted: review errors.csv together with the images._")
    return "\n".join(lines) + "\n"
