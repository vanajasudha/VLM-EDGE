"""Zero-shot open-vocabulary detector baseline on LOCO ground truth. No training.

COCO-pretrained closed-vocabulary detectors have no forklift/pallet classes, so the pretrained
baseline is an open-vocabulary model (OWLv2 through ``transformers``; no new dependency).
Backends declare ``kind``: 'real' (model weights) or 'mock' (tests/oracle). Mock output can never be
mistaken for a result: it is written under a ``mock_`` directory and flagged in every file.
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.benchmark_metrics import extended_binary_metrics, extended_count_metrics
from src.detection_metrics import evaluate_detections, operating_point

QUERY_VERSION = "loco_detector_queries_v1"
QUERIES = {7: "a photo of a pallet", 5: "a photo of a forklift", 11: "a photo of a pallet truck",
           10: "a photo of a stillage", 3: "a photo of a small load carrier"}
CATEGORY_NAMES = {3: "small_load_carrier", 5: "forklift", 7: "pallet", 10: "stillage", 11: "pallet_truck"}
DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"
GPU_LOCK_PATH = "/tmp/vlm-edge-evaluation-gpu.lock"   # must match evaluation_runner.QuantizedBackend


def xyxy_to_coco(box, width, height):
    """Clip an xyxy box to the image and convert to xywh; None if nothing is left."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    return [x1, y1, x2 - x1, y2 - y1] if x2 - x1 > 0 and y2 - y1 > 0 else None


def owlv2_target_size(width, height):
    """OWLv2 pads images to a square; boxes are relative to that square, so scale by the longer side."""
    side = max(width, height)
    return side, side


class Owlv2Backend:
    kind = "real"

    def __init__(self, model_id=DEFAULT_MODEL, revision=None, device="cuda", score_floor=0.05,
                 allow_download=False):
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A pinned 40-character model revision is required (never a moving branch)")
        self.model_id, self.revision, self.device = model_id, revision, device
        self.score_floor, self.allow_download = score_floor, allow_download

    def load(self):
        import os
        if not self.allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import fcntl
        self._lock = open(GPU_LOCK_PATH, "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluation process owns the GPU lock") from exc
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            free, _ = torch.cuda.mem_get_info()
            if free < 2 * 2**30:
                raise RuntimeError("Less than 2 GiB free CUDA memory; stop the operator app and other GPU workloads")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = Owlv2Processor.from_pretrained(self.model_id, revision=self.revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=dtype).to(self.device).eval()
        self.torch, self.dtype = torch, dtype
        self.category_ids = list(QUERIES)
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.allocated_after_load_mib = torch.cuda.memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()

    def metadata(self):
        torch = self.torch
        return {"model_id": self.model_id, "revision": self.revision, "device": self.device,
                "dtype": str(self.dtype), "queries": QUERIES, "query_version": QUERY_VERSION,
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
                "parameters": sum(p.numel() for p in self.model.parameters())}

    def detect(self, path):
        from PIL import Image
        torch = self.torch
        with Image.open(path) as im:
            image = im.convert("RGB")
        w, h = image.size
        inputs = self.processor(text=[[QUERIES[c] for c in self.category_ids]], images=image, return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            target = torch.tensor([owlv2_target_size(w, h)], device=self.device)
            post = getattr(self.processor, "image_processor", self.processor)   # method location differs across versions
            result = post.post_process_object_detection(
                outputs=outputs, threshold=self.score_floor, target_sizes=target)[0]
        detections = []
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()):
            coco = xyxy_to_coco(box, w, h)
            if coco:
                detections.append((self.category_ids[int(label)], float(score), coco))
        if self.device == "cuda":
            torch.cuda.synchronize()
        return detections

    def memory_summary(self):
        torch = self.torch
        if self.device != "cuda":
            return None
        return {"allocated_after_load_mib": self.allocated_after_load_mib,
                "peak_allocated_during_inference_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_during_inference_mib": torch.cuda.max_memory_reserved() / 2**20,
                "total_gpu_mib": torch.cuda.get_device_properties(0).total_memory / 2**20}

    def close(self):
        if hasattr(self, "model"):
            del self.model
            self.torch.cuda.empty_cache() if self.device == "cuda" else None
        if hasattr(self, "_lock"):
            self._lock.close()


class OracleBackend:
    """Mock: returns ground truth with score 1. Proves the pipeline/metrics, says nothing about any model."""
    kind = "mock"

    def __init__(self, gt):
        self.by_file = {}
        images = {im["id"]: im["file_name"] for im in gt["images"]}
        for a in gt["annotations"]:
            self.by_file.setdefault(images[a["image_id"]], []).append((a["category_id"], 1.0, a["bbox"]))

    def load(self): pass
    def metadata(self): return {"mock": "oracle returns ground truth"}
    def memory_summary(self): return None
    def close(self): pass

    def detect(self, path):
        for name, dets in self.by_file.items():
            if str(path).replace("\\", "/").endswith(name):
                return list(dets)
        return []


def _plan(gt, backend, *, score_floor, operating_threshold, min_area_frac, ambiguous_band, max_count, max_dets, limit):
    from src.evaluation_manifest import identity
    meta = backend.metadata()
    content = {"schema": 1, "ground_truth_sha256": identity(gt["images"]) , "n_images": len(gt["images"]),
               "n_annotations": len(gt["annotations"]), "backend": backend.kind, "backend_metadata": meta,
               "score_floor": score_floor, "operating_threshold": operating_threshold, "max_dets": max_dets,
               "count_rule": {"min_area_frac": min_area_frac, "ambiguous_band": ambiguous_band, "max_count": max_count},
               "limit": limit, "query_version": QUERY_VERSION, "gt_info": gt.get("info")}
    return {**content, "plan_sha256": identity(content)}


def derived_presence_count(gt_by_image, sizes, preds, *, threshold, min_area_frac, ambiguous_band, max_count):
    """Presence/count answers implied by detections, scored with the SAME rules as the VLM questions."""
    kept = [p for p in preds if p[2] >= threshold]
    out = {}
    for name, cat in (("forklift_presence", 5), ("pallet_presence", 7)):
        truth = ["yes" if any(c == cat for c, _ in g) else "no" for g in gt_by_image.values()]
        found = {p[0] for p in kept if p[1] == cat}
        pred = ["yes" if i in found else "no" for i in gt_by_image]
        out[name] = extended_binary_metrics(truth, pred)
    t, pc, skipped = [], [], Counter()
    for image_id, g in gt_by_image.items():
        area = sizes[image_id][0] * sizes[image_id][1]
        frac = [b[2] * b[3] / area for c, b in g if c == 7]
        if any(min_area_frac * ambiguous_band <= f < min_area_frac for f in frac):
            skipped["ambiguous"] += 1
            continue
        n = sum(f >= min_area_frac for f in frac)
        if n > max_count:
            skipped["above_max_count"] += 1
            continue
        t.append(n)
        pc.append(str(sum(1 for i, c, s, b in kept if i == image_id and c == 7 and b[2] * b[3] / area >= min_area_frac)))
    out["pallet_count"] = extended_count_metrics(t, pc) if t else None
    out["pallet_count_excluded"] = dict(skipped)
    return out


def run_detector_eval(gt, image_root, backend, out_root, *, score_floor=0.05, operating_threshold=0.3,
                      min_area_frac=0.005, ambiguous_band=0.5, max_count=30, max_dets=400, limit=None,
                      resume_dir=None):
    """Sequential, append-only, resumable. Failed images are recorded and counted, never skipped silently."""
    if backend.kind not in {"real", "mock"}:
        raise ValueError("backend.kind must be real or mock")
    images = gt["images"][:limit] if limit else gt["images"]
    backend.load()
    plan = _plan({**gt, "images": images}, backend, score_floor=score_floor, operating_threshold=operating_threshold,
                 min_area_frac=min_area_frac, ambiguous_band=ambiguous_band, max_count=max_count,
                 max_dets=max_dets, limit=limit)
    prefix = "mock_" if backend.kind == "mock" else ""
    if resume_dir:
        run = Path(resume_dir)
        if json.loads((run / "plan.json").read_text())["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Resume refused: the plan (data, model, settings) differs from the interrupted run")
    else:
        run = Path(out_root) / f"{prefix}loco_detector_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
        run.mkdir(parents=True, exist_ok=False)
        with (run / "plan.json").open("x", encoding="utf-8") as f:
            json.dump({**plan, "not_a_real_result": backend.kind == "mock", "started": datetime.now(timezone.utc).isoformat()}, f, indent=1)
    pred_path = run / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["image_id"] for l in pred_path.read_text().splitlines() if l.strip()}
    try:
        with pred_path.open("a", encoding="utf-8") as out:
            for im in images:
                if im["id"] in done:
                    continue
                started, error, dets, latency = perf_counter(), None, [], None
                try:
                    dets = backend.detect(Path(image_root) / im["file_name"])
                    latency = (perf_counter() - started) * 1000
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
                out.write(json.dumps({"image_id": im["id"], "file_name": im["file_name"], "status": "error" if error else "ok",
                                      "latency_ms": latency, "error": error,
                                      "detections": [{"category_id": c, "score": s, "bbox": b} for c, s, b in dets]}) + "\n")
                out.flush()
        memory = backend.memory_summary()
    finally:
        backend.close()
    records = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
    metrics = score_run(gt, records, plan, memory=memory)
    metrics["not_a_real_result"] = backend.kind == "mock"
    with (run / f"metrics_{datetime.now(timezone.utc):%H%M%S%f}.json").open("x", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    (run / f"report_{datetime.now(timezone.utc):%H%M%S%f}.md").write_text(render_report(metrics, plan), encoding="utf-8")
    return run, metrics


def score_run(gt, records, plan, *, memory=None):
    """Score a finished run.

    IMPORTANT: ALL planned images must be present in gt_by_image so that
    evaluate_detections() counts their ground-truth boxes in every denominator.
    Failed images contribute ZERO predictions (not absent from gt_by_image):
    this prevents selectively failing on hard images from inflating AP or
    presence/count accuracy.
    """
    planned_ids = {im["id"] for im in gt["images"]}
    ok = {r["image_id"]: r for r in records if r["status"] == "ok"}
    failed = [r["image_id"] for r in records if r["status"] != "ok"]
    not_recorded = sorted(planned_ids - {r["image_id"] for r in records})
    # Seed gt_by_image with ALL planned images (empty lists = zero predictions for failed ones).
    gt_by_image = {im["id"]: [] for im in gt["images"]}
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    for a in gt["annotations"]:
        if a["image_id"] in gt_by_image:
            gt_by_image[a["image_id"]].append((a["category_id"], a["bbox"]))
    # Only successful images contribute predictions.
    preds = [(i, d["category_id"], d["score"], d["bbox"]) for i, r in ok.items() for d in r["detections"]]
    cats = list(CATEGORY_NAMES)
    maxd = plan["max_dets"]
    det = {area: evaluate_detections(gt_by_image, preds, cats, area=area, max_dets=maxd)
           for area in ("all", "small", "medium", "large")}
    rule = plan["count_rule"]
    latencies = sorted(r["latency_ms"] for r in ok.values() if r["latency_ms"] is not None)
    return {
        "images_planned": plan["n_images"],
        "images_evaluated": len(ok),
        "images_failed": len(failed),
        "images_not_recorded": len(not_recorded),
        "failed_image_ids": failed[:50],
        "not_recorded_image_ids": not_recorded[:50],
        "denominator_note": (
            "All planned images contribute their GT boxes to detection/presence/count "
            "denominators. Failed images contribute zero predictions; they are NOT dropped."
        ),
        "detection": {"mAP@0.5": det["all"]["ap50"], "mAP@[0.5:0.95]": det["all"]["ap"], "AR": det["all"]["ar"],
                      "per_class": {CATEGORY_NAMES[c]: v for c, v in det["all"]["per_category"].items()},
                      "by_size": {a: {"mAP@[0.5:0.95]": det[a]["ap"], "per_class_ap": {CATEGORY_NAMES[c]: v["ap"] for c, v in det[a]["per_category"].items()}} for a in ("small", "medium", "large")},
                      "max_dets": maxd, "protocol": "COCO-style; validated against pycocotools"},
        "operating_point": operating_point(gt_by_image, preds, cats, score_threshold=plan["operating_threshold"], max_dets=maxd),
        "derived_from_detections": derived_presence_count(gt_by_image, sizes, preds, threshold=plan["operating_threshold"],
                                                          min_area_frac=rule["min_area_frac"], ambiguous_band=rule["ambiguous_band"], max_count=rule["max_count"]),
        "latency_ms": ({"n": len(latencies), "p50": latencies[len(latencies) // 2],
                        "p95": latencies[int(0.95 * (len(latencies) - 1))]} if latencies else None),
        "memory": memory,
        "caveat": "Zero-shot detector with fixed text queries; no training or tuning on LOCO."}


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def render_report(m, plan):
    d = m["detection"]
    lines = [f"# LOCO detector baseline ({'MOCK - NOT A RESULT' if m.get('not_a_real_result') else 'zero-shot, untrained'})", "",
             f"Images evaluated {m['images_evaluated']} / planned {m['images_planned']}; failed {m['images_failed']}.",
             f"Model: `{plan['backend_metadata'].get('model_id', 'mock')}` revision `{plan['backend_metadata'].get('revision', 'n/a')}`; "
             f"queries `{plan['query_version']}`; max detections per image {d['max_dets']}.", "",
             f"**mAP@0.5 = {_fmt(d['mAP@0.5'])}   mAP@[0.5:0.95] = {_fmt(d['mAP@[0.5:0.95]'])}   AR = {_fmt(d['AR'])}**", "",
             "| class | GT boxes | AP@0.5 | AP@[.5:.95] | precision@thr | recall@thr |", "|---|---|---|---|---|---|"]
    op = m["operating_point"]["per_category"]
    for cid, name in CATEGORY_NAMES.items():
        pc, o = d["per_class"][name], op[cid]
        lines.append(f"| {name} | {pc['n_gt']} | {_fmt(pc['ap50'])} | {_fmt(pc['ap'])} | {_fmt(o['precision'])} | {_fmt(o['recall'])} |")
    lines += ["", f"Operating point: score >= {m['operating_point']['score_threshold']}, IoU 0.5.", "",
              "| size bucket | mAP@[.5:.95] |", "|---|---|"]
    lines += [f"| {a} | {_fmt(v['mAP@[0.5:0.95]'])} |" for a, v in d["by_size"].items()]
    lines += ["", "## Presence / counting implied by the detections (same rules as the VLM questions)", ""]
    for task in ("forklift_presence", "pallet_presence"):
        b = m["derived_from_detections"][task]
        lines.append(f"- **{task}**: n={b['n']}, recall {_fmt(b['recall'])} (FNR {_fmt(b['false_negative_rate'])}), "
                     f"precision {_fmt(b['precision'])}, F1 {_fmt(b['f1'])}, TP/FP/FN/TN {b['tp']}/{b['fp']}/{b['fn']}/{b['tn']}"
                     + ("  _(small sample)_" if b["small_sample"] else ""))
    c = m["derived_from_detections"]["pallet_count"]
    if c:
        lines.append(f"- **pallet_count**: n={c['n']}, exact {_fmt(c['exact_match'])}, MAE {_fmt(c['mean_abs_error'])}, "
                     f"within-one {_fmt(c['within_one'])}, excluded {m['derived_from_detections']['pallet_count_excluded']}")
    lines += ["", f"Latency (ms): {m['latency_ms']}", f"GPU memory: {m['memory']}", "", f"_{m['caveat']}_"]
    return "\n".join(lines) + "\n"
