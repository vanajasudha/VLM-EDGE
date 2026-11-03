"""COCO-style detection metrics (AP, AR, precision/recall at an operating point). numpy only.

Follows the COCO evaluation protocol (greedy score-ordered matching, ignore regions for
area ranges, 101-point interpolated precision), without the pycocotools dependency.
Deliberate difference: ``max_dets`` defaults to 400, not COCO's 100, because LOCO images
contain up to ~345 pallets and a 100-detection cap would hard-limit recall.
Boxes are COCO ``[x, y, width, height]``. No ``iscrowd`` support (LOCO has none).
"""
import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
AREA_RANGES = {"all": (0.0, 1e10), "small": (0.0, 32.0 ** 2), "medium": (32.0 ** 2, 96.0 ** 2),
               "large": (96.0 ** 2, 1e10)}


def box_iou(a, b):
    """IoU matrix between (N,4) and (M,4) xywh arrays."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _match_image(gt_boxes, gt_areas, dets, thresholds, area_rng, max_dets):
    """COCO evaluateImg for one (image, category). Returns per-threshold (scores, tp, ignore) + n_gt."""
    gt_ignore = np.array([not (area_rng[0] <= a <= area_rng[1]) for a in gt_areas], dtype=bool)
    order = np.argsort(gt_ignore, kind="stable")                      # non-ignored GT first
    gt_boxes, gt_ignore = np.asarray(gt_boxes, float).reshape(-1, 4)[order], gt_ignore[order]
    dets = sorted(dets, key=lambda d: -d[0])[:max_dets]
    scores = np.array([d[0] for d in dets], float)
    boxes = np.array([d[1] for d in dets], float).reshape(-1, 4)
    ious = box_iou(boxes, gt_boxes)
    det_areas = boxes[:, 2] * boxes[:, 3] if len(boxes) else np.zeros(0)
    n_t, n_d, n_g = len(thresholds), len(dets), len(gt_boxes)
    tp = np.zeros((n_t, n_d), bool)
    ignore = np.zeros((n_t, n_d), bool)
    can_match = ious.max(axis=1) if n_g and n_d else np.zeros(n_d)
    for ti, thr in enumerate(thresholds):
        matched = np.zeros(n_g, bool)
        floor = min(thr, 1 - 1e-10)
        for di in range(n_d):
            m = -1
            if can_match[di] >= floor:                                 # fast path: nothing overlaps enough
                candidates = ~matched & (ious[di] >= floor)
                pool = candidates & ~gt_ignore
                if not pool.any():
                    pool = candidates & gt_ignore
                if pool.any():
                    best = ious[di][pool].max()
                    m = int(np.flatnonzero(pool & (ious[di] == best))[-1])   # COCO: ties go to the later GT
            if m == -1:
                ignore[ti, di] = not (area_rng[0] <= det_areas[di] <= area_rng[1])
            else:
                matched[m] = True
                ignore[ti, di] = gt_ignore[m]
                tp[ti, di] = not gt_ignore[m]
    return scores, tp, ignore, int((~gt_ignore).sum())


def evaluate_detections(ground_truth, predictions, categories, *, area="all", max_dets=400,
                        thresholds=IOU_THRESHOLDS):
    """Compute AP/AR per category.

    ground_truth: {image_id: [(category, [x, y, w, h]), ...]}  (every evaluated image must be a key,
                  including images without objects, so their false positives are counted)
    predictions:  [(image_id, category, score, [x, y, w, h]), ...]
    Returns {"per_category": {cat: {"ap", "ap50", "ar", "n_gt"}}, "ap", "ap50", "ar", ...}.
    Categories without ground truth are reported with None and excluded from means (COCO behaviour).
    """
    thresholds = np.asarray(thresholds, float)
    rng = AREA_RANGES[area]
    unknown = {p[0] for p in predictions} - set(ground_truth)
    if unknown:
        raise ValueError(f"Predictions reference images outside the evaluated set: {sorted(unknown)[:3]}")
    per_category = {}
    for cat in categories:
        scores_all, tp_all, ig_all, n_gt = [], [], [], 0
        for image_id, gts in ground_truth.items():
            g = [box for c, box in gts if c == cat]
            d = [(s, box) for i, c, s, box in predictions if i == image_id and c == cat] if predictions else []
            if not g and not d:
                continue
            areas = [b[2] * b[3] for b in g]
            s, tp, ig, n = _match_image(g, areas, d, thresholds, rng, max_dets)
            scores_all.append(s); tp_all.append(tp); ig_all.append(ig); n_gt += n
        per_category[cat] = _summarise(scores_all, tp_all, ig_all, n_gt, thresholds)
    valid = [v for v in per_category.values() if v["n_gt"] > 0]
    mean = lambda key: float(np.mean([v[key] for v in valid])) if valid else None
    ap50_values = [v["ap50"] for v in valid if v["ap50"] is not None]
    return {"per_category": per_category, "ap": mean("ap"),
            "ap50": float(np.mean(ap50_values)) if ap50_values else None, "ar": mean("ar"),
            "area": area, "max_dets": max_dets, "iou_thresholds": thresholds.tolist(),
            "n_categories_scored": len(valid)}


def _summarise(scores_all, tp_all, ig_all, n_gt, thresholds):
    if n_gt == 0:
        return {"ap": None, "ap50": None, "ar": None, "n_gt": 0}
    if scores_all and sum(len(s) for s in scores_all):
        scores = np.concatenate(scores_all)
        order = np.argsort(-scores, kind="mergesort")
        tp = np.concatenate(tp_all, axis=1)[:, order]
        ig = np.concatenate(ig_all, axis=1)[:, order]
    else:
        tp = ig = np.zeros((len(thresholds), 0), bool)
    aps, recalls = [], []
    for ti in range(len(thresholds)):
        keep = ~ig[ti]
        t = tp[ti][keep]
        ctp, cfp = np.cumsum(t), np.cumsum(~t)
        if not len(t):
            aps.append(0.0); recalls.append(0.0)
            continue
        recall = ctp / n_gt
        precision = ctp / (ctp + cfp)
        precision = np.maximum.accumulate(precision[::-1])[::-1]      # monotone envelope
        idx = np.searchsorted(recall, RECALL_POINTS, side="left")
        q = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
        aps.append(float(q.mean()))
        recalls.append(float(recall[-1]))
    return {"ap": float(np.mean(aps)), "ap50": aps[0] if abs(thresholds[0] - 0.5) < 1e-9 else None,
            "ar": float(np.mean(recalls)), "n_gt": n_gt}


def operating_point(ground_truth, predictions, categories, *, score_threshold, iou=0.5, max_dets=400):
    """Precision/recall/F1 at a fixed score threshold and IoU (a deployment-style number, not AP)."""
    out = {}
    kept = [p for p in predictions if p[2] >= score_threshold]
    for cat in categories:
        tp = fp = n_gt = 0
        for image_id, gts in ground_truth.items():
            g = [b for c, b in gts if c == cat]
            d = [(s, b) for i, c, s, b in kept if i == image_id and c == cat]
            scores, tps, ign, n = _match_image(g, [b[2] * b[3] for b in g], d, np.array([iou]), AREA_RANGES["all"], max_dets)
            tp += int(tps[0].sum()); fp += int((~tps[0]).sum()); n_gt += n
        out[cat] = {"tp": tp, "fp": fp, "fn": n_gt - tp, "n_gt": n_gt,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / n_gt if n_gt else None,
                    "f1": 2 * tp / (2 * tp + fp + (n_gt - tp)) if (2 * tp + fp + (n_gt - tp)) else None}
    return {"score_threshold": score_threshold, "iou": iou, "per_category": out}
