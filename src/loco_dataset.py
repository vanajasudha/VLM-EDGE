"""LOCO (Logistics Objects in Context) integration for VLM-Edge.

Mayershofer, Holm, Molter, Fottner. "LOCO: Logistics Objects in Context", IEEE ICMLA 2020.
Data: CC0-1.0, https://github.com/tum-fml/loco (annotations) - images are hosted by TUM.

Design rules
* The five official subset files are authoritative (image ids in loco-all-v1.json are renumbered).
  Paths, never ids, are the stable image key.
* Official environment split is preserved: subsets 2, 3, 5 = train; subsets 1, 4 = official val.
  Official val becomes the project's frozen ``test`` split. A project ``validation`` split is
  carved only from official-train, by whole recording group, never by frame.
* Presence/count labels are deterministic functions of LOCO's human boxes. No text is invented.
* No image is downloaded or copied into Git by this module.
No model or GPU imports.
"""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CITATION = ("Mayershofer, C., Holm, D.-M., Molter, B., Fottner, J. LOCO: Logistics Objects in Context. "
            "IEEE International Conference on Machine Learning and Applications (ICMLA), 2020.")
LICENSE = "CC0-1.0"
CLASSES = {3: "small_load_carrier", 5: "forklift", 7: "pallet", 10: "stillage", 11: "pallet_truck"}
SUBSETS = {  # key: (annotation file, official split)
    "sub1": ("loco-sub1-v1-val.json", "val"), "sub2": ("loco-sub2-v1-train.json", "train"),
    "sub3": ("loco-sub3-v1-train.json", "train"), "sub4": ("loco-sub4-v1-val.json", "val"),
    "sub5": ("loco-sub5-v1-train.json", "train")}
README_CLAIMS = {"images_total": 37988, "images_annotated": 5593, "annotations": 152421}
OFFICIAL_TO_PROJECT = {"val": "test"}          # official train is split further into train/validation
RULES = {"forklift_presence": "forklift_any_box_v1", "pallet_presence": "pallet_any_box_v1",
         "pallet_count": "pallet_boxes_min_area_v1"}
# COUNT RULE NOTE: pallet_count labels are annotation-derived proxies (LOCO human bounding boxes
# filtered by a minimum relative area threshold). They are NOT human-verified counts of
# *clearly visible* pallets. See build_index() and BENCHMARK_PROTOCOL.md for the rule definition.
COUNT_RULE_PROXY_NOTE = (
    "pallet_count is an annotation-based proxy: LOCO human bounding boxes whose relative area "
    "is >= min_area_frac. It is NOT a human-verified count of clearly visible pallets."
)
PROMPTS = {  # versioned; the version is embedded in every sample's label_provenance
    "version": "loco_prompts_v1",
    "forklift_presence": "Is a forklift visible in this image? Answer with only yes or no.",
    "pallet_presence": "Are any pallets visible in this image? Answer with only yes or no.",
    "pallet_count": "How many pallets are clearly visible in this image? Answer with only a single whole number."}
COUNT_BUCKETS = ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10 ** 9, "21+"))
INDEX_FIELDS = ["image_key", "subset", "official_split", "group_id", "sensor", "width", "height",
                "loco_image_id", "n_small_load_carrier", "n_forklift", "n_pallet", "n_stillage", "n_pallet_truck",
                "n_polygon_only", "forklift_present", "pallet_present", "pallet_count_min_area",
                "pallet_count_ambiguous", "min_area_frac", "ambiguous_band"]


class LocoError(ValueError):
    pass


class SplitFrozenError(LocoError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_path(path):
    """'/dataset/subset-2/2019-11-25_0/Kinect/color/x.jpg' -> key, group, sensor (no guessing)."""
    parts = [p for p in str(path).split("/") if p]
    if parts and parts[0] == "dataset":
        parts = parts[1:]
    if len(parts) < 2 or not parts[0].startswith("subset-"):
        raise LocoError(f"Unexpected LOCO image path: {path!r}")
    subset_dir = parts[0]
    session = parts[1] if len(parts) > 2 else ""
    sensor = parts[2] if len(parts) > 3 else ""
    return {"image_key": "/".join(parts), "subset_dir": subset_dir, "session": session,
            "group_id": f"{subset_dir}/{session}" if session else subset_dir, "sensor": sensor}


# ------------------------------------------------------------------ annotations
def load_annotation_sets(annotation_dir):
    base = Path(annotation_dir)
    sets = {}
    for key, (filename, _) in SUBSETS.items():
        path = base / filename
        if not path.is_file():
            raise LocoError(f"Missing annotation file: {path}")
        sets[key] = json.loads(path.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in sets[key]["categories"]}
        if names != CLASSES:
            raise LocoError(f"{filename}: unexpected categories {names}")
    return sets


def build_index(sets, *, min_area_frac=0.005, ambiguous_band=0.5):
    """One row per image. Count rule 'pallet_boxes_min_area_v1': annotated pallet boxes whose area is
    at least ``min_area_frac`` of the image. Pallets between ``band*T`` and ``T`` are 'ambiguous';
    images containing any are excluded from counting exports, not forced into an exact count."""
    if not 0 < min_area_frac < 1 or not 0 < ambiguous_band < 1:
        raise LocoError("min_area_frac and ambiguous_band must be in (0, 1)")
    rows = []
    for key, data in sets.items():
        official = SUBSETS[key][1]
        per_image = defaultdict(list)
        for a in data["annotations"]:
            per_image[a["image_id"]].append(a)
        for im in data["images"]:
            info = parse_path(im["path"])
            if info["subset_dir"] != f"subset-{key[-1]}":
                raise LocoError(f"{im['path']} is not inside {key}")
            anns = per_image.get(im["id"], [])
            counts = Counter(CLASSES[a["category_id"]] for a in anns)
            area = im["width"] * im["height"]
            fracs = [a["bbox"][2] * a["bbox"][3] / area for a in anns if a["category_id"] == 7]
            rows.append({
                "image_key": info["image_key"], "subset": key, "official_split": official,
                "group_id": info["group_id"], "sensor": info["sensor"], "width": im["width"],
                "height": im["height"], "loco_image_id": im["id"],
                **{f"n_{n}": counts[n] for n in CLASSES.values()},
                "n_polygon_only": sum(1 for a in anns if a.get("isbbox") is False),
                "forklift_present": counts["forklift"] > 0, "pallet_present": counts["pallet"] > 0,
                "pallet_count_min_area": sum(f >= min_area_frac for f in fracs),
                "pallet_count_ambiguous": sum(min_area_frac * ambiguous_band <= f < min_area_frac for f in fracs),
                "min_area_frac": min_area_frac, "ambiguous_band": ambiguous_band})
    keys = [r["image_key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise LocoError("Duplicate image paths across subset files")
    return rows


def write_csv(path, rows, fields):
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_index(path):
    ints = {"width", "height", "loco_image_id", "n_polygon_only", "pallet_count_min_area", "pallet_count_ambiguous",
            *(f"n_{n}" for n in CLASSES.values())}
    bools = {"forklift_present", "pallet_present"}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ints:
                r[k] = int(r[k])
            for k in bools:
                r[k] = r[k] == "True"
            r["min_area_frac"], r["ambiguous_band"] = float(r["min_area_frac"]), float(r["ambiguous_band"])
            rows.append(r)
    return rows


# ------------------------------------------------------------------ integrity / statistics
def integrity_report(sets, rows):
    """Everything checkable from annotation files alone (no images needed)."""
    img_ids, ann_ids, names, issues = Counter(), Counter(), Counter(), Counter()
    per_subset, areas = {}, defaultdict(lambda: Counter())
    rel_area = defaultdict(list)
    for key, data in sets.items():
        sizes = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
        with_ann = set()
        cls_ann, cls_img = Counter(), defaultdict(set)
        for im in data["images"]:
            img_ids[im["id"]] += 1
            names[im["file_name"]] += 1
        for a in data["annotations"]:
            ann_ids[a["id"]] += 1
            if a["image_id"] not in sizes:
                issues["orphan_annotation"] += 1
                continue
            w, h = sizes[a["image_id"]]
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                issues["nonpositive_box"] += 1
            if x < -1 or y < -1 or x + bw > w + 1 or y + bh > h + 1:
                issues["box_out_of_image"] += 1
            if a.get("iscrowd"):
                issues["iscrowd"] += 1
            if a.get("isbbox") is False:
                issues["polygon_only_annotation"] += 1
            if a["category_id"] not in CLASSES:
                issues["unknown_category"] += 1
                continue
            name = CLASSES[a["category_id"]]
            with_ann.add(a["image_id"])
            cls_ann[name] += 1
            cls_img[name].add(a["image_id"])
            bucket = "small" if bw * bh < 32 ** 2 else "medium" if bw * bh < 96 ** 2 else "large"
            areas[name][bucket] += 1
            rel_area[name].append(bw * bh / (w * h))
        per_subset[key] = {"official_split": SUBSETS[key][1], "images": len(data["images"]),
                           "annotations": len(data["annotations"]),
                           "images_without_annotations": len(data["images"]) - len(with_ann),
                           "annotations_per_class": {n: cls_ann[n] for n in CLASSES.values()},
                           "images_with_class": {n: len(cls_img[n]) for n in CLASSES.values()}}
    split_summary = {}
    for official in ("train", "val"):
        sub = [r for r in rows if r["official_split"] == official]
        positives = sum(r["forklift_present"] for r in sub)
        pallet_counts = sorted(r["n_pallet"] for r in sub)
        q = lambda p: pallet_counts[int(p * (len(pallet_counts) - 1))] if pallet_counts else None
        split_summary[official] = {
            "images": len(sub), "groups": len({r["group_id"] for r in sub}),
            "forklift_positive_images": positives,
            "forklift_positive_fraction": positives / len(sub) if sub else None,
            "pallet_negative_images": sum(not r["pallet_present"] for r in sub),
            "pallet_boxes_per_image": {"p50": q(.5), "p90": q(.9), "p99": q(.99), "max": pallet_counts[-1] if pallet_counts else None},
            "images_with_more_than_20_pallet_boxes": sum(r["n_pallet"] > 20 for r in sub),
            "images_countable_min_area_rule": sum(r["pallet_count_ambiguous"] == 0 for r in sub)}
    totals = {"images": sum(s["images"] for s in per_subset.values()),
              "annotations": sum(s["annotations"] for s in per_subset.values())}
    cls_totals = {n: sum(s["annotations_per_class"][n] for s in per_subset.values()) for n in CLASSES.values()}
    return {
        "totals": totals, "annotations_per_class": cls_totals,
        "readme_claims": README_CLAIMS,
        "readme_discrepancy": {"images_annotated": totals["images"] - README_CLAIMS["images_annotated"],
                               "annotations": totals["annotations"] - README_CLAIMS["annotations"]},
        "duplicate_image_ids": sum(v > 1 for v in img_ids.values()),
        "duplicate_file_names": sum(v > 1 for v in names.values()),
        "duplicate_annotation_ids": sum(v > 1 for v in ann_ids.values()),
        "annotation_issues": dict(issues), "per_subset": per_subset, "split_summary": split_summary,
        "box_size_buckets_coco_pixels": {n: dict(areas[n]) for n in CLASSES.values()},
        "median_relative_box_area": {n: sorted(v)[len(v) // 2] for n, v in rel_area.items() if v},
        "sensors": dict(Counter(r["sensor"] or "(none)" for r in rows)),
        "groups": len({r["group_id"] for r in rows}),
        "image_sizes": {f"{w}x{h}": c for (w, h), c in Counter((r["width"], r["height"]) for r in rows).most_common()},
        "notes": ["Every image has at least one annotation: images that contain none of the target objects "
                  "do not exist in LOCO, so negatives are 'target absent, other logistics objects present'.",
                  "Forklifts are rare, especially in official val subset 1.",
                  "Annotated pallets include very small, distant boxes; counts are not 'clearly visible' counts."]}


def expected_stats(annotation_dir, sets):
    """Fingerprint of a verified download (file hashes and per-subset counts)."""
    return {"source": "https://github.com/tum-fml/loco", "license": LICENSE, "citation": CITATION,
            "files": {SUBSETS[k][0]: sha256_file(Path(annotation_dir) / SUBSETS[k][0]) for k in SUBSETS},
            "counts": {k: {"images": len(d["images"]), "annotations": len(d["annotations"])} for k, d in sets.items()}}


def verify_expected(annotation_dir, sets, expected):
    problems = []
    for name, digest in expected["files"].items():
        if sha256_file(Path(annotation_dir) / name) != digest:
            problems.append(f"{name}: SHA-256 differs from the pinned verified download")
    for k, c in expected["counts"].items():
        got = {"images": len(sets[k]["images"]), "annotations": len(sets[k]["annotations"])}
        if got != c:
            problems.append(f"{k}: counts {got} != pinned {c}")
    return problems


# ------------------------------------------------------------------ splits
def assign_project_splits(rows, *, validation_fraction=0.15, seed=0):
    """official val -> test (frozen). official train -> train/validation by WHOLE group (never by frame)."""
    if not 0 <= validation_fraction < 1:
        raise LocoError("validation_fraction must be in [0, 1)")
    groups = defaultdict(int)
    for r in rows:
        if r["official_split"] == "train":
            groups[r["group_id"]] += 1
    total = sum(groups.values())
    ordered = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    validation, taken = set(), 0
    for g in ordered:
        if taken >= validation_fraction * total:
            break
        validation.add(g)
        taken += groups[g]
    out = {}
    for r in rows:
        out[r["image_key"]] = (OFFICIAL_TO_PROJECT["val"] if r["official_split"] == "val"
                               else "validation" if r["group_id"] in validation else "train")
    return out


def check_split_integrity(rows, splits):
    """Hard checks used by tests and the CLI. Returns a list of problems (empty = clean)."""
    problems, by_group = [], defaultdict(set)
    for r in rows:
        s = splits.get(r["image_key"])
        if s is None:
            problems.append(f"{r['image_key']}: no split assigned")
            continue
        by_group[r["group_id"]].add(s)
        if r["official_split"] == "val" and s != "test":
            problems.append(f"{r['image_key']}: official val image outside project test")
        if r["official_split"] == "train" and s == "test":
            problems.append(f"{r['image_key']}: official train image in project test")
    for g, s in by_group.items():
        if len(s) > 1:
            problems.append(f"group {g} spans splits {sorted(s)} (leakage)")
    subsets = defaultdict(set)
    for r in rows:
        subsets[r["subset"]].add(r["official_split"])
    for sub, off in subsets.items():
        if len(off) > 1:
            problems.append(f"subset {sub} has mixed official splits")
    return problems


def freeze_splits(path, rows, *, validation_fraction=0.15, seed=0):
    """Create splits.json once. Re-running verifies it is unchanged; it never silently rewrites."""
    path = Path(path)
    splits = assign_project_splits(rows, validation_fraction=validation_fraction, seed=seed)
    problems = check_split_integrity(rows, splits)
    if problems:
        raise LocoError("; ".join(problems[:5]))
    body = {"schema": 1, "seed": seed, "validation_fraction": validation_fraction,
            "rule": "official val->test; official train: whole groups -> validation by seeded hash; rest train",
            "counts": dict(Counter(splits.values())), "splits": dict(sorted(splits.items()))}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != digest:
            raise SplitFrozenError(f"{path} is frozen and differs from the requested split; refusing to overwrite")
        return existing
    record = {**body, "sha256": digest, "frozen_at": datetime.now(timezone.utc).isoformat()}
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    return record


# ------------------------------------------------------------------ image verification
IMAGE_CHECK_FIELDS = ["image_key", "status", "resolved_path", "resolution", "sha256", "dhash",
                      "width_px", "height_px", "detail"]


def _scan_files(image_root):
    by_name = defaultdict(list)
    for dirpath, _, files in os.walk(image_root):
        for name in files:
            by_name[name].append(os.path.relpath(os.path.join(dirpath, name), image_root).replace(os.sep, "/"))
    return by_name


def resolve_image(image_root, key, by_name):
    """Exact relative path first; else a UNIQUE file-name match (LOCO file names are globally unique)."""
    for candidate in (key, "dataset/" + key):
        if (Path(image_root) / candidate).is_file():
            return candidate, "exact"
    matches = by_name.get(key.rsplit("/", 1)[-1], [])
    if len(matches) == 1:
        return matches[0], "unique_filename"
    return None, ("ambiguous_filename" if matches else "missing")


def verify_images(rows, image_root, out_csv, *, progress=None):
    """Hash + decode + size-check every image. Resumable: rows already in out_csv are skipped."""
    from PIL import Image
    from src.warehouse_annotation import dhash
    image_root, out_csv = Path(image_root), Path(out_csv)
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            done = {r["image_key"] for r in csv.DictReader(f)}
    by_name = _scan_files(image_root)
    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=IMAGE_CHECK_FIELDS)
        if new_file:
            writer.writeheader()
        for n, r in enumerate(rows, 1):
            if r["image_key"] in done:
                continue
            rel, how = resolve_image(image_root, r["image_key"], by_name)
            rec = dict.fromkeys(IMAGE_CHECK_FIELDS, "")
            rec.update(image_key=r["image_key"], resolution=how)
            if rel is None:
                rec.update(status=how, detail="no file found under the image root")
            else:
                path = image_root / rel
                rec["resolved_path"] = rel
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        im.load()
                        w, h = im.size
                    rec.update(sha256=sha256_file(path), dhash=f"{dhash(path):016x}", width_px=w, height_px=h)
                    ok = (w, h) == (r["width"], r["height"])
                    rec.update(status="ok" if ok else "size_mismatch",
                               detail="" if ok else f"annotation says {r['width']}x{r['height']}, file is {w}x{h}")
                except Exception as exc:
                    rec.update(status="unreadable", detail=f"{type(exc).__name__}: {exc}")
            writer.writerow(rec)
            f.flush()
            if progress and n % 250 == 0:
                progress(n, len(rows))
    return read_image_checks(out_csv)


def read_image_checks(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image_key"]: r for r in csv.DictReader(f)}


def leakage_report(rows, checks, splits, *, near_threshold=5, max_listed=25):
    """Exact (SHA-256) and near (dHash Hamming) duplicates, classified by what they imply."""
    import numpy as np
    usable = [r for r in rows if checks.get(r["image_key"], {}).get("status") == "ok"]
    report = {"images_checked": len(usable), "images_not_ok": len(rows) - len(usable),
              "status_counts": dict(Counter(c["status"] for c in checks.values())),
              "near_threshold_hamming": near_threshold, "near_duplicates_are_candidates_only": True}
    by_sha = defaultdict(list)
    for r in usable:
        by_sha[checks[r["image_key"]]["sha256"]].append(r)
    exact = [g for g in by_sha.values() if len(g) > 1]
    report["exact_duplicate_groups"] = len(exact)
    report["exact_duplicate_cross_split_groups"] = sum(len({splits[r["image_key"]] for r in g}) > 1 for g in exact)
    report["exact_duplicate_examples"] = [[r["image_key"] for r in g] for g in exact[:max_listed]]
    if usable:
        hashes = np.array([int(checks[r["image_key"]]["dhash"], 16) for r in usable], dtype=np.uint64)
        table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        kinds, listed = Counter(), []
        for i in range(len(usable) - 1):
            dist = table[(hashes[i + 1:] ^ hashes[i]).view(np.uint8).reshape(-1, 8)].sum(axis=1)
            for j in np.flatnonzero(dist <= near_threshold):
                a, b = usable[i], usable[i + 1 + int(j)]
                kind = ("same_group" if a["group_id"] == b["group_id"] else
                        "cross_group_same_split" if splits[a["image_key"]] == splits[b["image_key"]] else "cross_split")
                kinds[kind] += 1
                if kind == "cross_split" and len(listed) < max_listed:
                    listed.append({"a": a["image_key"], "b": b["image_key"], "hamming": int(dist[j])})
        report["near_duplicate_pairs"] = dict(kinds)
        report["near_duplicate_cross_split_examples"] = listed
    return report


# ------------------------------------------------------------------ exports
def count_bucket(n):
    return next(label for lo, hi, label in COUNT_BUCKETS if lo <= n <= hi)


def _rank(seed, task, key):
    return hashlib.sha256(f"{seed}|{task}|{key}".encode()).hexdigest()


def select_samples(rows, splits, split, *, per_task=100, max_count=30, seed=0):
    """Seeded, deterministic selection. Binary tasks: balanced positives/negatives (prevalence is NOT
    representative - reported). Count task: stratified over count buckets, ambiguous images excluded."""
    pool = [r for r in rows if splits[r["image_key"]] == split]
    plan, report = [], {"split": split, "available_images": len(pool), "per_task_target": per_task, "tasks": {}}
    for task, flag in (("forklift_presence", "forklift_present"), ("pallet_presence", "pallet_present")):
        pos = sorted((r for r in pool if r[flag]), key=lambda r: _rank(seed, task, r["image_key"]))
        neg = sorted((r for r in pool if not r[flag]), key=lambda r: _rank(seed, task, r["image_key"]))
        take = len(pos) if per_task is None else per_task // 2
        chosen = pos[:take] + neg[:take if per_task is not None else len(neg)]
        plan += [(task, r, "yes" if r[flag] else "no") for r in chosen]
        report["tasks"][task] = {"available_positive": len(pos), "available_negative": len(neg),
                                 "selected_positive": min(len(pos), take), "selected_negative": len(chosen) - min(len(pos), take),
                                 "balanced_sample_prevalence_not_representative": per_task is not None,
                                 "subset_counts": dict(Counter(r["subset"] for t, r, _a in plan if t == task))}
    eligible = [r for r in pool if r["pallet_count_ambiguous"] == 0 and r["pallet_count_min_area"] <= max_count]
    excluded = {"ambiguous_small_pallets": sum(r["pallet_count_ambiguous"] > 0 for r in pool),
                "count_above_max": sum(r["pallet_count_ambiguous"] == 0 and r["pallet_count_min_area"] > max_count for r in pool)}
    buckets = defaultdict(list)
    for r in sorted(eligible, key=lambda r: _rank(seed, "pallet_count", r["image_key"])):
        buckets[count_bucket(r["pallet_count_min_area"])].append(r)
    chosen = []
    if per_task is None:
        chosen = [r for b in buckets.values() for r in b]
    else:
        queues = [list(b) for b in buckets.values()]
        while len(chosen) < per_task and any(queues):
            for q in queues:
                if q and len(chosen) < per_task:
                    chosen.append(q.pop(0))
    plan += [("pallet_count", r, str(r["pallet_count_min_area"])) for r in chosen]
    report["tasks"]["pallet_count"] = {"eligible_images": len(eligible), "excluded": excluded, "max_count": max_count,
                                       "selected": len(chosen), "selected_by_bucket": dict(Counter(count_bucket(r["pallet_count_min_area"]) for r in chosen)),
                                       "rule": RULES["pallet_count"]}
    return plan, report


def export_vlm_manifest(rows, checks, splits, out_dir, *, split="test", per_task=100, max_count=30, seed=0,
                        attested_by=None, statement=None):
    """Write a STANDARD manifest (src.evaluation_manifest compatible) plus a metadata sidecar.

    ``reference_verified`` is a human attestation in the existing pipeline, so it is written only when a
    named person supplies ``attested_by``; the tool never asserts it on its own.
    """
    if not attested_by or not str(attested_by).strip():
        raise LocoError("A named human attestation (attested_by) is required: the existing manifest treats "
                        "reference_verified as a human statement, and this tool cannot make it for you.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan, report = select_samples(rows, splits, split, per_task=per_task, max_count=max_count, seed=seed)
    missing = [r["image_key"] for _, r, _ in plan if checks.get(r["image_key"], {}).get("status") != "ok"]
    if missing:
        raise LocoError(f"{len(set(missing))} selected images are not verified-ok (run verify-images): {missing[:3]}")
    manifest, sidecar = [], []
    for task, r, answer in plan:
        c = checks[r["image_key"]]
        sid = f"loco-{task}-{hashlib.sha256(r['image_key'].encode()).hexdigest()[:12]}"
        manifest.append({
            "sample_id": sid, "image_path": c["resolved_path"], "source_id": r["group_id"],
            "question": PROMPTS[task], "question_type": "count" if task == "pallet_count" else "binary",
            "reference_answer": answer,
            "label_provenance": f"loco_v1_human_bbox_derived;rule={RULES[task]};{PROMPTS['version']};min_area_frac={r['min_area_frac']}",
            "reference_verified": "true", "split": split,
            "annotation_notes": f"LOCO {r['image_key']}"})
        sidecar.append({"sample_id": sid, "task": task, "image_key": r["image_key"], "subset": r["subset"],
                        "official_split": r["official_split"], "project_split": split, "group_id": r["group_id"],
                        "sensor": r["sensor"], "width": r["width"], "height": r["height"],
                        "image_sha256": c["sha256"], "n_forklift": r["n_forklift"], "n_pallet_all_boxes": r["n_pallet"],
                        "pallet_count_min_area": r["pallet_count_min_area"],
                        "count_bucket": count_bucket(r["pallet_count_min_area"]),
                        "n_pallet_truck": r["n_pallet_truck"], "reference_answer": answer})
    stamp = f"loco_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    manifest_path, sidecar_path = out_dir / f"{stamp}_manifest.csv", out_dir / f"{stamp}_sidecar.csv"
    write_csv(manifest_path, manifest, ["sample_id", "image_path", "source_id", "question", "question_type",
                                        "reference_answer", "label_provenance", "reference_verified", "split",
                                        "annotation_notes"])
    write_csv(sidecar_path, sidecar, list(sidecar[0]))
    attestation = {"attested_by": attested_by.strip(), "attested_at": datetime.now(timezone.utc).isoformat(),
                   "statement": statement or ("I accept LOCO's manual bounding-box annotations as the reference "
                                              "for the derived presence/count questions in this manifest."),
                   "citation": CITATION, "license": LICENSE}
    export = {**report, "manifest": manifest_path.name, "sidecar": sidecar_path.name, "seed": seed,
              "prompts": PROMPTS, "rules": RULES, "attestation": attestation, "n_samples": len(manifest),
              "count_rule_proxy_note": COUNT_RULE_PROXY_NOTE}
    with (out_dir / f"{stamp}_export.json").open("x", encoding="utf-8") as f:
        json.dump(export, f, indent=1)
    return manifest_path, sidecar_path, export


def export_detection_gt(sets, rows, checks, splits, out_path, *, split):
    """COCO-style ground truth for one project split; image file_name = verified resolved path."""
    chosen = {r["image_key"]: r for r in rows if splits[r["image_key"]] == split}
    images, annotations, key_to_id = [], [], {}
    for key in sorted(chosen):
        c = checks.get(key, {})
        if c.get("status") != "ok":
            raise LocoError(f"{key} is not verified-ok; run verify-images first")
        key_to_id[key] = len(key_to_id) + 1
        images.append({"id": key_to_id[key], "file_name": c["resolved_path"], "width": chosen[key]["width"],
                       "height": chosen[key]["height"], "loco_key": key, "subset": chosen[key]["subset"]})
    for sk, data in sets.items():
        loco_paths = {im["id"]: parse_path(im["path"])["image_key"] for im in data["images"]}
        for a in data["annotations"]:
            key = loco_paths[a["image_id"]]
            if key in key_to_id:
                annotations.append({"id": len(annotations) + 1, "image_id": key_to_id[key],
                                    "category_id": a["category_id"], "bbox": a["bbox"],
                                    "area": a["bbox"][2] * a["bbox"][3], "iscrowd": 0,
                                    "polygon_only": a.get("isbbox") is False})
    body = {"info": {"description": f"LOCO v1 ground truth, project split '{split}'", "license": LICENSE,
                     "citation": CITATION}, "images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in CLASSES.items()]}
    with Path(out_path).open("x", encoding="utf-8") as f:
        json.dump(body, f)
    return body
