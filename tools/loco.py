"""LOCO benchmark workflow. Nothing here trains a model or downloads images.

  fetch-annotations   download the CC0 annotation JSONs from GitHub (images are hosted by TUM; see data/loco/README.md)
  prepare             verify annotations, build the index, integrity report and FROZEN splits
  verify-images       hash/decode/size-check downloaded images, then exact + near-duplicate leakage report
  export-manifest     standard VLM-Edge manifest + metadata sidecar (needs a named human attestation)
  export-detection-gt COCO ground truth for one split
  eval-detector       zero-shot open-vocabulary detector baseline (real GPU run needs --revision)
  make-approval       write the approval JSON tools/evaluate.py requires (a human act: names are required)
  analyze             stratified metrics + error list for a finished Moondream2 run
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import loco_dataset as L  # noqa: E402

DATA = ROOT / "data" / "loco"
PINNED = DATA / "expected_annotation_stats.json"


def _paths(args):
    base = Path(args.data_dir)
    return {"ann": base / "annotations", "index": base / "index", "exports": base / "exports"}


def _splits(p):
    return json.loads((p["index"] / "splits.json").read_text())["splits"]


def cmd_fetch(args):
    p = _paths(args)
    p["ann"].mkdir(parents=True, exist_ok=True)
    ref = args.ref
    url = f"https://codeload.github.com/tum-fml/loco/zip/{ref}"
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    archive = zipfile.ZipFile(io.BytesIO(data))
    commit = archive.comment.decode("ascii", "replace")
    for key, (name, _) in L.SUBSETS.items():
        target = p["ann"] / name
        if target.exists():
            print(f"exists, not overwritten: {target}")
            continue
        member = next(m for m in archive.namelist() if m.endswith(f"rgb/{name}"))
        target.write_bytes(archive.read(member))
    (p["ann"] / "SOURCE.txt").write_text(f"{url}\nupstream commit: {commit}\nlicense: {L.LICENSE}\ncitation: {L.CITATION}\n") \
        if not (p["ann"] / "SOURCE.txt").exists() else None
    print("Upstream commit:", commit)
    return 0


def cmd_prepare(args):
    p = _paths(args)
    p["index"].mkdir(parents=True, exist_ok=True)
    sets = L.load_annotation_sets(p["ann"])
    fingerprint = L.expected_stats(p["ann"], sets)
    if args.skip_pin_check:
        print("WARNING: pin check skipped; these are NOT verified official LOCO files.", file=sys.stderr)
    elif PINNED.exists():
        problems = L.verify_expected(p["ann"], sets, json.loads(PINNED.read_text()))
        if problems:
            print("Annotation files differ from the pinned verified download:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        print("Annotation files match the pinned verified download.")
    elif args.pin:
        PINNED.parent.mkdir(parents=True, exist_ok=True)
        PINNED.write_text(json.dumps(fingerprint, indent=1))
        print("Pinned", PINNED)
    rows = L.build_index(sets, min_area_frac=args.min_area_frac)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    index_path = p["index"] / "index.csv"
    if not index_path.exists():
        L.write_csv(index_path, rows, L.INDEX_FIELDS)
    else:
        if {r["image_key"] for r in L.read_index(index_path)} != {r["image_key"] for r in rows}:
            print("index.csv exists and differs; refusing to overwrite", file=sys.stderr)
            return 2
    record = L.freeze_splits(p["index"] / "splits.json", rows, validation_fraction=args.validation_fraction, seed=args.seed)
    report = L.integrity_report(sets, rows)
    report["split_counts"] = record["counts"]
    report["splits_sha256"] = record["sha256"]
    report["annotation_fingerprint"] = fingerprint
    with (p["index"] / f"integrity_report_{stamp}.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("totals", "annotations_per_class", "readme_discrepancy", "annotation_issues",
                                             "split_counts", "groups")}, indent=1))
    return 0


def _checks(p):
    return L.read_image_checks(p["index"] / "image_checks.csv")


def cmd_verify_images(args):
    p = _paths(args)
    rows = L.read_index(p["index"] / "index.csv")
    checks = L.verify_images(rows, args.image_root, p["index"] / "image_checks.csv",
                             progress=lambda n, t: print(f"  {n}/{t}", flush=True))
    report = L.leakage_report(rows, checks, _splits(p), near_threshold=args.near_threshold)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with (p["index"] / f"image_verification_report_{stamp}.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1)[:3000])
    bad = report["status_counts"]
    return 0 if set(bad) <= {"ok"} else 1


def cmd_export_manifest(args):
    p = _paths(args)
    rows = L.read_index(p["index"] / "index.csv")
    m, s, export = L.export_vlm_manifest(rows, _checks(p), _splits(p), p["exports"], split=args.split,
                                         per_task=args.per_task, max_count=args.max_count, seed=args.seed,
                                         attested_by=args.attest_by)
    print(json.dumps({"manifest": str(m), "sidecar": str(s), "n_samples": export["n_samples"], "tasks": export["tasks"]}, indent=1))
    return 0


def cmd_export_gt(args):
    p = _paths(args)
    p["exports"].mkdir(parents=True, exist_ok=True)
    rows = L.read_index(p["index"] / "index.csv")
    sets = L.load_annotation_sets(p["ann"])
    out = p["exports"] / f"loco_detection_gt_{args.split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    body = L.export_detection_gt(sets, rows, _checks(p), _splits(p), out, split=args.split)
    print(f"{out}: {len(body['images'])} images, {len(body['annotations'])} annotations")
    return 0


def cmd_eval_detector(args):
    from src.detector_baseline import DEFAULT_MODEL, Owlv2Backend, OracleBackend, run_detector_eval
    gt = json.loads(Path(args.gt).read_text())
    if args.mock_oracle:
        backend = OracleBackend(gt)
    else:
        backend = Owlv2Backend(model_id=args.model, revision=args.revision, device=args.device,
                               score_floor=args.score_floor, allow_download=args.allow_download)
    run, metrics = run_detector_eval(gt, args.image_root, backend, args.output_root, score_floor=args.score_floor,
                                     operating_threshold=args.operating_threshold, limit=args.limit,
                                     resume_dir=args.resume)
    print("Run directory:", run)
    print(json.dumps({k: metrics[k] for k in ("images_planned", "images_evaluated", "images_failed", "latency_ms", "memory")}, indent=1))
    print("mAP@0.5", metrics["detection"]["mAP@0.5"], "| mAP@[0.5:0.95]", metrics["detection"]["mAP@[0.5:0.95]"])
    return 0 if metrics["images_failed"] == 0 else 1


def cmd_make_approval(args):
    plan = json.loads(Path(args.plan).read_text())
    if not args.confirm_human_labels:
        print("Refused: pass --confirm-human-labels only if you accept the attested LOCO reference labels.", file=sys.stderr)
        return 2
    body = {"dataset_sha256": plan["dataset_sha256"], "experiment_sha256": plan["experiment_sha256"],
            "approved_by": args.approved_by, "approved_at": datetime.now(timezone.utc).isoformat(),
            "human_labels_confirmed": True}
    with Path(args.output).open("x", encoding="utf-8") as f:
        json.dump(body, f, indent=1)
    print("Wrote", args.output)
    return 0


def cmd_analyze(args):
    from src.loco_analysis import analyze
    out, result = analyze(args.run, args.sidecar, args.output_root)
    print("Analysis:", out)
    return 0


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--data-dir", default=str(DATA))
    sub = cli.add_subparsers(dest="command", required=True)
    c = sub.add_parser("fetch-annotations"); c.add_argument("--ref", default="main"); c.set_defaults(fn=cmd_fetch)
    c = sub.add_parser("prepare"); c.add_argument("--min-area-frac", type=float, default=0.005)
    c.add_argument("--validation-fraction", type=float, default=0.15); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--pin", action="store_true", help="record file hashes/counts as the verified fingerprint")
    c.add_argument("--skip-pin-check", action="store_true", help="only for tests or a deliberately different LOCO version"); c.set_defaults(fn=cmd_prepare)
    c = sub.add_parser("verify-images"); c.add_argument("--image-root", required=True)
    c.add_argument("--near-threshold", type=int, default=5); c.set_defaults(fn=cmd_verify_images)
    c = sub.add_parser("export-manifest"); c.add_argument("--split", choices=("train", "validation", "test"), default="test")
    c.add_argument("--per-task", type=int, default=100, help="0 = every eligible image")
    c.add_argument("--max-count", type=int, default=30); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--attest-by", required=True, help="your name: you accept LOCO's human labels as reference"); c.set_defaults(fn=cmd_export_manifest)
    c = sub.add_parser("export-detection-gt"); c.add_argument("--split", choices=("train", "validation", "test"), default="test"); c.set_defaults(fn=cmd_export_gt)
    c = sub.add_parser("eval-detector"); c.add_argument("--gt", required=True); c.add_argument("--image-root", required=True)
    c.add_argument("--model", default="google/owlv2-base-patch16-ensemble"); c.add_argument("--revision")
    c.add_argument("--device", default="cuda"); c.add_argument("--score-floor", type=float, default=0.05)
    c.add_argument("--operating-threshold", type=float, default=0.3); c.add_argument("--limit", type=int)
    c.add_argument("--resume"); c.add_argument("--allow-download", action="store_true")
    c.add_argument("--mock-oracle", action="store_true", help="pipeline check only: returns ground truth, NOT a result")
    c.add_argument("--output-root", default=str(ROOT / "experiments")); c.set_defaults(fn=cmd_eval_detector)
    c = sub.add_parser("make-approval"); c.add_argument("--plan", required=True); c.add_argument("--approved-by", required=True)
    c.add_argument("--confirm-human-labels", action="store_true"); c.add_argument("--output", required=True); c.set_defaults(fn=cmd_make_approval)
    c = sub.add_parser("analyze"); c.add_argument("--run", required=True); c.add_argument("--sidecar", required=True)
    c.add_argument("--output-root", default=str(ROOT / "experiments")); c.set_defaults(fn=cmd_analyze)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "per_task", None) == 0:
        args.per_task = None
    try:
        return args.fn(args)
    except (L.LocoError, ValueError, OSError, FileExistsError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
