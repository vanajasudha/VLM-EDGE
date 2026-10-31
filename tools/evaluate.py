"""Validate/plan without model imports; real execution requires a bound approval."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_manifest import ManifestError, load_manifest
from src.evaluation_runner import Experiment, MODEL_REVISION, build_plan, run_evaluation, write_json


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("manifest", type=Path)
        cmd.add_argument("--image-root", type=Path, help="Default: manifest's directory")
        cmd.add_argument("--split", choices=("train", "validation", "test"), default="test")
        if name != "validate":
            cmd.add_argument("--revision", default=MODEL_REVISION)
            cmd.add_argument("--max-tokens", type=int, default=256)
            cmd.add_argument("--temperature", type=float, default=0.0)
            cmd.add_argument("--top-p", type=float, default=1.0)
            cmd.add_argument("--seed", type=int, default=0)
            cmd.add_argument("--warmup", type=int, default=1)
        if name == "plan":
            cmd.add_argument("--output", type=Path, help="Optional new JSON plan file; never overwritten")
        if name == "evaluate":
            cmd.add_argument("--approval", type=Path, required=True)
            cmd.add_argument("--output-root", type=Path, default=ROOT / "experiments")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        dataset = load_manifest(args.manifest, image_root=args.image_root)
        selected = dataset.select(args.split)
        if args.command == "validate":
            print(json.dumps({"status": "VALID", "n_total": len(dataset.samples),
                              "n_selected": len(selected), "split": args.split,
                              "dataset_sha256": dataset.dataset_sha256,
                              "manifest_sha256": dataset.manifest_sha256}, indent=2))
            return 0
        experiment = Experiment(split=args.split, revision=args.revision, max_tokens=args.max_tokens,
                                temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                                warmup=args.warmup)
        plan = build_plan(dataset, experiment)
        if args.command == "plan":
            if args.output:
                write_json(args.output, plan)
            print(json.dumps(plan, indent=2))
            return 0
        approval = json.loads(args.approval.read_text(encoding="utf-8"))
        run, summary = run_evaluation(dataset, experiment, args.output_root, approval=approval)
        print(json.dumps({"run_directory": str(run), **summary}, indent=2, allow_nan=False))
        return 0 if summary["status"] == "complete" else 1
    except (ValueError, OSError) as exc:
        print(f"Evaluation refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
