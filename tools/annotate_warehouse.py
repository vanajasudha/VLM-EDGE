"""Human-only local image annotation, source registration, QC and manifest export."""
import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import sys
from urllib.parse import parse_qs, urlencode, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.warehouse_annotation import (
    COUNT_RULE, COVERAGE, active_sources, dataset_lock, exclude_image, export_manifest,
    image_path, initialize, quality_report, read_events, register_image, review_pair, save_annotation,
)


def render(root, token, selected=None, message="", values=None):
    values = values or {}
    with dataset_lock(root):
        sources = active_sources(root)
        events = read_events(root)
    selected = selected or (sources[0]["image_id"] if sources else None)
    source = next((s for s in sources if s["image_id"] == selected), None)
    links = "".join(f'<li><a href="/?{urlencode({"image": s["image_id"]})}">{escape(s["image_id"])}</a> '
                    f'({escape(s["split"])}, session {escape(s["session_id"])})</li>' for s in sources)
    def field(name):
        return escape(str(values.get(name, "")), quote=True)
    body = ['<!doctype html><html><head><meta charset="utf-8"><title>Warehouse annotation</title></head><body>',
            '<h1>Warehouse pilot annotation</h1>',
            '<p>Human annotation only. No model predictions or GPU operations. Verification is unchecked by default.</p>',
            '<p>Read data/warehouse_pilot/ANNOTATION_GUIDE.md before labelling.</p>',
            f'<p role="status">{escape(message)}</p>', f'<ul>{links}</ul>']
    if source:
        body += [
            f'<h2>{escape(source["image_id"])}</h2>',
            f'<img src="/image?{urlencode({"image": selected})}" alt="Warehouse image being reviewed" width="900">',
            '<details><summary>Source and permission record</summary><pre>' +
            escape(json.dumps(source, indent=2)) + '</pre></details>',
            '<form method="post" action="/annotation">',
            f'<input type="hidden" name="csrf" value="{escape(token, quote=True)}">',
            f'<input type="hidden" name="image_id" value="{escape(selected, quote=True)}">',
            f'<p><label>Existing sample ID to revise (blank for new)<input name="sample_id" value="{field("sample_id")}"></label></p>',
            f'<p><label>Question<textarea name="question" required>{field("question")}</textarea></label></p>',
            '<p><label>Question type<select name="question_type">' +
            "".join(f'<option value="{kind}" {"selected" if values.get("question_type") == kind else ""}>{kind}</option>'
                    for kind in ("binary", "count", "description")) + '</select></label></p>',
            f'<p><label>Human reference answer<textarea name="reference_answer">{field("reference_answer")}</textarea></label></p>',
            f'<p><label>Annotation notes / count target and scope<textarea name="annotation_notes">{field("annotation_notes")}</textarea></label></p>',
            '<p><label>Counting rule (counts only)<select name="counting_rule"><option value="">Choose after reviewing guide</option>' +
            f'<option value="{COUNT_RULE}">{COUNT_RULE}: clearly identifiable visible instances</option></select></label></p>',
            f'<p><label>Annotator identity<input name="annotator_id" required value="{field("annotator_id")}"></label></p>',
            f'<p><label>Verifier identity<input name="reviewer_id" value="{field("reviewer_id")}"></label></p>',
            '<p><label><input type="checkbox" name="verified" value="yes">'
            'I personally inspected this image, checked this exact question and reference, and confirm the label. '
            'Do not confirm ambiguous/unanswerable cases.</label></p>',
            '<button type="submit">Save draft or explicitly verified label</button></form>',
            '<p>Notes/provenance are saved with the label. Drafts never enter exported manifests. '
            'Use the CLI quality and export commands after review.</p>',
            '<details><summary>Annotation event history (latest edits append; old records preserved)</summary><pre>' +
            escape(json.dumps([e for e in events if e.get("image_id") == selected], indent=2)) + '</pre></details>']
    else:
        body.append('<p>No permitted warehouse images registered. Register genuine human-supplied images using the CLI first.</p>')
    return "".join(body) + "</body></html>"


def create_server(root, port=8766):
    root = initialize(root)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, content, mime="text/html; charset=utf-8"):
            data = content.encode("utf-8") if isinstance(content, str) else content
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            selected = parse_qs(parsed.query).get("image", [None])[0]
            try:
                if parsed.path == "/":
                    self.respond(200, render(root, token, selected))
                elif parsed.path == "/image":
                    sources = {s["image_id"]: s for s in active_sources(root)}
                    if selected not in sources:
                        self.respond(404, "Unknown registered image")
                        return
                    path = image_path(root, sources[selected])
                    from src.warehouse_annotation import source_check
                    source_check(root, sources[selected])
                    self.respond(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream")
                else:
                    self.respond(404, "Not found")
            except (ValueError, OSError) as exc:
                self.respond(400, escape(str(exc)))

        def do_POST(self):
            if self.path != "/annotation":
                self.respond(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid form size")
                values = {k: v[0] for k, v in parse_qs(
                    self.rfile.read(length).decode("utf-8"), keep_blank_values=True).items()}
                if not secrets.compare_digest(values.get("csrf", ""), token):
                    self.respond(403, "Invalid local form token")
                    return
                result = save_annotation(
                    root, image_id=values.get("image_id", ""), question=values.get("question", ""),
                    question_type=values.get("question_type", ""), reference_answer=values.get("reference_answer", ""),
                    annotation_notes=values.get("annotation_notes", ""), annotator_id=values.get("annotator_id", ""),
                    reviewer_id=values.get("reviewer_id", ""), counting_rule=values.get("counting_rule", ""),
                    sample_id=values.get("sample_id") or None, verified=values.get("verified") == "yes")
                status = "human-verified" if result["verified"] else "draft"
                self.respond(200, render(root, token, result["image_id"],
                                         f"Saved {status} label {result['sample_id']}."))
            except (ValueError, OSError) as exc:
                self.respond(400, render(root, token, locals().get("values", {}).get("image_id"),
                                         str(exc), locals().get("values", {})))

    return HTTPServer(("127.0.0.1", port), Handler)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT / "data/warehouse_pilot")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    register = sub.add_parser("register")
    register.add_argument("image", type=Path)
    for name in ("image-id", "source-origin", "rights-basis", "permission-evidence",
                 "rights-reviewer", "session-id", "scene-id"):
        register.add_argument("--" + name, required=True)
    register.add_argument("--split", choices=("train", "validation", "test"), required=True)
    register.add_argument("--source-id", default="")
    register.add_argument("--tags", default="", help="Comma-separated human-assigned coverage tags")
    register.add_argument("--notes", default="")
    register.add_argument("--permission-confirmed", action="store_true")
    register.add_argument("--warehouse-confirmed", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8766)
    sub.add_parser("quality")
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path)
    review = sub.add_parser("review-pair")
    review.add_argument("image_a")
    review.add_argument("image_b")
    review.add_argument("--verdict", choices=("distinct", "same_scene", "duplicate"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", required=True)
    exclude = sub.add_parser("exclude")
    exclude.add_argument("image_id")
    exclude.add_argument("--reviewer", required=True)
    exclude.add_argument("--reason", required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = initialize(args.root)
        if args.command == "init":
            print(root)
        elif args.command == "register":
            result = register_image(
                root, args.image, image_id=args.image_id, source_origin=args.source_origin,
                rights_basis=args.rights_basis, permission_evidence=args.permission_evidence,
                rights_reviewer=args.rights_reviewer, session_id=args.session_id, scene_id=args.scene_id,
                split=args.split, source_id=args.source_id, coverage_tags=filter(None, args.tags.split(",")),
                notes=args.notes, permission_confirmed=args.permission_confirmed,
                warehouse_confirmed=args.warehouse_confirmed)
            print(json.dumps(result, indent=2))
        elif args.command == "serve":
            server = create_server(root, args.port)
            print(f"Local annotation: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        elif args.command == "quality":
            with dataset_lock(root):
                report = quality_report(root)
            print(json.dumps(report, indent=2))
            return 0 if not report["errors"] else 2
        elif args.command == "export":
            print(export_manifest(root, args.output))
        elif args.command == "review-pair":
            print(json.dumps(review_pair(root, args.image_a, args.image_b, verdict=args.verdict,
                                         reviewer_id=args.reviewer, notes=args.notes), indent=2))
        elif args.command == "exclude":
            exclude_image(root, args.image_id, reviewer_id=args.reviewer, reason=args.reason)
            print("Excluded from future exports; files and annotation history retained.")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Annotation refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
