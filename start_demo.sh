#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
demo_python="${VLM_EDGE_PYTHON:-.venv/bin/python}"
if ! "$demo_python" --version >/dev/null 2>&1; then
  printf '%s\n' 'Project Python is unavailable. Activate the working WSL environment and set VLM_EDGE_PYTHON to its interpreter.' >&2
  exit 1
fi
exec "$demo_python" tools/demo_runtime.py start --dashboard "$@"
