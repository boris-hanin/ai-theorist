#!/usr/bin/env bash
set -euo pipefail

# Run the CLI from the current worktree while reusing an explicitly selected
# Python environment. This keeps detached campaign worktrees isolated without
# changing the editable install used by an already-running campaign.
python="${AI_THEORIST_PYTHON:-.venv-forecast/bin/python}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m ai_theorist.autoscaler.cli "$@"
