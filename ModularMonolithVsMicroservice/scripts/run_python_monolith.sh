#!/usr/bin/env bash
# Runs the Python modular-monolith variant: one process, five stage
# modules imported at startup (see python/monolith/monolith_app.py).
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/gen_python_proto.sh

python3 python/monolith/monolith_app.py "$@"
