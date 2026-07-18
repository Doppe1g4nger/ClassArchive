#!/usr/bin/env bash
# Runs the numba variant: same five-stage pipeline as run_python_monolith.sh,
# with detector/spectrogram/jammer and IQ generation JIT-compiled instead
# of interpreted. Requires numpy + numba -- see
# python/requirements-numeric.txt (not needed by any other build in this
# repo).
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/gen_python_proto.sh

python3 python/numba_variant/numba_monolith_app.py "$@"
