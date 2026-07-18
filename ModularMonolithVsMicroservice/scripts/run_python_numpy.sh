#!/usr/bin/env bash
# Runs the numpy variant: same five-stage pipeline as run_python_monolith.sh,
# with detector/spectrogram/jammer rewritten as bulk numpy array
# operations. Requires numpy -- see python/requirements-numeric.txt (not
# needed by any other build in this repo). Note: this variant's numeric
# output is equivalent to, but not bit-identical with, every other build
# -- see python/numpy_variant/kernels.py's module docstring and
# python/numpy_variant/verify_numpy_variant.py for why and how that's
# checked.
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/gen_python_proto.sh

python3 python/numpy_variant/numpy_monolith_app.py "$@"
