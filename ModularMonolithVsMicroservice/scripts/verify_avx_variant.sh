#!/usr/bin/env bash
# Builds and runs verify_avx_variant: a standalone, full-precision
# correctness check for the AVX2 detector/jammer ports (see
# common/include/pulse_detector_avx.h and jammer_avx.h), since the usual
# printed-output diffs elsewhere in this repo only show three decimal
# places -- not enough to catch or rule out the floating-point
# reordering jammer_avx.h's docstring documents.
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_PULSES="${1:-50000}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

./build/bin/verify_avx_variant "$NUM_PULSES"
