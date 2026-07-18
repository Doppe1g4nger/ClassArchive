#!/usr/bin/env bash
# Runs the AVX2 C++ variant: same five-stage pipeline as run_monolith.sh,
# with the detector and jammer stages replaced by AVX2-vectorized ports.
# See common/include/pulse_detector_avx.h/jammer_avx.h and README.md's
# "Going further" section for what's different about this build (and,
# for the detector specifically, why it's still bit-identical to the
# scalar version despite being vectorized).
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

if [ ! -x ./build/bin/avx_monolith_app ]; then
  echo "avx_monolith_app wasn't built -- this toolchain likely doesn't support -mavx2 -mfma" >&2
  exit 1
fi

./build/bin/avx_monolith_app "$@"
