#!/usr/bin/env bash
# Builds and runs both unit-test suites: the C++ tests (via ctest --
# tests/pulsecore_tests.cpp, which includes the AVX2 parity tests when
# that variant was built) and the Python tests (python/tests/, stdlib
# unittest -- the numba/numpy parity tests skip cleanly when those
# optional dependencies aren't installed). See python/tests/__init__.py
# for how these relate to the repo's other correctness layers.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

echo "== C++ tests (ctest) =="
(cd build && ctest --output-on-failure)
echo

echo "== Python tests (unittest) =="
./scripts/gen_python_proto.sh >/dev/null
(cd python && python3 -m unittest discover -s tests -v 2>&1 | tail -5)
