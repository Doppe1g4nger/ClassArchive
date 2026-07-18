#!/usr/bin/env bash
# Builds (if needed) and runs the modular-monolith variant: one process,
# two modules loaded at runtime via dlopen().
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

./build/bin/monolith_app ./build/bin
