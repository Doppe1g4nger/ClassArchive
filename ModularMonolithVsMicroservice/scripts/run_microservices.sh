#!/usr/bin/env bash
# Builds (if needed) and runs the microservice variant: two separate
# executables, connected over TCP with length-prefixed protobuf messages.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

PORT="${1:-50051}"

./build/bin/stats_service "$PORT" &
STATS_PID=$!
trap 'kill "$STATS_PID" 2>/dev/null || true' EXIT

sleep 0.3
./build/bin/detector_service 127.0.0.1 "$PORT"
wait "$STATS_PID"
