#!/usr/bin/env bash
# Builds (if needed) and runs the microservice variant: five separate
# executables, connected over TCP with length-prefixed protobuf messages.
# detector_service is the data-plane hub -- it generates the synthetic IQ
# stream and fans both the raw batches and its own detected pulse events
# out to the other four, each listening at BASE_PORT + a fixed offset
# (see microservice/detector_service/main.cpp).
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

BASE_PORT="${1:-50051}"
NUM_PULSES="${2:-6}"

BIN=./build/bin
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

"$BIN/stats_service" "$BASE_PORT" &
PIDS+=("$!")
"$BIN/spectrogram_service" "$((BASE_PORT + 1))" &
PIDS+=("$!")
"$BIN/jammer_service" "$((BASE_PORT + 2))" &
PIDS+=("$!")
"$BIN/deinterleave_service" "$((BASE_PORT + 3))" &
PIDS+=("$!")

sleep 0.3
"$BIN/detector_service" 127.0.0.1 "$BASE_PORT" "$NUM_PULSES"

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
