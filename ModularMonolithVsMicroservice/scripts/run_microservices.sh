#!/usr/bin/env bash
# Builds (if needed) and runs the microservice variant: five separate
# executables wired as a linear chain over TCP with length-prefixed
# protobuf messages -- detector_service -> stats_service ->
# deinterleave_service -> spectrogram_service -> jammer_service. Each
# middle service is both a TCP server (accepting the stage before it) and
# a TCP client (connecting to the stage after it), so every service
# except the last must connect downstream *before* it can accept
# upstream, and every service except the first (a pure producer) must
# have its downstream target already listening. That forces startup in
# the reverse of data-flow order: jammer, spectrogram, deinterleave,
# stats, detector.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

# Default kept below the kernel's ephemeral port range (usually
# 32768-60999, check /proc/sys/net/ipv4/ip_local_port_range) -- every
# service in this chain also makes outbound connections, which get
# assigned ephemeral source ports by the OS, and a listener bound inside
# that range can randomly lose a bind() race against one of those.
BASE_PORT="${1:-20051}"
NUM_PULSES="${2:-6}"

# Chain order (data flow): stats=+0, deinterleave=+1, spectrogram=+2, jammer=+3.
PORT_STATS=$((BASE_PORT))
PORT_DEINTERLEAVE=$((BASE_PORT + 1))
PORT_SPECTROGRAM=$((BASE_PORT + 2))
PORT_JAMMER=$((BASE_PORT + 3))

BIN=./build/bin
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_for_port() {
  local port="$1"
  local port_hex
  port_hex=$(printf '%04X' "$port")
  for attempt in $(seq 1 100); do
    if awk -v p=":${port_hex}" '$2 ~ p && $4=="0A" {found=1} END{exit !found}' /proc/net/tcp; then
      return 0
    fi
    sleep 0.02
  done
  echo "timed out waiting for port $port to start listening" >&2
  return 1
}

# Reverse of data-flow order: each service's downstream target must
# already be listening before it starts.
"$BIN/jammer_service" "$PORT_JAMMER" &
PIDS+=("$!")
wait_for_port "$PORT_JAMMER"

"$BIN/spectrogram_service" "$PORT_SPECTROGRAM" 127.0.0.1 "$PORT_JAMMER" &
PIDS+=("$!")
wait_for_port "$PORT_SPECTROGRAM"

"$BIN/deinterleave_service" "$PORT_DEINTERLEAVE" 127.0.0.1 "$PORT_SPECTROGRAM" &
PIDS+=("$!")
wait_for_port "$PORT_DEINTERLEAVE"

"$BIN/stats_service" "$PORT_STATS" 127.0.0.1 "$PORT_DEINTERLEAVE" &
PIDS+=("$!")
wait_for_port "$PORT_STATS"

"$BIN/detector_service" 127.0.0.1 "$PORT_STATS" "$NUM_PULSES"

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
