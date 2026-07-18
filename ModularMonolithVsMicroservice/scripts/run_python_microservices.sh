#!/usr/bin/env bash
# Runs the Python microservice variant: five separate processes wired as
# the exact same linear chain as scripts/run_microservices.sh --
# detector_service.py -> spectrogram_service.py -> jammer_service.py ->
# stats_service.py -> deinterleave_service.py -- over the same
# length-prefixed protobuf-over-TCP wire format (see
# python/microservice/framing.py). Same reverse-of-data-flow startup
# order constraint as the C++ build, for the same reason: every service
# but the producer must connect downstream before it can accept upstream.
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/gen_python_proto.sh

# Default kept below the kernel's ephemeral port range, same reasoning as
# scripts/run_microservices.sh.
BASE_PORT="${1:-20151}"
NUM_PULSES="${2:-1000}"

# Chain order (data flow): spectrogram=+0, jammer=+1, stats=+2, deinterleave=+3.
PORT_SPECTROGRAM=$((BASE_PORT))
PORT_JAMMER=$((BASE_PORT + 1))
PORT_STATS=$((BASE_PORT + 2))
PORT_DEINTERLEAVE=$((BASE_PORT + 3))

PY=python3
DIR=python/microservice
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
"$PY" "$DIR/deinterleave_service.py" "$PORT_DEINTERLEAVE" &
PIDS+=("$!")
wait_for_port "$PORT_DEINTERLEAVE"

"$PY" "$DIR/stats_service.py" "$PORT_STATS" 127.0.0.1 "$PORT_DEINTERLEAVE" &
PIDS+=("$!")
wait_for_port "$PORT_STATS"

"$PY" "$DIR/jammer_service.py" "$PORT_JAMMER" 127.0.0.1 "$PORT_STATS" &
PIDS+=("$!")
wait_for_port "$PORT_JAMMER"

"$PY" "$DIR/spectrogram_service.py" "$PORT_SPECTROGRAM" 127.0.0.1 "$PORT_JAMMER" &
PIDS+=("$!")
wait_for_port "$PORT_SPECTROGRAM"

"$PY" "$DIR/detector_service.py" 127.0.0.1 "$PORT_SPECTROGRAM" "$NUM_PULSES"

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
