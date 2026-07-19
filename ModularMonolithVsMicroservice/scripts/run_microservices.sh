#!/usr/bin/env bash
# Builds (if needed) and runs the microservice variant: five separate
# executables wired as a linear chain over shared-memory rings carrying
# length-prefixed protobuf messages (see microservice/net/framing.h --
# this branch replaces the TCP transport) -- detector_service ->
# spectrogram_service -> jammer_service -> stats_service ->
# deinterleave_service. The first
# three all read the raw IQ batch, so they're grouped together and each
# hop after jammer_service drops it from the frame before forwarding
# (see microservice/jammer_service/main.cpp) -- the point being to avoid
# serializing and transmitting data no downstream stage will ever read.
#
# Each middle service is both a TCP server (accepting the stage before
# it) and a TCP client (connecting to the stage after it), so every
# service except the last must connect downstream *before* it can accept
# upstream, and every service except the first (a pure producer) must
# have its downstream target already listening. That forces startup in
# the reverse of data-flow order: deinterleave, stats, jammer,
# spectrogram, detector.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

# Default kept below the kernel's ephemeral port range (usually
# 32768-60999, check /proc/sys/net/ipv4/ip_local_port_range) -- every
# service in this chain also makes outbound connections, which get
# assigned ephemeral source ports by the OS, and a listener bound inside
# that range can randomly lose a bind() race against one of those.
# Default of 1000 pulses is exactly one buffer's worth at this repo's
# 1,000,000-pulse/sec, 1000-microsecond-buffer scale (see
# common/include/iq_source.h).
BASE_PORT="${1:-20051}"
NUM_PULSES="${2:-1000}"

# Chain order (data flow): spectrogram=+0, jammer=+1, stats=+2, deinterleave=+3.
PORT_SPECTROGRAM=$((BASE_PORT))
PORT_JAMMER=$((BASE_PORT + 1))
PORT_STATS=$((BASE_PORT + 2))
PORT_DEINTERLEAVE=$((BASE_PORT + 3))

BIN=./build/bin
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# Readiness = "the listener created its /dev/shm ring segment" (the
# transport is shared memory, not TCP -- see microservice/net/framing.h,
# which names each segment /pulse_ring_<port>). Connect() also waits on
# its own, so this only preserves the reverse-order startup story below.
wait_for_ring() {
  local port="$1"
  for attempt in $(seq 1 100); do
    if [ -e "/dev/shm/pulse_ring_${port}" ]; then
      return 0
    fi
    sleep 0.02
  done
  echo "timed out waiting for ring $port to be created" >&2
  return 1
}

# Reverse of data-flow order: each service's downstream target must
# already be listening before it starts.
"$BIN/deinterleave_service" "$PORT_DEINTERLEAVE" &
PIDS+=("$!")
wait_for_ring "$PORT_DEINTERLEAVE"

"$BIN/stats_service" "$PORT_STATS" 127.0.0.1 "$PORT_DEINTERLEAVE" &
PIDS+=("$!")
wait_for_ring "$PORT_STATS"

"$BIN/jammer_service" "$PORT_JAMMER" 127.0.0.1 "$PORT_STATS" &
PIDS+=("$!")
wait_for_ring "$PORT_JAMMER"

"$BIN/spectrogram_service" "$PORT_SPECTROGRAM" 127.0.0.1 "$PORT_JAMMER" &
PIDS+=("$!")
wait_for_ring "$PORT_SPECTROGRAM"

"$BIN/detector_service" 127.0.0.1 "$PORT_SPECTROGRAM" "$NUM_PULSES"

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
