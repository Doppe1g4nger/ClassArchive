#!/usr/bin/env bash
# Times 50 runs of each architecture against a synthetic input and reports
# min/median/mean/max wall-clock time per run. Default of 1,000,000 pulses
# is exactly one second of this repo's 1,000,000-pulse/sec, 1000-microsecond
# (1000 pulses/buffer) signal -- see common/include/iq_source.h.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-50}"
NUM_PULSES="${2:-1000000}"
# Default kept below the kernel's ephemeral port range (usually
# 32768-60999, check /proc/sys/net/ipv4/ip_local_port_range) -- every
# service in the microservice chain also makes outbound connections,
# which get assigned ephemeral source ports by the OS, and a listener
# bound inside that range can randomly lose a bind() race against one of
# those. Each run below claims its own block of 10 ports (four listeners
# plus headroom), so keep BASE_PORT + RUNS*10 + 3 under 32768 too.
BASE_PORT="${3:-20000}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

BIN=./build/bin
MONO_TIMES="$(mktemp)"
MICRO_TIMES="$(mktemp)"
trap 'rm -f "$MONO_TIMES" "$MICRO_TIMES"' EXIT

echo "Benchmarking: $RUNS runs each, $NUM_PULSES pulses per run"
echo

echo "== modular monolith (monolith_app) =="
for i in $(seq 1 "$RUNS"); do
  start=$(date +%s.%N)
  "$BIN/monolith_app" "$BIN" "$NUM_PULSES" >/dev/null
  end=$(date +%s.%N)
  echo "$end - $start" | bc >> "$MONO_TIMES"
done
echo "done"
echo

# Readiness = "the listener created its /dev/shm ring segment" -- the
# chain's transport is a shared-memory ring, not TCP, on this branch
# (see microservice/net/framing.h).
wait_for_ring() {
  local port="$1"
  for attempt in $(seq 1 100); do
    if [ -e "/dev/shm/pulse_ring_${port}" ]; then
      return 0
    fi
    sleep 0.02
  done
  return 1
}

echo "== microservices (detector -> spectrogram -> jammer -> stats -> deinterleave chain) =="
for i in $(seq 1 "$RUNS"); do
  # Each run gets its own block of 4 ports (base + i*10 + offset 0..3) so
  # consecutive runs can't collide even if a prior run's sockets are
  # still winding down.
  base=$((BASE_PORT + i * 10))
  port_spectrogram=$((base))
  port_jammer=$((base + 1))
  port_stats=$((base + 2))
  port_deinterleave=$((base + 3))

  # Timer starts before any service is even launched, so their process
  # startup and library load count toward the total -- the same way
  # monolith_app's single process-start-to-exit measurement above
  # includes its own startup cost. Timing only detector_service (as an
  # earlier version of this script did) would unfairly hide the
  # microservice architecture's process-startup overhead.
  start=$(date +%s.%N)

  # Each middle service connects downstream before it can accept
  # upstream, so startup order is the reverse of data flow (same
  # constraint as scripts/run_microservices.sh). Each wait_for_ring call
  # is itself counted as part of the microservice architecture's cost:
  # it's synchronization overhead the monolith never pays.
  "$BIN/deinterleave_service" "$port_deinterleave" >/dev/null 2>&1 &
  pid_deinterleave=$!
  wait_for_ring "$port_deinterleave"

  "$BIN/stats_service" "$port_stats" 127.0.0.1 "$port_deinterleave" >/dev/null 2>&1 &
  pid_stats=$!
  wait_for_ring "$port_stats"

  "$BIN/jammer_service" "$port_jammer" 127.0.0.1 "$port_stats" >/dev/null 2>&1 &
  pid_jammer=$!
  wait_for_ring "$port_jammer"

  "$BIN/spectrogram_service" "$port_spectrogram" 127.0.0.1 "$port_jammer" >/dev/null 2>&1 &
  pid_spectrogram=$!
  wait_for_ring "$port_spectrogram"

  "$BIN/detector_service" 127.0.0.1 "$port_spectrogram" "$NUM_PULSES" >/dev/null
  wait "$pid_spectrogram" "$pid_jammer" "$pid_stats" "$pid_deinterleave"
  end=$(date +%s.%N)
  echo "$end - $start" | bc >> "$MICRO_TIMES"
done
echo "done"
echo

python3 - "$MONO_TIMES" "$MICRO_TIMES" "$RUNS" "$NUM_PULSES" <<'PYEOF'
import sys, statistics

mono_path, micro_path, runs, num_pulses = sys.argv[1:5]

def load(path):
    with open(path) as f:
        return [float(line.strip()) * 1000.0 for line in f if line.strip()]  # -> ms

mono = load(mono_path)
micro = load(micro_path)

def summarize(name, xs):
    xs_sorted = sorted(xs)
    print(f"{name:>28}  min={min(xs):8.3f}ms  median={statistics.median(xs):8.3f}ms  "
          f"mean={statistics.mean(xs):8.3f}ms  stdev={statistics.pstdev(xs):7.3f}ms  max={max(xs):8.3f}ms")

print(f"Results over {runs} runs, {num_pulses} pulses/run:\n")
summarize("modular monolith", mono)
summarize("microservices", micro)

ratio = statistics.mean(micro) / statistics.mean(mono)
print(f"\nmicroservices mean is {ratio:.2f}x the modular monolith mean")
PYEOF
