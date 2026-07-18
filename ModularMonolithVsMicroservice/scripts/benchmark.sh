#!/usr/bin/env bash
# Times 50 runs of each architecture against a 1000-pulse synthetic input
# and reports min/median/mean/max wall-clock time per run.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-50}"
NUM_PULSES="${2:-1000}"
BASE_PORT="${3:-53100}"

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

echo "== microservices (detector_service + stats_service) =="
for i in $(seq 1 "$RUNS"); do
  port=$((BASE_PORT + i))
  # Timer starts before stats_service is even launched, so its process
  # startup and library load count toward the total -- the same way
  # monolith_app's single process-start-to-exit measurement above
  # includes its own startup cost. Timing only detector_service (as an
  # earlier version of this script did) would unfairly hide half of the
  # microservice architecture's process-startup overhead.
  start=$(date +%s.%N)
  "$BIN/stats_service" "$port" >/dev/null 2>&1 &
  spid=$!
  # Wait for the listener to actually bind rather than guessing with a
  # fixed sleep. Polls /proc/net/tcp for LISTEN state on the port instead
  # of probing with a real connect() -- stats_service only accept()s once
  # (backlog=1), so a throwaway probe connection could itself get
  # accepted and steal the slot detector_service needs. This wait is
  # itself counted as part of the microservice architecture's cost: it's
  # synchronization overhead the monolith never pays.
  port_hex=$(printf '%04X' "$port")
  for attempt in $(seq 1 100); do
    if awk -v p=":${port_hex}" '$2 ~ p && $4=="0A" {found=1} END{exit !found}' /proc/net/tcp; then
      break
    fi
    sleep 0.02
  done
  "$BIN/detector_service" 127.0.0.1 "$port" "$NUM_PULSES" >/dev/null
  wait "$spid"
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
