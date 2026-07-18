#!/usr/bin/env bash
# Times RUNS independent, fresh-process runs of all four implementations
# -- C++ monolith, C++ microservices, Python monolith, Python
# microservices -- against the same synthetic input, and reports
# min/median/mean/max wall-clock time per run for each.
#
# Defaults to a small scale (1000 pulses = one buffer) rather than the
# 1,000,000-pulse/1-second scale scripts/benchmark.sh uses for the C++-only
# comparison: pure-Python execution of this repo's per-sample loops
# (especially the spectrogram's O(bins * batch_size) inner loop, which
# dominates) is roughly two orders of magnitude slower than the equivalent
# C++, so a 50-run benchmark at the full-second scale would take on the
# order of an hour. See "Python vs C++" in README.md for a small number of
# single-run full-scale measurements taken separately instead.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-50}"
NUM_PULSES="${2:-1000}"
# Defaults kept below the kernel's ephemeral port range and given their
# own non-overlapping blocks, same reasoning as scripts/benchmark.sh.
CPP_BASE_PORT="${3:-20500}"
PY_BASE_PORT="${4:-20700}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null
./scripts/gen_python_proto.sh

BIN=./build/bin
PY=python3
PYDIR=python/microservice

CPP_MONO_TIMES="$(mktemp)"
CPP_MICRO_TIMES="$(mktemp)"
PY_MONO_TIMES="$(mktemp)"
PY_MICRO_TIMES="$(mktemp)"
trap 'rm -f "$CPP_MONO_TIMES" "$CPP_MICRO_TIMES" "$PY_MONO_TIMES" "$PY_MICRO_TIMES"' EXIT

echo "Benchmarking: $RUNS runs each, $NUM_PULSES pulses per run"
echo

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
  return 1
}

echo "== C++ modular monolith =="
for i in $(seq 1 "$RUNS"); do
  start=$(date +%s.%N)
  "$BIN/monolith_app" "$BIN" "$NUM_PULSES" >/dev/null
  end=$(date +%s.%N)
  echo "$end - $start" | bc >> "$CPP_MONO_TIMES"
done
echo "done"
echo

echo "== C++ microservices (chain) =="
for i in $(seq 1 "$RUNS"); do
  base=$((CPP_BASE_PORT + i * 10))
  port_spectrogram=$((base))
  port_jammer=$((base + 1))
  port_stats=$((base + 2))
  port_deinterleave=$((base + 3))

  start=$(date +%s.%N)
  "$BIN/deinterleave_service" "$port_deinterleave" >/dev/null 2>&1 &
  pid_d=$!
  wait_for_port "$port_deinterleave"
  "$BIN/stats_service" "$port_stats" 127.0.0.1 "$port_deinterleave" >/dev/null 2>&1 &
  pid_s=$!
  wait_for_port "$port_stats"
  "$BIN/jammer_service" "$port_jammer" 127.0.0.1 "$port_stats" >/dev/null 2>&1 &
  pid_j=$!
  wait_for_port "$port_jammer"
  "$BIN/spectrogram_service" "$port_spectrogram" 127.0.0.1 "$port_jammer" >/dev/null 2>&1 &
  pid_sp=$!
  wait_for_port "$port_spectrogram"
  "$BIN/detector_service" 127.0.0.1 "$port_spectrogram" "$NUM_PULSES" >/dev/null
  wait "$pid_sp" "$pid_j" "$pid_s" "$pid_d"
  end=$(date +%s.%N)
  echo "$end - $start" | bc >> "$CPP_MICRO_TIMES"
done
echo "done"
echo

echo "== Python modular monolith =="
for i in $(seq 1 "$RUNS"); do
  start=$(date +%s.%N)
  "$PY" python/monolith/monolith_app.py "$NUM_PULSES" >/dev/null
  end=$(date +%s.%N)
  echo "$end - $start" | bc >> "$PY_MONO_TIMES"
done
echo "done"
echo

echo "== Python microservices (chain) =="
for i in $(seq 1 "$RUNS"); do
  base=$((PY_BASE_PORT + i * 10))
  port_spectrogram=$((base))
  port_jammer=$((base + 1))
  port_stats=$((base + 2))
  port_deinterleave=$((base + 3))

  start=$(date +%s.%N)
  "$PY" "$PYDIR/deinterleave_service.py" "$port_deinterleave" >/dev/null 2>&1 &
  pid_d=$!
  wait_for_port "$port_deinterleave"
  "$PY" "$PYDIR/stats_service.py" "$port_stats" 127.0.0.1 "$port_deinterleave" >/dev/null 2>&1 &
  pid_s=$!
  wait_for_port "$port_stats"
  "$PY" "$PYDIR/jammer_service.py" "$port_jammer" 127.0.0.1 "$port_stats" >/dev/null 2>&1 &
  pid_j=$!
  wait_for_port "$port_jammer"
  "$PY" "$PYDIR/spectrogram_service.py" "$port_spectrogram" 127.0.0.1 "$port_jammer" >/dev/null 2>&1 &
  pid_sp=$!
  wait_for_port "$port_spectrogram"
  "$PY" "$PYDIR/detector_service.py" 127.0.0.1 "$port_spectrogram" "$NUM_PULSES" >/dev/null
  wait "$pid_sp" "$pid_j" "$pid_s" "$pid_d"
  end=$(date +%s.%N)
  echo "$end - $start" | bc >> "$PY_MICRO_TIMES"
done
echo "done"
echo

python3 - "$CPP_MONO_TIMES" "$CPP_MICRO_TIMES" "$PY_MONO_TIMES" "$PY_MICRO_TIMES" "$RUNS" "$NUM_PULSES" <<'PYEOF'
import statistics
import sys

cpp_mono_path, cpp_micro_path, py_mono_path, py_micro_path, runs, num_pulses = sys.argv[1:7]


def load(path):
    with open(path) as f:
        return [float(line.strip()) * 1000.0 for line in f if line.strip()]  # -> ms


cpp_mono = load(cpp_mono_path)
cpp_micro = load(cpp_micro_path)
py_mono = load(py_mono_path)
py_micro = load(py_micro_path)


def summarize(name, xs):
    print(
        f"{name:>24}  min={min(xs):9.3f}ms  median={statistics.median(xs):9.3f}ms  "
        f"mean={statistics.mean(xs):9.3f}ms  stdev={statistics.pstdev(xs):8.3f}ms  "
        f"max={max(xs):9.3f}ms"
    )


print(f"Results over {runs} runs, {num_pulses} pulses/run:\n")
summarize("C++ monolith", cpp_mono)
summarize("C++ microservices", cpp_micro)
summarize("Python monolith", py_mono)
summarize("Python microservices", py_micro)

print()
mc, mp = statistics.mean(cpp_mono), statistics.mean(py_mono)
sc, sp = statistics.mean(cpp_micro), statistics.mean(py_micro)
print(f"Python monolith is {mp / mc:.1f}x the C++ monolith mean")
print(f"Python microservices is {sp / sc:.1f}x the C++ microservices mean")
print(f"C++ microservices is {sc / mc:.2f}x the C++ monolith mean")
print(f"Python microservices is {sp / mp:.2f}x the Python monolith mean")
PYEOF
