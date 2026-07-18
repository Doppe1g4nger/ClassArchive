#!/usr/bin/env bash
# Times steady-state processing throughput for all four implementations --
# C++ monolith, C++ microservices, Python monolith, Python microservices --
# with process startup, dlopen()/import, and connection establishment
# excluded from the measurement. Each program/chain self-reports its own
# steady-state duration on a "STEADY_STATE_MS <value>" line (see
# monolith_main.cpp / monolith_app.py, where the timer wraps only the
# batch-processing loop, and
# microservice/deinterleave_service.{cpp,py}/detector_service.{cpp,py},
# where it's the sink's first-receive-to-last-receive span); this script
# launches each one N times, greps that self-reported number out of
# stdout, and summarizes it the same way the old wall-clock benchmarks did.
#
# This supersedes scripts/benchmark.sh and scripts/benchmark_python.sh,
# which timed each process from the outside (`date` before/after `exec`)
# and so necessarily included process startup and, for the microservice
# chain, four sequential TCP handshakes -- overhead this script is
# specifically designed to exclude. See "Steady-state benchmark" in
# README.md for the numbers and why the distinction matters.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-20}"
# 50,000 pulses = 50 batches/run, so each run's sink-side steady-state
# reading averages over 50 inter-arrival gaps instead of 1-2. A smaller
# default (e.g. 1000 pulses = 1-2 batches) was tried during calibration
# and produced wildly noisy, implausible run-to-run swings -- there just
# aren't enough batches for "first-recv-to-last-recv" to mean anything.
NUM_PULSES="${2:-50000}"
# Defaults kept below the kernel's ephemeral port range and given their
# own non-overlapping-in-practice blocks (the C++ and Python loops never
# run concurrently, so reusing nearby ranges is safe, but keeping them
# distinct makes log output easier to attribute while debugging).
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
PY_MULTI_TIMES="$(mktemp)"
SINK_OUT="$(mktemp)"
trap 'rm -f "$CPP_MONO_TIMES" "$CPP_MICRO_TIMES" "$PY_MONO_TIMES" "$PY_MICRO_TIMES" "$PY_MULTI_TIMES" "$SINK_OUT"' EXIT

# Pulls the numeric value out of a "... STEADY_STATE_MS <value>" line from
# stdin. Every program in this repo prints exactly one such line.
extract_ms() {
  grep -o 'STEADY_STATE_MS [0-9.]*' | awk '{print $2}'
}

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

echo "Benchmarking steady-state throughput: $RUNS runs each, $NUM_PULSES pulses per run"
echo

echo "== C++ modular monolith =="
for i in $(seq 1 "$RUNS"); do
  "$BIN/monolith_app" "$BIN" "$NUM_PULSES" | extract_ms >> "$CPP_MONO_TIMES"
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

  # Only the sink's (deinterleave_service's) output is captured -- its
  # STEADY_STATE_MS is the whole pipeline's steady-state drain time (see
  # this script's header comment). The other four services' stdout is
  # discarded the same way the old benchmark did.
  "$BIN/deinterleave_service" "$port_deinterleave" >"$SINK_OUT" 2>&1 &
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

  extract_ms <"$SINK_OUT" >>"$CPP_MICRO_TIMES"
done
echo "done"
echo

echo "== Python modular monolith =="
for i in $(seq 1 "$RUNS"); do
  "$PY" python/monolith/monolith_app.py "$NUM_PULSES" | extract_ms >> "$PY_MONO_TIMES"
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

  "$PY" "$PYDIR/deinterleave_service.py" "$port_deinterleave" >"$SINK_OUT" 2>&1 &
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

  extract_ms <"$SINK_OUT" >>"$PY_MICRO_TIMES"
done
echo "done"
echo

echo "== Python multiproc (forked chain) =="
for i in $(seq 1 "$RUNS"); do
  "$PY" python/multiproc/multiproc_monolith_app.py "$NUM_PULSES" | extract_ms >> "$PY_MULTI_TIMES"
done
echo "done"
echo

python3 - "$CPP_MONO_TIMES" "$CPP_MICRO_TIMES" "$PY_MONO_TIMES" "$PY_MICRO_TIMES" "$PY_MULTI_TIMES" "$RUNS" "$NUM_PULSES" <<'PYEOF'
import statistics
import sys

cpp_mono_path, cpp_micro_path, py_mono_path, py_micro_path, py_multi_path, runs, num_pulses = (
    sys.argv[1:8]
)


def load(path):
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]  # already ms


cpp_mono = load(cpp_mono_path)
cpp_micro = load(cpp_micro_path)
py_mono = load(py_mono_path)
py_micro = load(py_micro_path)
py_multi = load(py_multi_path)

for name, xs, expected in [
    ("C++ monolith", cpp_mono, int(runs)),
    ("C++ microservices", cpp_micro, int(runs)),
    ("Python monolith", py_mono, int(runs)),
    ("Python microservices", py_micro, int(runs)),
    ("Python multiproc", py_multi, int(runs)),
]:
    if len(xs) != expected:
        print(
            f"warning: {name} produced {len(xs)} STEADY_STATE_MS readings, expected {expected} "
            "(a run likely failed -- check for stray output above)",
            file=sys.stderr,
        )


def summarize(name, xs):
    if not xs:
        print(f"{name:>24}  no data")
        return
    print(
        f"{name:>24}  min={min(xs):9.3f}ms  median={statistics.median(xs):9.3f}ms  "
        f"mean={statistics.mean(xs):9.3f}ms  stdev={statistics.pstdev(xs):8.3f}ms  "
        f"max={max(xs):9.3f}ms"
    )


print(f"Steady-state results over {runs} runs, {num_pulses} pulses/run:\n")
summarize("C++ monolith", cpp_mono)
summarize("C++ microservices", cpp_micro)
summarize("Python monolith", py_mono)
summarize("Python microservices", py_micro)
summarize("Python multiproc", py_multi)

if cpp_mono and cpp_micro and py_mono and py_micro and py_multi:
    print()
    mc, mp = statistics.mean(cpp_mono), statistics.mean(py_mono)
    sc, sp = statistics.mean(cpp_micro), statistics.mean(py_micro)
    pm = statistics.mean(py_multi)
    print(f"Python monolith is {mp / mc:.1f}x the C++ monolith mean")
    print(f"Python microservices is {sp / sc:.1f}x the C++ microservices mean")
    print(f"C++ microservices is {sc / mc:.2f}x the C++ monolith mean")
    print(f"Python microservices is {sp / mp:.2f}x the Python monolith mean")
    print(f"Python multiproc is {pm / mp:.2f}x the Python monolith mean")
    print(f"Python multiproc is {pm / sp:.2f}x the Python microservices mean")
PYEOF
