#!/usr/bin/env bash
# Regenerates python/pulsecore/pulse_pb2.py from proto/pulse.proto so the
# Python port speaks the exact same wire format as the C++ builds. Not
# checked in, same as pulse.pb.cc/h for the C++ build (see CMakeLists.txt)
# -- every script that needs it calls this first.
set -euo pipefail
cd "$(dirname "$0")/.."

protoc --python_out=python/pulsecore --proto_path=proto proto/pulse.proto
