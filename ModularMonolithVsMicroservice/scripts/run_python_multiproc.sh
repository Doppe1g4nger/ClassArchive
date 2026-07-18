#!/usr/bin/env bash
# Runs the Python multiproc variant: one process that forks five worker
# processes wired by multiprocessing.Pipe(), instead of five independent
# executables wired by TCP sockets (run_python_microservices.sh) or one
# process running five imported modules in a single thread
# (run_python_monolith.sh). See
# python/multiproc/multiproc_monolith_app.py's docstring for what forking
# workers internally buys over both of those.
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/gen_python_proto.sh

python3 python/multiproc/multiproc_monolith_app.py "$@"
