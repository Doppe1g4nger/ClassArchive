"""Unit tests for the Python side, mirroring tests/pulsecore_tests.cpp
case for case (including the same pinned golden values, so the two
languages' generators can't silently drift apart -- see
test_pulsecore.py), plus parity tests for the numba/numpy variant
kernels that skip cleanly when those optional dependencies aren't
installed. stdlib unittest only -- no new dependencies.

Run via scripts/run_tests.sh, or directly:

    ./scripts/gen_python_proto.sh
    python3 -m unittest discover -s python/tests -v
"""
import os
import sys

# Same path bootstrap every entry point in python/ uses, so `pulsecore`
# and the variant packages import the same way here as everywhere else.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
