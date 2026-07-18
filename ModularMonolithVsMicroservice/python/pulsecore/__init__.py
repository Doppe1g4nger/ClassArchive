"""pulsecore: Python port of common/ -- the architecture-agnostic core
algorithms (detector, stats, spectrogram, jammer, deinterleaver) and the
synthetic IQ generator. Direct, line-for-line ports of the C++ originals,
including their already-established optimizations (e.g. spectrogram's
phasor rotation), so a Python-vs-C++ benchmark measures language/runtime
overhead on the same algorithm rather than an algorithmically-different
reimplementation.

pulse_pb2.py in this directory is generated from proto/pulse.proto by
scripts/gen_python_proto.sh -- it is not checked in, the same way
pulse.pb.cc/h are not checked in for the C++ build (see CMakeLists.txt).
"""
