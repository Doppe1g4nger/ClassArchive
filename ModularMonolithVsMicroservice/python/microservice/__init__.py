"""microservice: Python port of microservice/ -- the same length-prefixed
protobuf-over-TCP framing and the same five-stage chain topology as the
C++ build, wired the same way (detector_service.py -> spectrogram_service.py
-> jammer_service.py -> stats_service.py -> deinterleave_service.py).

Because the wire format (framing.py) is byte-for-byte identical to
microservice/net/framing.cpp and every message is the same
pulse.PipelineFrame protobuf, a Python service and a C++ service can talk
to each other without either one knowing the other's language -- protobuf
over TCP doesn't care what emitted the bytes. This repo doesn't test that
combination, but it's true by construction, and it's the whole point of
picking a wire format instead of a language-specific IPC mechanism.
"""
