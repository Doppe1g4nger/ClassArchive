"""multiproc: a third architecture, between the other two. One Python
entry point (multiproc_monolith_app.py) forks five worker processes --
detector_worker.py through deinterleave_worker.py, each mirroring its
microservice/*_service.py counterpart's algorithm and field-clearing
discipline exactly -- and wires them together with multiprocessing.Pipe()
connections instead of TCP sockets.

The point: "Python vs C++" (see the top-level README) found that the
microservices build beats the monolith at full scale because it's five
separate OS processes, each with its own GIL, so the pipeline can use
multiple cores. That raises a question the microservices build alone
can't answer -- is escaping the GIL what matters, or does the
network/serialization boundary matter too? This package is a multi-
process build with no network boundary at all, to isolate that variable.
See multiproc_monolith_app.py's docstring for the full comparison.
"""
