#pragma once

#include "pulse.pb.h"

// Modules are still dlopen()'d independently and only exposed through a
// handful of named symbols the host looks up with dlsym() -- that part of
// "modular monolith" is unchanged. What changed is *what* crosses that
// boundary.
//
// An earlier version of this header made every module speak only raw
// bytes, on the theory that it would mirror the microservice build's wire
// format and make the two architectures directly comparable. In practice
// that forced the host to serialize each IQBatch to bytes and the
// detector module to parse it straight back -- a real cost paid for no
// real reason, since host and module run in the same process and share
// one address space.
//
// That cost is avoidable, and here's why it's *safe* to avoid: every
// target in this build links the generated protobuf code from one shared
// library, pulse_proto (see CMakeLists.txt -- originally added to stop
// duplicate descriptor registration). Because of that, there is exactly
// one definition of pulse::PipelineFrame and friends -- one vtable, one
// layout -- loaded into the process no matter how many plugins reference
// it. That's precisely the condition under which passing a C++ reference
// to one of those types across a dlopen() boundary is well-defined
// instead of undefined behavior.
//
// The trade-off: this only works because every module in this build is
// compiled by the same toolchain against the same headers as the host.
// That's a reasonable assumption for a modular monolith shipped as one
// versioned bundle (which is the point of the pattern), but it is *not*
// an assumption the microservices get to make -- each service is a
// separate process with a separate address space, so it has no choice
// but to serialize onto the wire. That asymmetry is real, not an
// artifact of this demo: in-process module boundaries can avoid
// serialization if they share a build; true process boundaries cannot.
//
// Every stage shares one signature now instead of five different ones.
// That's a direct consequence of this being a linear chain rather than a
// fan-out: every stage receives the same evolving pulse::PipelineFrame,
// reads whatever fields it needs, and writes its own field before the
// frame moves to the next stage. A stage that reads a field an earlier
// stage hasn't populated yet is a wiring bug, not something the type
// system catches -- see monolith_main.cpp and
// scripts/run_microservices.sh for the chain order both architectures
// rely on: detector -> spectrogram -> jammer -> stats -> deinterleaver.
//
// That specific order isn't arbitrary: detector, spectrogram, and jammer
// are the three stages that read frame.iq() (the largest field by far,
// 4096 samples/batch), so they're grouped first. Once jammer -- the last
// of the three -- has read it, the microservice build clears frame.iq()
// before forwarding, so the last two hops (jammer->stats,
// stats->deinterleaver) never carry the raw samples at all. Each stage
// also clears its own summary field right after using it locally, since
// no later stage ever reads an earlier stage's summary -- see
// microservice/{spectrogram,jammer,stats}_service/main.cpp. The monolith
// doesn't need any of this: passing frame by reference costs nothing
// extra regardless of which fields are populated, so its modules never
// clear anything. Only a real process boundary has to care what it's
// carrying.
extern "C" {

typedef void* pulse_module_t;

typedef pulse_module_t (*pulse_module_create_fn)(const char* config);
typedef void (*pulse_module_destroy_fn)(pulse_module_t handle);

typedef void (*pulse_stage_process_fn)(pulse_module_t handle, pulse::PipelineFrame* frame);

}  // extern "C"

#define PULSE_MODULE_CREATE_SYM "pulse_module_create"
#define PULSE_MODULE_DESTROY_SYM "pulse_module_destroy"
#define PULSE_STAGE_PROCESS_SYM "pulse_stage_process"
