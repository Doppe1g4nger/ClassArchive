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
// one definition of pulse::IQBatch, pulse::PulseEventBatch, etc. -- one
// vtable, one layout -- loaded into the process no matter how many
// plugins reference it. That's precisely the condition under which
// passing a C++ reference to one of those types across a dlopen()
// boundary is well-defined instead of undefined behavior. So modules
// now take and return typed pulse:: messages directly: no serialize, no
// parse, no heap-allocated byte buffer to free.
//
// The trade-off: this only works because every module in this build is
// compiled by the same toolchain against the same headers as the host.
// That's a reasonable assumption for a modular monolith shipped as one
// versioned bundle (which is the point of the pattern), but it is *not*
// a assumption the microservices get to make -- detector_service and
// stats_service are separate processes with separate address spaces, so
// they have no choice but to serialize onto the wire. That asymmetry is
// real, not an artifact of this demo: in-process module boundaries can
// avoid serialization if they share a build; true process boundaries
// cannot.
extern "C" {

typedef void* pulse_module_t;

typedef pulse_module_t (*pulse_module_create_fn)(const char* config);
typedef void (*pulse_module_destroy_fn)(pulse_module_t handle);

// Detector modules: consume a batch of IQ samples, append any pulses
// found into *out (caller owns both; out is not cleared by the callee's
// caller, so implementations must clear it themselves if they don't want
// to accumulate across calls).
typedef void (*pulse_detector_process_fn)(pulse_module_t handle, const pulse::IQBatch& batch,
                                           pulse::PulseEventBatch* out);

// Stats modules: fold a batch of pulse events into running state and
// write the summary computed so far into *out.
typedef void (*pulse_stats_process_fn)(pulse_module_t handle, const pulse::PulseEventBatch& batch,
                                        pulse::PulseSummary* out);

// Spectrogram modules: fold a batch of IQ samples into a running
// magnitude spectrum and write the summary computed so far into *out.
typedef void (*pulse_spectrogram_process_fn)(pulse_module_t handle, const pulse::IQBatch& batch,
                                              pulse::SpectrogramSummary* out);

// Jammer-detection modules: fold a batch of IQ samples into running
// jam-detection state and write the summary computed so far into *out.
typedef void (*pulse_jammer_process_fn)(pulse_module_t handle, const pulse::IQBatch& batch,
                                         pulse::JamSummary* out);

// Deinterleaver modules: fold a batch of pulse events into candidate
// emitter tracks and write every track seen so far into *out.
typedef void (*pulse_deinterleaver_process_fn)(pulse_module_t handle,
                                                const pulse::PulseEventBatch& batch,
                                                pulse::DeinterleaveSummary* out);

}  // extern "C"

#define PULSE_MODULE_CREATE_SYM "pulse_module_create"
#define PULSE_MODULE_DESTROY_SYM "pulse_module_destroy"
#define PULSE_DETECTOR_PROCESS_SYM "pulse_detector_process"
#define PULSE_STATS_PROCESS_SYM "pulse_stats_process"
#define PULSE_SPECTROGRAM_PROCESS_SYM "pulse_spectrogram_process"
#define PULSE_JAMMER_PROCESS_SYM "pulse_jammer_process"
#define PULSE_DEINTERLEAVER_PROCESS_SYM "pulse_deinterleaver_process"
