#pragma once

#include <cstdint>

// Plain-C ABI so plugin modules can be dlopen()'d independently of C++
// name-mangling/ABI concerns, mirroring how a real modular monolith keeps
// its plugin boundary stable across independently built .so files.
//
// The key design point of this whole example: the host and every module
// exchange only two things across this boundary -- raw bytes and their
// length. Those bytes happen to be serialized protobuf messages, the very
// same message types (pulse.PulseEventBatch, etc.) that the microservice
// build sends over a TCP socket instead. Swapping dlopen()+function-call
// for connect()+send() is the entire difference between the two
// architectures in this repo.
extern "C" {

typedef void* pulse_module_t;

typedef pulse_module_t (*pulse_module_create_fn)(const char* config);
typedef void (*pulse_module_destroy_fn)(pulse_module_t handle);

// Returns 0 on success. On success the module allocates *out_bytes (the
// caller must release it via pulse_module_free_buffer).
typedef int (*pulse_module_process_fn)(pulse_module_t handle, const uint8_t* in_bytes,
                                        uint32_t in_len, uint8_t** out_bytes,
                                        uint32_t* out_len);

typedef void (*pulse_module_free_buffer_fn)(uint8_t* buffer);

}  // extern "C"

#define PULSE_MODULE_CREATE_SYM "pulse_module_create"
#define PULSE_MODULE_DESTROY_SYM "pulse_module_destroy"
#define PULSE_MODULE_PROCESS_SYM "pulse_module_process"
#define PULSE_MODULE_FREE_BUFFER_SYM "pulse_module_free_buffer"
