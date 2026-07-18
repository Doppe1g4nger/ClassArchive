cmake_minimum_required(VERSION 3.16)
project(ModularMonolithVsMicroservice CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Cross-TU inlining across pulsecore's small, frequently-called Process()
# methods and their callers (the plugins, monolith_main.cpp, each
# microservice's main.cpp) buys a modest additional win on top of the
# regular -O3 build with no source changes and no semantic risk -- LTO
# only affects codegen within each already-existing final link target
# (each .so, each executable), never linkage/registration behavior, so
# it can't disturb the pulse_proto single-registration invariant below.
# Only enabled when the toolchain actually supports it and only for
# Release, so a debug build's compile times aren't affected.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()

# Put every binary (executables and plugin .so's) in one directory so
# monolith_app can find its plugins next to itself with no extra setup.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/pulse.proto)

# ---------------------------------------------------------------------------
# pulse_proto: the generated protobuf message code, built as its own SHARED
# library. protobuf-generated code registers its message descriptors in a
# process-global database at static-init time, so it must exist exactly
# once per process. monolith_app links pulsecore directly *and* dlopen()s
# two plugins that also depend on pulsecore -- if pulse.pb.cc were baked
# statically into pulsecore, all three would each try to register
# "pulse.proto" and the process would abort on a duplicate-registration
# CHECK failure. Making it one shared library that everything links
# dynamically means the dynamic linker loads/registers it once, no matter
# how many places pull it in.
# ---------------------------------------------------------------------------
add_library(pulse_proto SHARED ${PROTO_SRCS})
set_target_properties(pulse_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pulse_proto PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(pulse_proto PUBLIC ${Protobuf_LIBRARIES})

# ---------------------------------------------------------------------------
# pulsecore: the shared business logic (detector, stats, synthetic source).
# Plain C++ with no global state, so it's fine for this to be statically
# linked into multiple binaries in the same process -- unlike pulse_proto,
# duplicating it doesn't cause a conflict.
# ---------------------------------------------------------------------------
add_library(pulsecore STATIC
  common/src/pulse_detector.cpp
  common/src/pulse_stats.cpp
  common/src/iq_source.cpp
  common/src/spectrogram.cpp
  common/src/jammer.cpp
  common/src/deinterleaver.cpp
)
set_target_properties(pulsecore PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pulsecore PUBLIC common/include)
target_link_libraries(pulsecore PUBLIC pulse_proto)

# ---------------------------------------------------------------------------
# Modular monolith: one host executable, two dynamically loaded modules.
# ---------------------------------------------------------------------------
add_library(pulse_detector_plugin SHARED monolith/plugins/pulse_detector_plugin.cpp)
target_include_directories(pulse_detector_plugin PRIVATE monolith/include)
target_link_libraries(pulse_detector_plugin PRIVATE pulsecore)

add_library(pulse_stats_plugin SHARED monolith/plugins/pulse_stats_plugin.cpp)
target_include_directories(pulse_stats_plugin PRIVATE monolith/include)
target_link_libraries(pulse_stats_plugin PRIVATE pulsecore)

add_library(pulse_spectrogram_plugin SHARED monolith/plugins/pulse_spectrogram_plugin.cpp)
target_include_directories(pulse_spectrogram_plugin PRIVATE monolith/include)
target_link_libraries(pulse_spectrogram_plugin PRIVATE pulsecore)

add_library(pulse_jammer_plugin SHARED monolith/plugins/pulse_jammer_plugin.cpp)
target_include_directories(pulse_jammer_plugin PRIVATE monolith/include)
target_link_libraries(pulse_jammer_plugin PRIVATE pulsecore)

add_library(pulse_deinterleaver_plugin SHARED monolith/plugins/pulse_deinterleaver_plugin.cpp)
target_include_directories(pulse_deinterleaver_plugin PRIVATE monolith/include)
target_link_libraries(pulse_deinterleaver_plugin PRIVATE pulsecore)

add_executable(monolith_app monolith/host/monolith_main.cpp)
target_include_directories(monolith_app PRIVATE monolith/include)
target_link_libraries(monolith_app PRIVATE pulsecore ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# AVX2 variant: a fourth C++ build, answering "optimize further with AVX
# instructions?" from the correctness-audit follow-up. Two of pulsecore's
# five algorithms (the detector's threshold decision, the jammer's power
# sum) get AVX2-vectorized ports -- pulse_detector_avx.cpp/jammer_avx.cpp
# -- built as their own small library so the original, unmodified
# pulsecore stays exactly as it was for every other target above (the
# same "add a variant, don't replace the baseline" approach the Python
# side's multiproc/numba_variant/numpy_variant builds already use). The
# spectrogram, stats, and deinterleaver stages are unmodified pulsecore
# code, reused directly -- see monolith/host/avx_monolith_main.cpp and
# common/include/pulse_detector_avx.h/jammer_avx.h for why AVX wasn't
# applied to the spectrogram (its per-sample cos()/sin() calls have no
# portable, dependency-free vectorized equivalent) and why the detector
# stays bit-identical to the scalar version while the jammer doesn't.
#
# Guarded behind an actual compiler-support check rather than assumed,
# same reasoning as the LTO check above: skip the target gracefully
# instead of failing the whole build on a toolchain/CPU that can't do it.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2_FMA)
if(COMPILER_SUPPORTS_AVX2_FMA)
  add_library(pulsecore_avx STATIC
    common/src/pulse_detector_avx.cpp
    common/src/jammer_avx.cpp
  )
  set_target_properties(pulsecore_avx PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(pulsecore_avx PUBLIC common/include)
  target_link_libraries(pulsecore_avx PUBLIC pulse_proto)
  # -ffp-contract=off: stop the compiler from silently fusing
  # pulse_detector_avx.cpp's separate multiply+add into one FMA
  # instruction -- that fusion rounds once instead of twice and would
  # break the detector's bit-identical guarantee. jammer_avx.cpp uses
  # FMA explicitly via intrinsics regardless of this flag (its output
  # isn't bit-identical to begin with -- see jammer_avx.h).
  target_compile_options(pulsecore_avx PRIVATE -mavx2 -mfma -ffp-contract=off)

  add_executable(avx_monolith_app monolith/host/avx_monolith_main.cpp)
  target_include_directories(avx_monolith_app PRIVATE monolith/include)
  target_link_libraries(avx_monolith_app PRIVATE pulsecore pulsecore_avx)

  add_executable(verify_avx_variant monolith/host/verify_avx_variant_main.cpp)
  target_link_libraries(verify_avx_variant PRIVATE pulsecore pulsecore_avx)
else()
  message(STATUS "Compiler doesn't support -mavx2 -mfma -- skipping avx_monolith_app")
endif()

# ---------------------------------------------------------------------------
# Microservice pair: two executables, connected by protobuf over TCP.
# ---------------------------------------------------------------------------
add_library(netutil STATIC microservice/net/framing.cpp)
target_include_directories(netutil PUBLIC microservice/net)

add_executable(detector_service microservice/detector_service/main.cpp)
target_link_libraries(detector_service PRIVATE pulsecore netutil)

add_executable(stats_service microservice/stats_service/main.cpp)
target_link_libraries(stats_service PRIVATE pulsecore netutil)

add_executable(spectrogram_service microservice/spectrogram_service/main.cpp)
target_link_libraries(spectrogram_service PRIVATE pulsecore netutil)

add_executable(jammer_service microservice/jammer_service/main.cpp)
target_link_libraries(jammer_service PRIVATE pulsecore netutil)

add_executable(deinterleave_service microservice/deinterleave_service/main.cpp)
target_link_libraries(deinterleave_service PRIVATE pulsecore netutil)
