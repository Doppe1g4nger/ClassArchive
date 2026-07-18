cmake_minimum_required(VERSION 3.16)
project(ModularMonolithVsMicroservice CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
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

add_executable(monolith_app monolith/host/monolith_main.cpp)
target_include_directories(monolith_app PRIVATE monolith/include)
target_link_libraries(monolith_app PRIVATE pulsecore ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# Microservice pair: two executables, connected by protobuf over TCP.
# ---------------------------------------------------------------------------
add_library(netutil STATIC microservice/net/framing.cpp)
target_include_directories(netutil PUBLIC microservice/net)

add_executable(detector_service microservice/detector_service/main.cpp)
target_link_libraries(detector_service PRIVATE pulsecore netutil)

add_executable(stats_service microservice/stats_service/main.cpp)
target_link_libraries(stats_service PRIVATE pulsecore netutil)
