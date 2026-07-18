#pragma once

#include <cstdint>
#include <string>

namespace netutil {

// Minimal blocking TCP helpers used to move length-prefixed protobuf
// messages between the two microservice executables. This is the entire
// "microservice framework" in this example -- no gRPC, no HTTP, just a
// 4-byte big-endian length prefix followed by a serialized protobuf
// message -- so the wire format is trivial to reason about and the
// contrast with the monolith's in-process byte hand-off stays clear.

int Listen(uint16_t port);
int Accept(int listen_fd);
int Connect(const std::string& host, uint16_t port);

// Both return false on a closed connection or I/O error.
bool SendMessage(int fd, const std::string& payload);
bool RecvMessage(int fd, std::string* payload);

}  // namespace netutil
