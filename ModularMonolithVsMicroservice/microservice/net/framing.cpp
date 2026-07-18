#include "framing.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>

namespace netutil {

namespace {

bool ReadFull(int fd, void* buf, size_t len) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    const ssize_t n = ::recv(fd, p, remaining, 0);
    if (n <= 0) return false;  // 0 = peer closed, <0 = error
    p += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

bool WriteFull(int fd, const void* buf, size_t len) {
  const auto* p = static_cast<const uint8_t*>(buf);
  size_t remaining = len;
  while (remaining > 0) {
    const ssize_t n = ::send(fd, p, remaining, 0);
    if (n <= 0) return false;
    p += n;
    remaining -= static_cast<size_t>(n);
  }
  return true;
}

}  // namespace

int Listen(uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  const int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  if (::listen(fd, /*backlog=*/1) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

int Accept(int listen_fd) { return ::accept(listen_fd, nullptr, nullptr); }

int Connect(const std::string& host, uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    ::close(fd);
    return -1;
  }

  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

bool SendMessage(int fd, const std::string& payload) {
  const uint32_t len = htonl(static_cast<uint32_t>(payload.size()));
  if (!WriteFull(fd, &len, sizeof(len))) return false;
  return WriteFull(fd, payload.data(), payload.size());
}

bool RecvMessage(int fd, std::string* payload) {
  uint32_t len_be = 0;
  if (!ReadFull(fd, &len_be, sizeof(len_be))) return false;

  const uint32_t len = ntohl(len_be);
  payload->resize(len);
  if (len == 0) return true;
  return ReadFull(fd, payload->data(), len);
}

}  // namespace netutil
