#include "framing.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

namespace netutil {

namespace {

void SetNoDelay(int fd) {
  // Disable Nagle's algorithm. This example only ever has one message in
  // flight at a time per direction, so without this a small write (like
  // the 4-byte length prefix, were it sent separately) can sit buffered
  // for up to ~40ms waiting to be coalesced with more outgoing data that
  // never comes. Cheap to set, and standard practice for latency-sensitive
  // socket code.
  const int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

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

int Accept(int listen_fd) {
  const int fd = ::accept(listen_fd, nullptr, nullptr);
  if (fd >= 0) SetNoDelay(fd);
  return fd;
}

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
  SetNoDelay(fd);
  return fd;
}

bool SendMessage(int fd, const std::string& payload) {
  const uint32_t len_be = htonl(static_cast<uint32_t>(payload.size()));

  // Header and payload go out as one writev() call instead of two send()
  // calls. Besides halving the syscall count, this guarantees the kernel
  // never sees the 4-byte header as a complete, sendable unit on its own
  // -- with TCP_NODELAY that distinction rarely matters, but it removes
  // any chance of the header and payload being flushed as two separate
  // TCP segments.
  iovec iov[2];
  iov[0].iov_base = const_cast<uint32_t*>(&len_be);
  iov[0].iov_len = sizeof(len_be);
  iov[1].iov_base = const_cast<char*>(payload.data());
  iov[1].iov_len = payload.size();

  size_t remaining = iov[0].iov_len + iov[1].iov_len;
  int iov_start = 0;
  int iov_count = 2;
  while (remaining > 0) {
    const ssize_t n = ::writev(fd, iov + iov_start, iov_count);
    if (n <= 0) return false;
    remaining -= static_cast<size_t>(n);

    // Almost always finishes in one call on loopback; this loop only
    // matters if the kernel ever accepts a partial write.
    size_t consumed = static_cast<size_t>(n);
    while (consumed > 0) {
      const size_t take = std::min(consumed, iov[iov_start].iov_len);
      iov[iov_start].iov_base = static_cast<uint8_t*>(iov[iov_start].iov_base) + take;
      iov[iov_start].iov_len -= take;
      consumed -= take;
      if (iov[iov_start].iov_len == 0 && iov_count > 1) {
        ++iov_start;
        --iov_count;
      }
    }
  }
  return true;
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
