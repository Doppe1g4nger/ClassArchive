#include "framing.h"

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>

namespace netutil {

namespace {

// Sized so the largest frame this repo produces (~200KB serialized
// packed PipelineFrame with iq + events) fits with headroom, times
// enough slots to keep the pipeline from stalling on a briefly slow
// consumer. 4 slots x 512KB = 2MB per link; four links = 8MB of shm.
constexpr uint32_t kSlotSize = 512 * 1024;
constexpr uint32_t kSlotCount = 4;  // power of two
constexpr uint32_t kMagic = 0x50524E47;  // "PRNG" -- pulse ring

struct RingHeader {
  uint32_t magic;
  uint32_t slot_size;
  uint32_t slot_count;
  alignas(64) std::atomic<uint64_t> head;         // producer-owned
  alignas(64) std::atomic<uint64_t> tail;         // consumer-owned
  alignas(64) std::atomic<uint32_t> writer_done;  // producer's FIN
};

constexpr size_t kRingBytes = sizeof(RingHeader) + size_t(kSlotCount) * kSlotSize;

void RingName(uint16_t port, char* out, size_t out_len) {
  std::snprintf(out, out_len, "/pulse_ring_%u", static_cast<unsigned>(port));
}

// Brief spin for the common fast case, then yield the core -- five
// processes share four cores in this repo's benchmark, so hogging a
// core while waiting would slow the very stage being waited on.
inline void WaitPause(int& spins) {
  if (++spins < 256) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  } else {
    ::sched_yield();
  }
}

}  // namespace

struct Channel {
  RingHeader* hdr = nullptr;
  uint8_t* slots = nullptr;
  uint16_t port = 0;
  bool is_producer = false;
  bool closed = false;
};

Channel* Listen(uint16_t port) {
  char name[64];
  RingName(port, name, sizeof(name));

  // Replace any stale segment from a crashed previous run, then create
  // fresh -- the consumer owns the segment's lifecycle.
  ::shm_unlink(name);
  const int fd = ::shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) return nullptr;
  if (::ftruncate(fd, static_cast<off_t>(kRingBytes)) != 0) {
    ::close(fd);
    ::shm_unlink(name);
    return nullptr;
  }
  void* mem = ::mmap(nullptr, kRingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (mem == MAP_FAILED) {
    ::shm_unlink(name);
    return nullptr;
  }

  auto* hdr = new (mem) RingHeader();
  hdr->slot_size = kSlotSize;
  hdr->slot_count = kSlotCount;
  hdr->head.store(0, std::memory_order_relaxed);
  hdr->tail.store(0, std::memory_order_relaxed);
  hdr->writer_done.store(0, std::memory_order_relaxed);
  // magic last, with release: a connecting producer that sees the magic
  // is guaranteed to see the initialized fields above.
  hdr->magic = 0;
  std::atomic_thread_fence(std::memory_order_release);
  hdr->magic = kMagic;

  auto* ch = new Channel();
  ch->hdr = hdr;
  ch->slots = static_cast<uint8_t*>(mem) + sizeof(RingHeader);
  ch->port = port;
  ch->is_producer = false;
  return ch;
}

Channel* Accept(Channel* listen_channel) { return listen_channel; }

Channel* Connect(const std::string& /*host*/, uint16_t port) {
  char name[64];
  RingName(port, name, sizeof(name));

  // Wait for the consumer to create the segment -- bounded so a
  // mis-wired chain still fails visibly instead of hanging forever.
  int fd = -1;
  for (int attempt = 0; attempt < 20000; ++attempt) {  // ~20s worst case
    fd = ::shm_open(name, O_RDWR, 0600);
    if (fd >= 0) break;
    ::usleep(1000);
  }
  if (fd < 0) return nullptr;

  // The name exists as soon as the consumer's shm_open(O_CREAT)
  // returns, which is BEFORE its ftruncate() has sized the segment.
  // mmap'ing a still-zero-length segment succeeds, but the first page
  // touch past EOF delivers SIGBUS -- even the magic-word spin below
  // would fault. So wait for the file to reach full size first; only
  // then is every page of the mapping backed. (Caught as a real
  // once-in-hundreds-of-runs Bus error during benchmarking, not
  // hypothesized.)
  struct stat st;
  for (int attempt = 0; attempt < 20000; ++attempt) {
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      return nullptr;
    }
    if (st.st_size >= static_cast<off_t>(kRingBytes)) break;
    ::usleep(1000);
  }
  if (st.st_size < static_cast<off_t>(kRingBytes)) {
    ::close(fd);
    return nullptr;
  }

  void* mem = ::mmap(nullptr, kRingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (mem == MAP_FAILED) return nullptr;

  auto* hdr = static_cast<RingHeader*>(mem);
  int spins = 0;
  while (hdr->magic != kMagic) WaitPause(spins);
  std::atomic_thread_fence(std::memory_order_acquire);

  auto* ch = new Channel();
  ch->hdr = hdr;
  ch->slots = static_cast<uint8_t*>(mem) + sizeof(RingHeader);
  ch->port = port;
  ch->is_producer = true;
  return ch;
}

bool SendMessage(Channel* ch, const std::string& payload) {
  if (ch == nullptr || ch->closed) return false;
  if (payload.size() + sizeof(uint32_t) > kSlotSize) return false;
  RingHeader* h = ch->hdr;

  const uint64_t head = h->head.load(std::memory_order_relaxed);
  int spins = 0;
  while (head - h->tail.load(std::memory_order_acquire) >= kSlotCount) {
    WaitPause(spins);
  }

  uint8_t* slot = ch->slots + (head & (kSlotCount - 1)) * size_t(kSlotSize);
  const uint32_t len = static_cast<uint32_t>(payload.size());
  std::memcpy(slot, &len, sizeof(len));
  std::memcpy(slot + sizeof(len), payload.data(), len);
  h->head.store(head + 1, std::memory_order_release);
  return true;
}

bool RecvMessage(Channel* ch, std::string* payload) {
  if (ch == nullptr || ch->closed) return false;
  RingHeader* h = ch->hdr;

  const uint64_t tail = h->tail.load(std::memory_order_relaxed);
  int spins = 0;
  while (tail == h->head.load(std::memory_order_acquire)) {
    if (h->writer_done.load(std::memory_order_acquire) != 0 &&
        tail == h->head.load(std::memory_order_acquire)) {
      return false;  // drained and producer closed: end of stream
    }
    WaitPause(spins);
  }

  const uint8_t* slot = ch->slots + (tail & (kSlotCount - 1)) * size_t(kSlotSize);
  uint32_t len = 0;
  std::memcpy(&len, slot, sizeof(len));
  payload->assign(reinterpret_cast<const char*>(slot + sizeof(len)), len);
  h->tail.store(tail + 1, std::memory_order_release);
  return true;
}

void Close(Channel* ch) {
  if (ch == nullptr || ch->closed) return;
  ch->closed = true;
  if (ch->is_producer) {
    ch->hdr->writer_done.store(1, std::memory_order_release);
    ::munmap(ch->hdr, kRingBytes);
  } else {
    char name[64];
    RingName(ch->port, name, sizeof(name));
    ::munmap(ch->hdr, kRingBytes);
    ::shm_unlink(name);
  }
  delete ch;
}

}  // namespace netutil
