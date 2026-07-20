#pragma once

#include <cstdint>
#include <string>

namespace google {
namespace protobuf {
class MessageLite;
}  // namespace protobuf
}  // namespace google

namespace netutil {

// Theoretical-limits branch: the chain's transport is a shared-memory
// SPSC ring per link instead of a TCP socket -- the "boundary cost
// approaches zero" endgame the main branch's README repeatedly priced
// but declined to build (it trades away the cross-host story and the
// capture-a-wire teaching prop; this branch's charter is speed, so it
// trades). The message *content* is unchanged: the same serialized
// pulse.PipelineFrame bytes that used to cross the socket are memcpy'd
// into a ring slot, so every service's parse/serialize logic, the
// end-of-stream semantics, and all printed output stay identical.
//
// The API shape deliberately mirrors the TCP version it replaced --
// Listen/Accept/Connect/Send/Recv, keyed by the same port numbers (a
// port now names the shm segment /pulse_ring_<port> instead of a TCP
// endpoint) -- so the five services and the startup-order invariant the
// steady-state benchmark leans on carry over unchanged: a service still
// "listens" (creates its inbound ring) only after its own downstream
// connect succeeded, so a producer's Connect() still can't succeed
// before the whole downstream chain is wired.
//
// Ring mechanics: single-producer/single-consumer, power-of-two slot
// count, head/tail release/acquire atomics, spin-then-yield waits (five
// processes share four cores here, so pure busy-spin would fight the
// pipeline it's carrying). The producer's Close() publishes a
// writer-closed flag, which is this transport's FIN: a consumer's
// RecvMessage() returns false once the ring is drained and the flag is
// set -- exactly when the TCP version returned false on peer close.

struct Channel;

// Consumer side: creates (replacing any stale segment) the inbound
// ring for this port. Returns nullptr on failure.
Channel* Listen(uint16_t port);
// Kept for call-site symmetry with the TCP version; the ring needs no
// per-connection accept step, so this is the identity.
Channel* Accept(Channel* listen_channel);
// Producer side: attaches to the ring named by port, waiting for the
// consumer to have created it (the TCP version's connect likewise
// blocked in the scripts' wait-for-listen loops). `host` is accepted
// and ignored so service CLIs stay unchanged. Returns nullptr on
// failure.
Channel* Connect(const std::string& host, uint16_t port);

// Both return false on a closed/failed channel, mirroring the TCP
// versions' semantics. Because a dead peer can't break a shm ring the
// way it breaks a socket, every blocking wait also carries a ~30s
// deadline and returns false on expiry -- otherwise a crashed neighbor
// would leave this process spinning forever (see framing.cpp's
// Deadline for the measured incident behind this).
bool SendMessage(Channel* ch, const std::string& payload);
bool RecvMessage(Channel* ch, std::string* payload);

// Zero-copy variants (theoretical-limits round three): the message
// serializes DIRECTLY into the ring slot and parses DIRECTLY out of
// it, eliminating the intermediate std::string entirely. Profiling
// caught what that string was costing the string-based path per hop:
// std::string::resize zero-fills the ~200KB payload before serialize
// overwrites it, then the bytes are copied once into the slot on send
// and once out of it on receive -- memcpy+memset were ~47% of the
// chain's pacing service's instructions. Send fails (false) if the
// serialized size can't fit a slot.
//
// RecvMessage parses with MERGE semantics, matching the in-place
// clear + merge-parse pattern every service already uses (see
// spectrogram_service/main.cpp): the caller clears the repeated-field
// carriers it knows about, then calls this, and parsed content lands
// in the cached, already-allocated objects.
bool SendMessage(Channel* ch, const google::protobuf::MessageLite& message);
bool RecvMessage(Channel* ch, google::protobuf::MessageLite* message);

// Producer close publishes end-of-stream; consumer close unmaps and
// unlinks the segment. Safe to call once per side.
void Close(Channel* ch);

}  // namespace netutil
