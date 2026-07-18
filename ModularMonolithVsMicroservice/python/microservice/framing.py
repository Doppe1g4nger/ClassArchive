"""Minimal length-prefixed protobuf-over-TCP framing -- Python port of
microservice/net/framing.cpp. Same wire format: a 4-byte big-endian
length prefix followed by a serialized protobuf message, and the same
TCP_NODELAY-on-every-socket policy, so a capture of this traffic is
indistinguishable from the C++ build's.
"""
import socket
import struct


def listen(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(1)
    return sock


def accept(listen_sock: socket.socket) -> socket.socket:
    conn, _ = listen_sock.accept()
    # Disable Nagle's algorithm -- see framing.cpp's SetNoDelay() for why
    # this matters even though this demo only ever has one message in
    # flight per direction at a time.
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn


def connect(host: str, port: int) -> socket.socket:
    sock = socket.create_connection((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def send_message(sock: socket.socket, payload: bytes) -> bool:
    """Sends a length-prefixed message. Header and payload go out as one
    sendall() call on one concatenated buffer -- the Python equivalent of
    framing.cpp's single writev() call, avoiding a second small send()
    for the 4-byte header alone."""
    header = struct.pack(">I", len(payload))
    try:
        sock.sendall(header + payload)
    except OSError:
        return False
    return True


def _read_full(sock: socket.socket, n: int) -> bytes | None:
    chunks = []
    remaining = n
    while remaining > 0:
        try:
            chunk = sock.recv(remaining)
        except OSError:
            return None
        if not chunk:
            return None  # peer closed
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> bytes | None:
    """Receives one length-prefixed message. Returns None on a closed
    connection or I/O error (mirroring RecvMessage's bool return, just
    spelled as Optional[bytes] instead of an out-parameter)."""
    header = _read_full(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0:
        return b""
    return _read_full(sock, length)
