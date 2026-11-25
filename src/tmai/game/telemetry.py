"""Background TCP client that keeps the newest telemetry packet from the plugin."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace

from tmai.game.protocol import ProtocolError, Telemetry, decode, newer_seq, packet_size_from_header


class TelemetryClient:
    """Connects to the plugin, reconnecting if the game restarts.

    The plugin sends one packet per rendered frame; only the newest is kept,
    so a slow consumer never works on stale data.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host, self.port = host, port
        self._latest: Telemetry | None = None
        self._received_at = 0.0
        self._seq_offset = 0
        self._wire_seq = None
        self._last_seq = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.connected = False
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)

    def start(self) -> "TelemetryClient":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=1.0) as sock:
                    sock.settimeout(1.0)
                    self.connected, self.error = True, None
                    self._read_loop(sock)
            except (OSError, ProtocolError) as exc:
                self.error = str(exc)
            finally:
                with self._cond:
                    self.connected = False
                    self._latest, self._received_at = None, 0.0
                    self._wire_seq = None
                    self._cond.notify_all()
            self._stop.wait(0.5)

    def _read_loop(self, sock: socket.socket) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            buf += chunk
            newest = None
            offset = 0
            while offset < len(buf):
                size = packet_size_from_header(memoryview(buf)[offset:])
                if size is None or len(buf) - offset < size:
                    break
                packet = decode(buf[offset:offset + size])
                offset += size
                if self._wire_seq is not None and not newer_seq(packet.seq, self._wire_seq):
                    continue
                if self._wire_seq is None and self._last_seq is not None:
                    # A reloaded plugin can restart its counter at zero.
                    seq = (packet.seq + self._seq_offset) & 0xffffffff
                    if not newer_seq(seq, self._last_seq):
                        self._seq_offset = self._last_seq + 1 - packet.seq
                self._wire_seq = packet.seq
                self._last_seq = (packet.seq + self._seq_offset) & 0xffffffff
                newest = packet
            if offset:
                buf = buf[offset:]
            if newest is None:
                continue
            with self._cond:
                self._latest = replace(newest, seq=self._last_seq)
                self._received_at = time.monotonic()
                self._cond.notify_all()

    def latest(self) -> Telemetry | None:
        with self._cond:
            return self._latest

    def age(self) -> float:
        """Seconds since the newest packet arrived (inf if none yet)."""
        with self._cond:
            return time.monotonic() - self._received_at if self._latest else float("inf")

    def wait(self, timeout: float = 5.0, after_seq: int | None = None,
             max_age: float | None = None) -> Telemetry:
        """Block until a packet arrives that is newer than `after_seq` (if given)
        and was received at most `max_age` seconds ago (if given), so a stalled
        or vanished game is never read as the present."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                pkt = self._latest
                fresh = max_age is None or time.monotonic() - self._received_at <= max_age
                if pkt is not None and fresh and (after_seq is None or newer_seq(pkt.seq, after_seq)):
                    return pkt
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    what = "no fresh telemetry" if pkt is not None else "no telemetry"
                    raise TimeoutError(
                        f"{what} from the plugin on {self.host}:{self.port} "
                        f"({self.error or 'connected but silent'})"
                    )
                # A stale packet is only replaced by a new one, which notifies.
                self._cond.wait(remaining)
