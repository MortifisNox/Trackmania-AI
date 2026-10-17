"""Background TCP client that keeps the newest telemetry packet from the plugin."""

from __future__ import annotations

import socket
import threading
import time

from tmai.game.protocol import PACKET_SIZE, ProtocolError, Telemetry, decode


class TelemetryClient:
    """Connects to the plugin, reconnecting if the game restarts.

    The plugin sends one packet per rendered frame; only the newest is kept,
    so a slow consumer never works on stale data.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host, self.port = host, port
        self._latest: Telemetry | None = None
        self._received_at = 0.0
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
            self.connected = False
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
            whole = len(buf) // PACKET_SIZE
            if whole == 0:
                continue
            newest = buf[(whole - 1) * PACKET_SIZE: whole * PACKET_SIZE]
            buf = buf[whole * PACKET_SIZE:]
            packet = decode(newest)
            with self._cond:
                self._latest, self._received_at = packet, time.monotonic()
                self._cond.notify_all()

    def latest(self) -> Telemetry | None:
        with self._cond:
            return self._latest

    def age(self) -> float:
        """Seconds since the newest packet arrived (inf if none yet)."""
        with self._cond:
            return time.monotonic() - self._received_at if self._latest else float("inf")

    def wait(self, timeout: float = 5.0, after_seq: int | None = None) -> Telemetry:
        """Block until a packet (newer than `after_seq`, if given) arrives."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                pkt = self._latest
                if pkt is not None and (after_seq is None or pkt.seq != after_seq):
                    return pkt
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"no telemetry from the plugin on {self.host}:{self.port} "
                        f"({self.error or 'connected but silent'})"
                    )
                self._cond.wait(remaining)
