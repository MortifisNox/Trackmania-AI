import socket
import threading
import time

import pytest

from tmai.game import protocol
from tmai.game.telemetry import TelemetryClient


def test_client_keeps_newest_packet_across_split_reads():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    sent = threading.Event()
    consumed = threading.Event()

    def serve():
        conn, _ = server.accept()
        sentinel = protocol.encode(seq=2, flags=protocol.FLAG_PLAYER | protocol.FLAG_PLAYING,
                                   api_race_time_ms=-(1 << 31), start_time_ms=protocol.INT32_MAX,
                                   net_now_ms=10_000, rpm=float("nan"))
        blob = b"".join((protocol.encode(seq=1), sentinel,
                         *(protocol.encode(seq=i, race_time_ms=i * 10) for i in range(3, 6))))
        conn.sendall(blob[:50])  # split mid-packet
        conn.sendall(blob[50:])
        sent.set()
        consumed.wait(5.0)
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    client = TelemetryClient(port=port).start()
    try:
        assert sent.wait(5.0)
        t = client.wait(5.0)
        deadline = 50
        while t.seq != 5 and deadline:
            t = client.wait(1.0, after_seq=t.seq)
            deadline -= 1
        assert t.seq == 5 and t.race_time_ms == 50
        assert client.connected and client.error is None
    finally:
        consumed.set()
        client.close()
        server.close()


def test_wait_refuses_old_packets():
    client = TelemetryClient(port=1)       # never connects; packet seeded by hand
    client._latest = protocol.decode(protocol.encode(seq=7))
    client._received_at = time.monotonic() - 30.0
    try:
        assert client.wait(0.01).seq == 7                   # no age limit: old behaviour
        with pytest.raises(TimeoutError, match="no fresh telemetry"):
            client.wait(0.05, max_age=0.1)
        client._received_at = time.monotonic()
        assert client.wait(0.05, max_age=0.1).seq == 7
    finally:
        client.close()


def test_observe_releases_the_pad_when_telemetry_is_stale():
    from tmai.game.interface import GameInterface

    class StaleClient:
        def wait(self, timeout, after_seq=None, max_age=None):
            assert max_age == 0.1
            raise TimeoutError("no fresh telemetry")

    class Pad:
        released = False

        def release(self):
            self.released = True

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad, game.capture, game.max_age = StaleClient(), Pad(), None, 0.1
    game._last_observed_seq = None
    with pytest.raises(TimeoutError):
        game.observe(0.01)
    assert game.pad.released


def test_observe_requires_strictly_newer_telemetry():
    from tmai.game.interface import GameInterface
    from test_game_logic import FakePad

    client = TelemetryClient(port=1)
    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad, game.capture, game.max_age = client, FakePad(), None, 0.1
    game._last_observed_seq = None
    for seq in (0xffffffff, 0xfffffffe, 0xffffffff, 0):
        client._latest = protocol.decode(protocol.encode(seq=seq))
        client._received_at = time.monotonic()
        if seq in (0xffffffff, 0) and seq != game._last_observed_seq:
            assert game.observe(0)[1].seq == seq
        else:
            with pytest.raises(TimeoutError):
                game.observe(0)


@pytest.mark.parametrize("failure", [b"", protocol.encode(magic=1) + protocol.encode(seq=8)])
def test_client_invalidates_latest_on_disconnect_or_bad_packet(monkeypatch, failure):
    client = TelemetryClient()

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            client._stop.set()

        def settimeout(self, timeout):
            pass

        def recv(self, size):
            if client.latest() is None:
                return protocol.encode(seq=7)
            assert client.latest().seq == 7
            return failure

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: Connection())
    client._run()
    assert client.latest() is None and not client.connected
    assert (client.error is not None) == bool(failure)
    with pytest.raises(TimeoutError):
        client.wait(0)


def test_client_receives_legacy_v2_packets_over_tcp():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    sent = threading.Event()
    consumed = threading.Event()

    def serve():
        conn, _ = server.accept()
        blob = b"".join(protocol.encode(version=2, seq=i, race_time_ms=i * 100) for i in range(1, 4))
        conn.sendall(blob[:70])
        conn.sendall(blob[70:])
        sent.set()
        consumed.wait(5.0)
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    client = TelemetryClient(port=port).start()
    try:
        assert sent.wait(5.0)
        t = client.wait(5.0)
        deadline = 50
        while t.seq != 3 and deadline:
            t = client.wait(1.0, after_seq=t.seq)
            deadline -= 1
        assert t.seq == 3 and t.race_time_ms == 300 and t.map_uid == ""
        assert client.error is None
    finally:
        consumed.set()
        client.close()
        server.close()
