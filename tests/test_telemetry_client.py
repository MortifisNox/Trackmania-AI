import socket
import threading

from tmai.game import protocol
from tmai.game.telemetry import TelemetryClient


def test_client_keeps_newest_packet_across_split_reads():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    sent = threading.Event()

    def serve():
        conn, _ = server.accept()
        blob = b"".join(protocol.encode(seq=i, race_time_ms=i * 10) for i in range(1, 6))
        conn.sendall(blob[:50])  # split mid-packet
        conn.sendall(blob[50:])
        sent.set()
        threading.Event().wait(1.0)
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
    finally:
        client.close()
        server.close()
