import pytest

from tmai.game import protocol


def test_round_trip():
    pkt = protocol.encode(flags=protocol.FLAG_PLAYER | protocol.FLAG_PLAYING, seq=7, race_time_ms=-1500,
                          cp_count=2, pos=(1.0, 2.0, 3.0), speed=25.0, gear=3.0,
                          wheel_material=(6, 6, 2, 2), wheel_slip=(0.1, 0.1, 0.5, 0.5))
    assert len(pkt) == protocol.PACKET_SIZE
    t = protocol.decode(pkt)
    assert t.has_player and t.playing and not t.finished
    assert t.seq == 7 and t.race_time_ms == -1500 and t.cp_count == 2
    assert t.pos == (1.0, 2.0, 3.0) and t.gear == 3
    assert t.speed_kmh == pytest.approx(90.0)
    assert t.wheel_material == (6, 6, 2, 2)


def test_size_matches_plugin_layout():
    # 4+2+2+4 header, 6 ints, 3+5 floats, 9+3 floats, 4 bytes, 4 floats
    assert protocol.PACKET_SIZE == 12 + 24 + 32 + 48 + 4 + 16


def test_race_time_prefers_game_clock_over_api_value():
    assert protocol.pick_race_time(0, 10_000, 15_000, -1) == 5_000
    assert protocol.pick_race_time(0, 10_000, 15_000, 12_000) == 2_000
    assert protocol.pick_race_time(0, 10_000, 9_000, -1) == -1_000  # countdown
    assert protocol.pick_race_time(4_321, 0, -1, -1) == 4_321
    t = protocol.decode(protocol.encode(api_race_time_ms=0, start_time_ms=1000, net_now_ms=3500))
    assert t.race_time_ms == 2500 and t.clocks == (0, 1000, 3500, -1)


def test_rejects_bad_packets():
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(b"\x00" * 3)
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(protocol.encode(magic=1))
    with pytest.raises(protocol.ProtocolError, match="reinstall"):
        protocol.decode(protocol.encode(version=99))
