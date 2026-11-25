import re
from pathlib import Path

import pytest

from tmai.game import protocol


def test_round_trip():
    pkt = protocol.encode(flags=protocol.FLAG_PLAYER | protocol.FLAG_PLAYING, seq=7, race_time_ms=-1500,
                          cp_count=2, pos=(1.0, 2.0, 3.0), speed=25.0, gear=3.0,
                          wheel_material=(6, 6, 2, 2), wheel_slip=(0.1, 0.1, 0.5, 0.5),
                          map_uid="MapUid_ABC123")
    assert len(pkt) == protocol.PACKET_SIZE
    t = protocol.decode(pkt)
    assert t.has_player and t.playing and not t.finished
    assert t.seq == 7 and t.race_time_ms == -1500 and t.cp_count == 2
    assert t.pos == (1.0, 2.0, 3.0) and t.gear == 3
    assert t.speed_kmh == pytest.approx(90.0)
    assert t.wheel_material == (6, 6, 2, 2)
    assert t.map_uid == "MapUid_ABC123"


def test_v2_packet_remains_backward_compatible():
    pkt_v2 = protocol.encode(version=2, flags=protocol.FLAG_PLAYER | protocol.FLAG_PLAYING, seq=11,
                             race_time_ms=2500, pos=(4.0, 5.0, 6.0), speed=15.0)
    assert len(pkt_v2) == protocol.PACKET_SIZE_V2 == 136
    assert protocol.packet_size_from_header(pkt_v2) == protocol.PACKET_SIZE_V2
    t = protocol.decode(pkt_v2)
    assert t.seq == 11 and t.race_time_ms == 2500 and t.pos == (4.0, 5.0, 6.0)
    assert t.map_uid == ""


def test_size_matches_plugin_layout():
    # v2: 4+2+2+4 header, 6 ints, 3+5 floats, 9+3 floats, 4 bytes, 4 floats = 136
    assert protocol.PACKET_SIZE_V2 == 12 + 24 + 32 + 48 + 4 + 16
    # v3 appends fixed 64-byte UTF-8 map UID
    assert protocol.PACKET_SIZE == protocol.PACKET_SIZE_V2 + protocol.MAP_UID_SIZE


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


@pytest.mark.parametrize(("field", "invalid", "expected"), [
    ("gear", float("nan"), 0), ("gear", 2.5, 0), ("rpm", float("inf"), 0.0), ("gas", 1.5, 0.0),
    ("pos", (0, 0, 1e30), (0.0, 0.0, 0.0)), ("cp_count", -1, 0), ("cp_count", 100_001, 0),
    ("last_cp_time_ms", protocol.INT32_MAX, 0), ("race_time_ms", 2_000_000_000, -1),
])
def test_sanitizes_invalid_numeric_fields(field, invalid, expected):
    packet = protocol.encode(seq=7, **{field: invalid})
    decoded = protocol.decode(packet)
    assert decoded.seq == 7 and getattr(decoded, field) == expected


@pytest.mark.parametrize("api_time", [-(1 << 31), protocol.INT32_MAX])
@pytest.mark.parametrize("start", [0, protocol.INT32_MAX])
def test_race_time_sentinels_are_unknown(api_time, start):
    flags = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    packet = protocol.encode(flags=flags, api_race_time_ms=api_time, start_time_ms=start, net_now_ms=10_000)
    t = protocol.decode(packet)
    assert t.race_time_ms == protocol.pick_race_time(api_time, start, 10_000, -1) == -1
    assert t.has_player and not t.playing
    assert t.clocks == (api_time, start, 10_000, -1)


def test_invalid_game_clock_falls_back_and_minus_one_countdown_remains_playing():
    flags = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    assert protocol.pick_race_time(0, 1000, 1200, protocol.INT32_MAX) == 200
    assert protocol.pick_race_time(0, 1000, 1200, 2_000_000_000) == 200
    assert protocol.pick_race_time(200, protocol.INT32_MAX, 1200, -1) == 200
    t = protocol.decode(protocol.encode(flags=flags, start_time_ms=1000, net_now_ms=999))
    assert t.race_time_ms == -1 and t.playing

@pytest.fixture
def telemetry_plugin_source():
    # Openplanet is unavailable in CI. These checks cover the source branches;
    # the API behavior and the plugin build still need a game-side check.
    plugin = Path(__file__).resolve().parents[1] / "openplanet" / "TMAI_Telemetry"
    return re.sub(r"//[^\n]*", "", (plugin / "main.as").read_text())


def test_plugin_normalizes_unset_race_clocks(telemetry_plugin_source):
    source = telemetry_plugin_source
    pick = source.partition("bool PickRaceTime(")[2].partition("void ReadWaypoints(")[0]
    assert re.search(r"if \(start > 0 && start < CLOCK_SENTINEL\)", pick)
    assert re.search(r"raceMs = -1;\s*return false;", pick)
    assert re.search(r"if \(!PickRaceTime\([^\n]+\)\)\s*flags &= ~FLAG_PLAYING;", source)


def test_plugin_preserves_already_counted_finish_when_ghost_result_is_missing(telemetry_plugin_source):
    source = telemetry_plugin_source
    fallback = source.partition("if (g_finishTime <= 0 && apiCount > 0)")[2]
    assert re.search(
        r"else if \(apiCount == g_cpCount && apiLast == g_lastCp && g_lastCp > 0\)"
        r".*?if \(apiLast == apiRaceMs .*?\)\s*g_finishTime = g_lastCp;",
        fallback, re.S,
    )
    assert "g_finishTime == g_lastCp" in fallback
    assert "g_cpCount + (finishAlreadyCounted ? 0 : 1)" in fallback
    assert "lastMs = finished ? Math::Max(0, g_finishTime) : g_lastCp;" in fallback


def test_plugin_releases_ghost_even_without_result_and_skips_multilap_start(telemetry_plugin_source):
    source = telemetry_plugin_source
    assert re.search(
        r"if \(ghost !is null\)\s*\{\s*try\s*\{\s*auto result = ghost.Result;"
        r"\s*if \(result !is null\)\s*g_finishTime = int\(result.Time\);\s*\}"
        r"\s*catch\s*\{\s*\}\s*rules.DataFileMgr.Ghost_Release\(ghost.Id\);",
        source,
    )
    assert re.search(
        r"if \(landmark.Waypoint.IsMultiLap\)\s*\{[^{}]*if \(raceMs > 0\)\s*\{\s*g_cpCount\+\+;",
        source,
    )
    assert "array<uint> g_seenLinkedOrders;" in source


def test_plugin_sends_v3_map_uid_and_info_toml_bumped(telemetry_plugin_source):
    source = telemetry_plugin_source
    assert "const uint16 VERSION = 3;" in source
    assert "const uint MAP_UID_SIZE = 64;" in source
    assert "map.MapInfo.MapUid" in source
    assert "map.EdChallengeId" in source
    assert "WriteFixedString(buf, MapUid(map), MAP_UID_SIZE);" in source
    info = (Path(__file__).resolve().parents[1] / "openplanet" / "TMAI_Telemetry" / "info.toml").read_text()
    assert 'version = "0.1.4"' in info
