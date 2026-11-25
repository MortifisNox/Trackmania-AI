"""Wire format of the TMAI Telemetry Openplanet plugin (openplanet/TMAI_Telemetry/main.as)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = 0x49414D54  # "TMAI" as little-endian uint32
VERSION = 3
SUPPORTED_VERSIONS = (2, 3)
MAP_UID_SIZE = 64

FLAG_PLAYER = 1
FLAG_PLAYING = 2
FLAG_FINISHED = 4
FLAG_GROUND = 8
FLAG_FINISH_TIME = 16  # optional v2 flag: last_cp_time_ms is the finish result

# Broad bounds for game data before it reaches feature and reward calculations.
_FLOAT_BOUNDS = {
    "pos": (-1e6, 1e6), "speed": (-1000, 1000), "gear": (-1, 10), "rpm": (0, 100_000),
    "steer": (-1.01, 1.01), "gas": (-0.01, 1.01), "brake": (-0.01, 1.01),
    "vel": (-1000, 1000), "dir": (-1.01, 1.01), "up": (-1.01, 1.01),
    "front_speed": (-1000, 1000), "side_speed": (-1000, 1000), "wheel_slip": (-1000, 1000),
}
INT32_MAX = 0x7fffffff
UNKNOWN_RACE_MS = -1
MAX_WAYPOINTS = 100_000
MAX_RACE_MS = 7 * 24 * 60 * 60 * 1000


def newer_seq(seq: int, previous: int) -> bool:
    """Compare uint32 sequence numbers, including wraparound."""
    return 0 < (seq - previous) % (1 << 32) < (1 << 31)

# Field order must match BuildPacket() in the plugin.
_V2_FIELDS = [
    ("magic", "I"), ("version", "H"), ("flags", "H"), ("seq", "I"),
    ("api_race_time_ms", "i"), ("start_time_ms", "i"), ("net_now_ms", "i"), ("rules_now_ms", "i"),
    ("cp_count", "i"), ("last_cp_time_ms", "i"),
    ("pos", "3f"), ("speed", "f"), ("gear", "f"), ("rpm", "f"),
    ("steer", "f"), ("gas", "f"),
    ("vel", "3f"), ("dir", "3f"), ("up", "3f"),
    ("front_speed", "f"), ("side_speed", "f"), ("brake", "f"),
    ("wheel_material", "4B"), ("wheel_slip", "4f"),
]
_FIELDS = [*_V2_FIELDS, ("map_uid", f"{MAP_UID_SIZE}s")]
HEADER_STRUCT = struct.Struct("<IH")
HEADER_SIZE = HEADER_STRUCT.size
STRUCT_V2 = struct.Struct("<" + "".join(fmt for _, fmt in _V2_FIELDS))
PACKET_SIZE_V2 = STRUCT_V2.size
STRUCT = struct.Struct("<" + "".join(fmt for _, fmt in _FIELDS))
PACKET_SIZE = STRUCT.size


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Telemetry:
    flags: int
    seq: int
    race_time_ms: int          # best available race clock; negative during the countdown
    cp_count: int              # waypoints crossed this run (finish included)
    last_cp_time_ms: int
    pos: tuple                 # world position (m)
    speed: float               # m/s
    gear: int
    rpm: float
    steer: float               # -1..1, as the game applied it
    gas: float                 # 0..1
    brake: float               # 0..1
    vel: tuple                 # world velocity (m/s)
    dir: tuple                 # car forward unit vector
    up: tuple                  # car up unit vector
    front_speed: float         # m/s along dir
    side_speed: float          # m/s sideways (drift)
    wheel_material: tuple      # EPlugSurfaceMaterialId per wheel FL, FR, RL, RR
    wheel_slip: tuple          # slip coefficient per wheel
    clocks: tuple = ()         # raw (api race time, start, network now, rules now) for diagnosis
    map_uid: str = ""          # v3+: loaded map UID (MapInfo.MapUid / EdChallengeId); empty for v2

    @property
    def has_player(self) -> bool:
        return bool(self.flags & FLAG_PLAYER)

    @property
    def playing(self) -> bool:
        return bool(self.flags & FLAG_PLAYING)

    @property
    def finished(self) -> bool:
        return bool(self.flags & FLAG_FINISHED)

    @property
    def exact_finish_ms(self) -> int | None:
        if self.finished and self.flags & FLAG_FINISH_TIME and self.last_cp_time_ms > 0:
            return self.last_cp_time_ms
        return None

    @property
    def on_ground(self) -> bool:
        return bool(self.flags & FLAG_GROUND)

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6


def _split(values: tuple, fields=_FIELDS) -> dict:
    out, i = {}, 0
    for name, fmt in fields:
        if fmt.endswith("s"):
            out[name] = values[i]
            i += 1
            continue
        n = int(fmt[:-1]) if len(fmt) > 1 else 1
        out[name] = tuple(values[i:i + n]) if n > 1 else values[i]
        i += n
    return out


def packet_size_from_header(buf: bytes | bytearray | memoryview) -> int | None:
    """Inspect the packet header to find its byte length for stream framing.
    Returns None when fewer than HEADER_SIZE bytes are buffered yet."""
    if len(buf) < HEADER_SIZE:
        return None
    magic, version = HEADER_STRUCT.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic:#x}; is another plugin using the port?")
    if version == 2:
        return PACKET_SIZE_V2
    if version == 3:
        return PACKET_SIZE
    raise ProtocolError(f"plugin protocol v{version}, expected v{VERSION}; reinstall the plugin")


def pick_race_time(api_race_ms: int, start_ms: int, net_now_ms: int, rules_now_ms: int) -> int:
    """CurrentRaceTime is 0 in some modes (the editor's test drive), so derive
    the race time from the game clock and the run's start time when possible."""
    race_ms = _select_race_time(api_race_ms, start_ms, net_now_ms, rules_now_ms)
    return UNKNOWN_RACE_MS if race_ms is None else race_ms


def _select_race_time(api_race_ms: int, start_ms: int, net_now_ms: int, rules_now_ms: int) -> int | None:
    if 0 < start_ms < INT32_MAX:
        for now_ms in (rules_now_ms, net_now_ms):
            if 0 <= now_ms < INT32_MAX:
                race_ms = now_ms - start_ms
                if -MAX_RACE_MS <= race_ms <= MAX_RACE_MS:
                    return race_ms
    return api_race_ms if -MAX_RACE_MS <= api_race_ms <= MAX_RACE_MS else None


def decode(packet: bytes) -> Telemetry:
    if len(packet) < HEADER_SIZE:
        raise ProtocolError(f"expected {PACKET_SIZE} bytes, got {len(packet)}")
    magic, version = HEADER_STRUCT.unpack_from(packet, 0)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic:#x}; is another plugin using the port?")
    if version == 2:
        if len(packet) != PACKET_SIZE_V2:
            raise ProtocolError(f"expected {PACKET_SIZE_V2} bytes for v2 packet, got {len(packet)}")
        f = _split(STRUCT_V2.unpack(packet), _V2_FIELDS)
        map_uid = ""
    elif version == 3:
        if len(packet) != PACKET_SIZE:
            raise ProtocolError(f"expected {PACKET_SIZE} bytes, got {len(packet)}")
        f = _split(STRUCT.unpack(packet), _FIELDS)
        raw_uid = f["map_uid"].split(b"\x00", 1)[0]
        map_uid = raw_uid.decode("utf-8", errors="replace").strip()
    else:
        raise ProtocolError(f"plugin protocol v{version}, expected v{VERSION}; reinstall the plugin")
    for name, (lo, hi) in _FLOAT_BOUNDS.items():
        value = f[name]
        values = value if isinstance(value, tuple) else (value,)
        if not all(lo <= x <= hi for x in values):
            f[name] = (0.0,) * len(value) if isinstance(value, tuple) else 0.0
    if not f["gear"].is_integer():
        f["gear"] = 0.0
    clocks = (f["api_race_time_ms"], f["start_time_ms"], f["net_now_ms"], f["rules_now_ms"])
    race_ms = _select_race_time(*clocks)
    if race_ms is None:
        race_ms = UNKNOWN_RACE_MS
        f["flags"] &= ~FLAG_PLAYING
    for name, hi in (("cp_count", MAX_WAYPOINTS), ("last_cp_time_ms", MAX_RACE_MS)):
        if not 0 <= f[name] <= hi:
            f[name] = 0
    return Telemetry(
        flags=f["flags"], seq=f["seq"], cp_count=f["cp_count"],
        race_time_ms=race_ms,
        clocks=clocks,
        last_cp_time_ms=f["last_cp_time_ms"], pos=f["pos"], speed=f["speed"], gear=int(f["gear"]),
        rpm=f["rpm"], steer=f["steer"], gas=f["gas"], brake=f["brake"], vel=f["vel"], dir=f["dir"],
        up=f["up"], front_speed=f["front_speed"], side_speed=f["side_speed"],
        wheel_material=f["wheel_material"], wheel_slip=f["wheel_slip"],
        map_uid=map_uid,
    )


def encode(race_time_ms: int | None = None, **fields) -> bytes:
    """Build a packet; used by tests. `race_time_ms` sets the API race time
    with no start time, so decode() reports it unchanged. Pass `version=2` for
    a legacy 136-byte packet without `map_uid`."""
    version = fields.get("version", VERSION)
    spec = _V2_FIELDS if version == 2 else _FIELDS
    packer = STRUCT_V2 if version == 2 else STRUCT
    defaults: dict = {}
    for name, fmt in spec:
        if fmt.endswith("s"):
            defaults[name] = b""
        elif len(fmt) > 1:
            defaults[name] = (0,) * int(fmt[:-1])
        else:
            defaults[name] = 0
    defaults.update(magic=MAGIC, version=version, net_now_ms=-1, rules_now_ms=-1)
    if race_time_ms is not None:
        defaults["api_race_time_ms"] = race_time_ms
    defaults.update(fields)
    if "map_uid" in defaults and isinstance(defaults["map_uid"], str):
        defaults["map_uid"] = defaults["map_uid"].encode("utf-8")[:MAP_UID_SIZE]
    values = []
    for name, fmt in spec:
        v = defaults[name]
        values.extend(v if isinstance(v, (tuple, list)) else [v])
    return packer.pack(*values)
