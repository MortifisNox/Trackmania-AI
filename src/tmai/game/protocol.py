"""Wire format of the TMAI Telemetry Openplanet plugin (openplanet/TMAI_Telemetry/main.as)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = 0x49414D54  # "TMAI" as little-endian uint32
VERSION = 2

FLAG_PLAYER = 1
FLAG_PLAYING = 2
FLAG_FINISHED = 4
FLAG_GROUND = 8

# Field order must match BuildPacket() in the plugin.
_FIELDS = [
    ("magic", "I"), ("version", "H"), ("flags", "H"), ("seq", "I"),
    ("api_race_time_ms", "i"), ("start_time_ms", "i"), ("net_now_ms", "i"), ("rules_now_ms", "i"),
    ("cp_count", "i"), ("last_cp_time_ms", "i"),
    ("pos", "3f"), ("speed", "f"), ("gear", "f"), ("rpm", "f"),
    ("steer", "f"), ("gas", "f"),
    ("vel", "3f"), ("dir", "3f"), ("up", "3f"),
    ("front_speed", "f"), ("side_speed", "f"), ("brake", "f"),
    ("wheel_material", "4B"), ("wheel_slip", "4f"),
]
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
    def on_ground(self) -> bool:
        return bool(self.flags & FLAG_GROUND)

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6


def _split(values: tuple) -> dict:
    out, i = {}, 0
    for name, fmt in _FIELDS:
        n = int(fmt[:-1]) if len(fmt) > 1 else 1
        out[name] = tuple(values[i:i + n]) if n > 1 else values[i]
        i += n
    return out


def pick_race_time(api_race_ms: int, start_ms: int, net_now_ms: int, rules_now_ms: int) -> int:
    """CurrentRaceTime is 0 in some modes (the editor's test drive), so derive
    the race time from the game clock and the run's start time when possible."""
    if start_ms > 0:
        if rules_now_ms >= 0:
            return rules_now_ms - start_ms
        if net_now_ms >= 0:
            return net_now_ms - start_ms
    return api_race_ms


def decode(packet: bytes) -> Telemetry:
    if len(packet) != PACKET_SIZE:
        raise ProtocolError(f"expected {PACKET_SIZE} bytes, got {len(packet)}")
    f = _split(STRUCT.unpack(packet))
    if f["magic"] != MAGIC:
        raise ProtocolError(f"bad magic {f['magic']:#x}; is another plugin using the port?")
    if f["version"] != VERSION:
        raise ProtocolError(f"plugin protocol v{f['version']}, expected v{VERSION}; reinstall the plugin")
    return Telemetry(
        flags=f["flags"], seq=f["seq"], cp_count=f["cp_count"],
        race_time_ms=pick_race_time(f["api_race_time_ms"], f["start_time_ms"], f["net_now_ms"], f["rules_now_ms"]),
        clocks=(f["api_race_time_ms"], f["start_time_ms"], f["net_now_ms"], f["rules_now_ms"]),
        last_cp_time_ms=f["last_cp_time_ms"], pos=f["pos"], speed=f["speed"], gear=int(f["gear"]),
        rpm=f["rpm"], steer=f["steer"], gas=f["gas"], brake=f["brake"], vel=f["vel"], dir=f["dir"],
        up=f["up"], front_speed=f["front_speed"], side_speed=f["side_speed"],
        wheel_material=f["wheel_material"], wheel_slip=f["wheel_slip"],
    )


def encode(race_time_ms: int | None = None, **fields) -> bytes:
    """Build a packet; used by tests. `race_time_ms` sets the API race time
    with no start time, so decode() reports it unchanged."""
    defaults = {name: ((0,) * int(fmt[:-1]) if len(fmt) > 1 else 0) for name, fmt in _FIELDS}
    defaults.update(magic=MAGIC, version=VERSION, net_now_ms=-1, rules_now_ms=-1)
    if race_time_ms is not None:
        defaults["api_race_time_ms"] = race_time_ms
    defaults.update(fields)
    values = []
    for name, fmt in _FIELDS:
        v = defaults[name]
        values.extend(v if isinstance(v, (tuple, list)) else [v])
    return STRUCT.pack(*values)
