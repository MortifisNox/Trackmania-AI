"""Where the game keeps its user files (OpenplanetNext, Documents/Trackmania).

On Windows that is the user profile. On Linux the game runs under Proton or
Wine, so the files live in the Windows user folder inside its prefix, for
Steam usually ~/.steam/steam/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser.
Set TMAI_GAME_USER_DIR to that folder to skip the search."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path

STEAM_APP_ID = "2225070"   # Trackmania (2020) on Steam
ENV_VAR = "TMAI_GAME_USER_DIR"


def steam_roots(home: Path) -> list[Path]:
    return [home / ".steam" / "steam", home / ".local" / "share" / "Steam",
            home / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam",
            home / ".var" / "app" / "com.valvesoftware.Steam" / "data" / "Steam"]


def steam_libraries(home: Path) -> list[Path]:
    """Steam roots plus every library folder listed in libraryfolders.vdf."""
    libs: list[Path] = []
    for root in steam_roots(home):
        if not root.is_dir():
            continue
        libs.append(root)
        vdf = root / "steamapps" / "libraryfolders.vdf"
        try:
            text = vdf.read_text(errors="replace")
        except OSError:
            continue
        libs += [Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', text)]
    out, seen = [], set()
    for lib in libs:
        key = os.path.realpath(lib)
        if key not in seen:
            seen.add(key)
            out.append(lib)
    return out


def _users_in(prefix: Path) -> list[Path]:
    users = prefix / "drive_c" / "users"
    if not users.is_dir():
        return []
    return sorted(u for u in users.iterdir() if u.is_dir() and u.name.lower() != "public")


def linux_candidates(home: Path) -> list[Path]:
    """Windows user folders of every prefix Trackmania may run in, best guesses first."""
    found: list[Path] = []
    for lib in steam_libraries(home):
        found += _users_in(lib / "steamapps" / "compatdata" / STEAM_APP_ID / "pfx")
    prefixes = [home / ".wine"]
    for pattern in ("Games/*", "Games/Heroic/Prefixes/*", "Games/Heroic/Prefixes/default/*",
                    ".local/share/lutris/prefixes/*"):
        prefixes += sorted(home.glob(pattern))
    for prefix in prefixes:
        found += _users_in(prefix)
    return found


def pick(candidates: list[Path]) -> Path | None:
    """The first folder with Openplanet in it, else the first with the game's
    Documents folder, else None."""
    for c in candidates:
        if (c / "OpenplanetNext").is_dir():
            return c
    for c in candidates:
        if any((c / "Documents" / n).is_dir() for n in ("Trackmania", "Trackmania2020")):
            return c
    return None


def game_user_dir(home: Path | None = None, system: str | None = None) -> Path:
    """The Windows user folder the game sees."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    system = system or platform.system()
    home = home or Path.home()
    if system == "Windows":
        return Path(os.environ.get("USERPROFILE") or home)
    found = pick(linux_candidates(home))
    if found is not None:
        return found
    # Nothing set up yet: point at the usual Steam location so messages say where to look.
    return home / ".steam" / "steam" / "steamapps" / "compatdata" / STEAM_APP_ID / "pfx" / "drive_c" / "users" / "steamuser"


def openplanet_dir() -> Path:
    return game_user_dir() / "OpenplanetNext"


def documents_dir() -> Path:
    """The Documents folder the game writes to. On Windows the real one (often
    redirected to OneDrive); under Proton or Wine the prefix's."""
    if platform.system() == "Windows" and not os.environ.get(ENV_VAR):
        try:
            from win32com.shell import shell, shellcon

            return Path(shell.SHGetFolderPath(0, shellcon.CSIDL_PERSONAL, None, 0))
        except ImportError:
            pass
    return game_user_dir() / "Documents"
