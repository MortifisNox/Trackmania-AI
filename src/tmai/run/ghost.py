"""The best lap as a Trackmania replay.

The TMAI Ghost Openplanet plugin (openplanet/TMAI_Ghost) saves every finished
run that beats the fastest one already saved for that map, as
Documents/Trackmania/Replays/TMAI/<map name>/<time>ms-<date>.Replay.Gbx. This
module finds the fastest of those for a run's track and copies it next to the
run report, so it is kept even if the game folder is cleaned up.

Watch it in game: Create > Replay editor > open TMAI/<map>/<file>, or race it
as a ghost with Play > Local > Against replay."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PLUGIN_FOLDER = "TMAI"
NAME = re.compile(r"(?P<ms>\d+)ms.*\.replay\.gbx$", re.IGNORECASE)
HOW_TO_WATCH = ("In Trackmania: Create > Replay editor, open TMAI > {folder} > {file}; "
                "or Play > Local > Against replay to race it as a ghost.")


def replays_root(documents: Path | None = None) -> Path | None:
    from tmai.game.maps import documents_dir

    documents = documents or documents_dir()
    for name in ("Trackmania", "Trackmania2020"):
        if (documents / name).is_dir():
            return documents / name / "Replays" / PLUGIN_FOLDER
    return None


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def map_folder(root: Path, track: str) -> Path | None:
    """The plugin names folders after the map's display name; the run names its
    track freely. Match ignoring case, spaces and punctuation."""
    if not root or not root.is_dir():
        return None
    want = _key(track)
    for folder in root.iterdir():
        if folder.is_dir() and _key(folder.name) == want:
            return folder
    return None


def replays_in(folder: Path) -> list[tuple[int, Path]]:
    found = []
    for p in folder.iterdir():
        m = NAME.search(p.name)
        if m and p.is_file():
            found.append((int(m.group("ms")), p))
    return sorted(found)


def preserve_best_replay(run_dir: str | Path, track: str, root: Path | None = None,
                         best_ms: int | None = None) -> dict:
    """Copy the fastest saved replay of `track` into runs/<name>/report/.
    Returns what was found, for the report; never raises for a missing file."""
    try:
        root = root or replays_root()
    except Exception as exc:   # no Documents folder (e.g. not on the game PC)
        return {"found": False, "detail": f"Trackmania folder not found: {exc}"}
    folder = map_folder(root, track) if root else None
    if folder is None:
        where = root if root else "Documents/Trackmania/Replays"
        return {"found": False, "detail": f"no replays in {where}/<{track}>; is the TMAI Ghost plugin loaded?"}
    replays = replays_in(folder)
    if not replays:
        return {"found": False, "detail": f"{folder} has no saved replays yet"}
    ms, src = replays[0]
    dst = Path(run_dir) / "report" / f"best-{ms}ms.Replay.Gbx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    info = {"found": True, "lap_ms": ms, "source": str(src), "copy": str(dst),
            "how_to_watch": HOW_TO_WATCH.format(folder=folder.name, file=src.name)}
    if best_ms is not None and abs(ms - best_ms) > 50:
        # The plugin saves the best run it saw, which can predate this run.
        info["note"] = (f"fastest saved replay is {ms / 1000:.3f}s; this run's best lap was "
                        f"{best_ms / 1000:.3f}s")
    return info
