"""Safe-stop requests and run status, shared by the trainer, the dashboard and
`tmai stop`.

Any process can ask a run to stop by calling `request_stop(run_dir)`: it writes
runs/<name>/control/stop.json. The trainer's StopController notices it (and
Ctrl+C, Ctrl+Break, closing the console window or SIGTERM) and sets the
trainer's stop event; the shutdown sequence then saves everything.
`read_status(run_dir)` tells a dashboard what the run is doing."""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path

from tmai.run import atomic

# Run states written to control/status.json.
RUNNING, STOPPING, STOPPED = "running", "stopping", "stopped"
POLL_S = 0.5
FORCE_AFTER = 3   # this many Ctrl+C presses abandon the clean stop


def control_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "control"


def request_stop(run_dir: str | Path, reason: str = "requested", source: str = "cli") -> Path:
    """Ask the run in `run_dir` to stop safely. Safe to call from any process,
    any number of times."""
    path = control_dir(run_dir) / "stop.json"
    atomic.write_json(path, {"reason": reason, "source": source, "time": time.time(), "pid": os.getpid()})
    return path


def read_status(run_dir: str | Path) -> dict:
    """{"state": running|stopping|stopped, "phase": ..., "steps": [...], ...}
    or {"state": "unknown"} when the run has not written a status yet."""
    return atomic.read_json(control_dir(run_dir) / "status.json", {"state": "unknown"})


class StopController:
    """Turns signals and stop.json into one stop event, and keeps
    control/status.json current."""

    def __init__(self, run_dir: str | Path, event: threading.Event | None = None, log=print):
        self.run_dir = Path(run_dir)
        self.event = event or threading.Event()
        self.log = log
        self.reason: str | None = None
        self.source: str | None = None
        self._presses = 0
        self._last_poll = 0.0
        self._previous: dict = {}
        self.status: dict = {}

    # -- requests ----------------------------------------------------------------
    def request(self, reason: str, source: str) -> None:
        if self.reason is None:
            self.reason, self.source = reason, source
            self.log(f"stop requested ({source}: {reason}); finishing the current step and saving")
            self.set_status(STOPPING, phase="stopping the driver")
        self.event.set()

    @property
    def stopping(self) -> bool:
        return self.event.is_set()

    def poll(self, force: bool = False) -> bool:
        """Check stop.json (at most every POLL_S unless forced). Cheap enough
        to call every control step. Returns True once a stop is requested."""
        now = time.monotonic()
        if force or now - self._last_poll >= POLL_S:
            self._last_poll = now
            req = atomic.read_json(control_dir(self.run_dir) / "stop.json")
            if req:
                self.request(req.get("reason", "requested"), req.get("source", "file"))
        return self.event.is_set()

    # -- signals -----------------------------------------------------------------
    def install(self) -> "StopController":
        """Install handlers; only works from the main thread. Clears any stop
        request left over from an earlier run."""
        (control_dir(self.run_dir) / "stop.json").unlink(missing_ok=True)
        sigs = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):   # Windows: Ctrl+Break
            sigs.append(signal.SIGBREAK)
        for s in sigs:
            self._previous[s] = signal.signal(s, self._on_signal)
        self.set_status(RUNNING, phase="training")
        return self

    def uninstall(self) -> None:
        for s, handler in self._previous.items():
            signal.signal(s, handler)
        self._previous.clear()

    def _on_signal(self, signum, frame) -> None:
        name = signal.Signals(signum).name
        if signum == signal.SIGINT:
            self._presses += 1
            if self._presses >= FORCE_AFTER:
                # Files are written atomically, so even this leaves the last
                # good checkpoint and snapshot intact; only unsaved progress is lost.
                self.uninstall()
                raise KeyboardInterrupt
            if self._presses > 1:
                left = FORCE_AFTER - self._presses
                self.log(f"still saving; press Ctrl+C {left} more time(s) to quit now (saved files stay intact)")
                return
        self.request(f"{name} received", "signal")

    # -- status ------------------------------------------------------------------
    def set_status(self, state: str, **fields) -> None:
        self.status.update(fields)
        self.status.update(state=state, updated=time.time(), pid=os.getpid(),
                           reason=self.reason, source=self.source)
        try:
            atomic.write_json(control_dir(self.run_dir) / "status.json", self.status)
        except OSError as exc:   # status is informational; never let it break a stop
            self.log(f"could not write run status: {exc}")

    def finish(self) -> None:
        (control_dir(self.run_dir) / "stop.json").unlink(missing_ok=True)
        self.uninstall()
