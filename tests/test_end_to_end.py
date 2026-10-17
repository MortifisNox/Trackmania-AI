"""Every feature together, through the real `tmai train` command with a fake
game: the dashboard's web API drives every control, the stop button ends the
run, the shutdown writes the checkpoint, snapshot and report, and a second
`tmai train` resumes from them."""

import threading
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from test_dashboard import wait_for  # noqa: E402
from test_rl import make_track  # noqa: E402
from test_rl_torch import FakeGame  # noqa: E402
from tmai import cli  # noqa: E402
from tmai.config import load_config  # noqa: E402
from tmai.dashboard.server import create_app  # noqa: E402
from tmai.dashboard.state import RunState  # noqa: E402
from tmai.run.stop import read_status  # noqa: E402


class Game(FakeGame):
    def __init__(self):
        super().__init__()
        self.steps = 0

    def observe(self):
        self.steps += 1
        return super().observe()

    def close(self):
        pass


def test_train_dashboard_stop_report_and_resume(tmp_path, monkeypatch):
    import tmai.game.interface

    games = []
    monkeypatch.setattr(tmai.game.interface, "GameInterface", lambda cfg, capture=True: games.append(Game()) or games[-1])
    monkeypatch.setattr("tmai.run.shutdown.openplanet_dir", lambda: tmp_path / "OpenplanetNext")
    state = RunState(None, 0)
    sets = [f"run.output_dir={tmp_path.as_posix()}", "run.track=T", "replay.capacity=4000", "train.batch_size=16",
            "model.hidden_dim=32", "model.encoder_channels=[8,8,8,8]", "env.step_hz=200", "env.episode_timeout_s=5",
            "train.warmup_steps=40", "train.updates_per_step=1", "replay.snapshot_every_min=0.01",
            f"dashboard.link_port={state.link_port}", "dashboard.tensorboard=false"]
    argv = [a for s in sets for a in ("--set", s)]
    make_track(tmp_path, load_config(None, sets))
    run_dir = tmp_path / "default"
    client = TestClient(create_app(state))
    snap = lambda: client.get("/api/snapshot").json()  # noqa: E731
    errors = []

    def command(kind, name=None, value=None):
        r = client.post("/api/command", json={"kind": kind, "name": name, "value": value})
        assert r.status_code == 200, r.text
        cid = r.json()["id"]
        assert wait_for(lambda: any(e["kind"] == "command" and e["data"]["id"] == cid and e["data"]["status"] == "done"
                                    for e in state.events_since(0) or []), timeout=10), kind

    def session(steer):
        def driver():
            try:
                assert wait_for(lambda: snap()["trainer"]["state"] == "training", timeout=60)
                steer()
            except BaseException as exc:
                errors.append(exc)
            finally:
                client.post("/api/command", json={"kind": "stop"})

        t = threading.Thread(target=driver, daemon=True)
        t.start()
        assert cli.main(argv + ["train"]) == 0
        t.join(30)
        if errors:
            raise errors[0]

    def first():
        for name, value in (("learning_rate", 1e-4), ("updates_per_step", 2), ("target_entropy", -2.0),
                            ("demo_ratio", 0.25)):
            command("set_param", name, value)
        assert wait_for(lambda: {p["name"]: p["value"] for p in snap()["params"]}["demo_ratio"] == 0.25)
        assert client.post("/api/command", json={"kind": "set_param", "name": "learning_rate",
                                                 "value": 5.0}).status_code == 400
        command("pause")
        assert wait_for(lambda: snap()["trainer"]["state"] == "paused")
        time.sleep(0.3)
        steps = games[-1].steps
        time.sleep(0.7)
        assert games[-1].steps == steps
        command("resume")
        assert wait_for(lambda: games[-1].steps > steps + 10)
        n = len(snap()["episodes"])
        command("reset_track")
        assert wait_for(lambda: any(e["end"] == "reset" for e in snap()["episodes"][n:]))
        command("save_checkpoint")
        assert wait_for(lambda: (run_dir / "checkpoints" / "latest.pt").exists())
        assert wait_for(lambda: snap()["best"] is not None and snap()["scalars"].get("critic_loss"))
        assert {"speed_kmh", "race_time_ms", "progress", "splits_ms"} <= set(snap()["telemetry"])

    session(first)
    status = read_status(run_dir)
    assert status["state"] == "stopped" and status["reason"] == "dashboard button"
    assert all(s["ok"] for s in status["steps"]), status["steps"]
    assert (run_dir / "checkpoints" / "final.pt").exists() and (run_dir / "best_laps.jsonl").exists()
    assert client.get("/report").status_code == 200
    first_ckpt = torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=False)

    def second():
        assert snap()["trainer"]["episodes"] >= first_ckpt["episodes"]
        assert wait_for(lambda: snap()["trainer"]["env_steps"] > first_ckpt["env_steps"] + 20)

    session(second)
    again = torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=False)
    assert again["episodes"] > first_ckpt["episodes"] and again["best_ms"] <= first_ckpt["best_ms"]
    assert again["agent"]["updates"] > first_ckpt["agent"]["updates"]
    assert cli.main(argv + ["report"]) == 0
    state.close()
