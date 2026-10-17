import json
import time
from types import SimpleNamespace

import pytest

from tmai.config import ConfigError, load_config
from tmai.dashboard.fake import FakeTrainer, _Opt
from tmai.dashboard.link import DashboardHook, ParamSpec, param_specs, read_jsonl
from tmai.dashboard.state import RunState
from tmai.run.hooks import HookList, TrainerHooks
from tmai.run.stop import read_status


def wait_for(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def live(tmp_path):
    """A dashboard RunState and a fast fake trainer talking over a real UDP link."""
    run_dir = tmp_path / "run"
    state = RunState(run_dir, 0)   # port 0: any free port
    cfg = load_config(overrides=[f"dashboard.link_port={state.link_port}", "train.warmup_steps=0"])
    fake = FakeTrainer(cfg, run_dir, step_hz=500)
    fake.start()
    assert wait_for(lambda: state.snapshot()["connected"] and state.snapshot()["params"])
    yield state, fake, run_dir
    fake.close()
    state.close()


def command_status(state, cid):
    for e in reversed(state.events_since(0) or []):
        if e["kind"] == "command" and e["data"]["id"] == cid:
            return e["data"]["status"]
    return None


def test_param_spec_checks_range_and_whole_numbers():
    spec = ParamSpec("updates_per_step", 4, 0, 16, integer=True)
    assert spec.coerce("8") == 8
    for bad in (17, 2.5, float("nan")):
        with pytest.raises(ValueError):
            spec.coerce(bad)


def test_params_are_read_from_and_applied_to_the_trainer():
    t = SimpleNamespace(agent=SimpleNamespace(critic_opt=_Opt(3e-4), actor_opt=_Opt(3e-4), alpha_opt=_Opt(3e-4),
                                              target_entropy=-3.0),
                        utd=4, batch_size=256, demo_n=128, demo=None)
    names = {s.name: s.value for s in param_specs(t)}
    assert names == {"learning_rate": 3e-4, "updates_per_step": 4, "target_entropy": -3.0}   # no demos
    from tmai.dashboard.link import PARAMS
    PARAMS["learning_rate"][2](t, 1e-3)
    assert all(o.param_groups[0]["lr"] == 1e-3 for o in (t.agent.critic_opt, t.agent.actor_opt, t.agent.alpha_opt))


def test_hook_list_resets_when_any_hook_asks():
    class Reset(TrainerHooks):
        def should_reset_episode(self):
            return True

    assert not HookList(TrainerHooks()).should_reset_episode()
    assert HookList(TrainerHooks(), Reset()).should_reset_episode()


def test_live_telemetry_metrics_and_episodes_reach_the_dashboard(live):
    state, fake, run_dir = live
    assert wait_for(lambda: state.snapshot()["telemetry"].get("speed_kmh") is not None)
    assert wait_for(lambda: state.snapshot()["scalars"].get("critic_loss"))
    state.submit("reset_track")
    assert wait_for(lambda: state.snapshot()["episodes"])
    row = state.snapshot()["episodes"][-1]
    assert {"episode", "reward", "end", "splits_ms"} <= set(row)
    # the hook keeps the history on disk too
    assert row["episode"] in [r["episode"] for r in read_jsonl(run_dir / "episodes.jsonl")]
    assert read_jsonl(run_dir / "metrics.jsonl")


def test_set_param_pause_resume_and_reset(live):
    state, fake, _ = live
    cmd = state.submit("set_param", "learning_rate", 1e-3)
    assert wait_for(lambda: fake.agent.actor_opt.param_groups[0]["lr"] == 1e-3)
    assert wait_for(lambda: command_status(state, cmd["id"]) == "done")
    assert wait_for(lambda: next(p for p in state.snapshot()["params"] if p["name"] == "learning_rate")["value"] == 1e-3)

    with pytest.raises(ValueError):
        state.submit("set_param", "updates_per_step", 2.5)
    with pytest.raises(ValueError):
        state.submit("set_param", "gamma", 0.9)

    state.submit("pause")
    assert wait_for(lambda: state.snapshot()["trainer"].get("paused"))
    eps = fake.episodes
    time.sleep(0.3)
    assert fake.episodes <= eps + 1   # the attempt in progress ends, then the driver waits
    state.submit("resume")
    assert wait_for(lambda: not state.snapshot()["trainer"].get("paused"))

    before = fake.episodes
    state.submit("reset_track")
    assert wait_for(lambda: fake.episodes > before)


def test_stop_button_stops_the_run_through_stop_json(live):
    state, fake, run_dir = live
    state.submit("stop")
    assert wait_for(lambda: read_status(run_dir).get("state") == "stopped", timeout=10)
    assert wait_for(lambda: state.snapshot()["run_status"].get("state") == "stopped")
    assert read_status(run_dir)["source"] == "dashboard"


def test_commands_need_a_running_trainer(tmp_path):
    state = RunState(tmp_path, 0)
    try:
        with pytest.raises(ValueError):
            state.submit("pause")
        with pytest.raises(ValueError):
            state.submit("launch_rockets")
    finally:
        state.close()


def test_history_is_reloaded_and_a_fresh_start_replaces_it(tmp_path):
    rows = [{"episode": 1, "reward": 1.0, "lap_ms": None}, {"episode": 2, "reward": 2.0, "lap_ms": 50_000},
            {"episode": 1, "reward": 3.0, "lap_ms": 60_000}]   # --fresh restarted the numbering
    (tmp_path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"episode": 2, "rew')
    (tmp_path / "metrics.jsonl").write_text('{"step": 10, "q": 1.0}\n{"step": 20, "q": 2.0}\n')
    state = RunState(tmp_path, 0)
    try:
        snap = state.snapshot()
        assert [e["reward"] for e in snap["episodes"]] == [3.0]
        assert snap["best"]["lap_ms"] == 60_000
        assert snap["scalars"]["q"] == [(10, 1.0), (20, 2.0)]
    finally:
        state.close()


def test_hook_finishes_a_line_cut_short_by_a_crash(tmp_path):
    (tmp_path / "episodes.jsonl").write_text('{"episode": 1}\n{"episo')
    hook = DashboardHook(tmp_path, 9)   # nothing listens on port 9; sends are dropped
    hook.on_episode({"episode": 2})
    hook.close()
    assert [r["episode"] for r in read_jsonl(tmp_path / "episodes.jsonl")] == [1, 2]


def test_ports_must_differ():
    with pytest.raises(ConfigError):
        load_config(overrides=["dashboard.link_port=8080"])


def test_web_api_and_websocket(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tmai.dashboard.server import create_app

    (tmp_path / "metrics.jsonl").write_text('{"step": 5, "critic_loss": 0.5}\n')
    state = RunState(tmp_path, 0)
    try:
        client = TestClient(create_app(state))
        assert "Trackmania AI training" in client.get("/").text
        assert client.get("/api/snapshot").json()["scalars"]["critic_loss"] == [[5, 0.5]]
        bad = client.post("/api/command", json={"kind": "pause"})
        assert bad.status_code == 400 and "not running" in bad.json()["detail"]
        assert client.get("/report").status_code == 404
        with client.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            assert first["type"] == "snapshot" and first["connected"] is False
    finally:
        state.close()


def test_dashboard_hook_drives_the_real_trainer(tmp_path):
    torch = pytest.importorskip("torch")
    from test_rl import make_track, small_cfg
    from test_rl_torch import FakeGame

    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path, warmup_steps=10, updates_per_step=1)
    make_track(tmp_path, cfg)
    state = RunState(None, 0)
    try:
        hook = DashboardHook(tmp_path / "default", state.link_port)
        trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
        import threading
        runner = threading.Thread(target=trainer.train, kwargs={"minutes": 0.15, "hooks": HookList(hook)})
        runner.start()
        assert wait_for(lambda: state.snapshot()["connected"] and state.snapshot()["params"], timeout=20)
        state.submit("set_param", "learning_rate", 1e-4)
        state.submit("set_param", "updates_per_step", 2)
        assert wait_for(lambda: trainer.utd == 2 and trainer.agent.actor_opt.param_groups[0]["lr"] == 1e-4,
                        timeout=20)
        state.submit("reset_track")
        assert wait_for(lambda: any(e.get("end") == "reset" for e in state.snapshot()["episodes"]), timeout=20)
        assert wait_for(lambda: state.snapshot()["scalars"].get("critic_loss"), timeout=20)
        runner.join(timeout=60)
    finally:
        state.close()


def test_episodes_before_learning_do_not_break_the_page(tmp_path):
    """Losses are NaN until the first learner update; NaN is not valid JSON."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tmai.dashboard.server import create_app

    (tmp_path / "episodes.jsonl").write_text('{"episode": 1, "reward": 1.0, "critic_loss": NaN}\n')
    state = RunState(tmp_path, 0)
    hook = DashboardHook(tmp_path, state.link_port)
    try:
        hook.trainer = SimpleNamespace(metrics={})
        hook.on_episode({"episode": 2, "reward": 2.0, "critic_loss": float("nan"), "q": float("inf")})
        assert wait_for(lambda: len(state.snapshot()["episodes"]) == 2)
        client = TestClient(create_app(state))
        snap = client.get("/api/snapshot").json()
        assert [e["critic_loss"] for e in snap["episodes"]] == [None, None] and snap["episodes"][1]["q"] is None
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())   # what the browser's JSON.parse sees
        assert "NaN" not in (tmp_path / "episodes.jsonl").read_text().splitlines()[1]
    finally:
        hook.close()
        state.close()
