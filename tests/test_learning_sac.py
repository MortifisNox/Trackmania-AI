import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tmai.config import load_config  # noqa: E402
from tmai.rl.features import state_dim  # noqa: E402
from tmai.rl.sac import Policy, SACAgent  # noqa: E402
from tmai.rl.train import Trainer  # noqa: E402
from test_rl import make_track  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides=[
        f"run.output_dir={tmp_path.as_posix()}", "run.track=T", "replay.capacity=2000",
        "train.batch_size=16", "model.hidden_dim=32", "env.observation=state",
        "env.lidar=false", "env.lookahead_points=20",
    ])


def batch_for(cfg):
    rng = np.random.default_rng(123)
    env = cfg["env"]
    size = env["frame_size"] if env["observation"] == "image" else 0
    frames = rng.integers(0, 256, (8, env["frame_stack"], size, size), dtype=np.uint8)
    state = rng.standard_normal((8, state_dim(cfg["env"])), dtype=np.float32)
    return {
        "frames": frames, "next_frames": frames, "state": state, "next_state": state[::-1].copy(),
        "action": rng.uniform(-1, 1, (8, 3)).astype(np.float32),
        "reward": np.zeros(8, np.float32), "discount": np.full(8, 0.98, np.float32),
    }


@pytest.mark.parametrize(("layer_norm", "lidar", "sdim"), [(False, False, 81), (True, True, 156)])
def test_actor_and_policy_use_same_layer_norm_and_three_actions(cfg, layer_norm, lidar, sdim):
    cfg["model"]["actor_layer_norm"] = layer_norm
    cfg["env"]["lidar"] = lidar
    agent, policy = SACAgent(cfg), Policy(cfg)
    assert state_dim(cfg["env"]) == sdim
    for actor in (agent.actor, policy.actor):
        assert sum(isinstance(layer, torch.nn.LayerNorm) for layer in actor.net) == 2 * layer_norm
    policy.load(agent.policy_state())
    batch = batch_for(cfg)
    result = policy.act(batch["frames"][0], batch["state"][0], deterministic=True)
    assert result.shape == (3,)


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_weight_decay_only_applies_to_adamw_networks(cfg, optimizer):
    cfg["env"]["observation"] = "image"
    cfg["train"].update(optimizer=optimizer, weight_decay=0.2, learning_rate=0.1)
    agent = SACAgent(cfg)
    expected_optimizer = torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam
    for opt, params in ((agent.actor_opt, list(agent.actor.parameters())),
                        (agent.critic_opt, [*agent.encoder.parameters(), *agent.critic.parameters()])):
        assert type(opt) is expected_optimizer
        assert len(opt.param_groups) == (2 if optimizer == "adamw" else 1)
        before = [p.detach().clone() for p in params]
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step()
        for param, original in zip(params, before):
            scale = 0.98 if optimizer == "adamw" and param.ndim >= 2 else 1.0
            torch.testing.assert_close(param, original * scale)
    assert type(agent.alpha_opt) is torch.optim.Adam
    alpha = agent.log_alpha.detach().clone()
    agent.log_alpha.grad = torch.zeros_like(agent.log_alpha)
    agent.alpha_opt.step()
    torch.testing.assert_close(agent.log_alpha, alpha)


@pytest.mark.parametrize(("legacy", "optimizer"), [(True, "adamw"), (False, "adamw"), (False, "adam")])
def test_checkpoint_restores_optimizer_moments_by_parameter(cfg, legacy, optimizer):
    cfg["env"]["observation"] = "image"
    cfg["train"].update(optimizer="adamw", weight_decay=0.2)
    source = SACAgent(cfg)
    param_lists = {
        "actor_opt": lambda agent: list(agent.actor.parameters()),
        "critic_opt": lambda agent: [*agent.encoder.parameters(), *agent.critic.parameters()],
    }
    for name, params_for in param_lists.items():
        params = params_for(source)
        if legacy:
            setattr(source, name, torch.optim.AdamW(params, weight_decay=0.2))
        opt = getattr(source, name)
        for i, group in enumerate(opt.param_groups, 1):
            group["lr"] = i * 1e-4
        for i, param in enumerate(params, 1):
            param.grad = torch.full_like(param, i)
        opt.step()

    cfg["train"].update(optimizer=optimizer, weight_decay=0.05)
    resumed = SACAgent(cfg)
    resumed.load_state_dict(source.state_dict())
    for name, params_for in param_lists.items():
        original, restored = getattr(source, name), getattr(resumed, name)
        expected = [(1e-4, 0.0)] if optimizer == "adam" else [
            (1e-4, 0.05), (1e-4 if legacy else 2e-4, 0.0)
        ]
        assert [(g["lr"], g["weight_decay"]) for g in restored.param_groups] == expected
        for before, after in zip(params_for(source), params_for(resumed)):
            torch.testing.assert_close(original.state[before], restored.state[after])


@pytest.mark.parametrize(("loss", "expected"), [("huber", 4.0), ("mse", 9.0)])
def test_scalar_critic_loss(cfg, loss, expected):
    cfg["model"]["critic_head"] = "scalar"
    cfg["train"].update(critic_loss=loss, huber_delta=2.0)
    agent = SACAgent(cfg)
    assert agent._critic_loss(torch.zeros(2, 2), torch.tensor([-3.0, 3.0])).item() == expected


def test_hl_gauss_projection(cfg):
    from tmai.rl.networks import Critic

    for key in ("q_bins", "q_min", "q_max"):
        cfg["model"].pop(key)
    critic = SACAgent(cfg).critic
    support = torch.linspace(-150.0, 200.0, 141)
    torch.testing.assert_close(critic.support, support)
    torch.testing.assert_close(Critic(0, 81, 3, 32, 2, 0.0).support, support)
    targets = torch.tensor([-300.0, -150.0, -110.5, 0.0, 102.4, 200.0, 250.0])
    probs = critic.project(targets)
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(targets))
    assert (probs >= 0).all()
    assert (probs[0] == probs[1]).all() and (probs[-1] == probs[-2]).all()
    assert probs[0].argmax() == 0 and probs[-1].argmax() == 140
    unit = critic.support[1] - critic.support[0]
    assert unit == 2.5
    assert ((probs @ critic.support - targets.clamp(-150, 200)).abs() < unit).all()


def test_hl_gauss_updates_and_reports_clipped_td_targets(cfg):
    cfg["model"].update(critic_head="hl_gauss", q_min=-150.0, q_max=200.0, q_bins=141)
    cfg["train"]["huber_delta"] = 0.0  # ignored for HL-Gauss
    agent = SACAgent(cfg)
    batch = batch_for(cfg)
    batch["discount"].fill(0)
    batch["reward"][:6] = [-500, -150, -100, -75, 200, 500]
    before = next(agent.critic.parameters()).detach().clone()
    metrics = agent.update(batch)
    assert np.isfinite(list(metrics.values())).all()
    assert metrics["q_clip_frac"] == 0.5
    assert not torch.equal(next(agent.critic.parameters()), before)


@pytest.mark.parametrize("camera", [False, True])
def test_shrink_and_perturb_occurs_on_schedule_after_resume(cfg, monkeypatch, camera):
    cfg["env"]["observation"] = "image" if camera else "state"
    cfg["train"].update(reset_every_updates=2, reset_shrink=0.8)
    agent = SACAgent(cfg)
    batch = batch_for(cfg)
    agent.update(batch)
    assert not agent.reset_due and agent.actor_opt.state
    saved = agent.state_dict()
    agent = SACAgent(cfg)
    agent.load_state_dict(saved)
    control = SACAgent(cfg)
    control.load_state_dict(copy.deepcopy(saved))
    control.reset_every_updates = 0

    def fixed_linear_init(layer):
        torch.nn.init.constant_(layer.weight, 0.25)
        torch.nn.init.constant_(layer.bias, -0.5)

    monkeypatch.setattr(torch.nn.Linear, "reset_parameters", fixed_linear_init)
    for learner in (control, agent):
        torch.manual_seed(42)
        learner.update(batch)
    assert agent.reset_due and agent.updates == 2
    assert not agent.actor_opt.state
    assert agent.critic_opt.state.keys() == set(agent.encoder.parameters())
    for param, other in zip(agent.encoder.parameters(), control.encoder.parameters()):
        torch.testing.assert_close(agent.critic_opt.state[param], control.critic_opt.state[other])
    for network in ("encoder", "encoder_target"):
        torch.testing.assert_close(getattr(agent, network).state_dict(), getattr(control, network).state_dict())
    assert agent.alpha_opt.state[agent.log_alpha]["step"].item() == 2
    for network in ("actor", "critic"):
        for layer, original in zip(getattr(agent, network).modules(), getattr(control, network).modules()):
            if isinstance(layer, torch.nn.Linear):
                torch.testing.assert_close(layer.weight, original.weight * 0.8 + 0.25 * 0.2)
                torch.testing.assert_close(layer.bias, original.bias * 0.8 - 0.5 * 0.2)
            elif isinstance(layer, torch.nn.LayerNorm):
                torch.testing.assert_close(layer.weight, original.weight * 0.8 + 0.2)
                torch.testing.assert_close(layer.bias, original.bias * 0.8)
    for name, weight in agent.critic.state_dict().items():
        torch.testing.assert_close(agent.critic_target.state_dict()[name], weight, atol=0, rtol=0)
    agent.update(batch)
    assert not agent.reset_due and agent.critic_opt.state


def test_trainer_logs_and_publishes_perturbed_policy(cfg, tmp_path, monkeypatch):
    make_track(tmp_path, cfg, with_demo=False)
    cfg["train"].update(reset_every_updates=1, warmup_steps=0)
    logs = []
    trainer = Trainer(cfg, log=logs.append, device=torch.device("cpu"))
    trainer.online.transitions = trainer.batch_size
    monkeypatch.setattr(trainer, "sample", lambda: batch_for(cfg))
    policy_state = trainer.agent.policy_state

    def snapshot():
        assert trainer.agent_lock.locked()
        return policy_state()

    monkeypatch.setattr(trainer.agent, "policy_state", snapshot)
    assert trainer.learn_once()
    assert "shrink-and-perturb applied at learner update 1" in logs
    for name, weight in trainer.agent.actor.state_dict().items():
        torch.testing.assert_close(trainer._pending_policy["actor"][name], weight)


def test_trainer_rejects_legacy_architecture_and_can_resume_with_legacy_settings(cfg, tmp_path):
    make_track(tmp_path, cfg, with_demo=False)
    cfg["model"].update(critic_head="scalar", actor_layer_norm=False)
    cfg["train"]["optimizer"] = "adam"
    device = torch.device("cpu")
    trainer = Trainer(cfg, device=device)
    trainer.agent.actor_opt.param_groups[0]["lr"] = 1e-4
    trainer.agent.update(batch_for(cfg))
    ckpt = trainer.save_checkpoint()
    saved = torch.load(ckpt, weights_only=False)
    assert saved["agent"].pop("architecture") == {"critic_head": "scalar", "actor_layer_norm": False}
    torch.save(saved, ckpt)

    cfg["train"]["optimizer"] = "adamw"
    resumed = Trainer(cfg, device=device)
    assert resumed.agent.updates == 1
    assert resumed.agent.actor_opt.param_groups[0]["lr"] == 1e-4
    assert resumed.agent.actor_opt.param_groups[0]["weight_decay"] == cfg["train"].get("weight_decay", 1e-4)
    resumed.agent.update(batch_for(cfg))

    cfg["model"].update(critic_head="hl_gauss", actor_layer_norm=True)
    with pytest.raises(FileNotFoundError, match=r"model\.critic_head='scalar'.*run_name.*fresh_start"):
        Trainer(cfg, device=device)
    assert Trainer(cfg, device=device, fresh=True).agent.updates == 0


@pytest.mark.parametrize(("key", "changed"), [("actor_layer_norm", False), ("q_bins", 51)])
def test_checkpoint_rejects_architecture_changes(cfg, key, changed):
    cfg["model"].update(critic_head="hl_gauss", actor_layer_norm=True, q_bins=141)
    saved = SACAgent(cfg).state_dict()
    cfg["model"][key] = changed
    with pytest.raises(ValueError, match=rf"model\.{key}=.*run_name.*fresh_start"):
        SACAgent(cfg).load_state_dict(saved)
