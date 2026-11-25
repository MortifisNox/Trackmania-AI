"""SAC with a DroQ critic ensemble and DrQ image augmentation.

The critic owns the image encoder and trains it; the actor reads detached
encoder features (as in DrQ). bf16 autocast runs on CUDA when the config asks
for it; master weights stay fp32."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from tmai.budget import ACTION_DIM
from tmai.rl.features import state_dim, uses_camera
from tmai.rl.networks import HL_GAUSS_BINS, HL_GAUSS_MAX, HL_GAUSS_MIN, Actor, Critic, make_encoder, random_shift


class CheckpointArchitectureError(ValueError):
    """The checkpoint's networks differ from the configured networks."""


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: _cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def _weight_decay_groups(params, weight_decay: float) -> list[dict]:
    weights, biases_and_norms = [], []
    for param in params:
        (weights if param.ndim >= 2 else biases_and_norms).append(param)
    return [{"params": weights, "weight_decay": weight_decay},
            {"params": biases_and_norms, "weight_decay": 0.0}]


def _load_network_optimizer(opt, saved: dict, params) -> None:
    current_groups = opt.param_groups
    saved_groups = saved["param_groups"]
    if len(saved_groups) != len(current_groups):
        # The old single group followed network parameter order. The two-group
        # layout orders by dimensionality, preserving network order within each group.
        saved_layout = [list(params)] if len(saved_groups) == 1 else [
            group["params"] for group in _weight_decay_groups(params, 0.0)
        ]
        saved_ids = {
            param: param_id
            for group, layout in zip(saved_groups, saved_layout, strict=True)
            for param, param_id in zip(layout, group["params"], strict=True)
        }
        saved_groups = [
            {**saved_groups[0], "params": [saved_ids[param] for param in group["params"]]}
            for group in current_groups
        ]
    # Preserve moments and saved learning rates while applying the configured Adam variant.
    groups = [
        {**group, **{key: current[key] for key in ("weight_decay", "fused", "decoupled_weight_decay")
                     if key in current}}
        for group, current in zip(saved_groups, current_groups, strict=True)
    ]
    opt.load_state_dict({**saved, "param_groups": groups})


class Policy:
    """Encoder + actor only: the copy the driving loop uses, so learner updates
    never change weights in the middle of choosing an action."""

    def __init__(self, cfg: dict, device: str | torch.device = "cpu"):
        m, env = cfg["model"], cfg["env"]
        self.device = torch.device(device)
        encoder, feat = make_encoder(env, m)
        self.encoder = encoder.to(self.device)
        self.actor = Actor(feat, state_dim(env), ACTION_DIM, m["hidden_dim"],
                           layer_norm=m.get("actor_layer_norm", True)).to(self.device)
        self.encoder.requires_grad_(False)
        self.actor.requires_grad_(False)

    def load(self, policy_state: dict) -> None:
        self.encoder.load_state_dict(policy_state["encoder"])
        self.actor.load_state_dict(policy_state["actor"])

    @torch.no_grad()
    def act(self, frames: np.ndarray, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        f = torch.as_tensor(frames, device=self.device).unsqueeze(0)
        s = torch.as_tensor(state, device=self.device).unsqueeze(0)
        a, _ = self.actor(self.encoder(f), s, deterministic)
        return a.squeeze(0).cpu().numpy()


class SACAgent:
    def __init__(self, cfg: dict, device: str | torch.device = "cpu"):
        m, t, env = cfg["model"], cfg["train"], cfg["env"]
        self.device = torch.device(device)
        self.gamma, self.tau = t["gamma"], t["tau"]
        encoder, feat = make_encoder(env, m)
        sdim = state_dim(env)
        self.augment = uses_camera(env)   # DrQ image shifts; nothing to shift in state mode
        self.encoder = encoder.to(self.device)
        head = m.get("critic_head", "hl_gauss")
        if head not in ("scalar", "hl_gauss"):
            raise ValueError(f"model.critic_head must be scalar or hl_gauss, got {head!r}")
        layer_norm = m.get("actor_layer_norm", True)
        self.architecture = {"critic_head": head, "actor_layer_norm": layer_norm}
        critic_kwargs = {"q_bins": 1}
        if head == "hl_gauss":
            q_bins = m.get("q_bins", HL_GAUSS_BINS)
            q_min, q_max = m.get("q_min", HL_GAUSS_MIN), m.get("q_max", HL_GAUSS_MAX)
            sigma_ratio = m.get("hl_sigma_ratio", 0.75)
            if q_bins < 2 or not (math.isfinite(q_max - q_min) and q_max > q_min):
                raise ValueError("hl_gauss requires at least two q_bins and finite q_min < q_max")
            if not math.isfinite(sigma_ratio) or sigma_ratio <= 0:
                raise ValueError("model.hl_sigma_ratio must be finite and positive")
            self.architecture.update(q_bins=q_bins, q_min=q_min, q_max=q_max)
            critic_kwargs = {"q_bins": q_bins, "q_min": q_min, "q_max": q_max, "hl_sigma_ratio": sigma_ratio}
        self.critic = Critic(feat, sdim, ACTION_DIM, m["hidden_dim"], m["num_critics"], m["critic_dropout"],
                             **critic_kwargs).to(self.device)
        self.actor = Actor(feat, sdim, ACTION_DIM, m["hidden_dim"], layer_norm=layer_norm).to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float32, device=self.device, requires_grad=True)
        self.target_entropy = float(t.get("target_entropy", -ACTION_DIM))
        lr = t["learning_rate"]
        optimizer = t.get("optimizer", "adamw")
        opt_kwargs = {"lr": lr, "fused": self.device.type == "cuda"}
        critic_params = [*self.encoder.parameters(), *self.critic.parameters()]
        actor_params = list(self.actor.parameters())
        if optimizer == "adamw":
            make_optimizer = torch.optim.AdamW
            decay = t.get("weight_decay", 1.0e-4)
            critic_params = _weight_decay_groups(critic_params, decay)
            actor_params = _weight_decay_groups(actor_params, decay)
        elif optimizer == "adam":
            make_optimizer = torch.optim.Adam
        else:
            raise ValueError(f"train.optimizer must be adam or adamw, got {optimizer!r}")
        self.critic_opt = make_optimizer(critic_params, **opt_kwargs)
        self.actor_opt = make_optimizer(actor_params, **opt_kwargs)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.critic_loss = t.get("critic_loss", "huber")
        self.huber_delta = t.get("huber_delta", 1.0)
        if head == "scalar":
            if self.critic_loss not in ("mse", "huber"):
                raise ValueError(f"train.critic_loss must be mse or huber, got {self.critic_loss!r}")
            if self.critic_loss == "huber" and self.huber_delta <= 0:
                raise ValueError("train.huber_delta must be positive")
        self.reset_every_updates = t.get("reset_every_updates", 400000)
        self.reset_shrink = t.get("reset_shrink", 0.8)
        if self.reset_every_updates < 0 or not 0 <= self.reset_shrink <= 1:
            raise ValueError("train.reset_every_updates must be nonnegative and train.reset_shrink in [0, 1]")
        self.use_bf16 = (self.augment and m["precision"] == "bf16" and self.device.type == "cuda"
                         and torch.cuda.is_bf16_supported())
        self.updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _encode(self, encoder, frames):
        if self.use_bf16:
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                return encoder(frames).float()
        return encoder(frames)

    def _tensors(self, batch: dict) -> dict:
        return {k: torch.as_tensor(v, device=self.device) if k.endswith("frames")
                else torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in batch.items()}

    def _critic_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.critic.distributional:
            probs = self.critic.project(target)
            return -(output.log_softmax(-1) * probs).sum(-1).mean()
        target = target.expand_as(output)
        if self.critic_loss == "huber":
            return F.huber_loss(output, target, delta=self.huber_delta)
        return F.mse_loss(output, target)

    @property
    def reset_due(self) -> bool:
        return bool(self.updates and self.reset_every_updates and self.updates % self.reset_every_updates == 0)

    @torch.no_grad()
    def _shrink_and_perturb(self) -> None:
        # Keep learned image features, the encoder's Adam moments, and its
        # Polyak-averaged target. Clear moments only for parameters we perturb.
        for net, opt in ((self.actor, self.actor_opt), (self.critic, self.critic_opt)):
            fresh = copy.deepcopy(net)
            for layer in fresh.modules():
                if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm)):
                    layer.reset_parameters()
            for param, fresh_param in zip(net.parameters(), fresh.parameters()):
                param.lerp_(fresh_param, 1 - self.reset_shrink)
                opt.state.pop(param, None)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update(self, batch: dict) -> dict:
        b = self._tensors(batch)
        frames, next_frames = b["frames"], b["next_frames"]
        if self.augment:
            frames, next_frames = random_shift(frames), random_shift(next_frames)
        with torch.no_grad():
            next_feat = self._encode(self.encoder, next_frames)
            next_a, next_logp = self.actor(next_feat, b["next_state"])
            next_output = self.critic_target(
                self._encode(self.encoder_target, next_frames), b["next_state"], next_a)
            q_next = self.critic_target.expected_q(next_output)
            v_next = q_next.min(0).values - self.alpha.detach() * next_logp
            # n-step batches carry their own discount (gamma**k, 0 after a terminal state)
            discount = b["discount"] if "discount" in b else self.gamma * (1.0 - b["terminal"])
            target = b["reward"] + discount * v_next
        feat = self._encode(self.encoder, frames)
        output = self.critic(feat, b["state"], b["action"])
        critic_loss = self._critic_loss(output, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        a, logp = self.actor(feat.detach(), b["state"])
        self.critic.requires_grad_(False)
        try:
            q_pi = self.critic.expected_q(self.critic(feat.detach(), b["state"], a)).mean(0)
        finally:
            self.critic.requires_grad_(True)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.critic, self.critic_target), (self.encoder, self.encoder_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.lerp_(p, self.tau)
            metrics = {
                "critic_loss": critic_loss, "actor_loss": actor_loss, "alpha": self.alpha,
                "q": self.critic.expected_q(output).mean(), "entropy": -logp.mean(),
            }
            if self.critic.distributional:
                clipped = (target <= self.critic.q_min) | (target >= self.critic.q_max)
                metrics["q_clip_frac"] = clipped.float().mean()
            metric_values = torch.stack(tuple(metrics.values()))
        self.updates += 1
        if self.reset_due:
            self._shrink_and_perturb()
        return dict(zip(metrics, metric_values.tolist()))

    def policy_state(self) -> dict:
        """Weights the driving loop needs (encoder and actor)."""
        return {"encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                "actor": {k: v.detach().clone() for k, v in self.actor.state_dict().items()}}

    def state_dict(self) -> dict:
        """Take an independent CPU snapshot. The caller holds the learner lock."""
        return _cpu_snapshot({
            "encoder": self.encoder.state_dict(), "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(), "encoder_target": self.encoder_target.state_dict(),
            "critic_target": self.critic_target.state_dict(), "log_alpha": self.log_alpha,
            "critic_opt": self.critic_opt.state_dict(), "actor_opt": self.actor_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(), "updates": self.updates, "architecture": self.architecture,
        })

    def load_state_dict(self, sd: dict) -> None:
        saved = sd.get("architecture", {"critic_head": "scalar", "actor_layer_norm": False})
        for key, expected in self.architecture.items():
            if saved.get(key) != expected:
                raise CheckpointArchitectureError(
                    f"checkpoint was trained with model.{key}={saved.get(key)!r}, this run uses {expected!r}; "
                    "give this run another run_name, or start it over with fresh_start")
        for name in ("encoder", "critic", "actor", "encoder_target", "critic_target"):
            getattr(self, name).load_state_dict(sd[name])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"])
        _load_network_optimizer(self.critic_opt, sd["critic_opt"],
                                [*self.encoder.parameters(), *self.critic.parameters()])
        _load_network_optimizer(self.actor_opt, sd["actor_opt"], list(self.actor.parameters()))
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
        self.updates = sd["updates"]
