"""SAC with a DroQ critic ensemble and DrQ image augmentation.

The critic owns the image encoder and trains it; the actor reads detached
encoder features (as in DrQ). bf16 autocast runs on CUDA when the config asks
for it; master weights stay fp32."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from tmai.budget import ACTION_DIM, STATE_DIM
from tmai.rl.networks import Actor, Critic, Encoder, random_shift


class Policy:
    """Encoder + actor only: the copy the driving loop uses, so learner updates
    never change weights in the middle of choosing an action."""

    def __init__(self, cfg: dict, device: str | torch.device = "cpu"):
        m, env = cfg["model"], cfg["env"]
        self.device = torch.device(device)
        self.encoder = Encoder(env["frame_stack"], env["frame_size"], m["encoder_channels"], m["hidden_dim"]).to(self.device)
        self.actor = Actor(m["hidden_dim"], STATE_DIM, ACTION_DIM, m["hidden_dim"]).to(self.device)
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
        feat = m["hidden_dim"]
        self.encoder = Encoder(env["frame_stack"], env["frame_size"], m["encoder_channels"], feat).to(self.device)
        self.critic = Critic(feat, STATE_DIM, ACTION_DIM, m["hidden_dim"], m["num_critics"], m["critic_dropout"]).to(self.device)
        self.actor = Actor(feat, STATE_DIM, ACTION_DIM, m["hidden_dim"]).to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(0.1), device=self.device, requires_grad=True)
        self.target_entropy = float(t.get("target_entropy", -ACTION_DIM))
        lr = t["learning_rate"]
        self.critic_opt = torch.optim.Adam([*self.encoder.parameters(), *self.critic.parameters()], lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.use_bf16 = (m["precision"] == "bf16" and self.device.type == "cuda"
                         and torch.cuda.is_bf16_supported())
        self.updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16)

    def _tensors(self, batch: dict) -> dict:
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    def update(self, batch: dict) -> dict:
        b = self._tensors(batch)
        frames, next_frames = random_shift(b["frames"]), random_shift(b["next_frames"])
        with self._autocast():
            with torch.no_grad():
                next_feat = self.encoder(next_frames)
                next_a, next_logp = self.actor(next_feat, b["next_state"])
                q_next = self.critic_target(self.encoder_target(next_frames), b["next_state"], next_a)
                v_next = q_next.min(0).values.float() - self.alpha.detach() * next_logp.float()
                # n-step batches carry their own discount (gamma**k, 0 after a terminal state)
                discount = b["discount"] if "discount" in b else self.gamma * (1.0 - b["terminal"])
                target = b["reward"] + discount * v_next
            feat = self.encoder(frames)
            q = self.critic(feat, b["state"], b["action"]).float()
            critic_loss = F.mse_loss(q, target.unsqueeze(0).expand_as(q))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        with self._autocast():
            a, logp = self.actor(feat.detach(), b["state"])
            q_pi = self.critic(feat.detach(), b["state"], a).float().mean(0)
            actor_loss = (self.alpha.detach() * logp.float() - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach().float() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.critic, self.critic_target), (self.encoder, self.encoder_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.lerp_(p, self.tau)
        self.updates += 1
        return {
            "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(), "q": q.mean().item(), "entropy": -logp.float().mean().item(),
        }

    def policy_state(self) -> dict:
        """Weights the driving loop needs (encoder and actor)."""
        return {"encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                "actor": {k: v.detach().clone() for k, v in self.actor.state_dict().items()}}

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(), "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(), "encoder_target": self.encoder_target.state_dict(),
            "critic_target": self.critic_target.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "critic_opt": self.critic_opt.state_dict(), "actor_opt": self.actor_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(), "updates": self.updates,
        }

    def load_state_dict(self, sd: dict) -> None:
        for name in ("encoder", "critic", "actor", "encoder_target", "critic_target"):
            getattr(self, name).load_state_dict(sd[name])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"])
        for name in ("critic_opt", "actor_opt", "alpha_opt"):
            getattr(self, name).load_state_dict(sd[name])
        self.updates = sd["updates"]
