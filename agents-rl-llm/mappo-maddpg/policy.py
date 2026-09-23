from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class CriticNetwork(nn.Module):
    def __init__(self, joint_obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.network(joint_obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def add(self, observation, action, reward, done, log_prob, value):
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def __len__(self):
        return len(self.rewards)

    def get_tensors(self):
        return {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
        }


def gae(rewards: Sequence[float], values: Sequence[float], dones: Sequence[float], gamma: float = 0.99, lam: float = 0.95):
    rewards = list(rewards)
    values = list(values)
    dones = list(dones)
    advantages = []
    returns = []
    gae_value = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * (1.0 - dones[step]) * next_value - values[step]
        gae_value = delta + gamma * lam * (1.0 - dones[step]) * gae_value
        advantages.append(gae_value)
        returns.append(rewards[step] + gamma * (1.0 - dones[step]) * next_value)
        next_value = values[step]
    advantages.reverse()
    returns.reverse()
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


class MAPPOPolicy:
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, action_names: Sequence[str] | None = None):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_names = list((action_names or ["move", "scan", "rescue", "deliver", "return", "wait"])[:action_dim])
        while len(self.action_names) < action_dim:
            self.action_names.append("wait")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        joint_obs_dim = max(obs_dim * max(1, num_agents), obs_dim)
        self.critic = CriticNetwork(joint_obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4)
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.ppo_epochs = 4
        self.policy_params = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def _prepare_obs(self, observations: Iterable[Sequence[float]]):
        if isinstance(observations, (list, tuple)) and observations and not isinstance(observations[0], (list, tuple)):
            return [list(observations)]
        return [list(obs) for obs in observations]

    def _as_tensor(self, observations):
        arr = self._prepare_obs(observations)
        tensor = torch.tensor(np.asarray(arr, dtype=np.float32), dtype=torch.float32, device=self.device)
        return tensor

    def _joint_obs(self, observations):
        obs = self._prepare_obs(observations)
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        return torch.tensor(flat[None, :], dtype=torch.float32, device=self.device)

    def forward(self, observations: Iterable[Sequence[float]]):
        obs = self._as_tensor(observations)
        logits = self.actor(obs)
        return logits

    def sample_action(self, observation: Sequence[float]) -> str:
        obs = torch.tensor(np.asarray(observation, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)
        action_index = dist.sample().item()
        return self.action_names[action_index]

    def decide_batch(self, observations: Sequence[Sequence[float]]) -> List[str]:
        obs = self._as_tensor(observations)
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)
        action_indices = dist.sample().detach().cpu().tolist()
        return [self.action_names[idx] for idx in action_indices]

    def sample_batch(self, observations: Sequence[Sequence[float]]):
        obs = self._as_tensor(observations)
        dist = Categorical(logits=self.actor(obs))
        action_indices = dist.sample()
        log_probs = dist.log_prob(action_indices)
        joint_obs = obs.reshape(1, -1)
        value = self.critic(joint_obs).expand(len(observations))
        return action_indices.detach().cpu().tolist(), log_probs.detach().cpu().tolist(), value.detach().cpu().tolist()

    def ppo_update(self, observations, actions, rewards, dones, log_probs, values, advantages=None, returns=None):
        obs_tensor = torch.tensor(np.asarray(self._prepare_obs(observations), dtype=np.float32), device=self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        if advantages is None:
            advantages = gae(rewards, values, dones, self.gamma, self.lam)[0].to(self.device)
        else:
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if returns is None:
            returns = gae(rewards, values, dones, self.gamma, self.lam)[1].to(self.device)
        else:
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        for _ in range(self.ppo_epochs):
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            clipped_adv = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages.detach()
            policy_loss = -(torch.min(ratio * advantages.detach(), clipped_adv)).mean()
            joint_obs = obs_tensor.reshape(1, -1)
            values_pred = self.critic(joint_obs).expand_as(returns)
            value_loss = F.mse_loss(values_pred, returns.detach())
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            policy_loss_total += float(policy_loss.detach().cpu().item())
            value_loss_total += float(value_loss.detach().cpu().item())
            entropy_total += float(entropy.detach().cpu().item())
        self.policy_params = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        return {
            "policy_loss": policy_loss_total / max(1, self.ppo_epochs),
            "value_loss": value_loss_total / max(1, self.ppo_epochs),
            "entropy": entropy_total / max(1, self.ppo_epochs),
        }

    def update(self, batch, rewards, advantages):
        if not batch:
            return self.policy_params
        obs = self._prepare_obs(batch)
        actions = list(range(len(obs)))
        return self.ppo_update(
            observations=obs,
            actions=actions,
            rewards=rewards,
            dones=[0.0 for _ in rewards],
            log_probs=[0.0 for _ in rewards],
            values=[0.0 for _ in rewards],
            advantages=advantages,
            returns=[float(r) for r in rewards],
        )

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "action_names": self.action_names,
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "lam": self.lam,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(payload.get("actor"), dict):
            self.actor.load_state_dict(payload["actor"])
        if isinstance(payload.get("critic"), dict):
            self.critic.load_state_dict(payload["critic"])
        if isinstance(payload.get("action_names"), list):
            self.action_names = payload["action_names"]
        self.policy_params = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        return self.policy_params
