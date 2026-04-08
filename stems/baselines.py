"""
Baseline agents for comparison with STEMS.

    RuleBasedAgent      – TOU scheduling heuristic
    SingleAgentSAC      – independent per-building SAC, no coordination
    DMAPPOAgent         – distributed MAPPO with soft CBF penalty
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Observation indices (matching OBS_NAMES in environment.py)
_IDX_HOUR = 1
_IDX_PRICE = 21
_IDX_SOC_ELEC = 19
_IDX_OCCUPANT = 26
_IDX_T_IN = 15
_IDX_NET = 20


# ==========================================================================
# Rule-Based Agent (TOU Scheduling)
# ==========================================================================

class RuleBasedAgent:
    """Time-of-Use heuristic control.

    Strategy:
        - Electrical storage: charge during off-peak hours [0, 6],
          discharge during peak hours [16, 21].
        - DHW storage: mirror electrical storage schedule.
        - Cooling: reduce (mild mode) when building is unoccupied.
    """

    def __init__(self, num_buildings: int = 3) -> None:
        self.B = num_buildings

    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = False,
    ) -> np.ndarray:
        """Return (B, 3) actions based on TOU schedule."""
        actions = np.zeros((self.B, 3), dtype=np.float32)

        for i, obs in enumerate(obs_list):
            hour = int(obs[_IDX_HOUR])
            occupant = float(obs[_IDX_OCCUPANT])

            # Electrical storage action (index 1)
            if 0 <= hour <= 6:
                elec_action = 0.8    # charge
            elif 16 <= hour <= 21:
                elec_action = -0.8   # discharge
            else:
                elec_action = 0.0

            # DHW storage action (index 0) – similar pattern
            dhw_action = elec_action * 0.5

            # Cooling action (index 2) – reduce if unoccupied
            if occupant == 0:
                cool_action = -0.3   # mild cooling (save energy)
            else:
                cool_action = 0.2    # normal cooling

            actions[i] = [dhw_action, elec_action, cool_action]

        return actions

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {}   # no learning

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


# ==========================================================================
# SAC helper networks
# ==========================================================================

class _SACNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SACPolicy(nn.Module):
    """Gaussian policy: outputs mean and log_std."""
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = (normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1)
        return action, log_prob


# ==========================================================================
# Single-Agent SAC (independent per building, no graph, no CBF)
# ==========================================================================

class SingleAgentSAC:
    """Independent Soft Actor-Critic per building.

    No cross-building coordination (no GCN), no safety shield.
    Uses standard SAC with entropy regularisation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int = 3,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_entropy: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.B = num_buildings
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha_ent = alpha_entropy
        self.device = torch.device(device)

        # Per-building networks
        self.policies = nn.ModuleList([
            _SACPolicy(obs_dim, hidden_dim, action_dim) for _ in range(num_buildings)
        ]).to(self.device)

        self.q1_nets = nn.ModuleList([
            _SACNet(obs_dim + action_dim, hidden_dim, 1) for _ in range(num_buildings)
        ]).to(self.device)

        self.q2_nets = nn.ModuleList([
            _SACNet(obs_dim + action_dim, hidden_dim, 1) for _ in range(num_buildings)
        ]).to(self.device)

        self.q1_target = copy.deepcopy(self.q1_nets).to(self.device)
        self.q2_target = copy.deepcopy(self.q2_nets).to(self.device)

        self.policy_opts = [
            optim.Adam(self.policies[i].parameters(), lr=lr)
            for i in range(num_buildings)
        ]
        self.q_opts = [
            optim.Adam(
                list(self.q1_nets[i].parameters()) + list(self.q2_nets[i].parameters()),
                lr=lr,
            )
            for i in range(num_buildings)
        ]

    # ------------------------------------------------------------------
    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if explore:
                    a, _ = self.policies[i].sample(x)
                else:
                    mean, _ = self.policies[i](x)
                    a = torch.tanh(mean)
            actions[i] = a.squeeze(0).cpu().numpy()
        return actions

    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}

        N = len(batch["obs"])
        losses = {"actor_loss": 0.0, "critic_loss": 0.0}

        for b in range(self.B):
            obs_b = torch.tensor(
                np.array([batch["obs"][n][b] for n in range(N)]), dtype=torch.float32
            ).to(self.device)
            next_obs_b = torch.tensor(
                np.array([batch["next_obs"][n][b] for n in range(N)]), dtype=torch.float32
            ).to(self.device)
            actions_b = torch.tensor(batch["actions"][:, b, :], dtype=torch.float32).to(self.device)
            rewards_b = torch.tensor(batch["rewards"][:, b], dtype=torch.float32).to(self.device)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32).to(self.device)

            # Q-function update
            with torch.no_grad():
                next_a, next_log_p = self.policies[b].sample(next_obs_b)
                q1_next = self.q1_target[b](torch.cat([next_obs_b, next_a], dim=-1)).squeeze(-1)
                q2_next = self.q2_target[b](torch.cat([next_obs_b, next_a], dim=-1)).squeeze(-1)
                q_target = rewards_b + self.gamma * (1 - dones_b) * (
                    torch.min(q1_next, q2_next) - self.alpha_ent * next_log_p
                )

            q1_pred = self.q1_nets[b](torch.cat([obs_b, actions_b], dim=-1)).squeeze(-1)
            q2_pred = self.q2_nets[b](torch.cat([obs_b, actions_b], dim=-1)).squeeze(-1)
            q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

            self.q_opts[b].zero_grad()
            q_loss.backward()
            self.q_opts[b].step()

            # Policy update
            a_new, log_p_new = self.policies[b].sample(obs_b)
            q1_new = self.q1_nets[b](torch.cat([obs_b, a_new], dim=-1)).squeeze(-1)
            q2_new = self.q2_nets[b](torch.cat([obs_b, a_new], dim=-1)).squeeze(-1)
            policy_loss = (self.alpha_ent * log_p_new - torch.min(q1_new, q2_new)).mean()

            self.policy_opts[b].zero_grad()
            policy_loss.backward()
            self.policy_opts[b].step()

            # Soft target update
            for param, target_param in zip(self.q1_nets[b].parameters(), self.q1_target[b].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2_nets[b].parameters(), self.q2_target[b].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            losses["critic_loss"] += q_loss.item()
            losses["actor_loss"] += policy_loss.item()

        losses["critic_loss"] /= self.B
        losses["actor_loss"] /= self.B
        return losses

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.policies.state_dict(), os.path.join(path, "sac_policies.pt"))
        torch.save(self.q1_nets.state_dict(), os.path.join(path, "sac_q1.pt"))
        torch.save(self.q2_nets.state_dict(), os.path.join(path, "sac_q2.pt"))

    def load(self, path: str) -> None:
        self.policies.load_state_dict(torch.load(os.path.join(path, "sac_policies.pt")))
        self.q1_nets.load_state_dict(torch.load(os.path.join(path, "sac_q1.pt")))
        self.q2_nets.load_state_dict(torch.load(os.path.join(path, "sac_q2.pt")))


# ==========================================================================
# DMAPPO with soft CBF penalty
# ==========================================================================

class _PPOActor(nn.Module):
    LOG_STD_MIN = -4
    LOG_STD_MAX = 1

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        mean = torch.tanh(self.mean_head(feat))
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std.exp()

    def log_prob(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1)


class _PPOCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DMAPPOAgent:
    """Distributed Multi-Agent PPO with soft CBF penalty.

    Unlike STEMS:
        - No GCN or Transformer encoder (uses raw observations)
        - Soft CBF: adds λ * max(0, -h(s,a)) to reward instead of projecting
        - PPO clip objective instead of advantage actor-critic
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int = 3,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        cbf_lambda: float = 1.0,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        device: str = "cpu",
    ) -> None:
        self.B = num_buildings
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.cbf_lambda = cbf_lambda
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.device = torch.device(device)

        self.actors = nn.ModuleList([
            _PPOActor(obs_dim, hidden_dim, action_dim) for _ in range(num_buildings)
        ]).to(self.device)

        self.critics = nn.ModuleList([
            _PPOCritic(obs_dim, hidden_dim) for _ in range(num_buildings)
        ]).to(self.device)

        self.actor_opts = [
            optim.Adam(self.actors[i].parameters(), lr=lr) for i in range(num_buildings)
        ]
        self.critic_opts = [
            optim.Adam(self.critics[i].parameters(), lr=lr) for i in range(num_buildings)
        ]

    # ------------------------------------------------------------------
    def _cbf_penalty(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Soft CBF penalty: λ * max(0, -h(s,a))."""
        soc = float(obs[_IDX_SOC_ELEC])
        delta = float(action[1]) * 0.1
        new_soc = soc + delta
        h_lo = new_soc - self.soc_min
        h_hi = self.soc_max - new_soc
        penalty = max(0.0, -h_lo) + max(0.0, -h_hi)
        return self.cbf_lambda * penalty

    # ------------------------------------------------------------------
    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mean, std = self.actors[i](x)
                if explore:
                    dist = torch.distributions.Normal(mean, std)
                    a = dist.sample()
                else:
                    a = mean
                a = torch.clamp(a, -1.0, 1.0)
            actions[i] = a.squeeze(0).cpu().numpy()
        return actions

    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}

        N = len(batch["obs"])
        losses = {"actor_loss": 0.0, "critic_loss": 0.0}

        for b in range(self.B):
            obs_b = torch.tensor(
                np.array([batch["obs"][n][b] for n in range(N)]), dtype=torch.float32
            ).to(self.device)
            next_obs_b = torch.tensor(
                np.array([batch["next_obs"][n][b] for n in range(N)]), dtype=torch.float32
            ).to(self.device)
            actions_b = torch.tensor(batch["actions"][:, b, :], dtype=torch.float32).to(self.device)
            rewards_b = torch.tensor(batch["rewards"][:, b], dtype=torch.float32).to(self.device)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32).to(self.device)

            # CBF soft penalty applied to rewards
            for n in range(N):
                penalty = self._cbf_penalty(batch["obs"][n][b], batch["actions"][n, b])
                rewards_b[n] -= penalty

            # Critic update
            values = self.critics[b](obs_b)
            with torch.no_grad():
                next_values = self.critics[b](next_obs_b)
                targets = rewards_b + self.gamma * (1 - dones_b) * next_values
                advantages = targets - values

            critic_loss = F.mse_loss(values, targets)
            self.critic_opts[b].zero_grad()
            critic_loss.backward()
            self.critic_opts[b].step()

            # PPO clip actor update
            old_log_prob = self.actors[b].log_prob(obs_b, actions_b).detach()
            new_log_prob = self.actors[b].log_prob(obs_b, actions_b)
            ratio = (new_log_prob - old_log_prob).exp()
            adv = advantages.detach()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_opts[b].zero_grad()
            actor_loss.backward()
            self.actor_opts[b].step()

            losses["actor_loss"] += actor_loss.item()
            losses["critic_loss"] += critic_loss.item()

        losses["actor_loss"] /= self.B
        losses["critic_loss"] /= self.B
        return losses

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actors.state_dict(), os.path.join(path, "ppo_actors.pt"))
        torch.save(self.critics.state_dict(), os.path.join(path, "ppo_critics.pt"))

    def load(self, path: str) -> None:
        self.actors.load_state_dict(torch.load(os.path.join(path, "ppo_actors.pt")))
        self.critics.load_state_dict(torch.load(os.path.join(path, "ppo_critics.pt")))
