"""
Baseline agents for comparison with STEMS.

    RuleBasedAgent      – TOU scheduling heuristic
    SingleAgentSAC      – independent per-building SAC, no coordination
    DMAPPOAgent         – distributed MAPPO with soft CBF penalty
    MPCAgent            – linear MPC using cvxpy QP at each step
    MADDPGAgent         – centralised critic, decentralised actors (DDPG-style)
    MARLISAAgent        – sequential SAC (buildings take turns seeing prior actions)
    MADCQAgent          – independent Q-networks with heuristic SOC clipping
    MetaEMSAgent        – simplified MAML wrapper around base RL agent
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
            actions[i] = np.array(a.squeeze(0).detach().tolist(), dtype=np.float32)
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
        ppo_epochs: int = 4,
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
        self.ppo_epochs = ppo_epochs  # number of PPO update epochs per batch
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
            actions[i] = np.array(a.squeeze(0).detach().tolist(), dtype=np.float32)
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

            # PPO clip actor update (multi-epoch).
            # Bug fix: old_log_prob must be computed from the policy BEFORE any
            # gradient update, then held fixed across ppo_epochs. Computing both
            # old and new inside the same forward pass (pre-step) gives ratio=1
            # always, making the clip dead. We now compute old_log_prob once with
            # no_grad, then iterate: after the first step() the actor weights
            # change, so subsequent epochs produce ratio != 1 and the clip fires.
            adv = advantages.detach()
            with torch.no_grad():
                old_log_prob = self.actors[b].log_prob(obs_b, actions_b)

            for _epoch in range(self.ppo_epochs):
                new_log_prob = self.actors[b].log_prob(obs_b, actions_b)
                ratio = (new_log_prob - old_log_prob).exp()
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


# ==========================================================================
# MPC Agent – linear Model Predictive Control
# ==========================================================================

class MPCAgent:
    """Rolling-horizon linear MPC using cvxpy.

    Solves a QP to minimise electricity cost + comfort penalty over a
    short prediction horizon, subject to SOC dynamics and power limits.
    Falls back to a rule-based heuristic if cvxpy is unavailable.
    """

    def __init__(
        self,
        num_buildings: int = 3,
        action_dim: int = 3,
        horizon: int = 6,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        P_building_max: float = 200.0,
        P_grid_max: float = 1000.0,
        eta: float = 0.1,
    ) -> None:
        self.B = num_buildings
        self.action_dim = action_dim
        self.horizon = horizon
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.P_building_max = P_building_max
        self.P_grid_max = P_grid_max
        self.eta = eta  # SOC update coefficient

        try:
            import cvxpy  # noqa: F401
            self._has_cvxpy = True
        except ImportError:
            self._has_cvxpy = False

    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = False,
    ) -> np.ndarray:
        if not self._has_cvxpy:
            return self._fallback(obs_list)

        import cvxpy as cp

        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        H = self.horizon

        for i, obs in enumerate(obs_list):
            soc = float(obs[_IDX_SOC_ELEC])
            price = max(float(obs[_IDX_PRICE]), 1e-6)

            # Decision variable: electrical storage actions over horizon
            u = cp.Variable(H)
            soc_traj = soc + self.eta * cp.cumsum(u)

            # Cost: electricity * price (simplified linear model)
            cost = cp.sum(cp.multiply(price, cp.pos(u)))

            constraints = [
                u >= -1.0,
                u <= 1.0,
                soc_traj >= self.soc_min,
                soc_traj <= self.soc_max,
            ]

            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=500)
                if u.value is not None:
                    elec_action = float(np.clip(u.value[0], -1.0, 1.0))
                else:
                    elec_action = 0.0
            except cp.SolverError:
                elec_action = 0.0

            actions[i] = [elec_action * 0.5, elec_action, 0.0]

        return actions

    def _fallback(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """Simple rule when cvxpy is unavailable."""
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            soc = float(obs[_IDX_SOC_ELEC])
            price = float(obs[_IDX_PRICE])
            if price < 0.05 and soc < 0.8:
                actions[i, 1] = 0.5
            elif price > 0.15 and soc > 0.3:
                actions[i, 1] = -0.5
        return actions

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


# ==========================================================================
# MADDPG – Multi-Agent DDPG with centralised critic
# ==========================================================================

class _DDPGActor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MADDPGAgent:
    """Multi-Agent DDPG: centralised critic seeing all agents' obs+actions,
    decentralised actors using only local observations."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int = 3,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.B = num_buildings
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.device = torch.device(device)

        # Decentralised actors
        self.actors = nn.ModuleList([
            _DDPGActor(obs_dim, hidden_dim, action_dim) for _ in range(num_buildings)
        ]).to(self.device)
        self.actors_target = copy.deepcopy(self.actors).to(self.device)

        # Centralised critics: input = all obs + all actions
        cent_input_dim = num_buildings * (obs_dim + action_dim)
        self.critics = nn.ModuleList([
            _SACNet(cent_input_dim, hidden_dim, 1) for _ in range(num_buildings)
        ]).to(self.device)
        self.critics_target = copy.deepcopy(self.critics).to(self.device)

        self.actor_opts = [
            optim.Adam(self.actors[i].parameters(), lr=lr) for i in range(num_buildings)
        ]
        self.critic_opts = [
            optim.Adam(self.critics[i].parameters(), lr=lr) for i in range(num_buildings)
        ]

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
                a = np.array(self.actors[i](x).squeeze(0).detach().tolist(), dtype=np.float32)
            if explore:
                a += np.random.normal(0, self.noise_std, size=a.shape)
            actions[i] = np.clip(a, -1.0, 1.0)
        return actions

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}

        N = len(batch["obs"])
        losses = {"actor_loss": 0.0, "critic_loss": 0.0}

        # Build joint obs / actions tensors
        all_obs = []
        all_next_obs = []
        all_actions = []
        per_obs = []
        per_next_obs = []

        for b in range(self.B):
            o_b = torch.tensor(
                np.array([batch["obs"][n][b] for n in range(N)]), dtype=torch.float32
            ).to(self.device)
            no_b = torch.tensor(
                np.array([batch["next_obs"][n][b] for n in range(N)]), dtype=torch.float32
            ).to(self.device)
            a_b = torch.tensor(batch["actions"][:, b, :], dtype=torch.float32).to(self.device)
            all_obs.append(o_b)
            all_next_obs.append(no_b)
            all_actions.append(a_b)
            per_obs.append(o_b)
            per_next_obs.append(no_b)

        joint_obs = torch.cat(all_obs, dim=-1)              # (N, B*obs)
        joint_actions = torch.cat(all_actions, dim=-1)       # (N, B*action)
        joint_obs_actions = torch.cat([joint_obs, joint_actions], dim=-1)

        # Target actions for next state
        with torch.no_grad():
            next_target_actions = []
            for b in range(self.B):
                next_target_actions.append(self.actors_target[b](per_next_obs[b]))
            joint_next_actions = torch.cat(next_target_actions, dim=-1)
            joint_next_obs = torch.cat(all_next_obs, dim=-1)
            joint_next = torch.cat([joint_next_obs, joint_next_actions], dim=-1)

        dones_t = torch.tensor(batch["dones"], dtype=torch.float32).to(self.device)

        for b in range(self.B):
            rewards_b = torch.tensor(batch["rewards"][:, b], dtype=torch.float32).to(self.device)

            # Critic update
            with torch.no_grad():
                q_next = self.critics_target[b](joint_next).squeeze(-1)
                q_target = rewards_b + self.gamma * (1 - dones_t) * q_next

            q_pred = self.critics[b](joint_obs_actions).squeeze(-1)
            critic_loss = F.mse_loss(q_pred, q_target)

            self.critic_opts[b].zero_grad()
            critic_loss.backward()
            self.critic_opts[b].step()

            # Actor update: replace agent b's action with current policy output
            new_actions = list(all_actions)
            new_actions[b] = self.actors[b](per_obs[b])
            joint_new_actions = torch.cat(new_actions, dim=-1)
            joint_for_actor = torch.cat([joint_obs, joint_new_actions], dim=-1)
            actor_loss = -self.critics[b](joint_for_actor).mean()

            self.actor_opts[b].zero_grad()
            actor_loss.backward()
            self.actor_opts[b].step()

            losses["actor_loss"] += actor_loss.item()
            losses["critic_loss"] += critic_loss.item()

        # Soft target updates
        for b in range(self.B):
            for p, tp in zip(self.actors[b].parameters(), self.actors_target[b].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critics[b].parameters(), self.critics_target[b].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        losses["actor_loss"] /= self.B
        losses["critic_loss"] /= self.B
        return losses

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actors.state_dict(), os.path.join(path, "maddpg_actors.pt"))
        torch.save(self.critics.state_dict(), os.path.join(path, "maddpg_critics.pt"))

    def load(self, path: str) -> None:
        self.actors.load_state_dict(torch.load(os.path.join(path, "maddpg_actors.pt")))
        self.critics.load_state_dict(torch.load(os.path.join(path, "maddpg_critics.pt")))


# ==========================================================================
# MARLISA – Sequential SAC (buildings take turns)
# ==========================================================================

class MARLISAAgent:
    """Multi-Agent RL with Iterative Sequential Action Selection (MARLISA).

    Each building selects its action sequentially, conditioned on previous
    buildings' chosen actions in the current step.  Uses per-building SAC
    with augmented observations that include previous agents' actions.
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

        # Augmented input: own obs + previous buildings' actions
        # Building i sees obs_dim + i * action_dim inputs
        self.policies = nn.ModuleList()
        self.q1_nets = nn.ModuleList()
        self.q2_nets = nn.ModuleList()
        for i in range(num_buildings):
            aug_dim = obs_dim + i * action_dim
            self.policies.append(_SACPolicy(aug_dim, hidden_dim, action_dim))
            self.q1_nets.append(_SACNet(aug_dim + action_dim, hidden_dim, 1))
            self.q2_nets.append(_SACNet(aug_dim + action_dim, hidden_dim, 1))

        self.policies = self.policies.to(self.device)
        self.q1_nets = self.q1_nets.to(self.device)
        self.q2_nets = self.q2_nets.to(self.device)
        self.q1_target = copy.deepcopy(self.q1_nets).to(self.device)
        self.q2_target = copy.deepcopy(self.q2_nets).to(self.device)

        self.policy_opts = [
            optim.Adam(self.policies[i].parameters(), lr=lr) for i in range(num_buildings)
        ]
        self.q_opts = [
            optim.Adam(
                list(self.q1_nets[i].parameters()) + list(self.q2_nets[i].parameters()), lr=lr
            ) for i in range(num_buildings)
        ]

    def _augment_obs(self, obs: np.ndarray, prev_actions: np.ndarray) -> np.ndarray:
        """Concatenate obs with flattened previous-agent actions."""
        return np.concatenate([obs, prev_actions.flatten()])

    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        for i in range(self.B):
            prev_a = actions[:i].flatten() if i > 0 else np.array([], dtype=np.float32)
            aug = self._augment_obs(obs_list[i], prev_a)
            x = torch.tensor(aug, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if explore:
                    a, _ = self.policies[i].sample(x)
                else:
                    mean, _ = self.policies[i](x)
                    a = torch.tanh(mean)
            actions[i] = np.array(a.squeeze(0).detach().tolist(), dtype=np.float32)
        return actions

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}

        N = len(batch["obs"])
        losses = {"actor_loss": 0.0, "critic_loss": 0.0}

        for b in range(self.B):
            # Build augmented observations with previous agents' actions
            obs_aug = []
            next_obs_aug = []
            for n in range(N):
                prev_a = batch["actions"][n, :b].flatten() if b > 0 else np.array([], dtype=np.float32)
                obs_aug.append(self._augment_obs(batch["obs"][n][b], prev_a))
                next_obs_aug.append(self._augment_obs(batch["next_obs"][n][b], prev_a))

            obs_b = torch.tensor(np.array(obs_aug), dtype=torch.float32).to(self.device)
            next_obs_b = torch.tensor(np.array(next_obs_aug), dtype=torch.float32).to(self.device)
            actions_b = torch.tensor(batch["actions"][:, b, :], dtype=torch.float32).to(self.device)
            rewards_b = torch.tensor(batch["rewards"][:, b], dtype=torch.float32).to(self.device)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32).to(self.device)

            # Q update
            with torch.no_grad():
                next_a, next_lp = self.policies[b].sample(next_obs_b)
                q1n = self.q1_target[b](torch.cat([next_obs_b, next_a], -1)).squeeze(-1)
                q2n = self.q2_target[b](torch.cat([next_obs_b, next_a], -1)).squeeze(-1)
                q_tgt = rewards_b + self.gamma * (1 - dones_b) * (
                    torch.min(q1n, q2n) - self.alpha_ent * next_lp
                )

            q1p = self.q1_nets[b](torch.cat([obs_b, actions_b], -1)).squeeze(-1)
            q2p = self.q2_nets[b](torch.cat([obs_b, actions_b], -1)).squeeze(-1)
            q_loss = F.mse_loss(q1p, q_tgt) + F.mse_loss(q2p, q_tgt)
            self.q_opts[b].zero_grad()
            q_loss.backward()
            self.q_opts[b].step()

            # Policy update
            a_new, lp_new = self.policies[b].sample(obs_b)
            q1n_ = self.q1_nets[b](torch.cat([obs_b, a_new], -1)).squeeze(-1)
            q2n_ = self.q2_nets[b](torch.cat([obs_b, a_new], -1)).squeeze(-1)
            p_loss = (self.alpha_ent * lp_new - torch.min(q1n_, q2n_)).mean()
            self.policy_opts[b].zero_grad()
            p_loss.backward()
            self.policy_opts[b].step()

            # Target update
            for p, tp in zip(self.q1_nets[b].parameters(), self.q1_target[b].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.q2_nets[b].parameters(), self.q2_target[b].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            losses["actor_loss"] += p_loss.item()
            losses["critic_loss"] += q_loss.item()

        losses["actor_loss"] /= self.B
        losses["critic_loss"] /= self.B
        return losses

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.policies.state_dict(), os.path.join(path, "marlisa_policies.pt"))
        torch.save(self.q1_nets.state_dict(), os.path.join(path, "marlisa_q1.pt"))

    def load(self, path: str) -> None:
        self.policies.load_state_dict(torch.load(os.path.join(path, "marlisa_policies.pt")))
        self.q1_nets.load_state_dict(torch.load(os.path.join(path, "marlisa_q1.pt")))


# ==========================================================================
# MADCQ – Multi-Agent Deep Constrained Q-learning
# ==========================================================================

class MADCQAgent:
    """Independent DQN-style agents with heuristic SOC clipping (no QP).

    Uses discrete action binning mapped back to continuous space,
    with a simple constraint: clip actions that would violate SOC bounds.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int = 3,
        hidden_dim: int = 128,
        n_bins: int = 11,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        eta: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.B = num_buildings
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.gamma = gamma
        self.tau = tau
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.eta = eta
        self.device = torch.device(device)

        # Per-building Q-networks over discretised action bins
        total_actions = n_bins ** action_dim
        self.q_nets = nn.ModuleList([
            _SACNet(obs_dim, hidden_dim, total_actions) for _ in range(num_buildings)
        ]).to(self.device)
        self.q_targets = copy.deepcopy(self.q_nets).to(self.device)

        self.optimizers = [
            optim.Adam(self.q_nets[i].parameters(), lr=lr) for i in range(num_buildings)
        ]

        # Precompute discrete action grid
        bins = np.linspace(-1.0, 1.0, n_bins)
        grids = np.meshgrid(*[bins] * action_dim, indexing="ij")
        self._action_table = np.stack([g.ravel() for g in grids], axis=-1).astype(np.float32)
        self._epsilon = 0.1

    def _soc_clamp(self, action: np.ndarray, soc: float) -> np.ndarray:
        """Clamp electrical storage action to stay within SOC limits."""
        a = action.copy()
        elec = a[1]
        new_soc = soc + self.eta * elec
        if new_soc > self.soc_max:
            a[1] = max((self.soc_max - soc) / self.eta, -1.0)
        elif new_soc < self.soc_min:
            a[1] = min((self.soc_min - soc) / self.eta, 1.0)
        return a

    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            if explore and np.random.random() < self._epsilon:
                idx = np.random.randint(len(self._action_table))
            else:
                with torch.no_grad():
                    q_vals = self.q_nets[i](x)
                idx = int(q_vals.argmax(dim=-1).item())
            a = self._action_table[idx]
            soc = float(obs[_IDX_SOC_ELEC])
            actions[i] = self._soc_clamp(a, soc)
        return actions

    def _action_to_idx(self, action: np.ndarray) -> int:
        """Map continuous action to nearest discrete index."""
        dists = np.linalg.norm(self._action_table - action, axis=-1)
        return int(np.argmin(dists))

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
            rewards_b = torch.tensor(batch["rewards"][:, b], dtype=torch.float32).to(self.device)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32).to(self.device)

            # Map actions to indices
            action_indices = torch.tensor(
                [self._action_to_idx(batch["actions"][n, b]) for n in range(N)],
                dtype=torch.long,
            ).to(self.device)

            # Q-learning update
            q_vals = self.q_nets[b](obs_b)
            q_pred = q_vals.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = self.q_targets[b](next_obs_b).max(dim=-1)[0]
                q_target = rewards_b + self.gamma * (1 - dones_b) * q_next

            q_loss = F.mse_loss(q_pred, q_target)
            self.optimizers[b].zero_grad()
            q_loss.backward()
            self.optimizers[b].step()

            # Target update
            for p, tp in zip(self.q_nets[b].parameters(), self.q_targets[b].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            losses["critic_loss"] += q_loss.item()

        losses["critic_loss"] /= self.B
        return losses

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.q_nets.state_dict(), os.path.join(path, "madcq_q.pt"))

    def load(self, path: str) -> None:
        self.q_nets.load_state_dict(torch.load(os.path.join(path, "madcq_q.pt")))


# ==========================================================================
# MetaEMS – Simplified MAML wrapper
# ==========================================================================

class MetaEMSAgent:
    """Model-Agnostic Meta-Learning (MAML) wrapper around independent SAC.

    Performs an inner-loop adaptation step on a fresh batch before
    the outer-loop policy update, allowing fast adaptation to new
    building/weather conditions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int = 3,
        hidden_dim: int = 128,
        lr_inner: float = 1e-3,
        lr_outer: float = 3e-4,
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
        self.lr_inner = lr_inner
        self.device = torch.device(device)

        # Base SAC agent which we meta-learn over
        self._base = SingleAgentSAC(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_buildings=num_buildings,
            hidden_dim=hidden_dim,
            lr=lr_outer,
            gamma=gamma,
            tau=tau,
            alpha_entropy=alpha_entropy,
            device=device,
        )

    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> np.ndarray:
        return self._base.select_action(obs_list, history, explore)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """MAML-style update: clone → inner adapt → outer update."""
        if len(batch["obs"]) == 0:
            return {}

        N = len(batch["obs"])
        half = max(1, N // 2)

        # Split batch: support set for inner loop, query set for outer loop
        support = {k: v[:half] if hasattr(v, '__getitem__') else v for k, v in batch.items()}
        query = {k: v[half:] if hasattr(v, '__getitem__') else v for k, v in batch.items()}

        # Save current params
        saved_states = {
            "policies": copy.deepcopy(self._base.policies.state_dict()),
            "q1": copy.deepcopy(self._base.q1_nets.state_dict()),
            "q2": copy.deepcopy(self._base.q2_nets.state_dict()),
        }

        # Inner loop: one gradient step on support set
        self._base.update(support)

        # Outer loop: update on query set with adapted params
        losses = self._base.update(query)

        # MAML: interpolate between adapted and original params (first-order approx)
        beta = 0.5
        for name, param in self._base.policies.named_parameters():
            if name in saved_states["policies"]:
                param.data.copy_(
                    beta * param.data + (1 - beta) * saved_states["policies"][name].to(self.device)
                )

        return losses

    def save(self, path: str) -> None:
        self._base.save(path)

    def load(self, path: str) -> None:
        self._base.load(path)
