#!/usr/bin/env python3
"""
Benchmark: Centralized SAC (pure PyTorch) on CityLearn / STEMS mock.

Trains a single SAC agent that controls ALL buildings jointly with a
flattened obs/action space.  Evaluates using the same MetricsCalculator
as evaluate.py so results are directly comparable for Table I.

No stable-baselines3 or gymnasium dependency — pure PyTorch only.

Usage:
    python -B benchmark_sac.py --episodes 15 --seed 42 --save-dir benchmarks/sac/
    python -B benchmark_sac.py --eval-only --save-dir benchmarks/sac/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.metrics import MetricsCalculator
from stems.utils import set_seed


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class _Actor(nn.Module):
    """Gaussian policy with tanh squashing."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(x)
        std = log_std.exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        action = torch.tanh(z)
        log_prob = (
            -0.5 * ((z - mu) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1)
        return action, log_prob

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self(x)
        return torch.tanh(mu)


class _Critic(nn.Module):
    """Twin Q-network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        inp = obs_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.cap = capacity
        self._ptr = 0
        self._size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self._ptr] = obs
        self.next_obs[self._ptr] = next_obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.cap
        self._size = min(self._size + 1, self.cap)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[idx]),
            "next_obs": torch.tensor(self.next_obs[idx]),
            "actions": torch.tensor(self.actions[idx]),
            "rewards": torch.tensor(self.rewards[idx]),
            "dones": torch.tensor(self.dones[idx]),
        }

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Centralized SAC Agent
# ---------------------------------------------------------------------------

class CentralSAC:
    """Centralized SAC: one actor/critic over all buildings jointly.

    obs_dim_total = B * obs_dim_per_building
    act_dim_total = B * action_dim_per_building
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        tune_alpha: bool = True,
        buffer_size: int = 500_000,
        batch_size: int = 512,
        learning_starts: int = 5000,
        device: str = "cpu",
    ) -> None:
        self.B = num_buildings
        self.obs_dim_per = obs_dim
        self.act_dim_per = action_dim
        self.flat_obs = num_buildings * obs_dim
        self.flat_act = num_buildings * action_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.device = torch.device(device)

        self.actor = _Actor(self.flat_obs, self.flat_act, hidden).to(self.device)
        self.critic = _Critic(self.flat_obs, self.flat_act, hidden).to(self.device)
        self.critic_target = _Critic(self.flat_obs, self.flat_act, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.tune_alpha = tune_alpha
        if tune_alpha:
            self.target_entropy = -float(self.flat_act)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2

        self.replay = _ReplayBuffer(buffer_size, self.flat_obs, self.flat_act)
        self._total_steps = 0

    # ------------------------------------------------------------------
    def _flatten(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(obs_list, axis=0).astype(np.float32)

    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> np.ndarray:
        if explore and self._total_steps < self.learning_starts:
            return np.random.uniform(-1, 1, size=(self.B, self.act_dim_per)).astype(np.float32)
        flat = torch.tensor(self._flatten(obs_list)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if explore:
                a, _ = self.actor.sample(flat)
            else:
                a = self.actor.deterministic(flat)
        return a.squeeze(0).cpu().numpy().reshape(self.B, self.act_dim_per)

    def store(
        self,
        obs_list: List[np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs_list: List[np.ndarray],
        done: bool,
    ) -> None:
        self.replay.add(
            self._flatten(obs_list), action.flatten(),
            reward, self._flatten(next_obs_list), done,
        )
        self._total_steps += 1

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.learning_starts:
            return {}
        batch = self.replay.sample(self.batch_size)
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            next_a, next_log_p = self.actor.sample(next_obs)
            q1_n, q2_n = self.critic_target(next_obs, next_a)
            q_target = rewards + self.gamma * (1 - dones) * (
                torch.min(q1_n, q2_n) - self.alpha * next_log_p
            )

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        a_new, log_p = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, a_new)
        actor_loss = (self.alpha * log_p - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        if self.tune_alpha:
            alpha_loss = -(self.log_alpha * (log_p + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.alpha}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))

    def load(self, path: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pt"), map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(path, "critic.pt"), map_location=self.device)
        )
        self.critic_target.load_state_dict(self.critic.state_dict())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    episodes: int,
    seed: int,
    save_dir: str,
    schema: Optional[str],
) -> None:
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    config = STEMSConfig()
    env = STEMSEnvironment(schema=schema, seed=seed)
    B, obs_dim, act_dim = env.num_buildings, env.obs_dim, env.action_dim

    agent = CentralSAC(obs_dim=obs_dim, action_dim=act_dim, num_buildings=B)

    inner = getattr(env, "_env", None)
    ep_len = getattr(inner, "EPISODE_LEN", 8760)
    total_steps = ep_len * episodes
    print(f"\n[CentralSAC] {episodes} eps × {ep_len} steps = {total_steps} total steps")
    print(f"[CentralSAC] B={B}  obs={obs_dim}  act={act_dim}  save={save_dir}\n")

    history: List[Dict[str, Any]] = []
    t0 = time.time()
    global_step = 0

    for ep in range(1, episodes + 1):
        obs_list, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            actions = agent.select_action(obs_list, explore=True)
            next_obs_list, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            total_r = float(np.sum(rewards))
            ep_reward += total_r
            agent.store(obs_list, actions, total_r, next_obs_list, done)
            agent.update()
            obs_list = next_obs_list
            global_step += 1
            if global_step % 2000 == 0:
                print(f"  step={global_step}  ep_reward_so_far={ep_reward:.1f}")

        row = {"episode": ep, "total_steps": global_step, "episode_reward": ep_reward}
        history.append(row)
        print(f"[CentralSAC] ep={ep:3d}/{episodes}  steps={global_step:7d}  "
              f"reward={ep_reward:.2f}  ({time.time()-t0:.0f}s)")

    agent.save(save_dir)
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[CentralSAC] Saved to {save_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    save_dir: str,
    schema: Optional[str],
    seed: int,
    eval_episodes: int = 1,
) -> Dict[str, float]:
    config = STEMSConfig()
    env = STEMSEnvironment(schema=schema, seed=seed + 100)
    B = env.num_buildings
    agent = CentralSAC(obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B)
    agent.load(save_dir)
    print(f"[CentralSAC eval] Loaded from {save_dir}")

    all_metrics: List[Dict[str, float]] = []
    for ep in range(eval_episodes):
        calc = MetricsCalculator(num_buildings=B, cbf_config=config.cbf)
        obs_list, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            actions = agent.select_action(obs_list, explore=False)
            next_obs_list, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            ep_reward += float(np.sum(rewards))
            calc.add_step(obs_list, actions, next_obs_list)
            obs_list = next_obs_list
        m = calc.compute_all()
        m["episode_reward"] = ep_reward
        all_metrics.append(m)
        print(f"[CentralSAC eval] ep={ep+1}  reward={ep_reward:.2f}  "
              f"cost={m.get('cost', 0):.4f}  viol={m.get('safety_violation_rate', 0):.4f}")

    return {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centralized SAC benchmark (pure PyTorch)")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="benchmarks/sac/")
    p.add_argument("--schema", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.eval_only:
        train(args.episodes, args.seed, args.save_dir, args.schema)
    metrics = evaluate(args.save_dir, args.schema, args.seed, args.eval_episodes)
    print("\n=== CentralSAC Benchmark Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:35s}: {v:.6f}")
    results_path = os.path.join(args.save_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
