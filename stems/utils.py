from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var",  torch.ones(dim,  dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x_flat = x.reshape(-1, x.shape[-1]).float()
        n = x_flat.shape[0]
        if n == 0:
            return
        batch_mean = x_flat.mean(dim=0)
        batch_var  = x_flat.var(dim=0, unbiased=False)

        total = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / total)
        m_a = self.var * self.count
        m_b = batch_var * n
        m2  = m_a + m_b + delta.pow(2) * self.count * n / total
        new_var = m2 / total

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + self.epsilon)


class EpisodeBuffer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._obs: List[List[np.ndarray]] = []
        self._actions: List[np.ndarray] = []
        self._raw_actions: List[np.ndarray] = []
        self._safe_actions: List[np.ndarray] = []
        self._rewards: List[List[float]] = []
        self._next_obs: List[List[np.ndarray]] = []
        self._dones: List[bool] = []
        self._history: List[np.ndarray] = []
        self._next_history: List[np.ndarray] = []
        self._constraint_costs: List[np.ndarray] = []
        self._pre_tanh: List[np.ndarray] = []
        self._behaviour_log_probs: List[np.ndarray] = []

    def add(
        self,
        obs: List[np.ndarray],
        actions: np.ndarray,
        rewards: List[float],
        next_obs: List[np.ndarray],
        done: bool,
        history: Optional[np.ndarray] = None,
        next_history: Optional[np.ndarray] = None,
        raw_actions: Optional[np.ndarray] = None,
        safe_actions: Optional[np.ndarray] = None,
        constraint_costs: Optional[np.ndarray] = None,
        pre_tanh: Optional[np.ndarray] = None,
        behaviour_log_probs: Optional[np.ndarray] = None,
    ) -> None:
        self._obs.append(obs)
        self._actions.append(actions)
        self._raw_actions.append(raw_actions if raw_actions is not None else actions)
        self._safe_actions.append(safe_actions if safe_actions is not None else
                                  (raw_actions if raw_actions is not None else actions))
        self._rewards.append(rewards)
        self._next_obs.append(next_obs)
        self._dones.append(done)
        if history is not None:
            self._history.append(history)
        if next_history is not None:
            self._next_history.append(next_history)
        if constraint_costs is not None:
            self._constraint_costs.append(constraint_costs)
        if pre_tanh is not None:
            self._pre_tanh.append(np.asarray(pre_tanh, dtype=np.float32))
        if behaviour_log_probs is not None:
            self._behaviour_log_probs.append(np.asarray(behaviour_log_probs, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._obs)

    def get_batch(self) -> Dict[str, Any]:
        batch: Dict[str, Any] = {
            "obs": self._obs,
            "actions": np.array(self._actions),
            "raw_actions": np.array(self._raw_actions),
            "safe_actions": np.array(self._safe_actions),
            "rewards": np.array(self._rewards),
            "next_obs": self._next_obs,
            "dones": np.array(self._dones, dtype=np.float32),
        }
        if self._history:
            batch["history"] = np.array(self._history)
            batch["next_history"] = np.array(self._next_history)
        if self._constraint_costs:
            batch["constraint_costs"] = np.array(self._constraint_costs)
        if self._pre_tanh:
            batch["pre_tanh"] = np.array(self._pre_tanh)
            batch["behaviour_log_probs"] = np.array(self._behaviour_log_probs)
        return batch


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._obs: deque = deque(maxlen=capacity)
        self._actions: deque = deque(maxlen=capacity)
        self._raw_actions: deque = deque(maxlen=capacity)
        self._rewards: deque = deque(maxlen=capacity)
        self._next_obs: deque = deque(maxlen=capacity)
        self._dones: deque = deque(maxlen=capacity)
        self._history: deque = deque(maxlen=capacity)
        self._next_history: deque = deque(maxlen=capacity)

    def add(
        self,
        obs: List[np.ndarray],
        actions: np.ndarray,
        rewards: List[float],
        next_obs: List[np.ndarray],
        done: bool,
        history: Optional[np.ndarray] = None,
        next_history: Optional[np.ndarray] = None,
        raw_actions: Optional[np.ndarray] = None,
    ) -> None:
        self._obs.append(obs)
        self._actions.append(actions)
        self._raw_actions.append(raw_actions)
        self._rewards.append(rewards)
        self._next_obs.append(next_obs)
        self._dones.append(done)
        self._history.append(history)
        self._next_history.append(next_history)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        batch: Dict[str, Any] = {
            "obs": [self._obs[i] for i in indices],
            "actions": np.array([self._actions[i] for i in indices]),
            "rewards": np.array([self._rewards[i] for i in indices]),
            "next_obs": [self._next_obs[i] for i in indices],
            "dones": np.array([self._dones[i] for i in indices], dtype=np.float32),
        }
        if self._raw_actions[indices[0]] is not None:
            batch["raw_actions"] = np.array([self._raw_actions[i] for i in indices])
        if self._history[indices[0]] is not None:
            batch["history"] = np.array([self._history[i] for i in indices])
            batch["next_history"] = np.array([self._next_history[i] for i in indices])
        return batch

    def __len__(self) -> int:
        return len(self._obs)

    @property
    def is_ready(self) -> bool:
        return len(self) > 0


class HistoryBuffer:
    def __init__(self, num_buildings: int, obs_dim: int, window_size: int) -> None:
        self.num_buildings = num_buildings
        self.obs_dim = obs_dim
        self.window_size = window_size
        self._buffer = np.zeros((num_buildings, window_size, obs_dim), dtype=np.float32)

    def update(self, obs_list: List[np.ndarray]) -> None:
        new_obs = np.array(obs_list, dtype=np.float32)
        self._buffer = np.roll(self._buffer, shift=-1, axis=1)
        self._buffer[:, -1, :] = new_obs

    def get(self) -> np.ndarray:
        return self._buffer.copy()

    def reset(self) -> None:
        self._buffer[:] = 0.0


def normalize_obs(
    obs: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if mean is None:
        mean = obs.mean(axis=-1, keepdims=True)
    if std is None:
        std = obs.std(axis=-1, keepdims=True) + 1e-8
    return (obs - mean) / (std + 1e-8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
