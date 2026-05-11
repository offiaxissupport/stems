"""
Utility classes and helpers for STEMS.

Provides:
    ReplayBuffer      – circular experience replay store (for off-policy baselines)
    EpisodeBuffer     – collects one full episode for on-policy training (Algorithm 2)
    HistoryBuffer     – rolling observation window for the Transformer
    RunningNormalizer – online Welford mean/std normalizer (nn.Module)
    normalize_obs     – zero-mean unit-variance observation normalisation
    set_seed          – global random seed helper
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RunningNormalizer (online zero-mean unit-std per feature)
# ---------------------------------------------------------------------------

class RunningNormalizer(nn.Module):
    """Online running mean/std normalization using Welford's algorithm.

    Implemented as an nn.Module so its buffers are moved with .to(device),
    included in state_dict for save/load, and shared cleanly between
    select_action and update.

    Parameters
    ----------
    dim     : int   – feature dimension
    epsilon : float – small constant for numerical stability
    """

    def __init__(self, dim: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        # Buffers are not parameters – they don't receive gradients.
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var",  torch.ones(dim,  dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running stats from a batch x of shape (..., dim).

        Uses Welford's parallel algorithm for numerically stable online variance.
        """
        x_flat = x.reshape(-1, x.shape[-1]).float()   # (N, dim)
        n = x_flat.shape[0]
        if n == 0:
            return
        batch_mean = x_flat.mean(dim=0)
        batch_var  = x_flat.var(dim=0, unbiased=False)

        total = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / total)
        # Chan et al. parallel formula for combined variance
        m_a = self.var * self.count
        m_b = batch_var * n
        m2  = m_a + m_b + delta.pow(2) * self.count * n / total
        new_var = m2 / total

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x to zero mean, unit std (element-wise per feature)."""
        return (x - self.mean) / (self.var.sqrt() + self.epsilon)


# ---------------------------------------------------------------------------
# EpisodeBuffer (on-policy, Algorithm 2)
# ---------------------------------------------------------------------------

class EpisodeBuffer:
    """Collects a single full episode trajectory for on-policy training.

    After collecting a complete episode, call ``get_batch()`` to retrieve the
    full trajectory as a training batch, then ``reset()`` for the next episode.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._obs: List[List[np.ndarray]] = []
        self._actions: List[np.ndarray] = []
        self._raw_actions: List[np.ndarray] = []
        self._safe_actions: List[np.ndarray] = []   # post-CBF actions in [-1,1], for Eq 24
        self._rewards: List[List[float]] = []
        self._next_obs: List[List[np.ndarray]] = []
        self._dones: List[bool] = []
        self._history: List[np.ndarray] = []
        self._next_history: List[np.ndarray] = []
        # Constraint cost signals (B, K) per step – k=0 SOC, k=1 building power, k=2 grid
        self._constraint_costs: List[np.ndarray] = []

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

    def __len__(self) -> int:
        return len(self._obs)

    def get_batch(self) -> Dict[str, Any]:
        """Return the full episode as a batch dict (same format as ReplayBuffer.sample)."""
        batch: Dict[str, Any] = {
            "obs": self._obs,
            "actions": np.array(self._actions),
            "raw_actions": np.array(self._raw_actions),
            "safe_actions": np.array(self._safe_actions),  # post-CBF, for Eq 24
            "rewards": np.array(self._rewards),
            "next_obs": self._next_obs,
            "dones": np.array(self._dones, dtype=np.float32),
        }
        if self._history:
            batch["history"] = np.array(self._history)
            batch["next_history"] = np.array(self._next_history)
        if self._constraint_costs:
            # shape (N, B, K): timestep × building × constraint
            batch["constraint_costs"] = np.array(self._constraint_costs)
        return batch


# ---------------------------------------------------------------------------
# ReplayBuffer (off-policy, for baselines)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer.

    Stores transitions as plain Python lists to avoid the memory overhead of
    pre-allocated NumPy arrays for variable-length multi-agent observations.
    """

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

    # ------------------------------------------------------------------
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
        """Push one transition into the buffer."""
        self._obs.append(obs)
        self._actions.append(actions)
        self._raw_actions.append(raw_actions)
        self._rewards.append(rewards)
        self._next_obs.append(next_obs)
        self._dones.append(done)
        self._history.append(history)
        self._next_history.append(next_history)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Return a random mini-batch as a dict of lists / arrays."""
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

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._obs)

    @property
    def is_ready(self) -> bool:
        return len(self) > 0


# ---------------------------------------------------------------------------
# HistoryBuffer
# ---------------------------------------------------------------------------

class HistoryBuffer:
    """Maintains a rolling window of observations for the Temporal Transformer.

    Shape convention:  (num_buildings, window_size, obs_dim)
    """

    def __init__(self, num_buildings: int, obs_dim: int, window_size: int) -> None:
        self.num_buildings = num_buildings
        self.obs_dim = obs_dim
        self.window_size = window_size
        # Initialise with zeros
        self._buffer = np.zeros((num_buildings, window_size, obs_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    def update(self, obs_list: List[np.ndarray]) -> None:
        """Shift window left by one step and append the newest observations."""
        new_obs = np.array(obs_list, dtype=np.float32)   # (B, obs_dim)
        # Roll along the time axis: drop oldest, append newest
        self._buffer = np.roll(self._buffer, shift=-1, axis=1)
        self._buffer[:, -1, :] = new_obs

    # ------------------------------------------------------------------
    def get(self) -> np.ndarray:
        """Return the current window; shape (num_buildings, window_size, obs_dim)."""
        return self._buffer.copy()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero-fill the buffer (call at the start of each episode)."""
        self._buffer[:] = 0.0


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalize_obs(
    obs: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return zero-mean, unit-variance observation.

    If *mean* / *std* are not provided the function normalises over the
    provided array itself (useful for single-step normalisation).
    """
    obs = np.asarray(obs, dtype=np.float32)
    if mean is None:
        mean = obs.mean(axis=-1, keepdims=True)
    if std is None:
        std = obs.std(axis=-1, keepdims=True) + 1e-8
    return (obs - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python / NumPy / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
