"""
Utility classes and helpers for STEMS.

Provides:
    ReplayBuffer   – circular experience replay store
    HistoryBuffer  – rolling observation window for the Transformer
    normalize_obs  – zero-mean unit-variance observation normalisation
    set_seed       – global random seed helper
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# ReplayBuffer
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
        self._rewards: deque = deque(maxlen=capacity)
        self._next_obs: deque = deque(maxlen=capacity)
        self._dones: deque = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    def add(
        self,
        obs: List[np.ndarray],
        actions: np.ndarray,
        rewards: List[float],
        next_obs: List[np.ndarray],
        done: bool,
    ) -> None:
        """Push one transition into the buffer."""
        self._obs.append(obs)
        self._actions.append(actions)
        self._rewards.append(rewards)
        self._next_obs.append(next_obs)
        self._dones.append(done)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Return a random mini-batch as a dict of lists / arrays."""
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        return {
            "obs": [self._obs[i] for i in indices],
            "actions": np.array([self._actions[i] for i in indices]),
            "rewards": np.array([self._rewards[i] for i in indices]),
            "next_obs": [self._next_obs[i] for i in indices],
            "dones": np.array([self._dones[i] for i in indices], dtype=np.float32),
        }

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
