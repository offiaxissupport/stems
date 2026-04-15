"""
Adaptive Building Similarity Graph construction (Eq 10-11).

The edge weight between buildings i and j combines:
    - geographic proximity (distance kernel)
    - functional similarity (feature kernel)

    w_ij = alpha * exp(-d_ij² / (2*sigma_d²))
           + beta  * exp(-||f_i - f_j||² / (2*sigma_f²))   (Eq 11)

The resulting adjacency matrix is row-normalised before being consumed by
the GCN layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from stems.config import GraphConfig


class BuildingGraph:
    """Constructs and maintains the building similarity graph.

    Parameters
    ----------
    num_buildings : int
        Number of buildings B.
    positions : np.ndarray, shape (B, 2)
        (x, y) geographic coordinates for each building.
    features : np.ndarray, shape (B, F)
        Functional feature vectors (e.g. load profile summary statistics).
    config : GraphConfig
        Graph hyper-parameters (alpha, beta, sigma_d, sigma_f).
    """

    def __init__(
        self,
        num_buildings: int,
        positions: np.ndarray,
        features: np.ndarray,
        config: Optional[GraphConfig] = None,
    ) -> None:
        self.num_buildings = num_buildings
        self.positions = np.asarray(positions, dtype=np.float32)   # (B, 2)
        self.features = np.asarray(features, dtype=np.float32)     # (B, F)
        self.config = config or GraphConfig()
        self._adj: Optional[torch.Tensor] = None   # cached adjacency matrix

    # ------------------------------------------------------------------
    # Adjacency matrix (Eq 11)
    # ------------------------------------------------------------------

    def compute_edge_weights(self) -> torch.Tensor:
        """Compute and cache the B×B adjacency matrix (Eq 11).

        Returns
        -------
        torch.Tensor, shape (B, B)
            Row-normalised edge-weight matrix.
        """
        B = self.num_buildings
        cfg = self.config

        # Pairwise squared Euclidean distances over geographic positions
        diff_pos = self.positions[:, None, :] - self.positions[None, :, :]  # (B, B, 2)
        d_sq = (diff_pos ** 2).sum(axis=-1)                                  # (B, B)

        # Pairwise squared distances over functional features
        diff_feat = self.features[:, None, :] - self.features[None, :, :]   # (B, B, F)
        f_sq = (diff_feat ** 2).sum(axis=-1)                                 # (B, B)

        # Eq 11
        w = (
            cfg.alpha * np.exp(-d_sq / (2.0 * cfg.sigma_d ** 2))
            + cfg.beta  * np.exp(-f_sq / (2.0 * cfg.sigma_f ** 2))
        )

        # Zero out self-loops; they are re-added via the identity in GCN
        np.fill_diagonal(w, 0.0)

        # Return raw edge weights; symmetric normalisation is handled by the GCN
        self._adj = torch.tensor(w, dtype=torch.float32)
        return self._adj

    # ------------------------------------------------------------------
    # Node feature extraction
    # ------------------------------------------------------------------

    def get_node_features(self, observations: List[np.ndarray]) -> torch.Tensor:
        """Process raw per-building observations into a node feature matrix.

        Parameters
        ----------
        observations : List[np.ndarray]
            List of B arrays each of shape (obs_dim,).

        Returns
        -------
        torch.Tensor, shape (B, obs_dim)
        """
        x = np.stack(observations, axis=0).astype(np.float32)   # (B, obs_dim)
        return torch.tensor(x, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def adj(self) -> torch.Tensor:
        """Return cached adjacency matrix, computing it if necessary."""
        if self._adj is None:
            self.compute_edge_weights()
        return self._adj  # type: ignore[return-value]
