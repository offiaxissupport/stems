from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from stems.config import GraphConfig


class BuildingGraph:
    def __init__(
        self,
        num_buildings: int,
        positions: np.ndarray,
        features: np.ndarray,
        config: Optional[GraphConfig] = None,
    ) -> None:
        self.num_buildings = num_buildings
        self.positions = np.asarray(positions, dtype=np.float32)
        self.features = np.asarray(features, dtype=np.float32)
        self.config = config or GraphConfig()
        self._adj: Optional[torch.Tensor] = None


    def compute_edge_weights(self) -> torch.Tensor:
        B = self.num_buildings
        cfg = self.config

        diff_pos = self.positions[:, None, :] - self.positions[None, :, :]
        d_sq = (diff_pos ** 2).sum(axis=-1)

        diff_feat = self.features[:, None, :] - self.features[None, :, :]
        f_sq = (diff_feat ** 2).sum(axis=-1)

        w = (
            cfg.alpha * np.exp(-d_sq / (2.0 * cfg.sigma_d ** 2))
            + cfg.beta  * np.exp(-f_sq / (2.0 * cfg.sigma_f ** 2))
        )

        np.fill_diagonal(w, 0.0)

        self._adj = torch.tensor(w, dtype=torch.float32)
        return self._adj


    def get_node_features(self, observations: List[np.ndarray]) -> torch.Tensor:
        x = np.stack(observations, axis=0).astype(np.float32)
        return torch.tensor(x, dtype=torch.float32)


    @property
    def adj(self) -> torch.Tensor:
        if self._adj is None:
            self.compute_edge_weights()
        return self._adj
