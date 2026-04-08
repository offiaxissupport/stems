"""
STEMS Agent: Actor, Critic, and STEMSAgent (Eq 22-26, Algorithm 2).

Actor  π_θ(r_i) : policy network – maps representation r_i to action (Eq 22)
Critic V_φ(r_i) : value network  – estimates state value V(r_i)        (Eq 23)

STEMSAgent orchestrates the encoder, actor/critic, and CBF shield for all B
buildings.  Training follows the advantage actor-critic update in Eq 24-26.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stems.config import STEMSConfig
from stems.encoder import STEncoder
from stems.cbf import CBFShield
from stems.graph import BuildingGraph


# --------------------------------------------------------------------------
# Actor network (Eq 22)
# --------------------------------------------------------------------------

class Actor(nn.Module):
    r"""Policy π_θ(r_i).

    Eq 22:  a_i = Tanh(W_2 * ReLU(W_1 * r_i + b_1) + b_2)
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # Initialise final layer with small weights for conservative initial policy
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


# --------------------------------------------------------------------------
# Critic network (Eq 23)
# --------------------------------------------------------------------------

class Critic(nn.Module):
    r"""State-value V_φ(r_i).

    Eq 23:  V_i = W_2 * ReLU(W_1 * r_i + b_1) + b_2
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r).squeeze(-1)   # (B,)


# --------------------------------------------------------------------------
# STEMSAgent
# --------------------------------------------------------------------------

class STEMSAgent:
    """Coordinates the encoder, per-building actors/critics, and CBF shield.

    Parameters
    ----------
    obs_dim      : int  – observation dimension
    action_dim   : int  – action dimension per building
    num_buildings: int  – number of buildings B
    building_graph: BuildingGraph – pre-built graph object
    config       : STEMSConfig   – full hyperparameter configuration
    use_cbf      : bool – whether to apply the CBF safety shield
    device       : str  – 'cpu' or 'cuda'
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int,
        building_graph: BuildingGraph,
        config: Optional[STEMSConfig] = None,
        use_cbf: bool = True,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.B = num_buildings
        self.graph = building_graph
        self.cfg = config or STEMSConfig()
        self.use_cbf = use_cbf
        self.device = torch.device(device)

        # Compute and cache adjacency matrix
        self.adj = self.graph.compute_edge_weights().to(self.device)

        # Shared spatial-temporal encoder
        self.encoder = STEncoder(
            obs_dim=obs_dim,
            spatial_dim=self.cfg.gcn.hidden_dim,
            temporal_dim=self.cfg.transformer.embed_dim,
            output_dim=self.cfg.fusion.output_dim,
            gcn_num_layers=self.cfg.gcn.num_layers,
            num_heads=self.cfg.transformer.num_heads,
            window_size=self.cfg.transformer.window_size,
        ).to(self.device)

        repr_dim = self.cfg.fusion.output_dim

        # Per-building actors and critics
        self.actors = nn.ModuleList([
            Actor(repr_dim, self.cfg.actor_critic.hidden_dim, action_dim)
            for _ in range(self.B)
        ]).to(self.device)

        self.critics = nn.ModuleList([
            Critic(repr_dim, self.cfg.actor_critic.hidden_dim)
            for _ in range(self.B)
        ]).to(self.device)

        lr = self.cfg.actor_critic.lr

        # Optimisers
        encoder_params = list(self.encoder.parameters())
        actor_params = list(self.actors.parameters())
        critic_params = list(self.critics.parameters())

        self.actor_optimizer = optim.Adam(encoder_params + actor_params, lr=lr)
        self.critic_optimizer = optim.Adam(encoder_params + critic_params, lr=lr)

        # CBF shield
        self.cbf = CBFShield(
            config=self.cfg.cbf,
            num_buildings=self.B,
        )

        # Training counters
        self._update_step = 0

    # ------------------------------------------------------------------
    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        """Encode observations → actor → add noise → CBF project → scale.

        Parameters
        ----------
        obs_list : List of B arrays, shape (obs_dim,)
        history  : np.ndarray, shape (B, T, obs_dim)
        explore  : bool – add exploration noise if True

        Returns
        -------
        actions : np.ndarray, shape (B, action_dim)  in [-1, 1]
        """
        self.encoder.eval()
        for actor in self.actors:
            actor.eval()

        with torch.no_grad():
            x = torch.tensor(
                np.stack(obs_list, axis=0), dtype=torch.float32
            ).to(self.device)                                              # (B, obs_dim)
            h = torch.tensor(history, dtype=torch.float32).to(self.device)  # (B, T, obs_dim)

            repr_mat = self.encoder(x, self.adj, h)                       # (B, repr_dim)

            actions_list = []
            for i in range(self.B):
                r_i = repr_mat[i].unsqueeze(0)                            # (1, repr_dim)
                a_i = self.actors[i](r_i).squeeze(0).cpu().numpy()        # (action_dim,)
                actions_list.append(a_i)

        actions = np.stack(actions_list, axis=0)   # (B, action_dim)

        # Exploration noise (Gaussian, σ = exploration_noise)
        if explore:
            noise = np.random.normal(
                0.0, self.cfg.training.exploration_noise, size=actions.shape
            ).astype(np.float32)
            actions = np.clip(actions + noise, -1.0, 1.0)

        # CBF safety projection (Algorithm 1)
        if self.use_cbf:
            actions = self.cbf.project(actions, obs_list)

        # Scale actions (paper uses 0.5 scaling for stability)
        actions = actions * self.cfg.training.action_scale

        return actions.astype(np.float32)

    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One gradient update step using a sampled mini-batch.

        Implements Eq 24-26 (advantage actor-critic update).

        Parameters
        ----------
        batch : dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'

        Returns
        -------
        dict with 'actor_loss' and 'critic_loss'
        """
        self.encoder.train()
        for actor in self.actors:
            actor.train()
        for critic in self.critics:
            critic.train()

        gamma = self.cfg.actor_critic.gamma

        # Stack observations across batch
        obs_batch = batch["obs"]             # list of N transitions, each is list of B arrays
        next_obs_batch = batch["next_obs"]
        rewards_batch = batch["rewards"]     # (N, B)
        dones_batch = batch["dones"]         # (N,)

        N = len(obs_batch)
        B = self.B

        # Build tensors (N*B, obs_dim)
        obs_tensor = torch.zeros(N, B, self.obs_dim, device=self.device)
        next_obs_tensor = torch.zeros(N, B, self.obs_dim, device=self.device)

        for n in range(N):
            for b in range(B):
                obs_tensor[n, b] = torch.tensor(
                    obs_batch[n][b], dtype=torch.float32
                )
                next_obs_tensor[n, b] = torch.tensor(
                    next_obs_batch[n][b], dtype=torch.float32
                )

        rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32).to(self.device)  # (N, B)
        dones_tensor = torch.tensor(dones_batch, dtype=torch.float32).to(self.device)     # (N,)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        T_win = self.cfg.transformer.window_size

        # Use the actual building adjacency during training to learn spatial coordination.
        # Each batch_forward call creates a fresh computation graph, so separate
        # critic and actor backward passes do not interfere.
        adj_id   = self.adj
        obs_nb   = obs_tensor.view(N, B, self.obs_dim)
        next_nb  = next_obs_tensor.view(N, B, self.obs_dim)
        hist_nb      = obs_nb.unsqueeze(2).expand(N, B, T_win, self.obs_dim).contiguous()
        next_hist_nb = next_nb.unsqueeze(2).expand(N, B, T_win, self.obs_dim).contiguous()

        # Critic update: compute all B critic losses, sum, then single backward.
        repr_nb = self.encoder.batch_forward(obs_nb, adj_id, hist_nb)           # (N, B, repr_dim)
        with torch.no_grad():
            repr_next_nb = self.encoder.batch_forward(next_nb, adj_id, next_hist_nb)

        critic_loss_total = torch.tensor(0.0, device=self.device)
        for b in range(B):
            repr_b      = repr_nb[:, b, :]
            repr_b_next = repr_next_nb[:, b, :]
            rewards_b   = rewards_tensor[:, b]
            values      = self.critics[b](repr_b)
            next_values = self.critics[b](repr_b_next)
            targets     = rewards_b + gamma * next_values * (1.0 - dones_tensor)
            c_loss      = F.mse_loss(values, targets.detach())
            critic_loss_total = critic_loss_total + c_loss
            total_critic_loss += c_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.critics.parameters()),
            max_norm=1.0,
        )
        self.critic_optimizer.step()

        # Actor update: fresh encode, accumulate all B actor losses, single backward.
        repr_nb_a = self.encoder.batch_forward(obs_nb, adj_id, hist_nb)
        with torch.no_grad():
            repr_next_nb_a = self.encoder.batch_forward(next_nb, adj_id, next_hist_nb)

        actor_loss_total = torch.tensor(0.0, device=self.device)
        for b in range(B):
            repr_b    = repr_nb_a[:, b, :]
            rewards_b = rewards_tensor[:, b]
            v_b   = self.critics[b](repr_b.detach())
            nv_b  = self.critics[b](repr_next_nb_a[:, b, :].detach())
            t_b   = rewards_b + gamma * nv_b * (1.0 - dones_tensor)
            adv   = (t_b - v_b).detach()
            actions_pred = self.actors[b](repr_b)
            a_loss = -(adv.unsqueeze(-1) * actions_pred).mean()
            actor_loss_total = actor_loss_total + a_loss
            total_actor_loss += a_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss_total.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actors.parameters()),
            max_norm=1.0,
        )
        self.actor_optimizer.step()

        self._update_step += 1

        return {
            "actor_loss": total_actor_loss / B,
            "critic_loss": total_critic_loss / B,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights to *path* (directory)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        torch.save(self.actors.state_dict(), os.path.join(path, "actors.pt"))
        torch.save(self.critics.state_dict(), os.path.join(path, "critics.pt"))

    def load(self, path: str) -> None:
        """Load model weights from *path* (directory)."""
        map_loc = self.device
        self.encoder.load_state_dict(
            torch.load(os.path.join(path, "encoder.pt"), map_location=map_loc)
        )
        self.actors.load_state_dict(
            torch.load(os.path.join(path, "actors.pt"), map_location=map_loc)
        )
        self.critics.load_state_dict(
            torch.load(os.path.join(path, "critics.pt"), map_location=map_loc)
        )
