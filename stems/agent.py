"""
STEMS Agent: Actor, Critic, and STEMSAgent (Eq 22-26, Algorithm 2).

Actor  π_θ(r_i) : policy network – maps representation r_i to action (Eq 22)
Critic V_φ(r_i) : value network  – estimates state value V(r_i)        (Eq 23)

STEMSAgent orchestrates the encoder, actor/critic, and CBF shield for all B
buildings.  Training follows the advantage actor-critic update in Eq 24-26.
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

from stems.config import STEMSConfig, LagrangianConfig
from stems.encoder import STEncoder
from stems.cbf import CBFShield, NeuralSafetyFilter
from stems.graph import BuildingGraph


# --------------------------------------------------------------------------
# Actor network (Eq 22) – stochastic SAC policy
# --------------------------------------------------------------------------

class Actor(nn.Module):
    r"""Stochastic SAC policy π_θ(r_i).

    Eq 22:  a_i = Tanh(z),  z ~ N(μ(r_i), σ²(r_i))

    log π(a|r) = log N(z; μ, σ) − Σ_d log(1 − tanh²(z_d))  [tanh correction]

    forward() returns (mean, log_std) of the pre-tanh Gaussian.
    sample()  returns (squashed_action, log_prob) via reparameterisation.
    log_prob_of() evaluates log π(stored_action | repr) using atanh inversion.
    """

    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) of the pre-tanh Gaussian."""
        feat = self.trunk(r)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample + tanh squash with log-prob correction.

        Returns
        -------
        action   : Tensor (*, action_dim) in [-1, 1]
        log_prob : Tensor (*,)
        """
        mean, log_std = self.forward(r)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()                        # reparameterised
        action = torch.tanh(z)
        log_prob = (
            normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(dim=-1)
        return action, log_prob

    def log_prob_of(self, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Log probability of a stored (tanh-squashed) action via atanh inversion.

        Parameters
        ----------
        r : representation (N, repr_dim)
        a : stored squashed action in (-1, 1) (N, action_dim)

        Returns
        -------
        log_prob : (N,)
        """
        mean, log_std = self.forward(r)
        std = log_std.exp()
        z = torch.atanh(a.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        normal = torch.distributions.Normal(mean, std)
        log_prob = (
            normal.log_prob(z) - torch.log(1.0 - a.pow(2) + 1e-6)
        ).sum(dim=-1)
        return log_prob


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
# CostCritic network – k outputs, one per safety constraint
# --------------------------------------------------------------------------

class CostCritic(nn.Module):
    r"""Per-building cost value network V^c_k(r_i).

    Estimates expected cumulative violation cost for each of the k=3 safety
    constraints independently (SOC bounds, per-building power, grid power).

    Output: (batch, NUM_CONSTRAINTS)
    """

    NUM_CONSTRAINTS: int = 3   # h1 (SOC), h2 (building power), h3 (grid power)

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.NUM_CONSTRAINTS),
        )
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)   # (batch, 3)


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

        # Optimisers – one per component (Eq 26: combined encoder gradient)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=lr)

        # Target critics: slow-moving EMA copies for stable Bellman bootstrap.
        # Without target networks the value estimate and its own bootstrap target
        # move together, causing oscillation and divergence.
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        for p in self.target_critics.parameters():
            p.requires_grad_(False)
        self._target_tau: float = 0.005  # polyak averaging rate

        # SAC temperature α (auto-tuning via log_alpha dual variable).
        # Target entropy H* = −|A| (standard SAC heuristic).
        self._target_entropy: float = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Multi-constraint Lagrangian safety (one CostCritic per building, k=3 outputs)
        # Addresses: separate cost signals + cost critics + independent λ per constraint.
        lag_cfg = self.cfg.lagrangian
        self._lag_cfg = lag_cfg
        self.cost_critics = nn.ModuleList([
            CostCritic(repr_dim, self.cfg.actor_critic.hidden_dim)
            for _ in range(self.B)
        ]).to(self.device)
        self.cost_critic_optimizer = optim.Adam(self.cost_critics.parameters(), lr=lr)

        # Lagrangian multipliers λ_k ≥ 0, one per constraint.
        # Updated by gradient ascent on the Lagrangian dual (independent per constraint).
        self._lambdas: torch.Tensor = torch.full(
            (lag_cfg.num_constraints,),
            lag_cfg.lambda_init,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.lambda_optimizer = optim.Adam([self._lambdas], lr=lag_cfg.lambda_lr)
        self._cost_limit = torch.tensor(
            [lag_cfg.cost_limit] * lag_cfg.num_constraints,
            dtype=torch.float32,
            device=self.device,
        )

        # CBF shield (verified fallback, used for offline data collection and
        # as fallback when neural filter uncertainty is high)
        self.cbf = CBFShield(
            config=self.cfg.cbf,
            num_buildings=self.B,
            action_scale=self.cfg.training.action_scale,
        )

        # Neural Safety Filter – differentiable replacement for the CBF QP.
        # Trained offline on (obs, a_nom, a_safe_qp) tuples; during online
        # training its gradients flow into the actor via select_action().
        self.neural_filter = NeuralSafetyFilter(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            num_ensemble=5,
            dropout_rate=0.1,
            uncertainty_threshold=0.05,
            cbf_config=self.cfg.cbf,
        ).to(self.device)
        self.neural_filter_optimizer = optim.Adam(
            self.neural_filter.parameters(), lr=lr
        )
        # Flag: use neural filter only after it has been pretrained
        self.use_neural_filter: bool = False

        # Running observation normalizer: zero-mean unit-std per feature.
        # Raw obs spans wildly different scales (hour 0-23, net power -300 to
        # 300 kW, SOC 0-1), which makes gradient magnitudes uneven and slows
        # learning.  The normalizer is updated online from each training batch.
        from stems.utils import RunningNormalizer
        self.obs_normalizer = RunningNormalizer(obs_dim).to(self.device)

        # Training counters
        self._update_step = 0

    # ------------------------------------------------------------------
    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        """Encode observations → actor → add noise → safety projection → scale.

        Safety projection uses:
          - NeuralSafetyFilter  (when pretrained, use_neural_filter=True)
          - CBFShield QP        (fallback when filter uncertainty is high, or
                                 when neural filter is not yet pretrained)

        The raw (QP) safe action is always stored in ``_last_qp_safe_actions``
        for offline training data collection.  The neural filter's prediction
        is stored in ``_last_safe_actions`` for policy gradient (Eq 24).

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

            # Normalize observations before encoding (zero-mean, unit-std)
            x_norm = self.obs_normalizer(x)
            h_norm = self.obs_normalizer(h.view(-1, self.obs_dim)).view(h.shape)

            repr_mat = self.encoder(x_norm, self.adj, h_norm)             # (B, repr_dim)

            actions_list = []
            for i in range(self.B):
                r_i = repr_mat[i].unsqueeze(0)                            # (1, repr_dim)
                if explore:
                    # Stochastic SAC sampling – exploration via policy entropy
                    a_i, _ = self.actors[i].sample(r_i)
                else:
                    # Deterministic: use mean action (no tanh noise)
                    mean, _ = self.actors[i](r_i)
                    a_i = torch.tanh(mean)
                actions_list.append(a_i.squeeze(0).cpu().numpy())

        actions = np.stack(actions_list, axis=0)   # (B, action_dim)

        # Store raw actions (pre-safety-projection)
        self._last_raw_actions = actions.copy().astype(np.float32)

        # Scale actions
        actions = actions * self.cfg.training.action_scale

        # --- Safety projection ---
        # QP path: always run to generate oracle labels for neural filter training
        if self.use_cbf:
            qp_safe = self.cbf.project(actions, obs_list)
        else:
            qp_safe = actions.copy()

        # Store QP-safe actions for offline neural filter training
        inv_scale = 1.0 / max(self.cfg.training.action_scale, 1e-8)
        self._last_qp_safe_actions = np.clip(
            qp_safe * inv_scale, -1.0, 1.0
        ).astype(np.float32)

        # Neural filter path: use when pretrained; fall back to QP on high uncertainty
        if self.use_neural_filter and self.use_cbf:
            obs_np = np.stack(obs_list, axis=0).astype(np.float32)  # (B, obs_dim)
            a_nom_np = (actions * inv_scale).astype(np.float32)       # (B, action_dim)
            nf_safe, used_fallback = self.neural_filter.predict(
                obs_np=obs_np,
                a_nom_np=a_nom_np,
                device=self.device,
                cbf_fallback=self.cbf,
                states=obs_list,
            )
            # Rescale to action_scale for environment
            final_actions = np.clip(nf_safe, -1.0, 1.0).astype(np.float32)
        else:
            final_actions = qp_safe.astype(np.float32)
            used_fallback = True  # using QP directly

        self._last_filter_used_fallback: bool = used_fallback

        # Policy gradient target: post-safety action in [-1,1]
        self._last_safe_actions = np.clip(
            final_actions * inv_scale, -1.0, 1.0
        ).astype(np.float32)

        return final_actions

    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One gradient update step using a sampled mini-batch.

        Implements Eq 24-26 (advantage actor-critic update).

        Parameters
        ----------
        batch : dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones',
                optionally 'history' and 'next_history'

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

        # Constraint cost tensor (N, B, K) – binary violation indicators per constraint.
        # Present when train.py passes constraint_costs; absent for legacy batches.
        has_costs = "constraint_costs" in batch and batch["constraint_costs"] is not None
        if has_costs:
            costs_tensor = torch.tensor(
                batch["constraint_costs"], dtype=torch.float32
            ).to(self.device)  # (N, B, 3): k=0 SOC, k=1 building power, k=2 grid power
        else:
            costs_tensor = None

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_cost_critic_loss = 0.0

        T_win = self.cfg.transformer.window_size

        # Use the actual building adjacency during training to learn spatial coordination.
        adj_id   = self.adj
        obs_nb   = obs_tensor.view(N, B, self.obs_dim)
        next_nb  = next_obs_tensor.view(N, B, self.obs_dim)

        # Update running normalizer from this batch (online Welford update)
        with torch.no_grad():
            self.obs_normalizer.update(obs_nb.view(-1, self.obs_dim))
        obs_nb_norm   = self.obs_normalizer(obs_nb)
        next_nb_norm  = self.obs_normalizer(next_nb)

        # Use stored history windows if available; otherwise fall back to dummy expansion
        if "history" in batch and batch["history"] is not None:
            hist_nb = torch.tensor(batch["history"], dtype=torch.float32).to(self.device)
            next_hist_nb = torch.tensor(batch["next_history"], dtype=torch.float32).to(self.device)
        else:
            hist_nb      = obs_nb.unsqueeze(2).expand(N, B, T_win, self.obs_dim).contiguous()
            next_hist_nb = next_nb.unsqueeze(2).expand(N, B, T_win, self.obs_dim).contiguous()

        hist_nb_norm      = self.obs_normalizer(hist_nb.view(-1, self.obs_dim)).view(hist_nb.shape)
        next_hist_nb_norm = self.obs_normalizer(next_hist_nb.view(-1, self.obs_dim)).view(next_hist_nb.shape)

        # --- Single forward pass through encoder (Eq 26: combined gradient) ---
        repr_nb = self.encoder.batch_forward(obs_nb_norm, adj_id, hist_nb_norm)  # (N, B, repr_dim)
        with torch.no_grad():
            repr_next_nb = self.encoder.batch_forward(next_nb_norm, adj_id, next_hist_nb_norm)

        # --- Critic losses (Eq 25) ---
        critic_loss_total = torch.tensor(0.0, device=self.device)
        for b in range(B):
            repr_b      = repr_nb[:, b, :]
            repr_b_next = repr_next_nb[:, b, :]
            rewards_b   = rewards_tensor[:, b]
            values      = self.critics[b](repr_b)
            # Use target critics for stable bootstrap – prevents moving-target instability.
            next_values = self.target_critics[b](repr_b_next)
            targets     = rewards_b + gamma * next_values * (1.0 - dones_tensor)
            c_loss      = F.mse_loss(values, targets.detach())
            critic_loss_total = critic_loss_total + c_loss
            total_critic_loss += c_loss.item()

        # --- Cost critic losses (Bellman update, one per constraint k=0,1,2) ---
        # Each CostCritic produces a (batch, 3) output; MSE against Bellman targets.
        cost_critic_loss_total = torch.tensor(0.0, device=self.device)
        # cost_advantages[b] = (N, K) detached cost advantage for building b.
        cost_advantages: Dict[int, torch.Tensor] = {}
        if has_costs and costs_tensor is not None:
            for b in range(B):
                cv_b = self.cost_critics[b](repr_nb[:, b, :])          # (N, K), grad enabled
                with torch.no_grad():
                    cnv_b = self.cost_critics[b](repr_next_nb[:, b, :])
                    cost_tgt_b = (
                        costs_tensor[:, b, :]
                        + gamma * cnv_b * (1.0 - dones_tensor.unsqueeze(-1))
                    )  # (N, K) Bellman target – detached
                cc_loss = F.mse_loss(cv_b, cost_tgt_b)
                cost_critic_loss_total = cost_critic_loss_total + cc_loss
                total_cost_critic_loss += cc_loss.item()
                # Detached cost advantage for actor: A_c_k = target - value
                cost_advantages[b] = (cost_tgt_b - cv_b.detach())   # (N, K)

        # --- Actor losses (Eq 24 + SAC entropy) ---
        # α = current SAC temperature (detached – only log_alpha gets its own update)
        alpha = self.log_alpha.exp().detach()

        # Use safe (post-CBF) actions for policy gradient – Eq 24 takes the
        # expectation over the safe action space.  Fall back to raw actions
        # if safe_actions are not stored (e.g. legacy batches).
        if "safe_actions" in batch and batch["safe_actions"] is not None:
            pg_actions_tensor = torch.tensor(
                batch["safe_actions"], dtype=torch.float32
            ).to(self.device)  # (N, B, action_dim)
        else:
            pg_actions_tensor = torch.tensor(
                batch["raw_actions"], dtype=torch.float32
            ).to(self.device)  # (N, B, action_dim)

        actor_loss_total = torch.tensor(0.0, device=self.device)
        # Accumulate alpha loss separately (needs only detached log_prob)
        alpha_loss_total = torch.tensor(0.0, device=self.device)

        for b in range(B):
            repr_b    = repr_nb[:, b, :]
            rewards_b = rewards_tensor[:, b]
            with torch.no_grad():
                v_b   = self.critics[b](repr_nb[:, b, :])
                # Use target critic for advantage bootstrap (same stable target as critic loss)
                nv_b  = self.target_critics[b](repr_next_nb[:, b, :])
                t_b   = rewards_b + gamma * nv_b * (1.0 - dones_tensor)
                adv   = (t_b - v_b)
                # Normalise advantage for training stability
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            pg_a_b = pg_actions_tensor[:, b, :]  # safe actions in (-1, 1), (N, action_dim)

            # True log π(a_safe | s) from the stochastic actor via atanh inversion
            log_prob = self.actors[b].log_prob_of(repr_b, pg_a_b)  # (N,)

            # Lagrangian penalty: Σ_k λ_k * A_c_k(s,a)
            if b in cost_advantages:
                lambdas_pos = torch.clamp(self._lambdas, min=0.0).detach()  # (K,)
                lambda_penalty = (lambdas_pos * cost_advantages[b]).sum(dim=-1)  # (N,)
            else:
                lambda_penalty = torch.zeros(N, device=self.device)

            # Correct REINFORCE+SAC-entropy actor loss (Eq 24):
            # L_actor = -E[(A - λ·A_c - α) · log_π(a|s)]
            # Subtracting α shifts the advantage baseline by the entropy temperature,
            # which is equivalent to adding an entropy bonus α·H(π) = -α·E[log_π].
            # Do NOT multiply α by log_prob inside the advantage — that creates a
            # runaway feedback (log_prob grows negative → advantage grows positive
            # → actor loss diverges to -∞).
            effective_adv = (adv - lambda_penalty - alpha).detach()  # (N,)
            a_loss = -(effective_adv * log_prob).mean()
            actor_loss_total = actor_loss_total + a_loss
            total_actor_loss += a_loss.item()

            # Alpha (temperature) update: detached log_prob so only log_alpha gets grad
            with torch.no_grad():
                log_prob_detached = self.actors[b].log_prob_of(repr_b, pg_a_b)
            alpha_loss_b = -(self.log_alpha * (log_prob_detached + self._target_entropy)).mean()
            alpha_loss_total = alpha_loss_total + alpha_loss_b

        # --- Neural filter safety gradient (online fine-tuning) ---
        # When the neural filter is active, run a forward pass through it using the
        # current batch and backpropagate its constraint penalty into the actor via
        # the shared representation.  This is the key novel contribution: safety
        # gradients flow end-to-end from the filter's differentiable constraint
        # penalty into the policy network.
        nf_loss_total = torch.tensor(0.0, device=self.device)
        if self.use_neural_filter and "safe_actions" in batch and batch["safe_actions"] is not None:
            # QP oracle labels for this batch (N, B, action_dim)
            a_safe_qp = torch.tensor(
                batch["safe_actions"], dtype=torch.float32
            ).to(self.device)
            # obs_nb: (N, B, obs_dim) – reuse the already-built tensor
            for b in range(B):
                obs_b = obs_nb[:, b, :]          # (N, obs_dim)
                # Actor mean as nominal action — grad enabled so safety flows back
                # forward() returns (mean, log_std); apply tanh to get bounded action
                _mean_b, _ = self.actors[b](repr_nb[:, b, :])
                a_nom_b = torch.tanh(_mean_b)                 # (N, action_dim)
                a_safe_b = a_safe_qp[:, b, :]    # (N, action_dim) QP oracle labels
                nf_loss_b = self.neural_filter.loss(obs_b, a_nom_b, a_safe_b, alpha=0.5)
                nf_loss_total = nf_loss_total + nf_loss_b

        # --- Single combined backward (avoids in-place tensor version conflicts) ---
        # Apply separate grad norm clips: tight (0.5) for actor to contain REINFORCE
        # variance, normal (1.0) for critics.
        combined_loss = (
            critic_loss_total
            + cost_critic_loss_total
            + actor_loss_total
            + nf_loss_total
        )
        self.encoder_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.cost_critic_optimizer.zero_grad()
        self.neural_filter_optimizer.zero_grad()
        combined_loss.backward()
        # Critics: generous clip — Bellman targets are bounded
        nn.utils.clip_grad_norm_(
            list(self.critics.parameters()) + list(self.cost_critics.parameters()),
            max_norm=1.0,
        )
        # Actor + encoder: tight clip — REINFORCE has high variance on long episodes
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actors.parameters())
            + list(self.neural_filter.parameters()),
            max_norm=0.5,
        )
        self.encoder_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.cost_critic_optimizer.step()
        self.neural_filter_optimizer.step()

        # --- Alpha (SAC temperature) update – separate backward ---
        # alpha_loss depends only on log_alpha (log_prob was detached above)
        self.alpha_optimizer.zero_grad()
        alpha_loss_total.backward()
        self.alpha_optimizer.step()

        # --- Lagrangian dual update: gradient ascent on λ · (J_c - d) ---
        # Each λ_k increases when its constraint is violated beyond cost_limit,
        # tightening the penalty on the policy in the next update.
        if has_costs and costs_tensor is not None:
            # Mean violation rate over all timesteps and buildings: (K,)
            mean_costs = costs_tensor.mean(dim=0).mean(dim=0)
            # Dual loss for gradient-descent optimizer → ascent on λ
            lambda_loss = -(self._lambdas * (mean_costs - self._cost_limit)).sum()
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()
            # Project λ to [0, lambda_max] (dual feasibility + anti-runaway cap)
            with torch.no_grad():
                self._lambdas.data.clamp_(min=0.0, max=self._lag_cfg.lambda_max)

        # Polyak EMA update for target critics: θ_target ← τ·θ + (1-τ)·θ_target
        with torch.no_grad():
            for p, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
                tp.data.mul_(1.0 - self._target_tau).add_(self._target_tau * p.data)

        self._update_step += 1

        return {
            "actor_loss": total_actor_loss / B,
            "critic_loss": total_critic_loss / B,
            "cost_critic_loss": total_cost_critic_loss / B,
            "neural_filter_loss": float(nf_loss_total.item()) / B,
            "lambdas": self._lambdas.detach().cpu().tolist(),
            "alpha": float(self.log_alpha.exp().item()),
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights to *path* (directory)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        torch.save(self.actors.state_dict(), os.path.join(path, "actors.pt"))
        torch.save(self.critics.state_dict(), os.path.join(path, "critics.pt"))
        torch.save(self.target_critics.state_dict(), os.path.join(path, "target_critics.pt"))
        torch.save(self.cost_critics.state_dict(), os.path.join(path, "cost_critics.pt"))
        torch.save({"lambdas": self._lambdas.data}, os.path.join(path, "lambdas.pt"))
        torch.save({"log_alpha": self.log_alpha.data}, os.path.join(path, "log_alpha.pt"))
        torch.save(self.obs_normalizer.state_dict(), os.path.join(path, "obs_normalizer.pt"))
        # Only save neural filter if it has actually been trained (use_neural_filter=True).
        # Saving the randomly-initialised filter and loading it later would silently
        # replace the real policy with a near-zero null controller.
        if self.use_neural_filter:
            torch.save(self.neural_filter.state_dict(), os.path.join(path, "neural_filter.pt"))

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
        target_critics_path = os.path.join(path, "target_critics.pt")
        if os.path.exists(target_critics_path):
            self.target_critics.load_state_dict(
                torch.load(target_critics_path, map_location=map_loc)
            )
        else:
            # Fallback: initialise target from live critics if no saved target exists
            self.target_critics.load_state_dict(self.critics.state_dict())
        cost_critics_path = os.path.join(path, "cost_critics.pt")
        if os.path.exists(cost_critics_path):
            self.cost_critics.load_state_dict(
                torch.load(cost_critics_path, map_location=map_loc)
            )
        lambdas_path = os.path.join(path, "lambdas.pt")
        if os.path.exists(lambdas_path):
            d = torch.load(lambdas_path, map_location=map_loc)
            with torch.no_grad():
                self._lambdas.data.copy_(d["lambdas"])
        log_alpha_path = os.path.join(path, "log_alpha.pt")
        if os.path.exists(log_alpha_path):
            d = torch.load(log_alpha_path, map_location=map_loc)
            with torch.no_grad():
                self.log_alpha.data.copy_(d["log_alpha"])
        obs_norm_path = os.path.join(path, "obs_normalizer.pt")
        if os.path.exists(obs_norm_path):
            self.obs_normalizer.load_state_dict(
                torch.load(obs_norm_path, map_location=map_loc)
            )
        nf_path = os.path.join(path, "neural_filter.pt")
        if os.path.exists(nf_path):
            self.neural_filter.load_state_dict(
                torch.load(nf_path, map_location=map_loc)
            )
            # Do NOT auto-enable use_neural_filter here.  The file is saved
            # unconditionally so its mere existence does not mean it was trained.
            # Callers must explicitly set agent.use_neural_filter = True after
            # verifying the filter is trained.
