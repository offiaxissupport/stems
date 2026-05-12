"""
CBF Safety Shield (Eq 16-20, Algorithm 1) and Neural Safety Filter.

Two complementary safety mechanisms:

CBFShield (original STEMS)
--------------------------
Solves a QP to find the minimal action correction satisfying hard constraints.
Used as a verified fallback and for offline data collection.

NeuralSafetyFilter (novel contribution)
----------------------------------------
A learned, differentiable safety correction network trained on (state, action)
pairs collected from the CBF QP oracle.  It maps unsafe nominal actions to safe
ones end-to-end, so safety gradients flow directly into policy learning.

Architecture:
    Input : [obs_i (D), a_nom_i (A)]  ← per-building concatenation
    Trunk : 3 × Linear-LayerNorm-ReLU (hidden_dim=128)
    Head  : Linear → Tanh → safe_action (A)  ← same range as actor

Uncertainty estimation:
    MC-Dropout ensemble of E=5 forward passes during inference.
    If ensemble std exceeds `uncertainty_threshold`, fall back to CBFShield QP.

Training:
    Offline, on a dataset of (obs, a_nom, a_safe) tuples where a_safe comes
    from the CBF QP oracle.  Loss = MSE(predicted_safe, a_safe_qp) +
    α · constraint_penalty(predicted_safe, obs).
    The constraint penalty is the sum of ReLU(−h_k) over all violated CBF
    constraints, making the loss differentiable w.r.t. the network weights.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from stems.config import CBFConfig

# --------------------------------------------------------------------------
# Optional cvxpy import
# --------------------------------------------------------------------------
_CVXPY_AVAILABLE = False
try:
    import cvxpy as cp  # type: ignore
    _CVXPY_AVAILABLE = True
except ImportError:
    pass

# Observation indices (matching OBS_NAMES in environment.py)
_IDX_SOC_ELEC = 19    # electrical_storage_soc
_IDX_NET = 20         # net_electricity_consumption


# --------------------------------------------------------------------------
# CBFShield
# --------------------------------------------------------------------------

class CBFShield:
    """Control Barrier Function safety shield.

    Parameters
    ----------
    config : CBFConfig
        CBF hyper-parameters.
    num_buildings : int
        Number of buildings B.
    """

    # SOC change per unit action per timestep.
    # Must match the mock environment's dynamics: _MockBuilding.step() applies
    # soc_elec += elec_action * 0.1, so the CBF must use the same coefficient.
    # A mismatch (e.g. 0.8 vs 0.1) makes the QP over-conservative by 8× and
    # causes check_violations to disagree with what the environment actually does.
    SOC_DELTA_RATE: float = 0.1

    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        num_buildings: int = 3,
        action_scale: float = 1.0,
    ) -> None:
        self.cfg = config or CBFConfig()
        self.B = num_buildings
        self.action_scale = action_scale

    # ------------------------------------------------------------------
    # Constraint functions
    # ------------------------------------------------------------------

    def _h_soc(self, soc: float, delta_soc: float) -> Tuple[float, float]:
        """Eq 16: Battery SOC safety margin.

        Returns (h_lower, h_upper) – both should be >= 0.
        """
        h_lower = soc + delta_soc - self.cfg.SOC_min
        h_upper = self.cfg.SOC_max - (soc + delta_soc)
        return h_lower, h_upper

    def _h_build(self, net: float) -> float:
        """Eq 17: Per-building power safety margin."""
        return self.cfg.P_building_max - abs(net)

    def _h_grid(self, total_positive_net: float) -> float:
        """Eq 18: Total grid power safety margin.

        Paper Eq 18: h_grid = P_grid_max - Σ_i max(0, e_i) ≥ 0
        Only grid *imports* (positive net) count against the grid limit.
        """
        return self.cfg.P_grid_max - total_positive_net

    # ------------------------------------------------------------------
    # Constraint violation check
    # ------------------------------------------------------------------

    def check_violations(
        self,
        actions: np.ndarray,
        states: List[np.ndarray],
    ) -> np.ndarray:
        """Return boolean mask (B,) – True where building i violates a constraint."""
        B = self.B
        violations = np.zeros(B, dtype=bool)

        # Eq 18: grid constraint uses Σ max(0, e_i) — only imports count
        total_positive_net = sum(max(0.0, float(s[_IDX_NET])) for s in states)
        grid_ok = self._h_grid(total_positive_net) >= 0.0

        for i in range(B):
            soc = float(states[i][_IDX_SOC_ELEC])
            delta_soc = float(actions[i, 1]) * self.SOC_DELTA_RATE   # rough SOC change per step
            net_i = float(states[i][_IDX_NET])

            h_lo, h_hi = self._h_soc(soc, delta_soc)
            soc_ok = (h_lo >= 0.0) and (h_hi >= 0.0)
            build_ok = self._h_build(net_i) >= 0.0

            violations[i] = not (soc_ok and build_ok and grid_ok)

        return violations

    # ------------------------------------------------------------------
    # QP projection (Algorithm 1)
    # ------------------------------------------------------------------

    def project(
        self,
        actions: np.ndarray,
        states: List[np.ndarray],
        adj: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project nominal actions onto the safe action set (Eq 19-20).

        Parameters
        ----------
        actions : np.ndarray, shape (B, action_dim)
        states  : List of B observation arrays
        adj     : ignored (kept for API consistency)

        Returns
        -------
        safe_actions : np.ndarray, shape (B, action_dim)
        """
        # Algorithm 1, lines 4-6: check if all constraints already satisfied.
        # If so, return the nominal action directly (no QP needed).
        if self._all_constraints_satisfied(actions, states):
            return actions.copy()

        if _CVXPY_AVAILABLE:
            return self._qp_project(actions, states)
        else:
            return self._clip_project(actions, states)

    # ------------------------------------------------------------------
    def _all_constraints_satisfied(
        self,
        actions: np.ndarray,
        states: List[np.ndarray],
    ) -> bool:
        """Return True iff all CBF constraints are satisfied (Algorithm 1, line 4)."""
        total_positive_net = sum(max(0.0, float(s[_IDX_NET])) for s in states)
        if self._h_grid(total_positive_net) < 0.0:
            return False
        for i in range(self.B):
            soc = float(states[i][_IDX_SOC_ELEC])
            delta_soc = float(actions[i, 1]) * self.SOC_DELTA_RATE
            h_lo, h_hi = self._h_soc(soc, delta_soc)
            if h_lo < 0.0 or h_hi < 0.0:
                return False
            if self._h_build(float(states[i][_IDX_NET])) < 0.0:
                return False
        return True

    # ------------------------------------------------------------------
    # Approximate power contribution per unit action (dhw, battery, cooling)
    POWER_FACTORS: list = [0.05, 0.1, 0.5]

    def _qp_project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        """QP-based projection using cvxpy with the SCS solver (Eq 19-20)."""
        B, action_dim = actions.shape
        safe_actions = actions.copy()
        # Eq 18: only positive (import) contributions count against P_grid_max
        total_positive_net = sum(max(0.0, float(s[_IDX_NET])) for s in states)

        for i in range(B):
            a_nom = actions[i]            # (action_dim,)
            soc = float(states[i][_IDX_SOC_ELEC])
            net_i = float(states[i][_IDX_NET])

            u = cp.Variable(action_dim)
            cost = cp.sum_squares(u - a_nom)
            constraints = []

            # SOC constraints (Eq 16): h_battery >= 0
            delta_soc = u[1] * self.SOC_DELTA_RATE
            constraints.append(soc + delta_soc >= self.cfg.SOC_min)
            constraints.append(soc + delta_soc <= self.cfg.SOC_max)

            # Building power constraint (Eq 17): P_building_max - |e_pred| >= 0
            pf = self.POWER_FACTORS
            predicted_delta = sum(
                pf[d] * (u[d] - float(a_nom[d]))
                for d in range(min(action_dim, len(pf)))
            )
            predicted_net = net_i + predicted_delta
            constraints.append(predicted_net <= self.cfg.P_building_max)
            constraints.append(predicted_net >= -self.cfg.P_building_max)

            # Grid power constraint (Eq 18): P_grid_max - Σmax(0,e) >= 0
            # Approximate: the building's predicted import is max(0, predicted_net)
            predicted_import = cp.maximum(predicted_net, 0)
            other_positive = total_positive_net - max(0.0, net_i)
            constraints.append(other_positive + predicted_import <= self.cfg.P_grid_max)

            # Action range
            constraints.append(u >= -self.action_scale)
            constraints.append(u <= self.action_scale)

            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
                    safe_actions[i] = np.clip(u.value, -self.action_scale, self.action_scale)
                else:
                    # Infeasible: emergency conservative action
                    safe_actions[i] = np.zeros(action_dim)
            except Exception:
                safe_actions[i] = np.zeros(action_dim)

        return safe_actions

    # ------------------------------------------------------------------
    def _clip_project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        """Analytical clipping fallback when cvxpy is unavailable.

        Clips all three action dimensions:
          index 0 (DHW storage)      : clipped to [-action_scale, action_scale]
          index 1 (elec storage)     : clipped to respect SOC bounds
          index 2 (cooling device)   : clipped to [0, action_scale] (cooling only)
        """
        B, action_dim = actions.shape
        safe_actions = actions.copy()

        for i in range(B):
            soc = float(states[i][_IDX_SOC_ELEC])

            # index 0: DHW storage – clip to valid action range
            if action_dim > 0:
                safe_actions[i, 0] = float(np.clip(
                    actions[i, 0], -self.action_scale, self.action_scale
                ))

            # index 1: electrical storage – respect SOC bounds
            if action_dim > 1:
                a1 = float(actions[i, 1])
                max_charge    = (self.cfg.SOC_max - soc) / self.SOC_DELTA_RATE
                max_discharge = (soc - self.cfg.SOC_min) / self.SOC_DELTA_RATE
                a1 = float(np.clip(a1, -max_discharge, max_charge))
                safe_actions[i, 1] = float(np.clip(a1, -self.action_scale, self.action_scale))

            # index 2: cooling device – only cooling allowed (no reverse heating via this action)
            if action_dim > 2:
                safe_actions[i, 2] = float(np.clip(actions[i, 2], 0.0, self.action_scale))

        return safe_actions


# ---------------------------------------------------------------------------
# NeuralSafetyFilter
# ---------------------------------------------------------------------------

class NeuralSafetyFilter(nn.Module):
    """Differentiable learned safety filter.

    Replaces the fixed CBF QP shield with a neural network trained offline on
    (obs, a_nominal) → a_safe pairs generated by the CBF oracle.  Because it is
    fully differentiable, safety gradients flow directly into actor learning
    during the policy update step.

    At inference time, MC-Dropout uncertainty is measured over E forward passes.
    If uncertainty (ensemble std) exceeds `uncertainty_threshold`, the module
    raises a flag and the caller falls back to the CBF QP.

    Parameters
    ----------
    obs_dim : int        – per-building observation dimension
    action_dim : int     – per-building action dimension
    hidden_dim : int     – width of hidden layers (default 128)
    num_ensemble : int   – MC-Dropout samples for uncertainty (default 5)
    dropout_rate : float – Dropout probability during MC sampling (default 0.1)
    uncertainty_threshold : float – fallback threshold on ensemble std (default 0.05)
    cbf_config : CBFConfig – constraint bounds for the differentiable penalty
    """

    SOC_DELTA_RATE: float = 0.1    # must match CBFShield and mock env dynamics

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_ensemble: int = 5,
        dropout_rate: float = 0.1,
        uncertainty_threshold: float = 0.05,
        cbf_config: Optional[CBFConfig] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.E = num_ensemble
        self.uncertainty_threshold = uncertainty_threshold
        self.cfg = cbf_config or CBFConfig()

        in_dim = obs_dim + action_dim

        # Trunk: 3 × (Linear → LayerNorm → ReLU → Dropout)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Head: projects to action space, Tanh to stay in [-1, 1]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        nn.init.uniform_(self.head[0].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head[0].bias, -3e-3, 3e-3)

        # Running flag set by forward() – True when last call fell back to QP
        self.last_used_fallback: bool = False

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor, a_nom: torch.Tensor) -> torch.Tensor:
        """Single deterministic forward pass (used during training backprop).

        Parameters
        ----------
        obs   : (B, obs_dim)
        a_nom : (B, action_dim)

        Returns
        -------
        a_safe : (B, action_dim) in [-1, 1]
        """
        x = torch.cat([obs, a_nom], dim=-1)   # (B, obs_dim + action_dim)
        return self.head(self.trunk(x))

    # ------------------------------------------------------------------
    def predict(
        self,
        obs_np: np.ndarray,
        a_nom_np: np.ndarray,
        device: torch.device,
        cbf_fallback: "CBFShield",
        states: List[np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        """MC-Dropout inference with uncertainty-triggered CBF fallback.

        Parameters
        ----------
        obs_np     : (B, obs_dim) float32
        a_nom_np   : (B, action_dim) float32
        device     : torch device
        cbf_fallback : CBFShield – used when uncertainty is too high
        states     : List[np.ndarray] – raw obs list for CBF

        Returns
        -------
        safe_actions : (B, action_dim)
        used_fallback : bool – True if uncertainty triggered QP fallback
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        a_t = torch.tensor(a_nom_np, dtype=torch.float32, device=device)

        # Enable Dropout for MC sampling
        self.train()
        with torch.no_grad():
            samples = torch.stack(
                [self.head(self.trunk(torch.cat([obs_t, a_t], dim=-1))) for _ in range(self.E)],
                dim=0,
            )  # (E, B, action_dim)

        mean = samples.mean(dim=0)        # (B, action_dim)
        std  = samples.std(dim=0)         # (B, action_dim)
        max_uncertainty = float(std.max().item())

        self.eval()
        self.last_used_fallback = False

        if max_uncertainty > self.uncertainty_threshold:
            # Uncertainty too high – fall back to verified CBF QP
            self.last_used_fallback = True
            return cbf_fallback.project(a_nom_np, states), True

        return mean.cpu().numpy(), False

    # ------------------------------------------------------------------
    # Differentiable constraint penalty (for offline training loss)
    # ------------------------------------------------------------------

    def constraint_penalty(
        self,
        obs: torch.Tensor,
        a_safe: torch.Tensor,
    ) -> torch.Tensor:
        """Soft constraint violation penalty – differentiable w.r.t. a_safe.

        Computes ReLU(−h_k) for each constraint k, averaged over the batch.
        This makes the training loss aware of constraint satisfaction so the
        network learns to be safe, not just to imitate the QP output.

        Constraints (per building i):
            h1_lo = SOC_i + a_{i,1}·δ − SOC_min  ≥ 0   (SOC lower bound)
            h1_hi = SOC_max − SOC_i − a_{i,1}·δ  ≥ 0   (SOC upper bound)
            h2    = P_build_max − |net_i|          ≥ 0   (building power, approx)

        Grid constraint is handled approximately: Σ relu(net_i) ≤ P_grid_max.

        Parameters
        ----------
        obs    : (B, obs_dim)
        a_safe : (B, action_dim) – the filter's predicted safe action

        Returns
        -------
        penalty : scalar tensor
        """
        soc   = obs[:, _IDX_SOC_ELEC]   # (B,)
        net   = obs[:, _IDX_NET]         # (B,)
        delta = a_safe[:, 1] * self.SOC_DELTA_RATE  # (B,) SOC change

        # SOC bounds
        h_soc_lo = soc + delta - self.cfg.SOC_min   # (B,)
        h_soc_hi = self.cfg.SOC_max - soc - delta    # (B,)

        # Building power (approximate: treat net as fixed, apply delta)
        pf_battery = 0.1
        net_pred = net + pf_battery * a_safe[:, 1]
        h_build_pos = self.cfg.P_building_max - net_pred
        h_build_neg = net_pred + self.cfg.P_building_max

        # Grid power (approximate sum)
        grid_import = torch.relu(net).sum()
        h_grid = torch.tensor(self.cfg.P_grid_max, device=obs.device) - grid_import

        violations = torch.cat([
            torch.relu(-h_soc_lo),
            torch.relu(-h_soc_hi),
            torch.relu(-h_build_pos),
            torch.relu(-h_build_neg),
            torch.relu(-h_grid).unsqueeze(0),
        ])
        return violations.mean()

    # ------------------------------------------------------------------
    # Pretraining loss (MSE imitation + constraint penalty)
    # ------------------------------------------------------------------

    def loss(
        self,
        obs: torch.Tensor,
        a_nom: torch.Tensor,
        a_safe_qp: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Offline training loss.

        L = MSE(filter(obs, a_nom), a_safe_qp) + α · constraint_penalty

        Parameters
        ----------
        obs       : (B, obs_dim)
        a_nom     : (B, action_dim) – nominal (unsafe) action
        a_safe_qp : (B, action_dim) – oracle safe action from CBF QP
        alpha     : weight on the constraint penalty term

        Returns
        -------
        scalar loss tensor
        """
        a_pred = self.forward(obs, a_nom)
        mse = nn.functional.mse_loss(a_pred, a_safe_qp)
        penalty = self.constraint_penalty(obs, a_pred)
        return mse + alpha * penalty
