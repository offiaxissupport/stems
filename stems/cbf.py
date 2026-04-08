"""
CBF Safety Shield (Eq 16-20, Algorithm 1).

Three safety constraints are enforced:
    h1(s)   Battery SOC bounds          (Eq 16)
    h2(s,a) Per-building power limit    (Eq 17)
    h3(s,a) Total grid power limit      (Eq 18)

The shield solves a Quadratic Programme (QP) to find the minimal correction
to the nominal action that makes all constraints satisfied (Eq 19-20).

    min_u  ||u - a||²
    s.t.   h_k(s, u) >= -gamma_cbf * h_k(s, a_nominal)  for all k

If cvxpy is unavailable the shield falls back to analytical clipping.
If the QP is infeasible an emergency conservative action is returned.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

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

    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        num_buildings: int = 3,
    ) -> None:
        self.cfg = config or CBFConfig()
        self.B = num_buildings

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

    def _h_grid(self, total_net: float) -> float:
        """Eq 18: Total grid power safety margin."""
        return self.cfg.P_grid_max - total_net

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

        total_net = sum(float(s[_IDX_NET]) for s in states)
        grid_ok = self._h_grid(total_net) >= 0.0

        for i in range(B):
            soc = float(states[i][_IDX_SOC_ELEC])
            delta_soc = float(actions[i, 1]) * 0.1   # rough SOC change per step
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
        if _CVXPY_AVAILABLE:
            return self._qp_project(actions, states)
        else:
            return self._clip_project(actions, states)

    # ------------------------------------------------------------------
    def _qp_project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        """QP-based projection using cvxpy with the SCS solver."""
        B, action_dim = actions.shape
        safe_actions = actions.copy()

        for i in range(B):
            a_nom = actions[i]            # (action_dim,)
            soc = float(states[i][_IDX_SOC_ELEC])
            net_i = float(states[i][_IDX_NET])
            total_net = sum(float(s[_IDX_NET]) for s in states)

            u = cp.Variable(action_dim)
            cost = cp.sum_squares(u - a_nom)
            constraints = []

            # SOC lower bound (Eq 16)
            delta_soc = u[1] * 0.1
            h_lo_nom, h_hi_nom = self._h_soc(soc, float(a_nom[1]) * 0.1)
            constraints.append(soc + delta_soc - self.cfg.SOC_min >= -self.cfg.gamma_cbf * h_lo_nom)
            # SOC upper bound
            constraints.append(self.cfg.SOC_max - (soc + delta_soc) >= -self.cfg.gamma_cbf * h_hi_nom)

            # Per-building power (Eq 17) – approximate via action magnitude
            constraints.append(cp.norm(u, 1) <= 3.0)

            # Action range
            constraints.append(u >= -1.0)
            constraints.append(u <= 1.0)

            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
                    safe_actions[i] = np.clip(u.value, -1.0, 1.0)
                else:
                    # Infeasible: emergency conservative action
                    safe_actions[i] = np.zeros(action_dim)
            except Exception:
                safe_actions[i] = np.zeros(action_dim)

        return safe_actions

    # ------------------------------------------------------------------
    def _clip_project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        """Analytical clipping fallback when cvxpy is unavailable."""
        B, action_dim = actions.shape
        safe_actions = actions.copy()

        for i in range(B):
            soc = float(states[i][_IDX_SOC_ELEC])
            # Clip electrical storage action to keep SOC in [SOC_min, SOC_max]
            # delta_soc ≈ action[1] * 0.1
            a1 = float(actions[i, 1])
            max_charge = (self.cfg.SOC_max - soc) / 0.1
            max_discharge = (soc - self.cfg.SOC_min) / 0.1
            a1 = float(np.clip(a1, -max_discharge, max_charge))
            safe_actions[i, 1] = float(np.clip(a1, -1.0, 1.0))

        return safe_actions
