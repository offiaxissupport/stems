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

    # SOC change per unit action per timestep (approximate physics model)
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
        if _CVXPY_AVAILABLE:
            return self._qp_project(actions, states)
        else:
            return self._clip_project(actions, states)

    # ------------------------------------------------------------------
    # Approximate power contribution per unit action (dhw, battery, cooling)
    POWER_FACTORS: list = [0.05, 0.1, 0.5]

    def _qp_project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        """QP-based projection using cvxpy with the SCS solver (Eq 19-20)."""
        B, action_dim = actions.shape
        safe_actions = actions.copy()
        total_net = sum(float(s[_IDX_NET]) for s in states)

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

            # Grid power constraint (Eq 18): P_grid_max - total >= 0
            predicted_total = total_net + predicted_delta
            constraints.append(predicted_total <= self.cfg.P_grid_max)

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
        """Analytical clipping fallback when cvxpy is unavailable."""
        B, action_dim = actions.shape
        safe_actions = actions.copy()

        for i in range(B):
            soc = float(states[i][_IDX_SOC_ELEC])
            # Clip electrical storage action to keep SOC in [SOC_min, SOC_max]
            a1 = float(actions[i, 1])
            max_charge    = (self.cfg.SOC_max - soc) / self.SOC_DELTA_RATE
            max_discharge = (soc - self.cfg.SOC_min) / self.SOC_DELTA_RATE
            a1 = float(np.clip(a1, -max_discharge, max_charge))
            safe_actions[i, 1] = float(np.clip(a1, -self.action_scale, self.action_scale))

        return safe_actions
