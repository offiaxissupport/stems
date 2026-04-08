"""
4-Part reward function for STEMS (Eq 3-9).

R_total = R_economic + R_stability + R_comfort + R_renewable

Each component is computed per-building; the tuple (obs, action, next_obs)
is expected to follow the 28-dim OBS_NAMES layout defined in environment.py.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from stems.config import RewardConfig

# Observation indices (matching OBS_NAMES in environment.py)
_IDX_T_IN = 15           # indoor_dry_bulb_temperature
_IDX_LOAD = 16           # non_shiftable_load
_IDX_SOLAR = 17          # solar_generation
_IDX_NET = 20            # net_electricity_consumption
_IDX_PRICE = 21          # electricity_pricing
_IDX_OCCUPANT = 26       # occupant_count
_IDX_T_SET = 27          # indoor_dry_bulb_temperature_cooling_set_point


class STEMSReward:
    """Computes the per-building 4-component reward (Eq 3-9).

    Parameters
    ----------
    config : RewardConfig
        Reward hyper-parameters from the paper.
    num_buildings : int
        Number of buildings B.
    P_grid_max : float
        Maximum total grid power used for stability normalisation.
    P_building_max : float
        Maximum per-building grid power used for stability normalisation.
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        num_buildings: int = 3,
        P_grid_max: float = 1000.0,
        P_building_max: float = 200.0,
    ) -> None:
        self.cfg = config or RewardConfig()
        self.B = num_buildings
        self.P_grid_max = P_grid_max
        self.P_building_max = P_building_max

    # ------------------------------------------------------------------
    def compute(
        self,
        obs_list: List[np.ndarray],
        actions: np.ndarray,
        next_obs_list: List[np.ndarray],
        prev_net_consumption: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute per-building rewards.

        Parameters
        ----------
        obs_list  : List of B arrays, shape (obs_dim,)
        actions   : np.ndarray, shape (B, action_dim)
        next_obs_list : List of B arrays, shape (obs_dim,)
        prev_net_consumption : optional previous-step net consumption per building

        Returns
        -------
        List[float] of length B
        """
        if prev_net_consumption is None:
            prev_net_consumption = [0.0] * self.B

        # Total grid draw at current step (sum across buildings)
        total_net = sum(float(o[_IDX_NET]) for o in next_obs_list)

        rewards: List[float] = []
        for i in range(self.B):
            obs_i = obs_list[i]
            next_i = next_obs_list[i]

            e_i = float(next_i[_IDX_NET])     # net electricity consumption
            v_t = float(next_i[_IDX_PRICE])   # electricity price
            t_in = float(next_i[_IDX_T_IN])
            solar_i = float(next_i[_IDX_SOLAR])
            e_prev = float(prev_net_consumption[i])

            # Eq 5: Economic reward
            r_econ = -self.cfg.mu * v_t * e_i

            # Eq 6-7: Stability reward
            # Grid term: penalise collective over-load
            grid_excess = max(0.0, total_net)
            grid_term = self.cfg.alpha_grid * (
                1.0 - (grid_excess / self.P_grid_max) ** 2
            )
            # Building load smoothness
            build_term = self.cfg.alpha_build * (
                1.0 - abs(e_i) / max(self.P_building_max, 1.0)
            )
            # Ramping penalty
            ramp_term = -self.cfg.beta_ramp * abs(e_i - e_prev) / max(
                self.P_building_max, 1.0
            )
            r_stab = grid_term + build_term + ramp_term

            # Eq 8: Comfort reward
            r_comfort = -self.cfg.lambda_indoor * (t_in - self.cfg.T_ref) ** 2

            # Eq 9: Renewable utilisation reward
            denom = solar_i + max(0.0, e_i)
            if denom > 1e-8:
                renewable_ratio = min(solar_i / denom, 1.0)
            else:
                renewable_ratio = 0.0
            r_renew = self.cfg.xi * renewable_ratio

            rewards.append(r_econ + r_stab + r_comfort + r_renew)

        return rewards
