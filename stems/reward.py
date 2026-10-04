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
        heating_setpoint_idx: Optional[int] = None,
        ev_layout: Optional[dict] = None,
    ) -> None:
        self.cfg = config or RewardConfig()
        self.B = num_buildings
        self.P_grid_max = P_grid_max
        self.P_building_max = P_building_max
        # When provided (heat-pump mode), comfort uses a dual-setpoint deadband:
        # penalise only t_in above the cooling setpoint or below the heating
        # setpoint. Otherwise the single cooling setpoint is used (paper default),
        # which incorrectly penalises comfortable winter temperatures.
        self.heating_setpoint_idx = heating_setpoint_idx
        # Observation indices of one charger bay (connected_state, soc,
        # required_soc_departure, battery_capacity). When absent the EV service
        # term is simply not applied, so building-only schemas are unaffected.
        self.ev_layout = ev_layout

    def _comfort_penalty(self, t_in: float, cond_i: np.ndarray) -> float:
        """Squared distance outside the comfort band during an occupied hour.

        ``cond_i`` is the pre-action observation, which carries the set points and
        occupancy of the hour being scored; ``t_in`` is that hour's simulated
        temperature. Unoccupied hours are not penalised, matching the discomfort
        KPI. In heat-pump mode the band is [heating set point, cooling set point];
        otherwise the single cooling set point (paper Eq. 8).
        """
        if float(cond_i[_IDX_OCCUPANT]) <= 0.0:
            return 0.0
        t_cool = float(cond_i[_IDX_T_SET])
        if self.heating_setpoint_idx is None:
            return (t_in - t_cool) ** 2
        t_heat = float(cond_i[self.heating_setpoint_idx])
        if not 0.0 < t_heat <= t_cool:
            raise ValueError(f"heating set point {t_heat} is missing or above the cooling "
                             f"set point {t_cool}; the comfort band is undefined")
        if t_in > t_cool:
            return (t_in - t_cool) ** 2
        if t_in < t_heat:
            return (t_heat - t_in) ** 2
        return 0.0

    def _ev_service_penalty(self, obs_i: np.ndarray, next_i: np.ndarray) -> float:
        """Penalty for a vehicle short of its departure requirement.

        Two components. The *departure* term fires on the step a connected
        vehicle disconnects and charges the normalised energy it left short --
        this is the service failure that matters and it cannot be undone. The
        *shaping* term is a small per-step penalty on the remaining shortfall,
        which keeps the signal from arriving only once per trip; without it the
        credit assignment over a ten-hour parking window is extremely sparse.

        Returns 0 when the schema has no charger, so building-only runs are
        numerically identical to before this term existed.
        """
        L = self.ev_layout
        if L is None:
            return 0.0
        was_connected = float(obs_i[L["connected_state"]]) > 0.5
        now_connected = float(next_i[L["connected_state"]]) > 0.5
        if not was_connected:
            return 0.0
        soc = float(obs_i[L["soc"]])
        required = float(obs_i[L["required_soc_departure"]])
        shortfall = max(required - soc, 0.0)
        if not now_connected:
            return self.cfg.ev_service * shortfall      # departed short
        return self.cfg.ev_shaping * shortfall          # still time to fix it

    # ------------------------------------------------------------------
    def compute(
        self,
        obs_list: List[np.ndarray],
        actions: np.ndarray,
        next_obs_list: List[np.ndarray],
        prev_net_consumption: Optional[List[float]] = None,
    ) -> List[float]:
        """Per-building rewards for the hour ``t`` the actions were applied to.

        Timing (see ``MetricsCalculator.add_step``): the conditions of hour ``t``
        -- price, set points, occupancy, solar, load -- are in ``obs_list``; its
        outcomes -- net consumption and simulated indoor temperature -- are in
        ``next_obs_list``. Reading the price from ``next_obs_list`` would reward
        hour ``t`` at hour ``t+1``'s tariff.

        Parameters
        ----------
        obs_list  : B pre-action observations (hour t conditions)
        actions   : (B, action_dim)
        next_obs_list : B post-action observations (hour t outcomes)
        prev_net_consumption : net consumption of hour t-1 per building

        Returns
        -------
        List[float] of length B
        """
        if prev_net_consumption is None:
            prev_net_consumption = [0.0] * self.B

        # District import, as scored by the grid constraint and peak KPIs.
        grid_draw = sum(max(0.0, float(o[_IDX_NET])) for o in next_obs_list)
        # Grid stability (Eq. 6). The paper's (1 - draw/P)^2 is minimised AT the
        # cap and rises again above it, i.e. it rewards overload. A convex,
        # monotone peak penalty keeps the intent (large district draws cost
        # more than proportionally) without that inversion.
        grid_term = -self.cfg.alpha_grid * (grid_draw / self.P_grid_max) ** 2

        rewards: List[float] = []
        for i in range(self.B):
            cond_i, next_i = obs_list[i], next_obs_list[i]
            e_i = float(next_i[_IDX_NET])
            price = float(cond_i[_IDX_PRICE])
            p_b = max(self.P_building_max, 1.0)

            # Eq 5: economic. Imports only, as the cost KPI and CityLearn's own cost
            # KPI: the tariff carries no export price, so none is invented.
            r_econ = -self.cfg.mu * price * max(e_i, 0.0)

            # Eq 6-7: stability (constant offsets of the paper's form dropped:
            # they change no decision).
            build_term = -self.cfg.alpha_build * abs(e_i) / p_b
            ramp_term = -self.cfg.beta_ramp * abs(e_i - float(prev_net_consumption[i])) / p_b
            r_stab = grid_term + build_term + ramp_term

            # Eq 8: comfort on the simulated temperature of hour t.
            r_comfort = -self.cfg.lambda_indoor * self._comfort_penalty(
                float(next_i[_IDX_T_IN]), cond_i)

            # Eq 9: renewable utilisation. A function of exogenous solar and load
            # only: no action changes it, so it shifts returns without changing any
            # decision. Off by default (xi = 0); kept for the paper's formulation.
            r_renew = 0.0
            if self.cfg.xi:
                solar_i = float(cond_i[_IDX_SOLAR])
                load_i = float(cond_i[_IDX_LOAD])
                if solar_i > 1e-8:
                    r_renew = self.cfg.xi * min(solar_i / max(load_i, 1e-8), 1.0)

            # EV service. Zero when the schema exposes no charger.
            r_ev = -self._ev_service_penalty(cond_i, next_i)

            rewards.append(r_econ + r_stab + r_comfort + r_renew + r_ev)

        return rewards
