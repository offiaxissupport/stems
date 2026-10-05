from __future__ import annotations

from typing import List, Optional

import numpy as np

from stems.config import RewardConfig

_IDX_T_IN = 15
_IDX_LOAD = 16
_IDX_SOLAR = 17
_IDX_NET = 20
_IDX_PRICE = 21
_IDX_OCCUPANT = 26
_IDX_T_SET = 27


class STEMSReward:
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
        self.heating_setpoint_idx = heating_setpoint_idx
        self.ev_layout = ev_layout

    def _comfort_penalty(self, t_in: float, cond_i: np.ndarray) -> float:
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

    def _ev_service_penalty(self, obs_i: np.ndarray, next_i: np.ndarray,
                            departed: Optional[List[dict]] = None) -> float:
        L = self.ev_layout
        if L is None:
            return 0.0
        was_connected = float(obs_i[L["connected_state"]]) > 0.5
        now_connected = float(next_i[L["connected_state"]]) > 0.5
        soc = float(obs_i[L["soc"]])
        required = float(obs_i[L["required_soc_departure"]])
        shortfall = max(required - soc, 0.0)
        if departed is not None:
            left_short = sum(max(float(d["required_soc"]) - float(d["soc"]), 0.0) for d in departed)
            shaping = self.cfg.ev_shaping * shortfall if was_connected and now_connected else 0.0
            return self.cfg.ev_service * left_short + shaping
        if not was_connected:
            return 0.0
        if not now_connected:
            return self.cfg.ev_service * shortfall
        return self.cfg.ev_shaping * shortfall

    def compute(
        self,
        obs_list: List[np.ndarray],
        actions: np.ndarray,
        next_obs_list: List[np.ndarray],
        prev_net_consumption: Optional[List[float]] = None,
        ev_departures: Optional[List[dict]] = None,
    ) -> List[float]:
        if prev_net_consumption is None:
            prev_net_consumption = [0.0] * self.B

        grid_draw = sum(max(0.0, float(o[_IDX_NET])) for o in next_obs_list)
        grid_term = -self.cfg.alpha_grid * (grid_draw / self.P_grid_max) ** 2

        rewards: List[float] = []
        for i in range(self.B):
            cond_i, next_i = obs_list[i], next_obs_list[i]
            e_i = float(next_i[_IDX_NET])
            price = float(cond_i[_IDX_PRICE])
            p_b = max(self.P_building_max, 1.0)

            r_econ = -self.cfg.mu * price * max(e_i, 0.0)

            build_term = -self.cfg.alpha_build * abs(e_i) / p_b
            ramp_term = -self.cfg.beta_ramp * abs(e_i - float(prev_net_consumption[i])) / p_b
            r_stab = grid_term + build_term + ramp_term

            r_comfort = -self.cfg.lambda_indoor * self._comfort_penalty(
                float(next_i[_IDX_T_IN]), cond_i)

            r_renew = 0.0
            if self.cfg.xi:
                solar_i = float(cond_i[_IDX_SOLAR])
                load_i = float(cond_i[_IDX_LOAD])
                if solar_i > 1e-8:
                    r_renew = self.cfg.xi * min(solar_i / max(load_i, 1e-8), 1.0)

            departed = (None if ev_departures is None
                        else [d for d in ev_departures if int(d["building"]) == i])
            r_ev = -self._ev_service_penalty(cond_i, next_i, departed)

            rewards.append(r_econ + r_stab + r_comfort + r_renew + r_ev)

        return rewards
