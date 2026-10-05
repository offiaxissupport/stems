from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.environment import T_OUT_PRED_LEAD_H

IDX_DAY_TYPE = 0
IDX_HOUR = 1
IDX_T_OUT = 2
IDX_T_OUT_PRED = 3
IDX_SOC_DHW = 18
IDX_DHW_DEMAND = 25

_MIN_COP = 0.5
_MAX_COP = 20.0


def hour_bin(obs: np.ndarray) -> int:
    return (int(round(float(obs[IDX_HOUR]))) - 1) % 24


def is_weekend(obs: np.ndarray) -> bool:
    return int(round(float(obs[IDX_DAY_TYPE]))) in (6, 7, 8)


def outdoor_temperature_forecast(obs: np.ndarray, horizon: int) -> np.ndarray:
    hours = np.arange(1, max(0, int(horizon)) + 1, dtype=np.float32)
    knots_t = (0.0,) + T_OUT_PRED_LEAD_H
    knots_v = [float(obs[IDX_T_OUT])] + [float(obs[IDX_T_OUT_PRED + k]) for k in range(3)]
    return np.interp(hours, knots_t, knots_v).astype(np.float32)


class DHWDynamics:
    def __init__(self, capacity: np.ndarray, nominal_power: np.ndarray,
                 efficiency: np.ndarray, loss_coefficient: np.ndarray,
                 action_bound: np.ndarray) -> None:
        self.capacity = np.asarray(capacity, dtype=np.float32).reshape(-1)
        self.nominal_power = np.asarray(nominal_power, dtype=np.float32).reshape(-1)
        self.efficiency = np.asarray(efficiency, dtype=np.float32).reshape(-1)
        self.loss = np.asarray(loss_coefficient, dtype=np.float32).reshape(-1)
        self.action_bound = np.asarray(action_bound, dtype=np.float32).reshape(-1)
        energy_limit = (self.nominal_power * self.efficiency
                        / np.maximum(self.capacity, 1e-6))
        self.charge_rate = np.clip(np.minimum(self.action_bound, energy_limit),
                                   1e-3, 1.0).astype(np.float32)

    @property
    def time_to_heat_h(self) -> np.ndarray:
        return (1.0 / self.charge_rate).astype(np.float32)

    def action_for_soc_gain(self, gain: np.ndarray) -> np.ndarray:
        gain = np.asarray(gain, dtype=np.float32)
        return np.clip(gain, 0.0, self.charge_rate) / np.maximum(self.charge_rate, 1e-6) \
            * self.action_bound

    def summary(self) -> Dict[str, np.ndarray]:
        return {"capacity": self.capacity.copy(),
                "nominal_power": self.nominal_power.copy(),
                "efficiency": self.efficiency.copy(),
                "charge_rate": self.charge_rate.copy(),
                "time_to_heat_h": self.time_to_heat_h}


class DHWDemandForecaster:
    def __init__(self, num_buildings: int, alpha: float = 0.2,
                 warmup: int = 24, temp_gain: float = 0.02) -> None:
        self.B = int(num_buildings)
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.temp_gain = float(temp_gain)
        self._mean = np.zeros((self.B, 24, 2), dtype=np.float32)
        self._count = np.zeros((self.B, 24, 2), dtype=np.int32)
        self._t_out_mean = np.nan
        self._steps = 0

    def update(self, obs_list: List[np.ndarray]) -> None:
        for i, o in enumerate(obs_list):
            h = hour_bin(o)
            w = int(is_weekend(o))
            d = max(0.0, float(o[IDX_DHW_DEMAND]))
            if self._count[i, h, w] == 0:
                self._mean[i, h, w] = d
            else:
                self._mean[i, h, w] += self.alpha * (d - self._mean[i, h, w])
            self._count[i, h, w] += 1
        t_out = float(obs_list[0][IDX_T_OUT])
        self._t_out_mean = t_out if np.isnan(self._t_out_mean) else \
            self._t_out_mean + 0.01 * (t_out - self._t_out_mean)
        self._steps += 1

    @property
    def ready(self) -> bool:
        return self._steps >= self.warmup

    def forecast(self, obs_list: List[np.ndarray], horizon: int) -> np.ndarray:
        horizon = max(0, int(horizon))
        out = np.zeros(self.B, dtype=np.float32)
        if horizon == 0:
            return out
        t_hat = outdoor_temperature_forecast(obs_list[0], horizon)
        for i, o in enumerate(obs_list):
            if not self.ready:
                out[i] = horizon * max(0.0, float(o[IDX_DHW_DEMAND]))
                continue
            h0 = hour_bin(o)
            w = int(is_weekend(o))
            total = 0.0
            for k in range(horizon):
                h = (h0 + k + 1) % 24
                base = float(self._mean[i, h, w]) if self._count[i, h, w] > 0 \
                    else float(self._mean[i].max())
                uplift = 1.0
                if not np.isnan(self._t_out_mean):
                    uplift += self.temp_gain * max(0.0, self._t_out_mean - t_hat[k])
                total += base * uplift
            out[i] = total
        return out


class CoPModel:
    def __init__(self, efficiency_heat: np.ndarray, target_heat: np.ndarray,
                 efficiency_cool: np.ndarray, target_cool: np.ndarray,
                 nominal_power_heat: np.ndarray, nominal_power_cool: np.ndarray) -> None:
        f = lambda x: np.asarray(x, dtype=np.float32).reshape(-1)
        self.eta_h, self.t_h = f(efficiency_heat), f(target_heat)
        self.eta_c, self.t_c = f(efficiency_cool), f(target_cool)
        self.p_h, self.p_c = f(nominal_power_heat), f(nominal_power_cool)

    def cop(self, t_out: np.ndarray, heating: bool) -> np.ndarray:
        t_out = np.asarray(t_out, dtype=np.float32).reshape(-1)
        if heating:
            denom = np.maximum(self.t_h - t_out, 1e-3)
            cop = self.eta_h * (self.t_h + 273.15) / denom
        else:
            denom = np.maximum(t_out - self.t_c, 1e-3)
            cop = self.eta_c * (self.t_c + 273.15) / denom
        return np.clip(cop, _MIN_COP, _MAX_COP).astype(np.float32)

    def electrical_draw(self, hvac_action: np.ndarray, t_out: np.ndarray,
                        heating: bool) -> np.ndarray:
        a = np.clip(np.abs(np.asarray(hvac_action, dtype=np.float32).reshape(-1)), 0.0, 1.0)
        p_nom = self.p_h if heating else self.p_c
        return (a * p_nom).astype(np.float32)

    def thermal_to_electrical(self, thermal_kwh: np.ndarray, t_out: np.ndarray,
                              heating: bool) -> np.ndarray:
        cop = self.cop(t_out, heating)
        return (np.asarray(thermal_kwh, dtype=np.float32).reshape(-1)
                / np.maximum(cop, 1e-3)).astype(np.float32)

    def cop_drop(self, obs_list: List[np.ndarray], horizon: int,
                 heating: bool) -> np.ndarray:
        t_now = np.array([float(o[IDX_T_OUT]) for o in obs_list], dtype=np.float32)
        cop_now = self.cop(t_now, heating)
        worst = cop_now.copy()
        if horizon > 0:
            t_ahead = np.stack([outdoor_temperature_forecast(o, horizon) for o in obs_list])
            for k in range(t_ahead.shape[1]):
                worst = np.minimum(worst, self.cop(t_ahead[:, k], heating))
        return np.clip((cop_now - worst) / np.maximum(cop_now, 1e-3),
                       0.0, 1.0).astype(np.float32)


class DHWReadinessBarrier(DeadlineStorageBarrier):
    def __init__(self, dynamics: DHWDynamics, forecaster: DHWDemandForecaster,
                 cop_model: Optional[CoPModel] = None, horizon: int = 2,
                 margin: float = 0.05, soc_cap: float = 0.95,
                 weather_gain: float = 0.0, weather_horizon: int = 3,
                 dhw_idx: int = 0) -> None:
        self.dyn = dynamics
        self.forecaster = forecaster
        self.cop = cop_model
        self.horizon = int(horizon)
        self.weather_gain = float(weather_gain)
        self.weather_horizon = int(weather_horizon)
        self.dhw_idx = int(dhw_idx)
        super().__init__(
            rate=dynamics.charge_rate,
            action_bound=dynamics.action_bound,
            action_index=dhw_idx,
            capacity=dynamics.capacity,
            efficiency=dynamics.efficiency,
            requirement_fn=self._requirement,
            soc_fn=lambda obs: np.array([float(o[IDX_SOC_DHW]) for o in obs],
                                        dtype=np.float32),
            margin=margin,
            soc_cap=soc_cap,
            name="dhw",
        )

    def _requirement(self, obs_list: List[np.ndarray]) -> DeadlineRequirement:
        demand = self.forecaster.forecast(obs_list, self.horizon)
        soc = demand / np.maximum(self.dyn.capacity, 1e-6)
        if self.weather_gain > 0.0 and self.cop is not None:
            soc = soc + self.weather_gain * self.cop.cop_drop(
                obs_list, self.weather_horizon, heating=True)
        n = len(obs_list)
        return DeadlineRequirement(soc=soc,
                                   steps_to_deadline=np.zeros(n, dtype=np.float32),
                                   active=np.ones(n, dtype=bool))


def build_thermal_stack(env, thermal_cfg, enable: bool = True):
    cop_params = env.heat_pump_info()
    cop = CoPModel(cop_params["efficiency_heat"], cop_params["target_heat"],
                   cop_params["efficiency_cool"], cop_params["target_cool"],
                   cop_params["nominal_power_heat"], cop_params["nominal_power_cool"])
    if not enable:
        return None, None
    barrier = None
    if thermal_cfg.dhw_readiness and env.dhw_action_index >= 0:
        di = env.dhw_info()
        dyn = DHWDynamics(di["capacity"], di["nominal_power"], di["efficiency"],
                          di["loss_coefficient"], di["action_bound"])
        forecaster = DHWDemandForecaster(env.num_buildings,
                                         thermal_cfg.forecast_alpha,
                                         thermal_cfg.forecast_warmup,
                                         thermal_cfg.forecast_temp_gain)
        barrier = DHWReadinessBarrier(
            dyn, forecaster, cop_model=cop,
            horizon=thermal_cfg.preheat_horizon, margin=thermal_cfg.dhw_margin,
            soc_cap=thermal_cfg.dhw_soc_cap,
            weather_gain=(thermal_cfg.weather_gain
                          if thermal_cfg.weather_anticipation else 0.0),
            weather_horizon=thermal_cfg.weather_horizon,
            dhw_idx=env.dhw_action_index)
    return barrier, (cop if thermal_cfg.cop_aware_power else None)
