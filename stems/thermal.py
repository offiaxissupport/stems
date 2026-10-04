"""Thermal-side models: DHW pre-heating dynamics, demand forecasting, and CoP.

This module supports the heat-pump / DHW study (Phase 4). It contains three
pieces, all calibrated from the *real* CityLearn devices rather than assumed
constants -- the same discipline that fixed the battery CBF:

``DHWDynamics``
    Per-building hot-water tank dynamics. CityLearn charges the tank with
    ``energy = action * capacity`` (``Building.update_dhw_storage``) but clamps
    that to the heater's one-hour output ``P_nom * eta``. The achievable SOC gain
    per step is therefore

        charge_rate_i = min(action_bound_i, P_nom_i * eta_i / C_i),

    which is 0.56-0.85 on the eight Travis buildings -- a **time-to-heat from
    empty of 1.2-1.8 hours**. Hot water cannot be produced on demand; it has to
    be started ahead of time. That lead time is the whole point of the study.

``DHWDemandForecaster``
    A causal, online hour-of-day climatology of each building's DHW demand.
    CityLearn exposes ``dhw_demand`` for the *current* step only (there is no
    DHW forecast observation), so the forecaster learns an EWMA per
    (building, hour-of-day, weekday/weekend) from what has already been observed
    and applies a cold-weather uplift. Nothing from the future is read -- the
    forecast is honest and would work online.

``CoPModel``
    Weather-dependent coefficient of performance of the space-conditioning heat
    pump, using each building's real ``efficiency`` and target supply
    temperatures, following CityLearn's Carnot model
    (``energy_model.HeatPump.get_cop``). Used to convert an HVAC action into the
    electrical draw the power barriers must respect, and to detect when an
    approaching cold front will make heating expensive.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.environment import T_OUT_PRED_LEAD_H

# STEMS observation indices (see OBS_NAMES in environment.py)
IDX_DAY_TYPE = 0
IDX_HOUR = 1
IDX_T_OUT = 2          # outdoor_dry_bulb_temperature (now)
IDX_T_OUT_PRED = 3     # outdoor_dry_bulb_temperature_predicted_1/2/3 at 3, 4, 5
IDX_SOC_DHW = 18
IDX_DHW_DEMAND = 25

_MIN_COP = 0.5
_MAX_COP = 20.0


def hour_bin(obs: np.ndarray) -> int:
    """0-based hour-of-day bin. CityLearn's ``hour`` runs 1..24, not 0..23."""
    return (int(round(float(obs[IDX_HOUR]))) - 1) % 24


def is_weekend(obs: np.ndarray) -> bool:
    """CityLearn ``day_type``: 1 = Monday .. 7 = Sunday, 8 = holiday."""
    return int(round(float(obs[IDX_DAY_TYPE]))) in (6, 7, 8)


def outdoor_temperature_forecast(obs: np.ndarray, horizon: int) -> np.ndarray:
    """Outdoor temperature 1..``horizon`` hours ahead.

    Linear interpolation between the current reading and the +6/+12/+24 h
    predictions; held at the +24 h value beyond that.
    """
    hours = np.arange(1, max(0, int(horizon)) + 1, dtype=np.float32)
    knots_t = (0.0,) + T_OUT_PRED_LEAD_H
    knots_v = [float(obs[IDX_T_OUT])] + [float(obs[IDX_T_OUT_PRED + k]) for k in range(3)]
    return np.interp(hours, knots_t, knots_v).astype(np.float32)


# ---------------------------------------------------------------------------
# Tank dynamics
# ---------------------------------------------------------------------------

class DHWDynamics:
    """Per-building DHW tank charge dynamics read from the real environment.

    Parameters
    ----------
    capacity : (B,) tank capacity [kWh]
    nominal_power : (B,) DHW heater rated power [kW]
    efficiency : (B,) heater efficiency
    loss_coefficient : (B,) tank standby loss per hour
    action_bound : (B,) CityLearn's per-building bound on ``|action[0]|``
    """

    def __init__(self, capacity: np.ndarray, nominal_power: np.ndarray,
                 efficiency: np.ndarray, loss_coefficient: np.ndarray,
                 action_bound: np.ndarray) -> None:
        self.capacity = np.asarray(capacity, dtype=np.float32).reshape(-1)
        self.nominal_power = np.asarray(nominal_power, dtype=np.float32).reshape(-1)
        self.efficiency = np.asarray(efficiency, dtype=np.float32).reshape(-1)
        self.loss = np.asarray(loss_coefficient, dtype=np.float32).reshape(-1)
        self.action_bound = np.asarray(action_bound, dtype=np.float32).reshape(-1)
        # Energy the heater can deliver in one step, as a fraction of capacity.
        energy_limit = (self.nominal_power * self.efficiency
                        / np.maximum(self.capacity, 1e-6))
        self.charge_rate = np.clip(np.minimum(self.action_bound, energy_limit),
                                   1e-3, 1.0).astype(np.float32)

    @property
    def time_to_heat_h(self) -> np.ndarray:
        """Hours needed to fill an empty tank at maximum charge (1 / charge_rate)."""
        return (1.0 / self.charge_rate).astype(np.float32)

    def action_for_soc_gain(self, gain: np.ndarray) -> np.ndarray:
        """Smallest action achieving the requested SOC gain (may saturate)."""
        gain = np.asarray(gain, dtype=np.float32)
        return np.clip(gain, 0.0, self.charge_rate) / np.maximum(self.charge_rate, 1e-6) \
            * self.action_bound

    def summary(self) -> Dict[str, np.ndarray]:
        return {"capacity": self.capacity.copy(),
                "nominal_power": self.nominal_power.copy(),
                "efficiency": self.efficiency.copy(),
                "charge_rate": self.charge_rate.copy(),
                "time_to_heat_h": self.time_to_heat_h}


# ---------------------------------------------------------------------------
# Causal DHW demand forecaster
# ---------------------------------------------------------------------------

class DHWDemandForecaster:
    """Online hour-of-day climatology of per-building DHW demand.

    The estimate for (building i, hour h, weekend flag w) is an exponentially
    weighted mean of every previously observed demand at that slot:

        m_{i,h,w} <- (1 - a) m_{i,h,w} + a * d_{i,t}

    A cold-weather uplift accounts for colder inlet water and higher draw when
    the outdoor temperature drops below the running mean:

        dhat_{i,t+k} = m_{i,h+k,w} * (1 + g * relu(Tbar - That_{t+k}))

    Only past observations enter, so this is usable online and introduces no
    look-ahead into the study.
    """

    def __init__(self, num_buildings: int, alpha: float = 0.2,
                 warmup: int = 24, temp_gain: float = 0.02) -> None:
        self.B = int(num_buildings)
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.temp_gain = float(temp_gain)
        # (B, 24, 2) mean demand and observation counts.
        self._mean = np.zeros((self.B, 24, 2), dtype=np.float32)
        self._count = np.zeros((self.B, 24, 2), dtype=np.int32)
        self._t_out_mean = np.nan
        self._steps = 0

    # -- ingest ---------------------------------------------------------
    def update(self, obs_list: List[np.ndarray]) -> None:
        """Absorb one observed timestep (call *after* stepping the env)."""
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
        """True once enough history exists for the climatology to be meaningful."""
        return self._steps >= self.warmup

    # -- query ----------------------------------------------------------
    def forecast(self, obs_list: List[np.ndarray], horizon: int) -> np.ndarray:
        """Return (B,) forecast total DHW demand [kWh] over the next ``horizon`` hours.

        Falls back to the currently observed demand scaled by the horizon while
        the climatology is still warming up, which is conservative (it never
        reports *less* than what is happening right now).
        """
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


# ---------------------------------------------------------------------------
# Weather-dependent CoP
# ---------------------------------------------------------------------------

class CoPModel:
    """Carnot CoP of each building's heat pump, per CityLearn's ``HeatPump.get_cop``.

        CoP_heat(T) = eta * (T_target_h + 273.15) / (T_target_h - T)
        CoP_cool(T) = eta * (T_target_c + 273.15) / (T - T_target_c)

    clipped to [0.5, 20]. Parameters come from the live environment
    (``STEMSEnvironment.heat_pump_info``), not from assumed values.
    """

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
        """Electrical power [kW] drawn by an HVAC action at outdoor temperature T.

        CityLearn delivers ``P_out = min(|a| P_nom, P_nom) * CoP`` thermal for
        ``P_in = P_out / CoP = min(|a| P_nom, P_nom)`` electrical -- so the
        electrical draw is CoP-independent for a given *action*, but the thermal
        service bought by that draw is not. What the power barrier needs is the
        draw required to deliver a given thermal *service*, which is what makes
        cold hours expensive; ``thermal_to_electrical`` provides that.
        """
        a = np.clip(np.abs(np.asarray(hvac_action, dtype=np.float32).reshape(-1)), 0.0, 1.0)
        p_nom = self.p_h if heating else self.p_c
        return (a * p_nom).astype(np.float32)

    def thermal_to_electrical(self, thermal_kwh: np.ndarray, t_out: np.ndarray,
                              heating: bool) -> np.ndarray:
        """Electrical energy needed to deliver ``thermal_kwh`` at temperature T."""
        cop = self.cop(t_out, heating)
        return (np.asarray(thermal_kwh, dtype=np.float32).reshape(-1)
                / np.maximum(cop, 1e-3)).astype(np.float32)

    def cop_drop(self, obs_list: List[np.ndarray], horizon: int,
                 heating: bool) -> np.ndarray:
        """Relative CoP loss between now and the worst of the next ``horizon`` hours.

        Returns a value in [0, 1): 0 when the weather is not getting worse,
        approaching 1 when an incoming cold front will roughly halve efficiency.
        This is the weather-anticipation signal.
        """
        t_now = np.array([float(o[IDX_T_OUT]) for o in obs_list], dtype=np.float32)
        cop_now = self.cop(t_now, heating)
        worst = cop_now.copy()
        if horizon > 0:
            t_ahead = np.stack([outdoor_temperature_forecast(o, horizon) for o in obs_list])
            for k in range(t_ahead.shape[1]):
                worst = np.minimum(worst, self.cop(t_ahead[:, k], heating))
        return np.clip((cop_now - worst) / np.maximum(cop_now, 1e-3),
                       0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Barrier h4: DHW readiness
# ---------------------------------------------------------------------------

class DHWReadinessBarrier(DeadlineStorageBarrier):
    """Anticipatory hot-water readiness barrier (h4).

    A ``DeadlineStorageBarrier`` whose requirement is *forecast* rather than
    declared. The tank must hold enough energy to cover the demand expected over
    the pre-heat horizon L, plus a robust margin, plus a weather-anticipation
    term:

        soc_req_i = clip( Dhat_i(L) / C_i + m + kappa * copdrop_i, 0, soc_cap )

    where ``Dhat_i(L)`` is the causally forecast DHW demand over the next L hours
    and ``copdrop_i`` is the relative CoP loss expected over the forecast window.

    The deadline is *implicit*: hot water is wanted as soon as the forecast says
    so, so ``steps_to_deadline = 0`` and any unmet requirement is immediately
    urgent. That reproduces the original hot-water projection exactly -- an
    intentional constraint, since heat-pump-only results are benchmarked
    elsewhere and must stay reproducible. An electric vehicle uses the same base
    class with a *declared* deadline instead (see ``stems.ev``), which is what
    opens the door to deferring a charge to a cheaper hour.

    Because the tank charges at a finite ``charge_rate``, a requirement of
    ``soc_req`` must be started at least ``(soc_req - soc)/charge_rate`` hours in
    advance -- L is chosen against the measured time-to-heat.
    """

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
        """Forecast demand over the horizon, plus the cold-front uplift.

        The margin is applied by the base class, so it is deliberately absent
        here.
        """
        demand = self.forecaster.forecast(obs_list, self.horizon)
        soc = demand / np.maximum(self.dyn.capacity, 1e-6)
        if self.weather_gain > 0.0 and self.cop is not None:
            soc = soc + self.weather_gain * self.cop.cop_drop(
                obs_list, self.weather_horizon, heating=True)
        n = len(obs_list)
        return DeadlineRequirement(soc=soc,
                                   steps_to_deadline=np.zeros(n, dtype=np.float32),
                                   active=np.ones(n, dtype=bool))


# ---------------------------------------------------------------------------
# Convenience builder used by train.py / evaluate.py / study_heatpump.py
# ---------------------------------------------------------------------------

def build_thermal_stack(env, thermal_cfg, enable: bool = True):
    """Return ``(dhw_barrier, cop_model)`` calibrated from a live environment.

    ``enable=False`` (or a config with the mechanisms switched off) returns
    ``(None, None)``, which leaves the CBF shield in its battery-only behaviour.
    """
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
