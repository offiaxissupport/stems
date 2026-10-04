#!/usr/bin/env python3
"""Heat-pump / hot-water study: anticipatory pre-heating and weather factors.

Question
--------
In the ``thermal`` isolation mode the agent drives the domestic-hot-water (DHW)
heater and the bidirectional space-conditioning heat pump, with the battery
frozen. Hot water cannot be produced on demand: CityLearn charges the tank by
``energy = action * capacity`` but caps it at the heater's hourly output, so the
achievable SOC gain per step is ``min(action_bound, P_nom * eta / C)`` -- 0.56 to
0.85 on these eight buildings, i.e. a **time-to-heat from empty of 1.2-1.8
hours**. Water therefore has to be heated *before* it is needed. This study asks:

  1. How much readiness does an anticipatory pre-heat barrier buy, as a function
     of the pre-heat horizon L?
  2. Does folding weather into the picture -- CoP-aware power barriers, and
     inflating the requirement ahead of a forecast cold front -- help further,
     and at what cost?

Method
------
The same seeded nominal policy is replayed through the real CityLearn
environment under different thermal-shield configurations, exactly as
``ablation.py`` does for the battery tricks. This isolates the shield's marginal
effect from RL training noise.

Scoring is done by a **fixed evaluation barrier** (horizon ``--eval-horizon``,
no weather term) that is identical across every configuration and is *not* the
barrier being ablated -- otherwise each configuration would be graded against
its own moving target. Its demand forecaster is fed the same observation stream
in every run, so the requirement at time t is the same number for all rows.

Everything is causal: the DHW demand forecaster is an online hour-of-day
climatology built only from already-observed steps, and the weather term uses
only the 1/2/3-step outdoor-temperature predictions already present in the
observation vector.

Usage
-----
    .venv/Scripts/python study_heatpump.py --steps 1500 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig, ThermalConfig
from stems.environment import STEMSEnvironment
from stems.metrics import MetricsCalculator
from stems.thermal import CoPModel, DHWDemandForecaster, DHWDynamics, DHWReadinessBarrier

_IDX_T_IN, _IDX_T_OUT, _IDX_SOC_DHW, _IDX_NET = 15, 2, 18, 20
_IDX_PRICE, _IDX_DHW_DEMAND, _IDX_T_SET = 21, 25, 27


# ---------------------------------------------------------------------------
# Nominal policies (the shield is what is being ablated, not the policy)
# ---------------------------------------------------------------------------

class ReactivePolicy:
    """A plausible price-aware *reactive* hot-water controller -- the strawman.

    It tops the tank up only once demand has already appeared, and it refuses to
    heat while electricity is above its running-mean price. That is exactly the
    behaviour a finite time-to-heat punishes: by the time the draw is observed,
    the tank needs 1.2-1.8 hours to refill, so the next hour's draw finds it
    empty. Space conditioning is a proportional controller on the comfort error.

    Using this rather than Gaussian noise makes the pre-heat barrier's marginal
    effect interpretable: it is measured against a controller that is already
    trying to be economical, just not anticipatory.
    """

    def __init__(self, num_buildings: int, dyn: DHWDynamics, seed: int = 0) -> None:
        self.B = num_buildings
        self.dyn = dyn
        self.rng = np.random.default_rng(seed)
        self._price_mean = np.nan

    def __call__(self, obs: List[np.ndarray], action_dim: int,
                 dhw_idx: int, hvac_idx: int) -> np.ndarray:
        a = np.zeros((self.B, action_dim), dtype=np.float32)
        price = float(obs[0][_IDX_PRICE])
        self._price_mean = (price if np.isnan(self._price_mean)
                            else self._price_mean + 0.05 * (price - self._price_mean))
        expensive = price > self._price_mean

        for i, o in enumerate(obs):
            soc = float(o[_IDX_SOC_DHW])
            demand = float(o[_IDX_DHW_DEMAND])
            cap = float(self.dyn.capacity[i])
            # Reactive top-up: refill only what this hour's draw just consumed.
            gap = max(0.0, demand / max(cap, 1e-6) - soc)
            a_dhw = 0.0 if expensive else min(gap, float(self.dyn.action_bound[i]))
            # Proportional space conditioning on the comfort error.
            t_in, t_set = float(o[_IDX_T_IN]), float(o[_IDX_T_SET])
            t_set = t_set if t_set > 0 else 22.0
            a_hvac = float(np.clip(0.5 * (t_in - t_set), -1.0, 1.0))
            a[i, dhw_idx] = a_dhw
            a[i, hvac_idx] = a_hvac
        return a


class RandomPolicy:
    """Zero-mean Gaussian nominal actions (matches ``ablation.py``'s methodology)."""

    def __init__(self, num_buildings: int, seed: int = 0) -> None:
        self.B = num_buildings
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: List[np.ndarray], action_dim: int,
                 dhw_idx: int, hvac_idx: int) -> np.ndarray:
        return np.clip(self.rng.normal(0.0, 0.5, (self.B, action_dim)),
                       -1, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Configurations under test
# ---------------------------------------------------------------------------

def _configs() -> List[Tuple[str, ThermalConfig]]:
    """(label, thermal config). Each row adds one mechanism to the row above."""
    off = dict(dhw_readiness=False, weather_anticipation=False, cop_aware_power=False)
    return [
        ("no pre-heat (reactive)",
         ThermalConfig(**off)),
        ("+ CoP-aware power guard",
         ThermalConfig(dhw_readiness=False, weather_anticipation=False,
                       cop_aware_power=True)),
        ("pre-heat L=1",
         ThermalConfig(dhw_readiness=True, preheat_horizon=1,
                       weather_anticipation=False, cop_aware_power=True)),
        ("pre-heat L=2",
         ThermalConfig(dhw_readiness=True, preheat_horizon=2,
                       weather_anticipation=False, cop_aware_power=True)),
        ("pre-heat L=3",
         ThermalConfig(dhw_readiness=True, preheat_horizon=3,
                       weather_anticipation=False, cop_aware_power=True)),
        ("+ weather anticipation (L=2)",
         ThermalConfig(dhw_readiness=True, preheat_horizon=2,
                       weather_anticipation=True, cop_aware_power=True)),
        ("FULL (L=2, weather + CoP)",
         ThermalConfig(dhw_readiness=True, preheat_horizon=2, dhw_margin=0.08,
                       weather_anticipation=True, cop_aware_power=True)),
    ]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _build_barrier(env: STEMSEnvironment, tcfg: ThermalConfig,
                   dyn: DHWDynamics, cop: CoPModel,
                   forecaster: DHWDemandForecaster,
                   weather: bool) -> DHWReadinessBarrier:
    return DHWReadinessBarrier(
        dynamics=dyn, forecaster=forecaster, cop_model=cop,
        horizon=tcfg.preheat_horizon, margin=tcfg.dhw_margin,
        soc_cap=tcfg.dhw_soc_cap,
        weather_gain=tcfg.weather_gain if weather else 0.0,
        weather_horizon=tcfg.weather_horizon,
        dhw_idx=env.dhw_action_index)


def rollout(env: STEMSEnvironment, tcfg: ThermalConfig, steps: int, seed: int,
            eval_horizon: int, eval_margin: float,
            policy_name: str = "reactive",
            cbf_cfg: Optional[CBFConfig] = None,
            temp_offset: float = 0.0,
            temp_gradient: float = 0.0) -> Dict[str, float]:
    """Replay a seeded nominal policy under one thermal-shield configuration."""
    cbf_cfg = cbf_cfg or CBFConfig()
    bi, di, hi = env.battery_info(), env.dhw_info(), env.heat_pump_info()
    dyn = DHWDynamics(di["capacity"], di["nominal_power"], di["efficiency"],
                      di["loss_coefficient"], di["action_bound"])
    cop = CoPModel(hi["efficiency_heat"], hi["target_heat"],
                   hi["efficiency_cool"], hi["target_cool"],
                   hi["nominal_power_heat"], hi["nominal_power_cool"])

    # Barrier under test (its own forecaster) ...
    ctrl_forecaster = DHWDemandForecaster(env.num_buildings, tcfg.forecast_alpha,
                                          tcfg.forecast_warmup, tcfg.forecast_temp_gain)
    barrier = (_build_barrier(env, tcfg, dyn, cop, ctrl_forecaster,
                              tcfg.weather_anticipation)
               if tcfg.dhw_readiness else None)
    # ... and the fixed scorer, identical in every configuration.
    eval_forecaster = DHWDemandForecaster(env.num_buildings, tcfg.forecast_alpha,
                                          tcfg.forecast_warmup, tcfg.forecast_temp_gain)
    eval_barrier = DHWReadinessBarrier(dyn, eval_forecaster, cop_model=None,
                                       horizon=eval_horizon, margin=eval_margin,
                                       weather_gain=0.0,
                                       dhw_idx=env.dhw_action_index)

    shield = CBFShield(
        cbf_cfg, env.num_buildings, soc_rate=bi["soc_rate"],
        nominal_power=bi["nominal_power"],
        elec_idx=env.electrical_storage_action_index,
        safety_cfg=SafetyConfig(),
        enforce_soc=False,                      # thermal mode: battery frozen
        dhw_barrier=barrier,
        cop_model=cop if tcfg.cop_aware_power else None,
        hvac_idx=env.hvac_action_index)

    metrics = MetricsCalculator(env.num_buildings, cbf_cfg,
                                soc_rate=bi["soc_rate"],
                                heating_setpoint_idx=env.heating_setpoint_idx,
                                count_soc=False, dhw_barrier=eval_barrier)

    control = [env.dhw_action_index, env.hvac_action_index]
    policy = (ReactivePolicy(env.num_buildings, dyn, seed) if policy_name == "reactive"
              else RandomPolicy(env.num_buildings, seed))
    env.set_temp_offset(temp_offset)
    env.set_weather_front(temp_gradient)
    obs, _ = env.reset()
    env.set_temp_offset(temp_offset)
    env.set_weather_front(temp_gradient)
    hvac_kwh = dhw_kwh = dhw_spend = 0.0
    for _ in range(steps):
        nominal = policy(obs, env.action_dim, env.dhw_action_index,
                         env.hvac_action_index)
        actions = shield.project(nominal, obs)
        mask = np.zeros(env.action_dim, dtype=np.float32)
        mask[control] = 1.0                     # freeze the battery (isolate=thermal)
        actions = actions * mask
        # Electrical energy the heat pump draws, for the weather-cost breakdown.
        t_out = np.array([float(o[_IDX_T_OUT]) for o in obs], dtype=np.float32)
        heating = t_out < 20.0
        p_nom = np.where(heating, cop.p_h, cop.p_c)
        hvac_kwh += float((np.abs(actions[:, env.hvac_action_index]) * p_nom).sum())
        # DHW electrical energy and what it was paid for, so load-shifting shows up.
        a_dhw = np.maximum(actions[:, env.dhw_action_index], 0.0)
        headroom = np.maximum(
            1.0 - np.array([float(o[_IDX_SOC_DHW]) for o in obs], dtype=np.float32), 0.0)
        charged = np.minimum(a_dhw, headroom) * dyn.capacity
        e_dhw = float((charged / np.maximum(dyn.efficiency, 1e-6)).sum())
        dhw_kwh += e_dhw
        dhw_spend += e_dhw * float(obs[0][_IDX_PRICE])

        next_obs, _, term, trunc, _ = env.step(actions)
        metrics.add_step(obs, actions, next_obs)
        # Both forecasters see the same stream, after the step (causal).
        ctrl_forecaster.update(next_obs)
        eval_forecaster.update(next_obs)
        obs = next_obs
        if term or trunc:
            break

    out = metrics.compute_all()
    out["hvac_electricity_kwh"] = hvac_kwh
    out["dhw_electricity_kwh"] = dhw_kwh
    # Mean price paid per kWh of hot-water heating: the load-shifting signal.
    out["dhw_price_per_kwh"] = dhw_spend / dhw_kwh if dhw_kwh > 1e-9 else 0.0
    return out


# ---------------------------------------------------------------------------

_REPORT_KEYS = ["dhw_readiness_rate", "dhw_demand_covered_rate", "dhw_deficit_kwh",
                "dhw_soc_mean", "cost", "emission", "avg_daily_peak",
                "electricity_consumption", "ramping_rate", "hvac_electricity_kwh",
                "dhw_electricity_kwh", "dhw_price_per_kwh",
                "discomfort_rate", "safety_violation_rate"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Heat-pump / DHW pre-heating and weather study on real CityLearn")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--schema", type=str, default=None)
    ap.add_argument("--eval-horizon", type=int, default=2,
                    help="pre-heat horizon of the fixed scoring barrier")
    ap.add_argument("--eval-margin", type=float, default=0.05)
    ap.add_argument("--policy", choices=["reactive", "random"], default="reactive",
                    help="nominal controller the shield corrects")
    ap.add_argument("--stress", action="store_true",
                    help="Cold-snap stress scenario: show the controller a "
                         "--temp-offset degC colder outdoor temperature and derate "
                         "the per-building power cap to --stress-cap kW, so the "
                         "weather-dependent mechanisms actually bind.")
    ap.add_argument("--temp-offset", type=float, default=-18.0,
                    help="degC added to the observed outdoor temperature under --stress")
    ap.add_argument("--temp-gradient", type=float, default=-4.0,
                    help="degC per hour of *forecast* cooling under --stress. A constant "
                         "offset cannot exercise the cold-front term, which keys on a "
                         "change rather than a level; this ramps the forecast only.")
    ap.add_argument("--stress-cap", type=float, default=12.0,
                    help="per-building power cap (kW) under --stress")
    ap.add_argument("--out", type=str, default="heatpump_study_results.json")
    args = ap.parse_args()

    # Calibration first: the physical lead time that motivates the study.
    probe = STEMSEnvironment(schema=args.schema, seed=args.seeds[0], heat_pump=True)
    probe.reset()
    di = probe.dhw_info()
    dyn = DHWDynamics(di["capacity"], di["nominal_power"], di["efficiency"],
                      di["loss_coefficient"], di["action_bound"])
    calib = {"capacity_kwh": di["capacity"].tolist(),
             "heater_kw": di["nominal_power"].tolist(),
             "action_bound": di["action_bound"].tolist(),
             "charge_rate_soc_per_h": dyn.charge_rate.tolist(),
             "time_to_heat_h": dyn.time_to_heat_h.tolist()}
    print(f"[study] real CityLearn | isolate=thermal | policy={args.policy} | "
          f"steps={args.steps} | seeds={args.seeds}")
    print(f"[study] simulator patches active: {probe.citylearn_patches or 'none'}")
    print("[study] DHW calibration (per building):")
    for i in range(len(dyn.charge_rate)):
        print(f"   b{i}: tank={di['capacity'][i]:6.2f} kWh  heater={di['nominal_power'][i]:5.2f} kW"
              f"  charge_rate={dyn.charge_rate[i]:.3f} SOC/h  time_to_heat={dyn.time_to_heat_h[i]:.2f} h")
    print(f"[study] mean time-to-heat = {float(dyn.time_to_heat_h.mean()):.2f} h "
          f"-> a pre-heat horizon of L>=2 is the physically motivated choice\n")

    # Stress scenario. The temperature offset perturbs the outdoor temperature the
    # controller *observes* (and forecasts from), not CityLearn's internal physics,
    # so this tests whether the weather-anticipation logic responds correctly to a
    # cold-snap signal -- it is not a claim about true energy use in a cold snap.
    # The derated power cap is a genuine constraint change: at nominal 80 kW the
    # per-building barrier never binds on this dataset (loads are 10-40 kW), so the
    # CoP-aware guard has nothing to do until the cap is tightened.
    cbf_cfg = CBFConfig(P_building_max=args.stress_cap) if args.stress else CBFConfig()
    temp_offset = args.temp_offset if args.stress else 0.0
    temp_gradient = args.temp_gradient if args.stress else 0.0
    if args.stress:
        print(f"[study] STRESS: observed T_out {temp_offset:+.1f} degC with a "
              f"{temp_gradient:+.1f} degC/h forecast front, "
              f"per-building cap {args.stress_cap:.0f} kW (nominal 80 kW)\n")

    results: Dict[str, Dict[str, float]] = {}
    for label, tcfg in _configs():
        per_seed = []
        for s in args.seeds:
            env = STEMSEnvironment(schema=args.schema, seed=s, heat_pump=True)
            per_seed.append(rollout(env, tcfg, args.steps, s,
                                    args.eval_horizon, args.eval_margin,
                                    args.policy, cbf_cfg, temp_offset,
                                    temp_gradient))
        agg = {k: float(np.mean([r[k] for r in per_seed])) for k in _REPORT_KEYS}
        agg["readiness_std"] = float(np.std([r["dhw_readiness_rate"] for r in per_seed]))
        results[label] = agg
        print(f"{label:30s} ready={agg['dhw_readiness_rate']:.3f}+/-{agg['readiness_std']:.3f} "
              f"covered={agg['dhw_demand_covered_rate']:.3f} "
              f"deficit={agg['dhw_deficit_kwh']:8.1f}kWh  "
              f"cost={agg['cost']:8.1f}  dhw_kwh={agg['dhw_electricity_kwh']:7.1f}  "
              f"$/kWh={agg['dhw_price_per_kwh']:.4f}  "
              f"viol={agg['safety_violation_rate']:.3f}")

    payload = {"meta": {"env_type": "CityLearn", "isolate": "thermal",
                        "steps": args.steps, "seeds": args.seeds,
                        "policy": args.policy, "stress": args.stress,
                        "temp_offset": temp_offset,
                        "temp_gradient": temp_gradient,
                        "p_building_max": cbf_cfg.P_building_max,
                        "citylearn_patches": probe.citylearn_patches,
                        "eval_horizon": args.eval_horizon,
                        "eval_margin": args.eval_margin},
               "dhw_calibration": calib,
               "results": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[study] wrote {args.out}")


if __name__ == "__main__":
    main()
