from __future__ import annotations

import numpy as np
import pytest

from stems.config import ThermalConfig
from stems.thermal import (CoPModel, DHWDemandForecaster, DHWDynamics,
                           DHWReadinessBarrier, IDX_DHW_DEMAND, IDX_HOUR,
                           IDX_SOC_DHW, IDX_T_OUT, IDX_T_OUT_PRED)

OBS_DIM = 30


def _obs(soc_dhw=0.5, demand=0.0, hour=8, t_out=15.0, t_pred=(15.0, 15.0, 15.0)):
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[IDX_HOUR] = hour
    o[IDX_T_OUT] = t_out
    for k in range(3):
        o[IDX_T_OUT_PRED + k] = t_pred[k]
    o[IDX_SOC_DHW] = soc_dhw
    o[IDX_DHW_DEMAND] = demand
    return o


def _dynamics(B=2, cap=(6.76, 11.20), nom=(5.86, 6.41), eff=(0.98, 0.98),
              bound=(0.87, 0.57)):
    return DHWDynamics(np.array(cap), np.array(nom), np.array(eff),
                       np.zeros(B), np.array(bound))


def test_charge_rate_is_the_binding_of_two_limits():
    dyn = _dynamics()
    assert dyn.charge_rate[0] == pytest.approx(0.850, abs=2e-3)
    assert dyn.charge_rate[1] == pytest.approx(0.561, abs=2e-3)


def test_time_to_heat_is_more_than_one_hour():
    dyn = _dynamics()
    assert np.all(dyn.time_to_heat_h > 1.0)
    assert dyn.time_to_heat_h[1] > dyn.time_to_heat_h[0]


def test_forecaster_is_causal_and_conservative_during_warmup():
    f = DHWDemandForecaster(num_buildings=2, warmup=24)
    obs = [_obs(demand=1.5, hour=7), _obs(demand=2.0, hour=7)]
    assert not f.ready
    out = f.forecast(obs, horizon=2)
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx(4.0)


def test_forecaster_learns_hour_of_day_structure():
    f = DHWDemandForecaster(num_buildings=1, alpha=0.5, warmup=24, temp_gain=0.0)
    for day in range(4):
        for hour in range(24):
            d = 3.0 if hour == 7 else 0.0
            f.update([_obs(demand=d, hour=hour)])
    assert f.ready
    at_six = f.forecast([_obs(hour=6)], horizon=2)[0]
    at_ten = f.forecast([_obs(hour=10)], horizon=2)[0]
    assert at_six > 1.0
    assert at_ten < 0.1
    assert at_six > 10 * max(at_ten, 1e-6)


def test_forecaster_cold_weather_uplift():
    f = DHWDemandForecaster(num_buildings=1, alpha=0.5, warmup=1, temp_gain=0.05)
    for hour in range(24):
        f.update([_obs(demand=1.0, hour=hour, t_out=20.0)])
    mild = f.forecast([_obs(hour=6, t_pred=(20.0,) * 3)], horizon=2)[0]
    cold = f.forecast([_obs(hour=6, t_pred=(0.0,) * 3)], horizon=2)[0]
    assert cold > mild


def test_cop_falls_as_it_gets_colder_when_heating():
    cop = CoPModel(np.array([0.29]), np.array([46.0]),
                   np.array([0.29]), np.array([9.0]),
                   np.array([8.7]), np.array([4.8]))
    warm = cop.cop(np.array([15.0]), heating=True)[0]
    cold = cop.cop(np.array([-5.0]), heating=True)[0]
    assert cold < warm


def test_cop_drop_flags_an_incoming_cold_front_only():
    cop = CoPModel(np.array([0.29]), np.array([46.0]),
                   np.array([0.29]), np.array([9.0]),
                   np.array([8.7]), np.array([4.8]))
    steady = cop.cop_drop([_obs(t_out=10.0, t_pred=(10.0,) * 3)], 12, heating=True)[0]
    front = cop.cop_drop([_obs(t_out=10.0, t_pred=(2.0, -4.0, -8.0))], 12, heating=True)[0]
    warming = cop.cop_drop([_obs(t_out=0.0, t_pred=(8.0,) * 3)], 12, heating=True)[0]
    assert steady == pytest.approx(0.0, abs=1e-6)
    assert warming == pytest.approx(0.0, abs=1e-6)
    assert front > 0.1


def test_cop_drop_only_counts_the_front_inside_the_horizon():
    cop = CoPModel(np.array([0.29]), np.array([46.0]),
                   np.array([0.29]), np.array([9.0]),
                   np.array([8.7]), np.array([4.8]))
    o = [_obs(t_out=10.0, t_pred=(2.0, -4.0, -8.0))]
    drops = [cop.cop_drop(o, L, heating=True)[0] for L in (1, 2, 6, 12, 24)]
    assert all(a < b for a, b in zip(drops, drops[1:])), drops


def _barrier(weather_gain=0.0, horizon=2, margin=0.05):
    dyn = _dynamics()
    f = DHWDemandForecaster(num_buildings=2, alpha=0.5, warmup=1, temp_gain=0.0)
    cop = CoPModel(np.full(2, 0.29), np.full(2, 46.0), np.full(2, 0.29),
                   np.full(2, 9.0), np.full(2, 8.7), np.full(2, 4.8))
    return DHWReadinessBarrier(dyn, f, cop, horizon=horizon, margin=margin,
                               weather_gain=weather_gain, dhw_idx=0), f, dyn


def test_barrier_raises_the_dhw_action_when_the_tank_is_short():
    barrier, f, _ = _barrier()
    for hour in range(24):
        f.update([_obs(demand=2.0, hour=hour), _obs(demand=2.0, hour=hour)])
    obs = [_obs(soc_dhw=0.0, hour=6), _obs(soc_dhw=0.0, hour=6)]
    nominal = np.zeros((2, 3), dtype=np.float32)
    safe = barrier.project(nominal, obs)
    assert np.all(safe[:, 0] > 0.0), "an empty tank facing demand must be pre-heated"


def test_barrier_never_lowers_a_nominal_action():
    barrier, f, dyn = _barrier()
    for hour in range(24):
        f.update([_obs(demand=0.1, hour=hour), _obs(demand=0.1, hour=hour)])
    obs = [_obs(soc_dhw=0.9, hour=6), _obs(soc_dhw=0.9, hour=6)]
    nominal = np.tile(np.array([0.6, 0.0, 0.0], dtype=np.float32), (2, 1))
    safe = barrier.project(nominal, obs)
    assert np.all(safe[:, 0] >= nominal[:, 0] - 1e-6)


def test_longer_horizon_never_lowers_the_requirement():
    reqs = []
    for L in (1, 2, 3):
        barrier, f, _ = _barrier(horizon=L)
        for hour in range(24):
            f.update([_obs(demand=1.0, hour=hour), _obs(demand=1.0, hour=hour)])
        reqs.append(barrier.required_soc([_obs(hour=6), _obs(hour=6)]))
    assert np.all(reqs[1] >= reqs[0] - 1e-6)
    assert np.all(reqs[2] >= reqs[1] - 1e-6)


def test_weather_anticipation_raises_the_requirement_before_a_cold_front():
    plain, f1, _ = _barrier(weather_gain=0.0)
    weathered, f2, _ = _barrier(weather_gain=0.15)
    for hour in range(24):
        for f in (f1, f2):
            f.update([_obs(demand=1.0, hour=hour), _obs(demand=1.0, hour=hour)])
    front = [_obs(hour=6, t_out=10.0, t_pred=(2.0, -4.0, -8.0))] * 2
    assert np.all(weathered.required_soc(front) > plain.required_soc(front))


def test_requirement_is_capped_below_a_full_tank():
    barrier, f, _ = _barrier(margin=0.5)
    for hour in range(24):
        f.update([_obs(demand=99.0, hour=hour), _obs(demand=99.0, hour=hour)])
    req = barrier.required_soc([_obs(hour=6), _obs(hour=6)])
    assert np.all(req <= ThermalConfig().dhw_soc_cap + 1e-6)


@pytest.fixture(scope="module")
def real_env():
    from stems.environment import STEMSEnvironment
    try:
        env = STEMSEnvironment(seed=0, heat_pump=True)
    except Exception as exc:
        pytest.skip(f"real CityLearn unavailable: {exc!r}")
    env.reset()
    return env


def test_dhw_patch_is_applied_and_recorded(real_env):
    assert "dhw_storage_capacity" in real_env.citylearn_patches


def test_real_dhw_action_actually_charges_the_tank(real_env):
    obs, _ = real_env.reset()
    soc_before = np.array([o[IDX_SOC_DHW] for o in obs], dtype=np.float32)
    a = np.zeros((real_env.num_buildings, real_env.action_dim), dtype=np.float32)
    a[:, real_env.dhw_action_index] = 0.8
    obs, *_ = real_env.step(a)
    soc_after = np.array([o[IDX_SOC_DHW] for o in obs], dtype=np.float32)
    assert np.all(soc_after > soc_before + 0.1), "DHW action is a no-op"


def test_predicted_charge_rate_matches_the_simulator(real_env):
    di = real_env.dhw_info()
    dyn = DHWDynamics(di["capacity"], di["nominal_power"], di["efficiency"],
                      di["loss_coefficient"], di["action_bound"])
    obs, _ = real_env.reset()
    soc0 = np.array([o[IDX_SOC_DHW] for o in obs], dtype=np.float32)
    a = np.zeros((real_env.num_buildings, real_env.action_dim), dtype=np.float32)
    a[:, real_env.dhw_action_index] = 1.0
    obs, *_ = real_env.step(a)
    soc1 = np.array([o[IDX_SOC_DHW] for o in obs], dtype=np.float32)
    gain = soc1 - soc0
    assert np.allclose(gain, dyn.charge_rate, rtol=0.15), \
        f"predicted {dyn.charge_rate} vs simulated {gain}"


def test_real_time_to_heat_exceeds_one_hour(real_env):
    di = real_env.dhw_info()
    dyn = DHWDynamics(di["capacity"], di["nominal_power"], di["efficiency"],
                      di["loss_coefficient"], di["action_bound"])
    assert np.all(dyn.time_to_heat_h > 1.0)
    assert 1.0 < float(dyn.time_to_heat_h.mean()) < 3.0


def test_heat_pump_info_is_physically_plausible(real_env):
    hi = real_env.heat_pump_info()
    assert np.all((hi["efficiency_heat"] > 0.15) & (hi["efficiency_heat"] < 0.4))
    assert np.all((hi["target_heat"] > 40.0) & (hi["target_heat"] < 60.0))
    assert np.all((hi["target_cool"] > 5.0) & (hi["target_cool"] < 15.0))
