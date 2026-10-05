from __future__ import annotations

import numpy as np
import pytest

from stems.deadline import (DeadlineRequirement, DeadlineStorageBarrier,
                            coupled_feasibility, prioritise)
from stems.ev import (EVChargerSpec, EVObsLayout, EVReadinessBarrier,
                      steps_to_departure)
from stems.thermal import (CoPModel, DHWDemandForecaster, DHWDynamics,
                           DHWReadinessBarrier, IDX_DHW_DEMAND, IDX_HOUR,
                           IDX_SOC_DHW, IDX_T_OUT, IDX_T_OUT_PRED)

OBS_DIM = 40


def _obs(soc_dhw=0.5, demand=0.0, hour=8, t_out=15.0, t_pred=(15.0, 15.0, 15.0)):
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[IDX_HOUR] = hour
    o[IDX_T_OUT] = t_out
    for k in range(3):
        o[IDX_T_OUT_PRED + k] = t_pred[k]
    o[IDX_SOC_DHW] = soc_dhw
    o[IDX_DHW_DEMAND] = demand
    return o


def _dhw_dynamics(B=2):
    return DHWDynamics(np.array([6.76, 11.20]), np.array([5.86, 6.41]),
                       np.array([0.98, 0.98]), np.zeros(B),
                       np.array([0.87, 0.57]))


def _legacy_dhw_projection(actions, obs_list, dyn, forecaster, cop,
                           horizon, margin, soc_cap, weather_gain,
                           weather_horizon, dhw_idx=0):
    actions = np.asarray(actions, dtype=np.float32).copy()
    soc = np.array([float(o[IDX_SOC_DHW]) for o in obs_list], dtype=np.float32)
    demand = forecaster.forecast(obs_list, horizon)
    req = demand / np.maximum(dyn.capacity, 1e-6) + margin
    if weather_gain > 0.0 and cop is not None:
        req = req + weather_gain * cop.cop_drop(obs_list, weather_horizon,
                                                heating=True)
    req = np.clip(req, 0.0, soc_cap).astype(np.float32)
    gap = np.maximum(req - soc, 0.0)
    a_min = np.minimum(
        np.clip(gap, 0.0, dyn.charge_rate) / np.maximum(dyn.charge_rate, 1e-6)
        * dyn.action_bound, dyn.action_bound)
    actions[:, dhw_idx] = np.maximum(actions[:, dhw_idx], a_min)
    return actions


@pytest.mark.parametrize("weather_gain", [0.0, 0.15])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_refactor_matches_legacy_dhw_to_float32_precision(horizon, weather_gain):
    rng = np.random.default_rng(0)
    dyn = _dhw_dynamics()
    cop = CoPModel(np.full(2, 0.29), np.full(2, 46.0), np.full(2, 0.29),
                   np.full(2, 9.0), np.full(2, 8.7), np.full(2, 4.8))
    margin, soc_cap = 0.05, 0.95

    f_new = DHWDemandForecaster(2, alpha=0.5, warmup=1, temp_gain=0.02)
    f_old = DHWDemandForecaster(2, alpha=0.5, warmup=1, temp_gain=0.02)
    barrier = DHWReadinessBarrier(dyn, f_new, cop, horizon=horizon,
                                  margin=margin, soc_cap=soc_cap,
                                  weather_gain=weather_gain, dhw_idx=0)

    for _ in range(200):
        obs = [_obs(soc_dhw=float(rng.uniform(0, 1)),
                    demand=float(rng.uniform(0, 4)),
                    hour=int(rng.integers(0, 24)),
                    t_out=float(rng.uniform(-10, 30)),
                    t_pred=tuple(rng.uniform(-15, 30, 3)))
               for _ in range(2)]
        f_new.update(obs)
        f_old.update(obs)
        nominal = rng.uniform(-1, 1, (2, 3)).astype(np.float32)

        got = barrier.project(nominal, obs)
        want = _legacy_dhw_projection(nominal, obs, dyn, f_old, cop, horizon,
                                      margin, soc_cap, weather_gain, 3)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_refactored_required_soc_matches_legacy():
    dyn = _dhw_dynamics()
    f1 = DHWDemandForecaster(2, alpha=0.5, warmup=1, temp_gain=0.0)
    f2 = DHWDemandForecaster(2, alpha=0.5, warmup=1, temp_gain=0.0)
    b = DHWReadinessBarrier(dyn, f1, None, horizon=2, margin=0.05, soc_cap=0.95)
    for hour in range(24):
        o = [_obs(demand=1.5, hour=hour), _obs(demand=2.5, hour=hour)]
        f1.update(o)
        f2.update(o)
    obs = [_obs(hour=6), _obs(hour=6)]
    legacy = np.clip(f2.forecast(obs, 2) / dyn.capacity + 0.05, 0.0, 0.95)
    np.testing.assert_allclose(b.required_soc(obs), legacy, rtol=1e-6)


def test_dhw_deadline_is_always_immediate():
    dyn = _dhw_dynamics()
    f = DHWDemandForecaster(2, alpha=0.5, warmup=1)
    b = DHWReadinessBarrier(dyn, f, None, horizon=2)
    for hour in range(24):
        f.update([_obs(demand=2.0, hour=hour)] * 2)
    u = b.urgency([_obs(soc_dhw=0.0, hour=6)] * 2)
    assert np.all(u["steps_to_deadline"] == 0.0)
    assert np.all(u["slack"] <= 0.0)


def _generic(rate, bound, req_soc, steps, active=True, margin=0.0, soc_cap=1.0,
             soc=0.0, capacity=50.0, efficiency=0.9):
    n = len(np.atleast_1d(req_soc))
    arr = lambda v: np.full(n, v, dtype=np.float32) if np.isscalar(v) else np.asarray(v, np.float32)
    return DeadlineStorageBarrier(
        rate=arr(rate), action_bound=arr(bound), action_index=0,
        capacity=arr(capacity), efficiency=arr(efficiency),
        requirement_fn=lambda obs: DeadlineRequirement(
            soc=arr(req_soc), steps_to_deadline=arr(steps), active=arr(active).astype(bool)),
        soc_fn=lambda obs: arr(soc), margin=margin, soc_cap=soc_cap, name="test")


def test_positive_slack_defers_charging():
    b = _generic(rate=0.25, bound=1.0, req_soc=0.8, steps=10, soc=0.2)
    obs = [np.zeros(OBS_DIM, np.float32)]
    out = b.project(np.zeros((1, 3), np.float32), obs)
    assert out[0, 0] == pytest.approx(0.0), "should defer: 10h available, ~3h needed"


def test_zero_slack_forces_charging():
    b = _generic(rate=0.25, bound=1.0, req_soc=0.8, steps=2, soc=0.2)
    obs = [np.zeros(OBS_DIM, np.float32)]
    out = b.project(np.zeros((1, 3), np.float32), obs)
    assert out[0, 0] > 0.9, "needs 3 steps, has 2 -> must charge at maximum"


def test_projection_is_monotone():
    b = _generic(rate=0.25, bound=1.0, req_soc=0.8, steps=0, soc=0.2)
    obs = [np.zeros(OBS_DIM, np.float32)]
    nominal = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    assert b.project(nominal, obs)[0, 0] >= nominal[0, 0] - 1e-6


def test_inactive_requirement_is_ignored():
    b = _generic(rate=0.25, bound=1.0, req_soc=0.9, steps=0, active=False, soc=0.0)
    obs = [np.zeros(OBS_DIM, np.float32)]
    out = b.project(np.zeros((1, 3), np.float32), obs)
    assert out[0, 0] == pytest.approx(0.0)


def test_energy_accounting_uses_capacity_and_efficiency():
    b = _generic(rate=0.5, bound=1.0, req_soc=0.8, steps=0, soc=0.3,
                 capacity=40.0, efficiency=0.8)
    obs = [np.zeros(OBS_DIM, np.float32)]
    assert b.energy_still_required_kwh(obs)[0] == pytest.approx(25.0, rel=1e-5)


EV_LAYOUT = EVObsLayout(connected_state=30, departure_time=31,
                        required_soc_departure=32, soc=33,
                        battery_capacity=34)


def _ev_obs(connected=1.0, departure=8.0, req_soc=0.8, soc=0.3,
            capacity=60.0, hour=22.0):
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[IDX_HOUR] = hour
    o[30], o[31], o[32], o[33], o[34] = connected, departure, req_soc, soc, capacity
    return o


def _ev_barrier(p_kw=7.4, bound=1.0):
    spec = EVChargerSpec(max_charging_power_kw=np.array([p_kw]),
                         efficiency=np.array([0.95]),
                         action_bound=np.array([bound]), action_index=0)
    return EVReadinessBarrier(EV_LAYOUT, spec)


def test_departure_countdown_is_used_verbatim():
    conn = np.array([True])
    assert steps_to_departure(np.array([10.0]), conn)[0] == pytest.approx(10.0)
    assert steps_to_departure(np.array([-1.0]), conn)[0] == pytest.approx(0.0)
    assert steps_to_departure(np.array([10.0]), np.array([False]))[0] == pytest.approx(0.0)


def test_ev_rate_tracks_the_connected_car():
    b = _ev_barrier()
    small = b.current_rate([_ev_obs(capacity=30.0)])[0]
    large = b.current_rate([_ev_obs(capacity=90.0)])[0]
    assert small > large
    assert small == pytest.approx(0.85 * 7.4 * 0.95 / 30.0, rel=1e-4)


def test_ev_rate_is_conservative_against_nameplate():
    b = _ev_barrier()
    nameplate = 7.4 * 0.95 / 60.0
    assert b.current_rate([_ev_obs(capacity=60.0)])[0] < nameplate


def test_ev_defers_overnight_then_commits():
    b = _ev_barrier()
    obs_early = [_ev_obs(departure=10.0, soc=0.3, req_soc=0.8)]
    assert b.project(np.zeros((1, 3), np.float32), obs_early)[0, 0] == pytest.approx(0.0)

    obs_late = [_ev_obs(departure=4.0, soc=0.3, req_soc=0.8)]
    assert b.project(np.zeros((1, 3), np.float32), obs_late)[0, 0] > 0.9


def test_ev_disconnected_bay_is_inert():
    b = _ev_barrier()
    obs = [_ev_obs(connected=0.0, soc=0.1, req_soc=0.9)]
    assert b.project(np.zeros((1, 3), np.float32), obs)[0, 0] == pytest.approx(0.0)
    assert not b.deadline_report(obs)["active"][0]


def test_ev_at_risk_is_reported_not_hidden():
    b = _ev_barrier(p_kw=3.0)
    obs = [_ev_obs(departure=1.0, soc=0.1, req_soc=1.0, capacity=90.0)]
    rep = b.deadline_report(obs)
    assert rep["at_risk"][0]
    assert rep["slack"][0] < 0


def test_coupled_feasibility_detects_an_empty_safe_set():
    obs = [np.zeros(OBS_DIM, np.float32)]
    b1 = _generic(rate=0.5, bound=1.0, req_soc=0.5, steps=1, soc=0.0,
                  capacity=40.0, efficiency=1.0)
    b2 = _generic(rate=0.5, bound=1.0, req_soc=0.5, steps=1, soc=0.0,
                  capacity=40.0, efficiency=1.0)
    b1.name, b2.name = "a", "b"
    res = coupled_feasibility([b1, b2], obs, power_cap_kw=10.0)
    assert res["energy_required_kwh"] == pytest.approx(40.0)
    assert not res["feasible"]
    assert res["shortfall_kwh"] == pytest.approx(30.0)


def test_coupled_feasibility_passes_when_the_cap_is_ample():
    obs = [np.zeros(OBS_DIM, np.float32)]
    b = _generic(rate=0.5, bound=1.0, req_soc=0.5, steps=1, soc=0.0,
                 capacity=40.0, efficiency=1.0)
    assert coupled_feasibility([b], obs, power_cap_kw=100.0)["feasible"]


def test_empty_barrier_set_is_trivially_feasible():
    assert coupled_feasibility([], [np.zeros(OBS_DIM, np.float32)], 10.0)["feasible"]


def test_prioritise_serves_the_earliest_deadline_first():
    obs = [np.zeros(OBS_DIM, np.float32)]
    urgent = _generic(rate=1.0, bound=1.0, req_soc=0.5, steps=1, soc=0.0,
                      capacity=20.0, efficiency=1.0)
    later = _generic(rate=1.0, bound=1.0, req_soc=0.5, steps=9, soc=0.0,
                     capacity=20.0, efficiency=1.0)
    urgent.name, later.name = "urgent", "later"
    res = prioritise([later, urgent], obs, power_cap_kw=10.0)
    assert res["allocation_kwh"]["urgent"][0] == pytest.approx(10.0)
    assert res["allocation_kwh"]["later"][0] == pytest.approx(0.0)
    assert [m[0] for m in res["missed"]] == ["later"]
