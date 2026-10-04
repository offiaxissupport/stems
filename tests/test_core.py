"""Core correctness tests for the rewritten STEMS pipeline.

Run directly (prints PASS/FAIL) or under pytest:
    .venv/Scripts/python tests/test_core.py
    .venv/Scripts/python -m pytest tests/test_core.py -q

The real-CityLearn tests share a single env (reset ~17s) to stay fast.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.environment import STEMSEnvironment
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.metrics import MetricsCalculator


def test_mock_is_loud_and_flagged():
    """force_mock must yield a clearly-flagged synthetic env (never silent)."""
    env = STEMSEnvironment(force_mock=True, seed=0)
    assert env.env_type == "mock"
    assert env.using_mock is True
    obs, _ = env.reset()
    assert len(obs) == env.num_buildings


def test_real_env_battery_calibration():
    """Real env exposes per-building soc_rate that conservatively bounds the
    observed SOC change (the calibration that fixes the violation rate)."""
    env = STEMSEnvironment(seed=0)
    assert env.env_type == "CityLearn"
    bi = env.battery_info()
    soc_rate = bi["soc_rate"]
    assert soc_rate.shape == (env.num_buildings,)
    obs, _ = env.reset()
    a = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (env.num_buildings, 1))
    nxt, *_ = env.step(a)
    observed = np.array([nxt[i][19] - obs[i][19] for i in range(env.num_buildings)])
    assert np.all(soc_rate >= observed - 1e-3), "soc_rate must conservatively bound dSOC"


def test_cbf_feasibility_and_safety_on_real_data():
    """The CBF must (a) never return all-zeros, and (b) keep observed SOC inside
    [SOC_min, SOC_max] under an adversarial random policy on real data."""
    env = STEMSEnvironment(seed=0)
    bi = env.battery_info()
    cbf = CBFShield(CBFConfig(), env.num_buildings, soc_rate=bi["soc_rate"],
                    nominal_power=bi["nominal_power"],
                    elec_idx=env.electrical_storage_action_index,
                    safety_cfg=SafetyConfig())
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    violations = 0
    saw_nonzero = False
    for _ in range(200):
        nominal = np.clip(rng.normal(0.3, 0.7, (env.num_buildings, env.action_dim)), -1, 1).astype(np.float32)
        safe = cbf.project(nominal, obs)
        # Recovery action is never the degenerate all-zeros vector.
        if np.any(np.abs(safe) > 1e-6):
            saw_nonzero = True
        obs, *_ = env.step(safe)
        soc = np.array([o[19] for o in obs])
        violations += int(((soc < CBFConfig().SOC_min) | (soc > CBFConfig().SOC_max)).sum())
    assert saw_nonzero, "CBF must produce non-zero actions"
    assert violations == 0, f"CBF should hold SOC in band on real data, got {violations} violations"


def test_metrics_single_source_violation_rate():
    """MetricsCalculator.safety_violation_rate equals the manual any-violation
    fraction over observed (step, building) outcomes."""
    cbf_cfg = CBFConfig()
    B = 4
    mc = MetricsCalculator(B, cbf_cfg, soc_rate=np.full(B, 0.2, np.float32))
    rng = np.random.default_rng(1)
    socs, nets = [], []
    for _ in range(50):
        prev = np.full((B,), 0.5, np.float32)
        nxt = []
        soc = rng.uniform(0.0, 1.0, B).astype(np.float32)   # some out of band on purpose
        net = rng.uniform(-20, 20, B).astype(np.float32)
        for i in range(B):
            o = np.zeros(28, np.float32); o[19] = soc[i]; o[20] = net[i]; o[26] = 1.0
            nxt.append(o)
        obs = [np.array([0]*19 + [prev[i]] + [0]*8, np.float32) for i in range(B)]
        mc.add_step(obs, np.zeros((B, 3), np.float32), nxt)
        socs.append(soc); nets.append(net)
    res = mc.compute_all()
    socs = np.stack(socs); nets = np.stack(nets)
    soc_v = (socs < cbf_cfg.SOC_min) | (socs > cbf_cfg.SOC_max)
    pow_v = np.abs(nets) > cbf_cfg.P_building_max
    grid_v = np.maximum(nets, 0).sum(1, keepdims=True) > cbf_cfg.P_grid_max
    manual = float((soc_v | pow_v | grid_v).mean())
    assert abs(res["safety_violation_rate"] - manual) < 1e-6


def test_season_aware_comfort():
    """A temperature inside the heating/cooling deadband is comfortable under the
    dual-setpoint model but wrongly penalised by the single cooling setpoint."""
    from stems.reward import STEMSReward
    obs = np.zeros(30, np.float32)
    obs[15] = 21.0    # indoor temperature
    obs[27] = 22.0    # cooling setpoint
    obs[28] = 20.0    # heating setpoint (heat-pump obs)
    obs[26] = 1.0     # occupied: comfort counts only when someone is home
    single = STEMSReward(num_buildings=1)
    dual = STEMSReward(num_buildings=1, heating_setpoint_idx=28)
    assert single._comfort_penalty(21.0, obs) > 0.0     # penalised vs cooling-only
    assert dual._comfort_penalty(21.0, obs) == 0.0      # comfortable in the deadband
    assert dual._comfort_penalty(24.0, obs) > 0.0       # too hot still penalised
    assert dual._comfort_penalty(18.0, obs) > 0.0       # too cold still penalised
    empty = obs.copy()
    empty[26] = 0.0
    assert dual._comfort_penalty(18.0, empty) == 0.0     # unoccupied: as the KPI


def _main():
    tests = [test_mock_is_loud_and_flagged, test_real_env_battery_calibration,
             test_cbf_feasibility_and_safety_on_real_data,
             test_metrics_single_source_violation_rate,
             test_season_aware_comfort]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _main()


def test_reward_bills_the_hour_at_its_own_price_and_imports_only():
    from stems.reward import STEMSReward
    r = STEMSReward(num_buildings=1)
    pre = np.zeros(30, np.float32)
    pre[21] = 0.10                      # tariff of the hour being simulated
    post = np.zeros(30, np.float32)
    post[21] = 0.90                     # next hour's tariff: must not be used
    post[20] = 2.0
    base = r.compute([pre], np.zeros((1, 3), np.float32), [np.zeros(30, np.float32)], [2.0])[0]
    got = r.compute([pre], np.zeros((1, 3), np.float32), [post], [2.0])[0]
    grid = -r.cfg.alpha_grid * (2.0 / r.P_grid_max) ** 2
    build = -r.cfg.alpha_build * 2.0 / r.P_building_max
    assert got - base == pytest.approx(-0.10 * 2.0 + grid + build + r.cfg.beta_ramp * 2.0 / r.P_building_max)
    post[20] = -3.0                     # exporting earns nothing (no export tariff)
    exported = r.compute([pre], np.zeros((1, 3), np.float32), [post], [-3.0])[0]
    assert exported <= base


def test_grid_reward_never_favours_more_draw():
    """The paper's (1 - draw/P)^2 is minimised at the cap and rewards overload."""
    from stems.reward import STEMSReward
    r = STEMSReward(num_buildings=1, P_grid_max=10.0, P_building_max=1e9)
    pre = np.zeros(30, np.float32)
    rewards = []
    for draw in (0.0, 5.0, 10.0, 15.0, 30.0):
        post = np.zeros(30, np.float32)
        post[20] = draw
        rewards.append(r.compute([pre], np.zeros((1, 3), np.float32), [post], [draw])[0])
    assert all(a > b for a, b in zip(rewards, rewards[1:])), rewards


def test_departure_penalty_is_on_what_the_car_left_with():
    """The charge of a car's last connected hour is never observed: the reward must
    take the state of charge at departure from the simulator's record."""
    from stems.reward import STEMSReward

    layout = {"connected_state": 30, "soc": 31, "required_soc_departure": 32, "departure_time": 33}
    r = STEMSReward(num_buildings=1, ev_layout=layout)
    before, after = np.zeros(36), np.zeros(36)
    before[30], before[31], before[32] = 1.0, 0.70, 0.80     # last hour starts 0.10 short
    after[30] = 0.0                                          # gone
    act = np.zeros((1, 4))
    full = r.compute([before], act, [after], ev_departures=[
        {"building": 0, "soc": 0.80, "required_soc": 0.80, "capacity_kwh": 60.0}])[0]
    short = r.compute([before], act, [after], ev_departures=[
        {"building": 0, "soc": 0.75, "required_soc": 0.80, "capacity_kwh": 60.0}])[0]
    blind = r.compute([before], act, [after])[0]             # observations only
    assert full - short == pytest.approx(r.cfg.ev_service * 0.05)
    assert full - blind == pytest.approx(r.cfg.ev_service * 0.10)   # charged as if the last hour did nothing
    nobody = r.compute([before], act, [after], ev_departures=[
        {"building": 3, "soc": 0.1, "required_soc": 0.9, "capacity_kwh": 60.0}])[0]
    assert nobody == pytest.approx(full)                     # another building's car
