from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.battery import BatteryModel
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.environment import STEMSEnvironment

SOC = 19
WEEK = {"episode_time_steps": [(0, 167)]}


def test_linear_model_is_soc_plus_rate_times_action():
    m = BatteryModel.linear(np.array([0.1, 0.3]))
    np.testing.assert_allclose(m.next_soc(np.array([0.5, 0.5]), np.array([0.5, -1.0])),
                               [0.55, 0.2])


def test_linear_interval_is_the_closed_form_clip():
    rate = np.array([0.1, 0.3, 0.5])
    soc = np.array([0.15, 0.5, 0.85])
    a_lo, a_hi = BatteryModel.linear(rate).safe_interval(soc, 0.1, 0.9)
    np.testing.assert_allclose(a_lo, np.maximum((0.1 - soc) / rate, -1.0), atol=1e-6)
    np.testing.assert_allclose(a_hi, np.minimum((0.9 - soc) / rate, 1.0), atol=1e-6)


def test_out_of_reach_band_gives_the_recovery_action():
    m = BatteryModel.linear(np.array([0.1, 0.1]))
    a_lo, a_hi = m.safe_interval(np.array([0.0, 1.0]), 0.3, 0.7)
    np.testing.assert_allclose(a_lo, [1.0, -1.0])
    np.testing.assert_allclose(a_hi, [1.0, -1.0])


@pytest.fixture(scope="module")
def rollout():
    env = STEMSEnvironment(seed=0, heat_pump=True, env_kwargs=WEEK)
    model = env.battery_model()
    e = env.electrical_storage_action_index
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        soc = np.array([o[SOC] for o in obs], dtype=np.float64)
        a = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        a[:, e] = rng.uniform(-1, 1, env.num_buildings)
        out = env.step(a)
        obs, done = out[0], out[2] or out[3]
        executed = env.executed_actions[:, e]
        rows.append((soc, executed, model.next_soc(soc, executed),
                     np.array([o[SOC] for o in obs], dtype=np.float64)))
    floor = np.array([1.0 - b.electrical_storage.depth_of_discharge for b in env._env.buildings])
    return tuple(np.stack(x) for x in zip(*rows)) + (floor,)


def test_model_is_exact_inside_the_band(rollout):
    soc, action, predicted, real, floor = rollout
    limiter_inactive = real > floor[None, :] + 0.02
    inside = (real < 0.95) & limiter_inactive
    assert inside.mean() > 0.5
    assert np.abs(predicted - real)[inside].max() < 1e-4


def test_model_errs_toward_the_bound_being_protected(rollout):
    soc, action, predicted, real, floor = rollout
    discharging, charging = action < 0, action > 0
    assert (real - predicted)[discharging].min() > -1e-4
    assert (predicted - real)[charging].min() > -1e-4


def test_standing_still_at_the_lower_bound_needs_a_charge():
    env = STEMSEnvironment(seed=0, heat_pump=True, env_kwargs=WEEK)
    env.reset()
    a_lo, _ = env.battery_model().safe_interval(np.full(env.num_buildings, 0.1), 0.1, 0.9)
    assert np.all(a_lo > 0.0)


def test_barrier_keeps_the_band_and_uses_all_of_it():
    env = STEMSEnvironment(seed=0, heat_pump=True, env_kwargs=WEEK)
    cfg = CBFConfig()
    shield = CBFShield(cfg, env.num_buildings, battery_model=env.battery_model(),
                       elec_idx=env.electrical_storage_action_index,
                       safety_cfg=SafetyConfig(anticipatory=False, robust_margins=False))
    rng = np.random.default_rng(1)
    obs, _ = env.reset()
    socs, done = [], False
    while not done:
        raw = rng.uniform(-1, 1, (env.num_buildings, env.action_dim)).astype(np.float32)
        raw[:, env.hvac_action_index] = 0.0
        out = env.step(shield.project(raw, obs))
        obs, done = out[0], out[2] or out[3]
        socs.append([o[SOC] for o in obs])
    socs = np.array(socs)
    recovered = socs[3:]
    assert recovered.min() >= cfg.SOC_min and recovered.max() <= cfg.SOC_max
    floor = np.array([1.0 - b.electrical_storage.depth_of_discharge for b in env._env.buildings])
    reachable = np.maximum(cfg.SOC_min, floor - 0.01)
    assert np.all(recovered.min(axis=0) < reachable + 0.02), "the lower end must be usable"
    assert recovered.max(axis=0).min() > cfg.SOC_max - 0.02, "the upper end must be usable"
