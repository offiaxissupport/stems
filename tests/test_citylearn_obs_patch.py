"""CityLearn's state observations, and our correction (see ENDOGENOUS_OBS).

Unpatched, the indoor temperature observation is the dataset's uncontrolled
value for the next hour, so a controller cannot see its own HVAC actions; and
the final transition of an episode repeats the previous hour's state. Real
CityLearn only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.environment import STEMSEnvironment

WINDOW = {"episode_time_steps": [(0, 47)]}


def _rollout(patch: bool, hvac: float):
    env = STEMSEnvironment(seed=0, heat_pump=True, env_kwargs=WINDOW, patch_endogenous_obs=patch)
    j = env.obs_names.index("indoor_dry_bulb_temperature")
    env.reset()
    observed, applied, done = [], [], False
    while not done:
        applied.append(env._env.time_step)
        action = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        action[:, env.hvac_action_index] = hvac
        out = env.step(action)
        observed.append([o[j] for o in out[0]])
        done = out[2] or out[3]
    return env, np.array(observed), applied


def _series(env, applied, attr):
    return np.array([[getattr(b.energy_simulation, attr)[t] for b in env._env.buildings]
                     for t in applied])


def test_unpatched_temperature_is_the_uncontrolled_next_hour():
    env, observed, applied = _rollout(patch=False, hvac=-1.0)
    uncontrolled = np.array([[b.energy_simulation.indoor_dry_bulb_temperature_without_control[t + 1]
                              for b in env._env.buildings] for t in applied[:-1]])
    np.testing.assert_allclose(observed[:-1], uncontrolled, atol=1e-4)
    simulated = _series(env, applied, "indoor_dry_bulb_temperature")
    assert np.abs(simulated - observed).max() > 5.0, "full cooling in winter must be visible somewhere"


def test_patched_temperature_is_the_simulated_hour():
    env, observed, applied = _rollout(patch=True, hvac=-1.0)
    assert "endogenous_obs_from_simulated_hour" in env.citylearn_patches
    np.testing.assert_allclose(observed, _series(env, applied, "indoor_dry_bulb_temperature"), atol=1e-4)


def test_patched_final_transition_reports_its_own_hour():
    env = STEMSEnvironment(seed=0, heat_pump=True, env_kwargs=WINDOW)
    j = env.obs_names.index("net_electricity_consumption")
    rng = np.random.default_rng(0)
    env.reset()
    done = False
    while not done:
        t = env._env.time_step
        out = env.step(rng.uniform(-1, 1, (env.num_buildings, env.action_dim)).astype(np.float32))
        done = out[2] or out[3]
    expected = [b.net_electricity_consumption[t] for b in env._env.buildings]
    np.testing.assert_allclose([o[j] for o in out[0]], expected, atol=1e-4)


def test_metrics_match_citylearn_ground_truth():
    """Every scored quantity, recomputed from CityLearn's own series for the
    hours the actions were applied to, must equal MetricsCalculator's value."""
    from stems.config import CBFConfig
    from stems.metrics import MetricsCalculator

    env = STEMSEnvironment(seed=0, heat_pump=True, env_kwargs=WINDOW)
    m = MetricsCalculator(env.num_buildings, CBFConfig(), soc_rate=env.battery_info()["soc_rate"],
                          heating_setpoint_idx=env.heating_setpoint_idx)
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    hours, done = [], False
    while not done:
        hours.append(env._env.time_step)
        a = rng.uniform(-1, 1, (env.num_buildings, env.action_dim)).astype(np.float32)
        out = env.step(a)
        m.add_step(obs, a, out[0])
        obs, done = out[0], out[2] or out[3]
    k, H, bs = m.compute_all(), np.array(hours), env._env.buildings
    col = lambda f: np.array([np.asarray(f(b))[H] for b in bs]).T
    net = col(lambda b: b.net_electricity_consumption)
    imports = np.maximum(net, 0.0)
    es = lambda name: col(lambda b: getattr(b.energy_simulation, name))
    t_in, occ = es("indoor_dry_bulb_temperature"), es("occupant_count") > 0
    cool = es("indoor_dry_bulb_temperature_cooling_set_point")
    heat = es("indoor_dry_bulb_temperature_heating_set_point")
    expected = {
        "cost": (imports * col(lambda b: b.pricing.electricity_pricing)).sum(),
        "emission": (imports * col(lambda b: b.carbon_intensity.carbon_intensity)).sum(),
        "electricity_consumption": imports.sum(),
        "peak_import_kw": imports.sum(axis=1).max(),
        "discomfort_rate": (occ & ((t_in - cool > 2) | (heat - t_in > 2))).sum() / occ.sum(),
    }
    for key, value in expected.items():
        assert k[key] == pytest.approx(value, rel=1e-5), key
