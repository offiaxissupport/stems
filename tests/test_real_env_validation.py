from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.environment import STEMSEnvironment, OBS_NAMES, HEATPUMP_OBS_NAMES, ACTION_DIM

EXPECTED_BUILDINGS = 8
EXPECTED_ACTION_NAMES = ["dhw_storage", "electrical_storage", "cooling_or_heating_device"]


def test_env_type_is_real_not_mock():
    env = STEMSEnvironment(seed=0)
    assert env.env_type == "CityLearn", (
        f"Expected real CityLearn, got env_type={env.env_type!r}. "
        "A silent mock fallback would invalidate any result.")
    assert env.using_mock is False


def test_building_count():
    env = STEMSEnvironment(seed=0)
    assert env.num_buildings == EXPECTED_BUILDINGS, (
        f"Expected {EXPECTED_BUILDINGS} buildings from the TX Travis 8B schema, "
        f"got {env.num_buildings}.")


def test_action_layout():
    env = STEMSEnvironment(seed=0)
    assert env.action_dim == ACTION_DIM == 3
    assert env.action_names == EXPECTED_ACTION_NAMES, (
        f"Action layout drifted: expected {EXPECTED_ACTION_NAMES}, got {env.action_names}.")
    assert env.electrical_storage_action_index == 1
    assert env.hvac_action_index == 2


def test_observation_layout():
    env = STEMSEnvironment(seed=0)
    assert env.obs_dim == len(OBS_NAMES) == 28
    assert env.obs_names == OBS_NAMES

    env_hp = STEMSEnvironment(seed=0, heat_pump=True)
    assert env_hp.obs_dim == len(OBS_NAMES) + len(HEATPUMP_OBS_NAMES) == 30
    assert env_hp.heating_setpoint_idx == 28


def test_real_env_survives_a_short_rollout():
    env = STEMSEnvironment(seed=0)
    obs_list, _ = env.reset()
    assert len(obs_list) == EXPECTED_BUILDINGS
    for o in obs_list:
        assert o.shape == (env.obs_dim,)
        assert np.all(np.isfinite(o))

    no_op = np.zeros((EXPECTED_BUILDINGS, env.action_dim), dtype=np.float32)
    for _ in range(30):
        obs_list, rewards, terminated, truncated, _ = env.step(no_op)
        assert len(obs_list) == EXPECTED_BUILDINGS
        assert len(rewards) == EXPECTED_BUILDINGS
        for o in obs_list:
            assert np.all(np.isfinite(o))
        if terminated or truncated:
            break


def test_battery_info_is_per_building_and_plausible():
    env = STEMSEnvironment(seed=0)
    bi = env.battery_info()
    assert bi["soc_rate"].shape == (EXPECTED_BUILDINGS,)
    assert np.all(bi["soc_rate"] > 0.1), (
        f"soc_rate looks like the old hard-coded placeholder: {bi['soc_rate']}")


def _main():
    tests = [test_env_type_is_real_not_mock, test_building_count, test_action_layout,
             test_observation_layout, test_real_env_survives_a_short_rollout,
             test_battery_info_is_per_building_and_plausible]
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
