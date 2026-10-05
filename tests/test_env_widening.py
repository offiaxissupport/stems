from __future__ import annotations

import numpy as np
import pytest

from stems.environment import (EV_SLOT_FIELDS, OBS_NAMES, STEMSEnvironment,
                               ev_slot_obs_names)

EV_DATASET = "citylearn_challenge_2022_phase_all_plus_evs"


@pytest.fixture(scope="module")
def travis():
    try:
        env = STEMSEnvironment(seed=0, heat_pump=True)
    except Exception as exc:
        pytest.skip(f"tx_travis_8b unavailable: {exc!r}")
    env.reset()
    return env


def test_travis_layout_is_unchanged(travis):
    assert travis.num_buildings == 8
    assert travis.obs_dim == 30
    assert travis.action_dim == 3
    assert travis.action_names == ["dhw_storage", "electrical_storage",
                                   "cooling_or_heating_device"]


def test_travis_has_no_ev_padding(travis):
    assert travis.ev_slots == 0
    assert travis.ev_action_indices() == []
    assert travis.ev_obs_layout() == []


def test_travis_has_no_absent_observations(travis):
    assert travis.absent_observations == []


def test_travis_every_building_owns_every_actuator(travis):
    mask = travis.action_presence_mask()
    assert mask.shape == (8, 3)
    assert mask.all()


def test_travis_isolation_modes_unchanged(travis):
    assert travis.resolve_control_indices("thermal") == [0, 2]
    assert travis.resolve_control_indices("heatpump") == [2]
    assert travis.resolve_control_indices("dhw") == [0]
    assert travis.resolve_control_indices("none") is None


def test_travis_ev_group_still_fails_loud(travis):
    with pytest.raises(RuntimeError, match="electric_vehicle_storage"):
        travis.resolve_control_indices("ev")


def test_missing_observations_raise_without_opt_in():
    try:
        with pytest.raises(RuntimeError, match="missing required observations"):
            STEMSEnvironment(schema=EV_DATASET, seed=0)
    except Exception as exc:
        if "not found" in str(exc):
            pytest.skip(f"EV dataset unavailable: {exc!r}")
        raise


@pytest.fixture(scope="module")
def ev_env():
    try:
        env = STEMSEnvironment(schema=EV_DATASET, seed=0, allow_missing_obs=True)
    except Exception as exc:
        pytest.skip(f"EV dataset unavailable: {exc!r}")
    env.reset()
    return env


def test_heterogeneous_buildings_become_one_shape(ev_env):
    obs, _ = ev_env.reset()
    assert ev_env.num_buildings == 17
    assert len({np.shape(o) for o in obs}) == 1
    assert np.shape(obs[0]) == (ev_env.obs_dim,)


def test_ev_slots_discovered_and_named(ev_env):
    assert ev_env.ev_slots >= 1
    ev_actions = ev_env.ev_action_indices()
    assert len(ev_actions) == ev_env.ev_slots
    for slot, idx in enumerate(ev_actions):
        assert ev_env.action_names[idx] == f"electric_vehicle_storage_{slot}"


def test_obs_dim_accounts_for_every_ev_slot(ev_env):
    base = len(OBS_NAMES)
    assert ev_env.obs_dim == base + ev_env.ev_slots * len(EV_SLOT_FIELDS)
    names = ev_env.obs_names
    for slot in range(ev_env.ev_slots):
        for n in ev_slot_obs_names(slot):
            assert n in names


def test_absent_thermal_features_are_reported(ev_env):
    absent = ev_env.absent_observations
    assert "indoor_dry_bulb_temperature" in absent
    assert "dhw_storage_soc" in absent


def test_only_some_buildings_own_a_charger(ev_env):
    mask = ev_env.action_presence_mask()
    ev_slot = ev_env.ev_action_indices()[0]
    owners = int(mask[:, ev_slot].sum())
    assert 0 < owners < ev_env.num_buildings


def test_padded_slots_read_zero_and_are_inert(ev_env):
    obs, _ = ev_env.reset()
    layout = ev_env.ev_obs_layout()[0]
    mask = ev_env.action_presence_mask()
    ev_slot = ev_env.ev_action_indices()[0]
    for b in range(ev_env.num_buildings):
        if not mask[b, ev_slot]:
            assert obs[b][layout["connected_state"]] == pytest.approx(0.0)
            assert obs[b][layout["battery_capacity"]] == pytest.approx(0.0)


def test_padded_action_never_reaches_citylearn(ev_env):
    ev_env.reset()
    actions = np.ones((ev_env.num_buildings, ev_env.action_dim), dtype=np.float32)
    native = ev_env._remap_actions(actions)
    for b in range(ev_env.num_buildings):
        assert len(native[b]) == ev_env._env.action_space[b].shape[0]
        assert np.all(np.isfinite(native[b]))


def test_charging_moves_soc_only_on_occupied_bays(ev_env):
    obs, _ = ev_env.reset()
    layout = ev_env.ev_obs_layout()[0]
    ev_slot = ev_env.ev_action_indices()[0]
    before = np.array([o[layout["soc"]] for o in obs], dtype=np.float32)
    connected = np.array([o[layout["connected_state"]] for o in obs]) > 0.5

    actions = np.zeros((ev_env.num_buildings, ev_env.action_dim), dtype=np.float32)
    actions[:, ev_slot] = 1.0
    for _ in range(3):
        obs, _, term, trunc, _ = ev_env.step(actions)
        if term or trunc:
            break
    after = np.array([o[layout["soc"]] for o in obs], dtype=np.float32)

    assert np.all(after[~connected] == pytest.approx(0.0)), "empty bay charged"
    if connected.any():
        assert np.any(after[connected] > before[connected] - 1e-6)


def test_rollout_is_stable_on_the_heterogeneous_schema(ev_env):
    obs, _ = ev_env.reset()
    actions = np.zeros((ev_env.num_buildings, ev_env.action_dim), dtype=np.float32)
    for _ in range(24):
        obs, _, term, trunc, _ = ev_env.step(actions)
        assert all(np.all(np.isfinite(o)) for o in obs)
        if term or trunc:
            break
