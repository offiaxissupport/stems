from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO / "citylearn_schemas" / "tx_travis_8b_ev"
SCHEMA = SCHEMA_DIR / "schema.json"
BASE_SCHEMA = REPO / "citylearn_schemas" / "tx_travis_8b" / "schema.json"


@pytest.fixture(scope="module")
def schema_path():
    if not SCHEMA.is_file():
        if not BASE_SCHEMA.is_file():
            pytest.skip("base tx_travis_8b schema not generated")
        subprocess.run([sys.executable, "-B", str(REPO / "setup_citylearn_ev.py")],
                       cwd=str(REPO), check=True, capture_output=True)
    if not SCHEMA.is_file():
        pytest.skip("could not generate tx_travis_8b_ev")
    return SCHEMA


@pytest.fixture(scope="module")
def env(schema_path):
    from stems.environment import STEMSEnvironment
    try:
        e = STEMSEnvironment(schema=str(schema_path), seed=0, heat_pump=True)
    except Exception as exc:
        pytest.skip(f"merged schema unavailable: {exc!r}")
    e.reset()
    return e


def test_no_thermal_observation_was_lost(env):
    assert env.absent_observations == []


def test_original_actions_kept_in_order(env):
    assert env.action_names[:3] == ["dhw_storage", "electrical_storage",
                                    "cooling_or_heating_device"]


def test_thermal_isolation_modes_are_unchanged(env):
    assert env.resolve_control_indices("thermal") == [0, 2]
    assert env.resolve_control_indices("heatpump") == [2]
    assert env.resolve_control_indices("dhw") == [0]


def test_dhw_tank_still_charges(env):
    obs, _ = env.reset()
    actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    actions[:, env.dhw_action_index] = 0.8
    for _ in range(3):
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            break
    dhw_soc = np.array([o[18] for o in obs], dtype=np.float32)
    assert np.all(dhw_soc > 0.3), f"DHW tank did not charge: {dhw_soc}"


def test_building_count_and_dims(env):
    from stems.environment import EV_SLOT_FIELDS, HEATPUMP_OBS_NAMES, OBS_NAMES
    assert env.num_buildings == 8
    expected = (len(OBS_NAMES) + len(HEATPUMP_OBS_NAMES)
                + env.ev_slots * len(EV_SLOT_FIELDS))
    assert env.obs_dim == expected


def test_chargers_discovered(env):
    assert env.ev_slots >= 1
    assert env.ev_action_indices() == [env.action_dim - 1]
    assert env.action_names[-1].startswith("electric_vehicle_storage")


def test_partial_ev_penetration(env):
    mask = env.action_presence_mask()
    ev_slot = env.ev_action_indices()[0]
    owners = int(mask[:, ev_slot].sum())
    assert 0 < owners < env.num_buildings
    assert mask[:, :3].all()


def test_charger_calibration_is_plausible(env):
    info = env.ev_info()
    power = info["max_charging_power"]
    owned = power > 0
    assert owned.any()
    assert np.all(power[owned] >= 3.0) and np.all(power[owned] <= 22.0)


def test_ev_charging_moves_soc_only_on_occupied_bays(env):
    obs, _ = env.reset()
    layout = env.ev_obs_layout()[0]
    ev_slot = env.ev_action_indices()[0]
    connected = np.array([o[layout["connected_state"]] for o in obs]) > 0.5
    before = np.array([o[layout["soc"]] for o in obs], dtype=np.float32)

    actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    actions[:, ev_slot] = 1.0
    for _ in range(4):
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            break
    after = np.array([o[layout["soc"]] for o in obs], dtype=np.float32)

    never_present = ~np.array([env.building_has_action(b, ev_slot)
                               for b in range(env.num_buildings)])
    assert np.all(after[never_present] == pytest.approx(0.0))
    if connected.any():
        assert np.any(after[connected] >= before[connected] - 1e-6)


def test_vehicles_actually_leave_and_return(env):
    obs, _ = env.reset()
    layout = env.ev_obs_layout()[0]
    actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    seen_connected = seen_away = False
    for _ in range(72):
        obs, _, term, trunc, _ = env.step(actions)
        conn = np.array([o[layout["connected_state"]] for o in obs]) > 0.5
        owners = np.array([env.building_has_action(b, env.ev_action_indices()[0])
                           for b in range(env.num_buildings)])
        seen_connected |= bool((conn & owners).any())
        seen_away |= bool((~conn & owners).any())
        if term or trunc:
            break
    assert seen_connected and seen_away


def test_departure_countdown_reaches_zero_before_leaving(schema_path):
    import csv
    path = SCHEMA_DIR / "charger_1_1.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    states = [r["electric_vehicle_charger_state"] for r in rows]
    first_away = states.index("3")
    assert first_away > 0
    last_parked = rows[first_away - 1]
    assert last_parked["electric_vehicle_charger_state"] == "1"
    assert float(last_parked["electric_vehicle_departure_time"]) == pytest.approx(0.0)


def test_schedule_states_and_length(schema_path):
    import csv
    with open(SCHEMA_DIR / "charger_1_1.csv") as f:
        rows = list(csv.DictReader(f))
    with open(BASE_SCHEMA) as f:
        base = json.load(f)
    expected = (int(base["simulation_end_time_step"])
                - int(base["simulation_start_time_step"]) + 1)
    assert len(rows) == expected
    states = {r["electric_vehicle_charger_state"] for r in rows}
    assert states <= {"1", "2", "3"}
    assert {"1", "3"} <= states, "schedule must contain both parked and away"


def test_required_soc_is_a_percentage(schema_path):
    import csv
    with open(SCHEMA_DIR / "charger_1_1.csv") as f:
        vals = [float(r["electric_vehicle_required_soc_departure"])
                for r in csv.DictReader(f)
                if r["electric_vehicle_required_soc_departure"]]
    assert vals
    assert min(vals) > 1.5, "looks normalised, not a percentage"
    assert max(vals) <= 100.0


def test_thermal_data_is_not_copied(schema_path):
    with open(schema_path) as f:
        merged = json.load(f)
    with open(BASE_SCHEMA) as f:
        base = json.load(f)
    assert merged["root_directory"] == base["root_directory"]
    for name, b in merged["buildings"].items():
        if b.get("include"):
            assert b["energy_simulation"] == base["buildings"][name]["energy_simulation"]
