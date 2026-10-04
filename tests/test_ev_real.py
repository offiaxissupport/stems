"""Validate the EV deadline barrier against real CityLearn EV data.

Run:  .venv/Scripts/python -m pytest tests/test_ev_real.py -q

``STEMSEnvironment`` is currently pinned to the tx_travis_8b observation subset,
which has no chargers, so these tests drive the raw CityLearn EV dataset
directly. That is deliberate: the point is to prove the barrier's calibration
against the actual simulator before the wrapper is widened, exactly as the
hot-water calibration was checked against the real tank.

Dataset: ``citylearn_challenge_2022_phase_all_plus_evs`` (17 buildings, one
charger per building, actions ``[electrical_storage,
electric_vehicle_storage_charger_<id>, washing_machine_<id>]``).
"""

from __future__ import annotations

import numpy as np
import pytest

from stems.ev import EVChargerSpec, EVObsLayout, EVReadinessBarrier

DATASET = "citylearn_challenge_2022_phase_all_plus_evs"


@pytest.fixture(scope="module")
def ev_env():
    try:
        from citylearn.citylearn import CityLearnEnv
        env = CityLearnEnv(DATASET, central_agent=False)
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"CityLearn EV dataset unavailable: {exc!r}")
    env.reset()
    return env


def _layout_and_spec(env, building_index: int = 0):
    b = env.buildings[building_index]
    charger = b.electric_vehicle_chargers[0]
    cid = charger.charger_id
    names = list(env.observation_names[building_index])
    idx = {n: i for i, n in enumerate(names)}
    base = f"connected_electric_vehicle_at_charger_{cid}"
    layout = EVObsLayout(
        connected_state=idx[f"electric_vehicle_charger_{cid}_connected_state"],
        departure_time=idx[f"{base}_departure_time"],
        required_soc_departure=idx[f"{base}_required_soc_departure"],
        soc=idx[f"{base}_soc"],
        battery_capacity=idx[f"{base}_battery_capacity"],
    )
    action_names = list(env.action_names[building_index])
    ev_action = [i for i, n in enumerate(action_names)
                 if n.startswith("electric_vehicle_storage")][0]
    spec = EVChargerSpec(
        max_charging_power_kw=np.array([float(charger.max_charging_power)]),
        efficiency=np.array([float(charger.efficiency)]),
        action_bound=np.array([1.0]),
        action_index=ev_action,
    )
    return layout, spec, charger


def test_ev_observations_are_all_present(ev_env):
    """Deadline, requirement, state and capacity must all be observable."""
    layout, _, _ = _layout_and_spec(ev_env)
    for field in ("connected_state", "departure_time", "required_soc_departure",
                  "soc", "battery_capacity"):
        assert getattr(layout, field) >= 0


def test_ev_charge_rate_matches_the_real_charger(ev_env):
    """rho = P_charge * eta / C_battery, read from the plant, not assumed."""
    layout, spec, charger = _layout_and_spec(ev_env)
    barrier = EVReadinessBarrier(layout, spec)
    obs, _ = ev_env.reset()
    rate = barrier.current_rate([obs[0]])[0]
    capacity = barrier.current_capacity([obs[0]])[0]
    nameplate = float(charger.max_charging_power) * float(charger.efficiency) / capacity
    # Deliberately derated: see EVReadinessBarrier._rate_derate. A latest-start
    # trigger must not be fed an optimistic rate.
    assert rate == pytest.approx(barrier._rate_derate * nameplate, rel=1e-4)
    assert rate < nameplate


def test_ev_lead_time_far_exceeds_the_hot_water_tank(ev_env):
    """An EV needs hours, not minutes -- the deadline barrier matters more here.

    The DHW tank fills in 1.2-1.9 h; a car on a domestic charger needs several
    times that, so the horizon rule that mattered marginally for hot water is
    decisive for vehicles.
    """
    layout, spec, _ = _layout_and_spec(ev_env)
    barrier = EVReadinessBarrier(layout, spec)
    obs, _ = ev_env.reset()
    hours_to_full = 1.0 / barrier.current_rate([obs[0]])[0]
    assert hours_to_full > 2.0, f"expected a multi-hour fill, got {hours_to_full:.2f} h"


def test_ev_barrier_runs_over_a_real_rollout(ev_env):
    """Barrier stays well-defined across arrivals, departures and empty bays."""
    layout, spec, _ = _layout_and_spec(ev_env)
    barrier = EVReadinessBarrier(layout, spec)
    obs, _ = ev_env.reset()
    action_dim = ev_env.action_space[0].shape[0]

    seen_connected = seen_empty = False
    for _ in range(72):
        o = [obs[0]]
        rep = barrier.deadline_report(o)
        assert np.all(np.isfinite(rep["slack"]))
        assert np.all(rep["gap"] >= 0.0)
        seen_connected |= bool(rep["active"][0])
        seen_empty |= not bool(rep["active"][0])

        safe = barrier.project(np.zeros((1, action_dim), np.float32), o)
        assert np.all(np.isfinite(safe))
        assert -1.0 <= safe[0, spec.action_index] <= 1.0
        # An empty bay must never be commanded to charge.
        if not rep["active"][0]:
            assert safe[0, spec.action_index] == pytest.approx(0.0)

        actions = [list(np.zeros(ev_env.action_space[i].shape[0], np.float32))
                   for i in range(len(ev_env.buildings))]
        obs, _, terminated, truncated, _ = ev_env.step(actions)
        if terminated or truncated:
            break

    assert seen_connected, "no vehicle connected during the window"
    assert seen_empty, "bay never empty during the window"


def test_ev_barrier_commands_charge_when_the_deadline_closes(ev_env):
    """With the departure near and the battery short, the barrier must act."""
    layout, spec, _ = _layout_and_spec(ev_env)
    barrier = EVReadinessBarrier(layout, spec)
    obs, _ = ev_env.reset()
    o = obs[0].copy()
    o[layout.connected_state] = 1.0
    o[layout.battery_capacity] = 60.0
    o[layout.soc] = 0.2
    o[layout.required_soc_departure] = 0.9
    o[layout.departure_time] = 1.0        # one step left, ~4 steps of charge needed
    safe = barrier.project(np.zeros((1, len(o)), np.float32), [o])
    assert safe[0, spec.action_index] > 0.9
    assert barrier.deadline_report([o])["at_risk"][0], "unreachable deadline must be flagged"
