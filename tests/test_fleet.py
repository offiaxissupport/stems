"""``stems.fleet``: the EV plant model, the myopic rules and the joint-feasibility programme.

The scheduling tests use a lossless, curve-free fleet whose answers can be worked
out by hand; the last tests run on the real simulator.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.battery import BatteryModel, TankModel
from stems.fleet import (BaseLoadForecaster, EVFleetModel, FleetShield, FleetState,
                         HouseStorage, allocate, apply_dead_band, laxity, schedule)

EV_SCHEMA = "citylearn_schemas/tx_travis_8b_ev/schema.json"


def simple_fleet(n=3, p_max=10.0, p_min=1.0, capacity=50.0):
    """n identical vehicles: 50 kWh, 10 kW, no losses, no taper, perfect charger."""
    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    battery = BatteryModel([capacity] * n, [p_max] * n, [0.0] * n, [flat] * n, [flat] * n)
    return EVFleetModel([True] * n, [p_max] * n, [p_min] * n, [1.0] * n, battery)


def fleet_state(slots, cap, soc=0.4, target=0.8, base=None, horizon=8, price=None):
    n = len(slots)
    base = np.zeros((horizon, n)) if base is None else np.asarray(base, dtype=float)
    return FleetState([True] * n, [soc] * n, [target] * n, slots, base, cap, price)


# ---------------------------------------------------------------------------
# Laxity and the myopic rules
# ---------------------------------------------------------------------------

def test_laxity_is_slots_minus_hours_of_full_charging():
    model = simple_fleet()                       # 20 kWh to go = 2 h at 10 kW
    assert laxity(model, fleet_state([2, 3, 4], 100)).tolist() == [0, 1, 2]
    done = FleetState([True] * 3, [0.9] * 3, [0.8] * 3, [2, 3, 4], np.zeros((4, 3)), 100)
    assert np.all(np.isinf(laxity(model, done)))


def test_rules_enforce_the_same_total_and_differ_in_who_is_served():
    model, req = simple_fleet(), np.array([10.0, 10.0, 10.0])
    state = fleet_state([4, 2, 3], cap=15.0)
    assert allocate(model, state, req, "independent").tolist() == [10, 10, 10]
    assert allocate(model, state, req, "proportional").tolist() == [5, 5, 5]
    assert allocate(model, state, req, "edf").tolist() == [0, 10, 5]      # leaves soonest first
    late = FleetState([True] * 3, [0.4, 0.7, 0.4], [0.8] * 3, [3, 2, 3], np.zeros((4, 3)), 15.0)
    # Vehicle 1 leaves first but needs 1 h (laxity 1); the others need 2 of 3 (laxity 1).
    assert allocate(model, late, req, "llf").sum() == pytest.approx(15.0)


def test_own_pv_surplus_charges_the_car_without_using_the_shared_budget():
    model = simple_fleet(2)
    base = np.array([[-6.0, 3.0]] * 4)           # house 0 exports 6 kW, house 1 imports 3
    state = FleetState([True, True], [0.4, 0.4], [0.8, 0.8], [3, 3], base, cap=5.0)
    out = allocate(model, state, np.array([10.0, 10.0]), "proportional")
    # Shared budget = 5 - 3 = 2 kW. House 0: 6 free + its share; house 1: its share.
    assert out[0] == pytest.approx(6.0 + 2.0 * 4.0 / 14.0)
    assert out[1] == pytest.approx(2.0 * 10.0 / 14.0)
    assert np.maximum(base[0] + out, 0.0).sum() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# The linear programme
# ---------------------------------------------------------------------------

def test_feasible_fleet_keeps_the_request_when_it_is_safe():
    model = simple_fleet()
    state = fleet_state([4, 4, 4], cap=30.0)
    sol = schedule(model, state, np.array([10.0, 0.0, 5.0]))
    assert sol["feasible"] and not sol["binding"]
    np.testing.assert_allclose(sol["now_kw"], [10.0, 0.0, 5.0], atol=1e-6)


def test_programme_forces_only_the_charging_the_deadlines_need():
    """Policy asks for nothing. 60 kWh are owed; the cap delivers 15 kW for the 4
    hours all three are parked, i.e. exactly 60: every hour must be used in full."""
    model = simple_fleet()
    sol = schedule(model, fleet_state([4, 4, 4], cap=15.0), np.zeros(3))
    assert sol["feasible"]
    assert sol["now_kw"].sum() == pytest.approx(15.0, abs=1e-6)
    relaxed = schedule(model, fleet_state([8, 8, 8], cap=15.0), np.zeros(3))
    assert relaxed["feasible"] and relaxed["now_kw"].sum() == pytest.approx(0.0, abs=1e-6)


def test_joint_infeasibility_is_detected_where_each_vehicle_alone_is_fine():
    """Each car alone has laxity >= 0, so every per-vehicle barrier is satisfied;
    together they need 60 kWh and the cap can deliver 55 before they leave."""
    model = simple_fleet()
    state = fleet_state([2, 3, 4], cap=15.0)
    assert np.all(laxity(model, state) >= 0)
    sol = schedule(model, state, np.array([10.0, 10.0, 10.0]))
    assert not sol["feasible"]
    assert sol["total_shortfall_kwh"] == pytest.approx(5.0, abs=1e-4)
    generous = schedule(model, fleet_state([2, 3, 4], cap=17.5), np.array([10.0] * 3))
    assert generous["feasible"]


def test_unavoidable_shortfall_is_spread_not_dumped_on_one_vehicle():
    model = simple_fleet()
    sol = schedule(model, fleet_state([2, 2, 2], cap=15.0), np.array([10.0] * 3))
    assert sol["total_shortfall_kwh"] == pytest.approx(30.0, abs=1e-4)   # 60 owed, 30 deliverable
    np.testing.assert_allclose(sol["shortfall_soc"], [0.2, 0.2, 0.2], atol=1e-4)
    assert sol["worst_shortfall_share"] == pytest.approx(0.5, abs=1e-4)


def test_cost_objective_charges_in_the_cheap_hours():
    model = simple_fleet(1)
    price = np.array([0.5, 0.5, 0.1, 0.1, 0.5, 0.5])
    state = fleet_state([6], cap=100.0, price=price, horizon=6)
    assert schedule(model, state, objective="cost")["now_kw"][0] == pytest.approx(0.0, abs=1e-6)
    cheap_now = fleet_state([6], cap=100.0, price=price[2:], horizon=4)
    assert schedule(model, cheap_now, objective="cost")["now_kw"][0] == pytest.approx(10.0, abs=1e-6)


def test_dead_band_rounds_toward_the_deadline():
    model = simple_fleet(2, p_min=1.4)
    urgent_and_relaxed = FleetState([True, True], [0.4, 0.4], [0.8, 0.8], [2, 8],
                                    np.zeros((8, 2)), 100.0)
    out = apply_dead_band(model, urgent_and_relaxed, np.array([0.5, 0.5]))
    assert out.tolist() == [1.4, 0.0]


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

def test_replay_forecast_is_exact_and_needs_no_margin():
    replay = np.arange(12, dtype=float).reshape(6, 2)
    f = BaseLoadForecaster(2, replay=replay)
    obs = [np.zeros(30), np.zeros(30)]
    np.testing.assert_allclose(f.predict(obs, 3), replay[:3])
    f.observe(replay[0])
    np.testing.assert_allclose(f.predict(obs, 2), replay[1:3])
    assert f.margin == 0.0


def test_causal_margin_covers_the_recent_forecast_errors():
    f = BaseLoadForecaster(1, quantile=0.9, window=50)
    obs = [np.zeros(30)]
    rng = np.random.default_rng(0)
    for _ in range(60):
        f.predict(obs, 2)
        f.observe(np.array([rng.uniform(0.0, 4.0)]))
    recent = np.asarray(f.errors[-50:])
    assert f.margin == pytest.approx(max(np.quantile(recent, 0.9), 0.0))
    assert np.mean(recent <= f.margin) >= 0.88


# ---------------------------------------------------------------------------
# Against the simulator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ev_env():
    from stems.environment import STEMSEnvironment

    return STEMSEnvironment(schema=EV_SCHEMA, seed=0, heat_pump=True,
                            env_kwargs={"episode_time_steps": [(0, 239)]},
                            hvac_control="setpoint")


def test_fleet_model_matches_the_simulator(ev_env):
    env, cl = ev_env, ev_env._env
    model = env.ev_fleet_model()
    layout, e = env.ev_obs_layout()[0], env.ev_action_indices()[0]
    vehicles = {ev.name: ev for ev in cl.electric_vehicles}
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    soc_err, draw_err, floor_draw, done = [], [], [], False
    while not done:
        t = cl.time_step
        conn = np.array([o[layout["connected_state"]] for o in obs]) > 0.5
        soc = np.array([o[layout["soc"]] for o in obs], dtype=float)
        a = np.where(conn, rng.choice([0.0, 0.05, 0.3, 0.7, 1.0], env.num_buildings), 0.0)
        actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        actions[:, e] = a
        predicted, draw = model.next_soc(soc, a), model.draw_kw(soc, a)
        out = env.step(actions)
        obs, done = out[0], out[2] or out[3]
        for i in np.flatnonzero(conn & model.has_ev):
            charger = cl.buildings[i].electric_vehicle_chargers[0]
            name = str(np.asarray(charger.charger_simulation.electric_vehicle_id)[t])
            soc_err.append(abs(predicted[i] - vehicles[name].battery.soc[t]))
            draw_err.append(abs(draw[i] - env.ev_draw_kwh[i]))
            if a[i] == 0.05 and soc[i] < 0.7:
                floor_draw.append(env.ev_draw_kwh[i])
    assert len(soc_err) > 500
    assert max(soc_err) < 2e-4
    assert max(draw_err) < 2e-2
    np.testing.assert_allclose(floor_draw, 1.4, atol=1e-3)     # the dead band


def test_departures_are_scored_on_the_final_hour(ev_env):
    """A vehicle charged only in its last connected hour must leave with that charge."""
    env = ev_env
    model = env.ev_fleet_model()
    layout, e = env.ev_obs_layout()[0], env.ev_action_indices()[0]
    obs, _ = env.reset()
    events, last_obs_soc, done = [], {}, False
    while not done:
        conn = np.array([o[layout["connected_state"]] for o in obs]) > 0.5
        last = conn & (np.array([o[layout["departure_time"]] for o in obs]) == 0)
        for i in np.flatnonzero(last):
            last_obs_soc[i] = float(obs[i][layout["soc"]])
        actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        actions[:, e] = np.where(last & model.has_ev, 1.0, 0.0)
        out = env.step(actions)
        obs, done = out[0], out[2] or out[3]
        for d in env.ev_departures:
            events.append((d, last_obs_soc[d["building"]]))
    assert len(events) >= 6
    gains = [d["soc"] - before for d, before in events]
    assert min(gains) > 0.05, "the last hour's charge must be in the departure record"


def test_lp_shield_meets_every_deadline_under_a_cap_that_binds(ev_env):
    env = ev_env
    model = env.ev_fleet_model()
    layout, e = env.ev_obs_layout()[0], env.ev_action_indices()[0]
    zeros = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)

    obs, _ = env.reset()                          # replay: the load without charging
    base, done = [], False
    while not done:
        out = env.step(zeros)
        base.append([o[20] for o in out[0]])
        done = out[2] or out[3]
    base = np.array(base)
    cap = float(np.maximum(base, 0.0).sum(axis=1).max()) + 8.0    # < one 11 kW charger above the peak

    shield = FleetShield(model, layout, e, cap, "lp", BaseLoadForecaster(env.num_buildings, replay=base))
    obs, _ = env.reset()
    departures, imports, done = [], [], False
    while not done:
        ask = zeros.copy()
        ask[:, e] = 1.0                           # every car asks for full power
        out = env.step(shield.project(ask, obs))
        shield.observe(out[0], env.ev_draw_kwh)
        obs, done = out[0], out[2] or out[3]
        departures += env.ev_departures
        imports.append(np.maximum([o[20] for o in obs], 0.0).sum())
    assert len(departures) >= 20
    missed = [d for d in departures if d["soc"] + 1e-3 < d["required_soc"]]
    assert not missed, missed[:3]
    assert max(imports) <= cap + model.p_min.max() + 1e-6       # dead-band rounding at most
    assert np.mean(np.array(imports) > cap + 1e-6) < 0.02


# ---------------------------------------------------------------------------
# Rules from the scheduling literature
# ---------------------------------------------------------------------------

def test_static_split_wastes_what_a_neighbour_leaves_unused():
    model, req = simple_fleet(), np.array([10.0, 10.0, 0.0])
    state = fleet_state([4, 4, 4], cap=15.0)
    assert allocate(model, state, req, "static").tolist() == [5, 5, 0]     # 5 kW idle
    assert allocate(model, state, req, "proportional").sum() == pytest.approx(15.0)


def test_sllf_equalises_laxity_after_the_hour():
    """Chen et al. (2022): served vehicles end the hour with the same laxity."""
    model = simple_fleet()
    # Owed 20, 15 and 20 kWh with 3, 3 and 6 hours left: laxity 1, 1.5 and 4.
    state = FleetState([True] * 3, [0.4, 0.5, 0.4], [0.8] * 3, [3, 3, 6], np.zeros((8, 3)), 12.0)
    kw = allocate(model, state, np.array([10.0] * 3), "sllf")
    np.testing.assert_allclose(kw, [8.5, 3.5, 0.0], atol=1e-6)
    owed = np.array([20.0, 15.0, 20.0])
    after = (state.slots - 1) - (owed - kw) / 10.0           # laxity one hour later
    assert after[0] == pytest.approx(after[1]) == pytest.approx(0.85)


def test_llf_breaks_ties_toward_the_longer_remaining_charge():
    """Xu, Pan and Tong (2016): equal laxity -> the vehicle with more left to charge."""
    model = simple_fleet(2)
    # Both have laxity 1: vehicle 0 needs 3 h of its 4, vehicle 1 needs 1 h of its 2.
    state = FleetState([True, True], [0.2, 0.6], [0.8, 0.8], [4, 2], np.zeros((6, 2)), 10.0)
    assert laxity(model, state).tolist() == [1, 1]
    assert allocate(model, state, np.array([10.0, 10.0]), "llf").tolist() == [10, 0]
    assert allocate(model, state, np.array([10.0, 10.0]), "edf").tolist() == [0, 10]


def test_flexibility_interval_brackets_every_safe_charging_level():
    from stems.fleet import fleet_power_bounds

    model = simple_fleet()
    tight = fleet_power_bounds(model, fleet_state([4, 4, 4], cap=15.0))
    assert tight["feasible"] and tight["u_min"] == pytest.approx(15.0, abs=1e-6)
    loose = fleet_power_bounds(model, fleet_state([8, 8, 8], cap=15.0))
    assert loose["u_min"] == pytest.approx(0.0, abs=1e-6)
    assert loose["u_max"] == pytest.approx(15.0, abs=1e-6)
    empty = fleet_power_bounds(model, fleet_state([2, 2, 2], cap=15.0))
    assert not empty["feasible"] and empty["total_shortfall_kwh"] == pytest.approx(30.0, abs=1e-4)


def test_executable_schedule_never_hands_out_less_than_the_minimum_power():
    """Three cars, 21 kW: the programme alone would give one car 1 kW, which a charger
    with a 1.4 kW minimum cannot deliver; the executable schedule must not."""
    from stems.fleet import schedule_executable

    model = simple_fleet(p_min=1.4)
    state = fleet_state([6, 6, 6], cap=21.0)
    sol = schedule_executable(model, state, np.array([10.0, 10.0, 10.0]))
    kw = sol["now_kw"]
    assert sol["feasible"] and kw.sum() <= 21.0 + 1e-6
    assert np.all((kw < 1e-6) | (kw >= 1.4 - 1e-6)), kw


def test_useful_power_stops_at_the_target_and_at_what_the_battery_takes():
    model = simple_fleet(2)                       # 50 kWh, 10 kW
    # Vehicle 0 needs 20 kWh (two full hours); vehicle 1 needs 2 kWh.
    kw = model.useful_kw(np.array([0.4, 0.76]), np.array([0.8, 0.8]))
    np.testing.assert_allclose(kw, [10.0, 2.0], atol=1e-3)
    assert model.useful_kw(np.array([0.9, 0.9]), np.array([0.8, 0.8])).tolist() == [0.0, 0.0]


def test_executable_schedule_survives_a_request_to_charge_a_full_car():
    """A learning policy can ask for anything; P_min does not fit into a full battery."""
    from stems.fleet import schedule_executable

    model = simple_fleet(2, p_min=1.4)
    state = FleetState([True, True], [0.999, 0.4], [0.8, 0.8], [5, 5], np.zeros((8, 2)), 30.0)
    sol = schedule_executable(model, state, np.array([10.0, 10.0]))
    assert sol["feasible"] and sol["now_kw"][0] < 1e-6 and sol["now_kw"][1] == pytest.approx(10.0)


def test_time_reserve_plans_departures_early():
    model = simple_fleet(1)
    layout = {"connected_state": 0, "soc": 1, "required_soc_departure": 2, "departure_time": 3}
    obs = [np.array([1.0, 0.4, 0.8, 3.0] + [0.0] * 26)]        # 4 charging hours, needs 2
    zeros = np.zeros((1, 1), dtype=np.float32)
    def now(reserve):
        shield = FleetShield(model, layout, 0, 100.0, "lp",
                             BaseLoadForecaster(1, replay=np.zeros((24, 1))), reserve_hours=reserve)
        return float(shield.project(zeros, obs)[0, 0])
    assert now(0) == 0.0 and now(1) == 0.0       # laxity 2, then 1: still room to wait
    assert now(2) == pytest.approx(1.0)          # planned to leave 2 h early: must start now


def test_a_granted_draw_becomes_the_command_that_produces_it():
    """With a battery that accepts 80% of the command, 4 kW of draw needs a 5 kW command."""
    flat, limited = np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [0.8, 0.8]])
    battery = BatteryModel([50.0], [10.0], [0.0], [flat], [limited])
    model = EVFleetModel([True], [10.0], [1.0], [1.0], battery)
    soc = np.array([0.4])
    assert model.draw_kw(soc, np.array([1.0]))[0] == pytest.approx(8.0)
    a = model.action_for_draw(soc, np.array([4.0]))
    assert model.draw_kw(soc, a)[0] == pytest.approx(4.0, abs=1e-4)
    assert model.action_for_draw(soc, np.array([9.5]))[0] == 1.0       # more than it can take
    assert model.action_for_draw(soc, np.array([0.0]))[0] == 0.0


def test_requirement_holds_at_departure_not_at_the_start_of_the_last_hour():
    """A parked battery loses charge: at its requirement with an hour to go, a
    vehicle leaves below it unless it is topped up."""
    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    battery = BatteryModel([50.0], [10.0], [0.01], [flat], [flat])     # loses 1% per hour
    model = EVFleetModel([True], [10.0], [1.0], [1.0], battery)
    at_target = FleetState([True], [0.80], [0.80], [1], np.zeros((4, 1)), 100.0)
    assert model.idle_soc(at_target.soc, at_target.slots)[0] == pytest.approx(0.792)
    assert laxity(model, at_target)[0] == 0                 # must charge in its last hour
    assert model.useful_kw(at_target.soc, at_target.target)[0] > 0.0
    comfortable = FleetState([True], [0.85], [0.80], [3], np.zeros((4, 1)), 100.0)
    assert np.isinf(laxity(model, comfortable)[0])          # 0.85 * 0.99^3 = 0.825 >= 0.80
    sol = schedule(model, at_target, np.zeros(1))
    assert sol["feasible"] and sol["now_kw"][0] > 0.0       # the programme tops it up too


# ---------------------------------------------------------------------------
# One cap for the vehicles and the house batteries
# ---------------------------------------------------------------------------

LAYOUT = {"connected_state": 0, "soc": 1, "required_soc_departure": 2, "departure_time": 3}


def house_obs(connected, ev_soc, target, countdown, load, batt_soc):
    """One building's observation with the columns the cap shield reads."""
    o = np.zeros(30)
    o[0], o[1], o[2], o[3] = connected, ev_soc, target, countdown
    o[16], o[19] = load, batt_soc              # non-shiftable load, battery state of charge
    return o


def ideal_batteries(n=1, capacity=20.0, power=5.0, tank=None):
    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    model = BatteryModel([capacity] * n, [power] * n, [0.0] * n, [flat] * n, [flat] * n)
    return HouseStorage(model, battery_action=1, battery_soc=19, soc_lo=0.1, soc_hi=0.9,
                        tank=tank, tank_action=2)


def ideal_tank(n=1, capacity=10.0, power=8.0):
    """Lossless tank behind a perfect heater: charging by ``a`` draws ``a * 10`` kW."""
    one = [1.0] * n
    return TankModel([capacity] * n, [power] * n, one, one, [0.0] * n)


def test_forecast_uses_the_battery_action_it_already_knows():
    """A battery that stops discharging is not forecast to keep discharging."""
    fc = BaseLoadForecaster(1)
    obs = [house_obs(0, 0, 0, 0, load=4.0, batt_soc=0.5)]
    assert fc.predict(obs, 3, known_kw=np.array([-3.0]))[0, 0] == pytest.approx(1.0)
    fc.observe(np.array([1.0]))                        # the house drew 4 - 3
    assert fc.predict(obs, 3, known_kw=np.array([0.0]))[0, 0] == pytest.approx(4.0)
    blind = BaseLoadForecaster(1)                       # the same hours without the knowledge
    blind.predict(obs, 3)
    blind.observe(np.array([1.0]))
    assert blind.predict(obs, 3)[0, 0] == pytest.approx(1.0)      # 3 kW too low


def test_unknown_draw_reduces_to_the_plain_persistence_forecast():
    """Without ``known_kw`` the forecast is, bit for bit, the persistence forecast
    the EV study was run with: observed load plus last hour's remainder."""
    rng = np.random.default_rng(0)
    fc = BaseLoadForecaster(2)
    prev_exo = prev_real = None
    for _ in range(30):
        exo = rng.uniform(1, 5, size=2)
        obs = [house_obs(0, 0, 0, 0, load=float(x), batt_soc=0.5) for x in exo]
        expected = exo if prev_real is None else exo + (prev_real - prev_exo)
        assert np.array_equal(fc.predict(obs, 24)[0], expected)
        realised = rng.uniform(0, 6, size=2)
        fc.observe(realised)
        prev_exo, prev_real = exo, realised


def test_battery_charging_is_cut_back_to_what_the_cap_leaves():
    model = simple_fleet(1)
    shield = FleetShield(model, LAYOUT, 0, 6.0, "lp", BaseLoadForecaster(1), house=ideal_batteries())
    obs = [house_obs(0, 0, 0, 0, load=4.0, batt_soc=0.5)]          # no car; the house draws 4 kW
    out = shield.project(np.array([[0.0, 1.0]], dtype=np.float32), obs)   # battery asks for 5 kW
    assert out[0, 1] == pytest.approx(0.4, abs=1e-4)               # 2 kW fits under 6
    assert shield.last["predicted_import_kw"] == pytest.approx(6.0, abs=1e-3)
    assert shield.last["storage_shed_kw"] == pytest.approx(3.0, abs=1e-3)
    roomy = FleetShield(model, LAYOUT, 0, 20.0, "lp", BaseLoadForecaster(1), house=ideal_batteries())
    assert roomy.project(np.array([[0.0, 1.0]], dtype=np.float32), obs)[0, 1] == 1.0
    assert roomy.last["storage_shed_kw"] == 0.0


def test_a_vehicle_with_a_deadline_outranks_battery_charging():
    model = simple_fleet(1)                                         # 10 kW charger, 50 kWh
    shield = FleetShield(model, LAYOUT, 0, 12.0, "lp", BaseLoadForecaster(1), house=ideal_batteries())
    # The car needs 20 kWh in its 2 remaining hours: full power now. House load 1 kW.
    obs = [house_obs(1, 0.4, 0.8, 1, load=1.0, batt_soc=0.5)]
    out = shield.project(np.array([[0.0, 1.0]], dtype=np.float32), obs)
    assert out[0, 0] == pytest.approx(1.0)                          # the car gets its 10 kW
    assert out[0, 1] == pytest.approx(0.2, abs=1e-4)                # the battery the 1 kW left
    assert shield.last["predicted_import_kw"] == pytest.approx(12.0, abs=1e-3)


def test_battery_discharge_makes_room_for_the_vehicle():
    model = simple_fleet(1)
    obs = [house_obs(1, 0.6, 0.8, 0, load=6.0, batt_soc=0.5)]      # last hour: 10 kW or it leaves short
    with_batt = FleetShield(model, LAYOUT, 0, 12.0, "lp", BaseLoadForecaster(1), house=ideal_batteries())
    with_batt.project(np.array([[0.0, -0.8]], dtype=np.float32), obs)     # battery covers 4 of the 6 kW
    assert with_batt.last["feasible"] and with_batt.last["predicted_import_kw"] <= 12.0 + 1e-6
    alone = FleetShield(model, LAYOUT, 0, 12.0, "lp", BaseLoadForecaster(1))
    alone.project(np.array([[0.0, 0.0]], dtype=np.float32), obs)
    assert not alone.last["feasible"]                               # 6 + 10 > 12 without it


def test_recovery_charge_below_the_band_is_not_shed():
    model = simple_fleet(1)
    house = ideal_batteries()
    shield = FleetShield(model, LAYOUT, 0, 4.0, "lp", BaseLoadForecaster(1), house=house)
    obs = [house_obs(0, 0, 0, 0, load=4.0, batt_soc=0.05)]         # below the 0.1 floor; cap already full
    floor = float(house.floor_action(np.array([0.05]))[0])                # of the battery
    assert floor == pytest.approx(0.2, abs=1e-3)                    # 1 kWh of 20 back to 0.1 = 0.2 * 5 kW
    out = shield.project(np.array([[0.0, 1.0]], dtype=np.float32), obs)
    assert out[0, 1] == pytest.approx(floor, abs=1e-3)              # shed down to the recovery, no further


def test_house_batteries_refuse_a_replayed_load():
    with pytest.raises(ValueError):
        FleetShield(simple_fleet(1), LAYOUT, 0, 10.0, "lp",
                    BaseLoadForecaster(1, replay=np.zeros((24, 1))), house=ideal_batteries())


def test_daily_pattern_anticipates_a_scheduled_step():
    """A load that steps up by 5 kW at the same hour every day is forecast to do
    so from the second day on; plain persistence is 5 kW low every time."""
    def run(days_of_pattern):
        fc = BaseLoadForecaster(1, daily_pattern_days=days_of_pattern)
        errors = []
        for t in range(24 * 4):
            hour = t % 24
            actual = 2.0 + (5.0 if 10 <= hour < 16 else 0.0)       # the step is not in the observed load
            predicted = fc.predict([house_obs(0, 0, 0, 0, load=2.0, batt_soc=0.5)], 2)[0, 0]
            if t >= 48 and hour == 10:
                errors.append(actual - predicted)
            fc.observe(np.array([actual]))
        return errors
    assert run(0) == pytest.approx([5.0, 5.0])
    assert run(7) == pytest.approx([0.0, 0.0])


def test_tank_model_limits():
    tank = ideal_tank()
    half, zero = np.array([0.5]), np.array([0.0])
    assert tank.drawn_kwh(half, np.array([0.3]), zero)[0] == pytest.approx(3.0)       # 0.3 * 10 kWh
    assert tank.drawn_kwh(half, np.array([1.0]), zero)[0] == pytest.approx(5.0)       # only 5 kWh of room
    assert tank.drawn_kwh(np.array([0.1]), np.array([1.0]), zero)[0] == pytest.approx(8.0)   # heater: 8 kW
    assert tank.drawn_kwh(half, np.array([-0.5]), np.array([2.0]))[0] == pytest.approx(-2.0)  # no more than the draw
    assert tank.drawn_kwh(half, np.array([-0.5]), zero)[0] == 0.0                     # nothing to serve


def test_tank_charging_is_known_and_shed_with_the_battery():
    model = simple_fleet(1)
    house = ideal_batteries(tank=ideal_tank())
    obs = [house_obs(0, 0, 0, 0, load=4.0, batt_soc=0.5)]
    obs[0][18] = 0.2                                                # tank 20% full
    ask = np.array([[0.0, 1.0, 0.3]], dtype=np.float32)            # battery 5 kW, tank 3 kW
    roomy = FleetShield(model, LAYOUT, 0, 20.0, "lp", BaseLoadForecaster(1), house=house)
    out = roomy.project(ask.copy(), obs)
    assert out[0, 1] == 1.0 and out[0, 2] == pytest.approx(0.3)
    assert roomy.last["predicted_import_kw"] == pytest.approx(12.0, abs=1e-3)     # 4 + 5 + 3: no surprise
    tight = FleetShield(model, LAYOUT, 0, 8.0, "lp", BaseLoadForecaster(1), house=house)
    out = tight.project(ask.copy(), obs)                            # 4 kW of room for 8 kW of charging
    assert out[0, 1] == pytest.approx(0.5, abs=1e-4) and out[0, 2] == pytest.approx(0.15, abs=1e-4)
    assert tight.last["predicted_import_kw"] == pytest.approx(8.0, abs=1e-3)


def test_tank_model_matches_the_simulator(ev_env):
    """Random hot-water actions for two days: the model's draw equals the heater's
    electricity minus what it draws with the tank left alone."""
    env = ev_env
    tank = env.dhw_tank_model()
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    buildings = env._env.buildings
    worst = 0.0
    for _ in range(48):
        soc = np.array([float(o[18]) for o in obs])
        a = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        a[:, 0] = rng.uniform(-0.6, 0.6, size=env.num_buildings)
        ts = env._env.time_step
        demand = np.array([b.dhw_demand[ts] for b in buildings])
        predicted = tank.drawn_kwh(soc, a[:, 0], demand)
        obs = env.step(a)[0]
        actual = np.array([b.dhw_device.electricity_consumption[ts] for b in buildings])
        worst = max(worst, float(np.abs(actual - demand / tank.heater_efficiency - predicted).max()))
    assert worst < 1e-4


# ---------------------------------------------------------------------------
# A shield that can be leaned on
# ---------------------------------------------------------------------------

def lossy_fleet(loss=0.01):
    """One 50 kWh vehicle on a 10 kW charger that loses 1% of its charge per hour."""
    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    battery = BatteryModel([50.0], [10.0], [loss], [flat], [flat])
    return EVFleetModel([True], [10.0], [1.0], [1.0], battery)


def test_a_departure_planned_early_still_holds_at_the_door():
    """Planned two hours early, the requirement is raised by two hours of standby
    loss: at 0.8 two hours before leaving, the car would leave at 0.784."""
    model = lossy_fleet()
    obs = [house_obs(1, 0.5, 0.8, 5, load=0.0, batt_soc=0.5)]      # six hours left
    plain = FleetShield(model, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1)).state(obs)
    assert plain.slots[0] == 6 and plain.target[0] == pytest.approx(0.8)
    early = FleetShield(model, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1), reserve_hours=2).state(obs)
    assert early.slots[0] == 4
    assert early.target[0] == pytest.approx(0.8 / 0.99 ** 2)
    assert model.idle_soc(early.target, np.array([2]))[0] == pytest.approx(0.8)   # and it holds
    last = [house_obs(1, 0.8, 0.8, 0, load=0.0, batt_soc=0.5)]      # the real last hour
    assert FleetShield(model, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1),
                       reserve_hours=2).state(last).target[0] == pytest.approx(0.8)


def test_later_margin_is_the_day_ahead_error_and_never_below_the_hourly_one():
    fc = BaseLoadForecaster(1)
    rng = np.random.default_rng(1)
    for t in range(24 * 6):
        day = t // 24
        load = 5.0 + (4.0 if day % 2 else 0.0) + rng.normal(0.0, 0.1)   # every other day is 4 kW higher
        fc.predict([house_obs(0, 0, 0, 0, load=load, batt_soc=0.5)], 24)
        fc.observe(np.array([load]))
    assert len(fc.day_errors) == 24 * 5
    assert fc.margin < 1.0                       # one hour ahead the load is observed
    assert 3.5 < fc.later_margin < 4.5           # a day ahead it is 4 kW off on the bad days
    assert BaseLoadForecaster(1, replay=np.zeros((24, 1))).later_margin == 0.0


def test_the_programme_plans_the_later_hours_against_their_own_cap():
    """20 kWh to deliver in two hours on a 10 kW charger: fits under 10 kW in both
    hours, and is 5 kWh short if the second hour is only trusted for 5 kW."""
    model = simple_fleet(1)
    state = FleetState([True], [0.4], [0.8], [2], np.zeros((4, 1)), 10.0)
    assert schedule(model, state, np.zeros(1))["feasible"]
    wary = FleetState([True], [0.4], [0.8], [2], np.zeros((4, 1)), 10.0, cap_later=5.0)
    sol = schedule(model, wary, np.zeros(1))
    assert not sol["feasible"] and sol["total_shortfall_kwh"] == pytest.approx(5.0, abs=1e-6)
    assert sol["now_kw"][0] == pytest.approx(10.0)       # everything it can, now


def test_lead_margin_makes_a_deferred_plan_start_while_there_is_room():
    """A car that could wait under the hourly margin is started now when the later
    hours are only trusted up to the day-ahead margin."""
    model = simple_fleet(1)
    obs = [house_obs(1, 0.6, 0.8, 1, load=2.0, batt_soc=0.5)]      # 10 kWh in two hours
    zeros = np.zeros((1, 3), dtype=np.float32)

    def shield(lead):
        fc = BaseLoadForecaster(1)
        fc.day_errors = [8.0] * 48                       # the day-ahead forecast is 8 kW off
        fc.errors = [0.0] * 48
        return FleetShield(model, LAYOUT, 0, 12.0, "lp", fc, lead_margin=lead)
    relaxed, wary = shield(False), shield(True)
    assert relaxed.project(zeros.copy(), obs)[0, 0] == 0.0          # 2 + 10 fits next hour
    assert wary.state(obs).cap_later == pytest.approx(4.0)
    assert wary.project(zeros.copy(), obs)[0, 0] > 0.0              # next hour is trusted for 2 kW only


def test_the_shield_reports_what_it_forced_and_what_it_cut():
    model = simple_fleet(2)
    layout = LAYOUT
    obs = [house_obs(1, 0.6, 0.8, 0, load=0.0, batt_soc=0.5),      # must have 10 kW now
           house_obs(1, 0.4, 0.8, 9, load=0.0, batt_soc=0.5)]      # can wait
    shield = FleetShield(model, layout, 0, 14.0, "lp", BaseLoadForecaster(2))
    ask = np.array([[0.0], [1.0]], dtype=np.float32)               # the wrong way round
    shield.project(ask, obs)
    assert shield.last["forced_kw"] == pytest.approx([10.0, 0.0])
    assert shield.last["cut_kw"] == pytest.approx([0.0, 6.0])
