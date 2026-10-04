"""The HVAC action's sign picks the mode: positive heats, negative cools.

CityLearn splits the one action as ``a_heat = max(a, 0)``, ``a_cool = |min(a, 0)|``
with ``hvac_mode = 3`` at every hour of the Travis data. These tests pin every
consumer of that convention that previously assumed otherwise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.baselines import RuleBasedAgent
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.environment import (HVAC_DEADBAND, HVAC_LOOP_GAIN, HVAC_OFFSET_RANGE,
                               thermostat_step)
from stems.thermal import CoPModel

HVAC = 2
P_BATT = np.array([3.0], dtype=np.float32)


def _obs(t_in=22.0, t_cool=24.0, t_heat=20.0, t_out=10.0, net=0.0, hour=12, heat_pump=True):
    o = np.zeros(30 if heat_pump else 28, dtype=np.float32)
    o[1], o[2], o[15], o[20], o[27] = hour, t_out, t_in, net, t_cool
    if heat_pump:
        o[28] = t_heat
    return o


# ---------------------------------------------------------------------------
# Integral thermostat (shared by the environment's set-point mode and the baseline)
# ---------------------------------------------------------------------------

def _step(u, t_in, offset=0.0, t_heat=20.0, t_cool=24.0):
    f = lambda x: np.array([x], dtype=np.float32)
    return float(thermostat_step(f(u), f(t_in), f(t_heat), f(t_cool), f(offset))[0])


def test_thermostat_integrates_the_distance_to_the_band():
    assert _step(0.0, 18.0) == pytest.approx(HVAC_LOOP_GAIN * 2.0)          # cold: more heat
    assert _step(0.2, 26.0) == pytest.approx(0.2 - HVAC_LOOP_GAIN * 2.0)    # hot: less / cool
    assert _step(0.3, 22.0) == pytest.approx(0.3)                           # inside: hold
    assert _step(0.3, 20.0 - 0.5 * HVAC_DEADBAND) == pytest.approx(0.3)     # deadband: hold


def test_thermostat_is_bounded_and_never_cools_a_cold_house_from_rest():
    assert _step(0.98, 10.0) == 1.0 and _step(-0.98, 35.0) == -1.0
    for t_in in np.linspace(10.0, 19.6, 12):
        assert _step(0.0, float(t_in)) > 0.0


def test_offset_shifts_the_band_by_at_most_the_offset_range():
    assert HVAC_OFFSET_RANGE < 2.0, "the offset must stay inside the 2 degC comfort tolerance"
    assert _step(0.0, 20.5, offset=1.0) == pytest.approx(
        HVAC_LOOP_GAIN * (20.0 + HVAC_OFFSET_RANGE - 20.5))                 # pre-heat
    assert _step(0.0, 23.5, offset=-1.0) == pytest.approx(
        -HVAC_LOOP_GAIN * (23.5 - (24.0 - HVAC_OFFSET_RANGE)))              # pre-cool
    assert _step(0.0, 20.5, offset=5.0) == _step(0.0, 20.5, offset=1.0)     # clipped


def test_baseline_runs_the_same_loop_in_power_mode_and_idles_in_setpoint_mode():
    cold = _obs(t_in=17.0, t_heat=20.0, t_cool=24.0)
    power = RuleBasedAgent(num_buildings=1, hvac_control="power", battery_nominal_power=P_BATT)
    first = power.select_action([cold])[0, HVAC]
    second = power.select_action([cold])[0, HVAC]
    assert first == pytest.approx(HVAC_LOOP_GAIN * 3.0) and second == pytest.approx(2 * first)
    supervisory = RuleBasedAgent(num_buildings=1, hvac_control="setpoint",
                                 battery_nominal_power=P_BATT)
    assert supervisory.select_action([cold])[0, HVAC] == 0.0


def test_baseline_integrates_from_the_executed_action():
    rbc = RuleBasedAgent(num_buildings=1, hvac_control="power", battery_nominal_power=P_BATT)
    cold = _obs(t_in=17.0, t_heat=20.0, t_cool=24.0)
    rbc.select_action([cold])
    rbc.notify_executed(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))      # shield cut it to 0
    assert rbc.select_action([cold])[0, HVAC] == pytest.approx(HVAC_LOOP_GAIN * 3.0)


def test_thermostat_refuses_to_guess_a_missing_heating_set_point():
    with pytest.raises(ValueError, match="heat_pump=True"):
        RuleBasedAgent(num_buildings=1, battery_nominal_power=P_BATT).select_action(
            [_obs(heat_pump=False)])


# ---------------------------------------------------------------------------
# CoP-aware HVAC power guard
# ---------------------------------------------------------------------------

def _shield(cap_kw=3.0):
    # Heating nameplate 8.7 kW, cooling 4.8 kW: the same |a| draws very
    # different power depending on which device the sign selects.
    cop = CoPModel(np.array([0.29]), np.array([46.0]), np.array([0.29]),
                   np.array([9.0]), np.array([8.7]), np.array([4.8]))
    return CBFShield(CBFConfig(P_building_max=cap_kw, P_grid_max=100.0), 1,
                     safety_cfg=SafetyConfig(robust_margins=False), enforce_soc=False,
                     cop_model=cop, hvac_idx=HVAC)


def test_cooling_command_on_a_cold_day_is_priced_at_the_cooling_device():
    """0.5 x 4.8 kW = 2.4 kW fits a 3 kW cap; priced as heating it would not."""
    safe = np.array([[0.0, 0.0, -0.5]], dtype=np.float32)
    out = _shield()._apply_hvac_power_guard(safe.copy(), [_obs(t_out=-5.0)])
    assert out[0, HVAC] == pytest.approx(-0.5)


def test_heating_command_is_capped_and_keeps_its_sign():
    safe = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    out = _shield()._apply_hvac_power_guard(safe.copy(), [_obs(t_out=-5.0)])
    assert 0.0 < out[0, HVAC] < 0.5
    assert out[0, HVAC] * 8.7 <= 3.0 + 1e-5


# ---------------------------------------------------------------------------
# Time-of-use storage rule
# ---------------------------------------------------------------------------

def _hourly(hour, load=2.0, solar=0.5):
    o = _obs(hour=hour)
    o[16], o[17] = load, solar
    return o


def test_rule_charges_before_the_peak_and_idles_overnight():
    rbc = RuleBasedAgent(num_buildings=1, hvac_control="setpoint", battery_nominal_power=P_BATT)
    assert rbc.select_action([_hourly(3)])[0, 1] == 0.0
    assert rbc.select_action([_hourly(13)])[0, 1] == RuleBasedAgent.CHARGE_ACTION
    assert rbc.select_action([_hourly(23)])[0, 1] == 0.0


def test_peak_discharge_never_exceeds_the_houses_own_net_load():
    """Exports earn nothing, so the battery covers the load and no more."""
    rbc = RuleBasedAgent(num_buildings=1, hvac_control="setpoint", battery_nominal_power=P_BATT)
    a = rbc.select_action([_hourly(18, load=2.0, solar=0.5)])[0, 1]
    assert a == pytest.approx(-(2.0 - 0.5) / 3.0)
    assert rbc.select_action([_hourly(18, load=0.4, solar=1.0)])[0, 1] == 0.0     # PV surplus
    assert rbc.select_action([_hourly(18, load=9.0, solar=0.0)])[0, 1] == -1.0    # capped


def test_grid_guard_scales_charging_to_the_cap_exactly():
    """Two buildings importing 10 kW each, both asking to charge 10 kW, cap 30 kW:
    half the charging fits. (cap / total = 0.75 would leave 35 kW.)"""
    from stems.battery import BatteryModel
    from stems.config import SafetyConfig

    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    shield = CBFShield(CBFConfig(P_building_max=1e9, P_grid_max=30.0), 2,
                       battery_model=BatteryModel([100.0] * 2, [10.0] * 2, [0.0] * 2,
                                                  [flat] * 2, [flat] * 2),
                       nominal_power=np.array([10.0, 10.0]), elec_idx=1,
                       safety_cfg=SafetyConfig(anticipatory=False, robust_margins=False))
    obs = [np.zeros(30) for _ in range(2)]
    for o in obs:
        o[19], o[20] = 0.5, 10.0                 # state of charge, net load last hour
    ask = np.zeros((2, 3), dtype=np.float32)
    ask[:, 1] = 1.0
    out = shield.project(ask, obs)
    assert out[:, 1] == pytest.approx([0.5, 0.5], abs=1e-6)
    shield.grid_guard = False                     # a cap shield with a forecast takes over
    assert shield.project(ask, obs)[:, 1] == pytest.approx([1.0, 1.0])
