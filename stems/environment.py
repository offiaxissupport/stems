"""CityLearn environment wrapper for STEMS.

Real CityLearn is the only path used for results. The wrapper is **fail-loud**:
if CityLearn (or the requested schema) cannot be constructed it raises, rather
than silently substituting synthetic data. A synthetic mock exists *only* for
fast unit/smoke tests and is reachable solely via ``force_mock=True``; it prints
a prominent banner and reports ``env_type == "mock"`` so no result is ever
mistaken for real.

The wrapper maps CityLearn's rich 45-feature observation down to the 28-feature
STEMS subset (``OBS_NAMES``, paper Section III-A2) and exposes the per-building
battery dynamics needed to calibrate the CBF (``battery_info``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from citylearn.citylearn import CityLearnEnv  # type: ignore
    _CITYLEARN_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as exc:  # pragma: no cover - import-time environment issue
    CityLearnEnv = None  # type: ignore
    _CITYLEARN_IMPORT_ERROR = exc


# --------------------------------------------------------------------------
# 28-feature STEMS observation subset (selected from CityLearn's 45 features)
# --------------------------------------------------------------------------
OBS_NAMES: List[str] = [
    "day_type",                                       # 0
    "hour",                                           # 1
    "outdoor_dry_bulb_temperature",                   # 2
    "outdoor_dry_bulb_temperature_predicted_1",       # 3
    "outdoor_dry_bulb_temperature_predicted_2",       # 4
    "outdoor_dry_bulb_temperature_predicted_3",       # 5
    "diffuse_solar_irradiance",                       # 6
    "diffuse_solar_irradiance_predicted_1",           # 7
    "diffuse_solar_irradiance_predicted_2",           # 8
    "diffuse_solar_irradiance_predicted_3",           # 9
    "direct_solar_irradiance",                        # 10
    "direct_solar_irradiance_predicted_1",            # 11
    "direct_solar_irradiance_predicted_2",            # 12
    "direct_solar_irradiance_predicted_3",            # 13
    "carbon_intensity",                               # 14
    "indoor_dry_bulb_temperature",                    # 15
    "non_shiftable_load",                             # 16
    "solar_generation",                               # 17
    "dhw_storage_soc",                                # 18
    "electrical_storage_soc",                         # 19
    "net_electricity_consumption",                    # 20
    "electricity_pricing",                            # 21
    "electricity_pricing_predicted_1",                # 22
    "electricity_pricing_predicted_2",                # 23
    "cooling_demand",                                 # 24
    "dhw_demand",                                      # 25
    "occupant_count",                                 # 26
    "indoor_dry_bulb_temperature_cooling_set_point",  # 27
]

# Heat-pump extension features (appended when heat_pump=True). Both are present
# in the real Travis schema; there is no "heating_demand" feature, so the heating
# electrical draw is used as the heating-load signal instead.
# Lead times of outdoor_dry_bulb_temperature_predicted_1/2/3, measured against
# the tx_travis data files: the actuals shifted by 6, 12 and 24 hours (MAE
# 0.15 / 0.33 / 0.69 degC), not by 1, 2 and 3.
T_OUT_PRED_LEAD_H = (6.0, 12.0, 24.0)

HEATPUMP_OBS_NAMES: List[str] = [
    "indoor_dry_bulb_temperature_heating_set_point",  # 28
    "heating_electricity_consumption",                # 29
]

OBS_DIM = len(OBS_NAMES)   # 28
ACTION_DIM = 3             # dhw_storage, electrical_storage, cooling_or_heating_device

# Named subsets of the action space for restricted-control studies (e.g.
# training on the heat pump only while the battery stays frozen at no-op).
# Resolved to indices via STEMSEnvironment.resolve_control_indices, so this
# generalises to future actuators (e.g. an EV action) without code changes.
# ---------------------------------------------------------------------------
# Electric-vehicle charger slots
# ---------------------------------------------------------------------------
# A charger bay is described by five observations whose CityLearn names embed
# the charger id, so they differ building to building. STEMS needs a
# homogeneous per-building vector (the GCN stacks buildings into one matrix), so
# bays are mapped into a fixed number of *slots* and buildings with fewer bays
# are zero-padded. A padded slot reports ``connected_state = 0``, which the EV
# barrier already treats as "no vehicle" -- so an absent charger and an empty bay
# are handled by the same code path rather than by a special case.

EV_SLOT_FIELDS: List[str] = [
    "connected_state",
    "departure_time",
    "required_soc_departure",
    "soc",
    "battery_capacity",
]


def ev_slot_obs_names(slot: int) -> List[str]:
    """Canonical STEMS observation names for EV slot ``slot``."""
    return [f"ev{slot}_{f}" for f in EV_SLOT_FIELDS]


def ev_native_obs_names(charger_id: str) -> Dict[str, str]:
    """CityLearn's own observation names for one charger id."""
    base = f"connected_electric_vehicle_at_charger_{charger_id}"
    return {
        "connected_state": f"electric_vehicle_charger_{charger_id}_connected_state",
        "departure_time": f"{base}_departure_time",
        "required_soc_departure": f"{base}_required_soc_departure",
        "soc": f"{base}_soc",
        "battery_capacity": f"{base}_battery_capacity",
    }


EV_ACTION_PREFIX = "electric_vehicle_storage"


def ev_slot_action_name(slot: int) -> str:
    """Canonical STEMS action name for EV slot ``slot``."""
    return f"{EV_ACTION_PREFIX}_{slot}"


ACTION_GROUPS: Dict[str, List[str]] = {
    "dhw": ["dhw_storage"],
    "heatpump": ["cooling_or_heating_device"],
    "thermal": ["dhw_storage", "cooling_or_heating_device"],
    # Electric vehicles. CityLearn names a charger action per bay
    # ("electric_vehicle_storage_charger_1_1"), so the entry is matched as a
    # *prefix* and may resolve to several indices in one building -- see
    # ``_find_action_indices``. A schema without chargers raises rather than
    # silently controlling nothing.
    "ev": ["electric_vehicle_storage"],
    "ev+thermal": ["dhw_storage", "cooling_or_heating_device",
                   "electric_vehicle_storage"],
    "battery+ev": ["electrical_storage", "electric_vehicle_storage"],
    "all": ["dhw_storage", "electrical_storage", "cooling_or_heating_device",
            "electric_vehicle_storage"],
}


# --------------------------------------------------------------------------
# Synthetic mock (unit/smoke tests only -- never used for reported results)
# --------------------------------------------------------------------------

class _MockBuilding:
    """One building with simple, self-consistent physics for testing.

    The mock's battery uses ``soc += elec_action * MOCK_SOC_RATE`` so the CBF,
    calibrated from ``battery_info``, stays consistent with it.
    """

    MOCK_SOC_RATE = 0.1
    _TYPE_PARAMS = {
        "residential": (10.0, 5.0, 1.5, 5.0),
        "commercial":  (15.0, 5.0, 1.5, 8.0),
        "mixed":       (12.0, 4.0, 1.2, 6.0),
    }

    def __init__(self, rng: np.random.Generator, building_id: int) -> None:
        self.rng = rng
        self.id = building_id
        self._type = ("residential" if building_id <= 4
                      else "commercial" if building_id <= 6 else "mixed")
        p = self._TYPE_PARAMS[self._type]
        self._base_load, self._load_var, self._cool_coeff, self._solar_cap = p
        self._soc_dhw = 0.5
        self._soc_elec = 0.5
        self._t_indoor = 22.0
        self._t = 0
        self._temp_offset = 0.0

    def reset(self) -> None:
        self._soc_dhw = 0.5 + self.rng.uniform(-0.1, 0.1)
        self._soc_elec = 0.5 + self.rng.uniform(-0.1, 0.1)
        self._t_indoor = 22.0 + self.rng.uniform(-1.0, 1.0)
        self._t = 0

    def step(self, action: np.ndarray, heat_pump: bool = False) -> np.ndarray:
        """Advance one hour with a 3-dim action; return the 28-dim observation
        (30-dim in heat-pump mode). Conventions follow CityLearn: ``hour`` runs
        1..24 and the HVAC action heats when positive, cools when negative."""
        self._t += 1
        hour = (self._t - 1) % 24 + 1
        day_type = 1 + int(self._t / 24) % 7

        t_out = 15.0 + 10.0 * np.sin(2 * np.pi * hour / 24) + self.rng.normal(0, 1) + self._temp_offset
        solar = max(0.0, 500.0 * np.sin(np.pi * (hour - 6) / 12)) + self.rng.normal(0, 20)
        carbon = 0.3 + 0.1 * np.sin(2 * np.pi * hour / 24) + self.rng.normal(0, 0.02)
        price = float(np.clip(0.12 + 0.08 * (1.0 if 16 <= hour <= 21 else 0.0)
                              + self.rng.normal(0, 0.005), 0.05, 0.30))

        load = float(np.clip(self._base_load + self._load_var * np.sin(2 * np.pi * hour / 24)
                             + self.rng.normal(0, self._load_var * 0.3), 0.5, self._base_load * 4.0))
        cooling = max(0.0, self._cool_coeff * (t_out - 18.0) + self.rng.normal(0, self._cool_coeff * 0.2))
        dhw = max(0.0, self._base_load * 0.05 + self._base_load * 0.02 * self.rng.normal())
        occupant = float(self.rng.integers(0, 5))
        solar_gen = max(0.0, self._solar_cap * solar / 500.0 + self.rng.normal(0, self._solar_cap * 0.05))

        dhw_action = float(np.clip(action[0], -1, 1))
        elec_action = float(np.clip(action[1], -1, 1))
        hvac_action = float(np.clip(action[2], -1, 1))

        self._soc_dhw = float(np.clip(self._soc_dhw + dhw_action * 0.05, 0.05, 0.95))
        self._soc_elec = float(np.clip(self._soc_elec + elec_action * self.MOCK_SOC_RATE, 0.0, 1.0))

        # Thermal model. Cooling (hvac_action < 0) removes heat; in heat-pump
        # mode, heating (hvac_action > 0) adds heat. Passive exchange always on.
        cool_effect = 0.5 * max(0.0, -hvac_action)
        heat_effect = (0.5 * max(0.0, hvac_action)) if heat_pump else 0.0
        self._t_indoor += (0.1 * (t_out - self._t_indoor) - cool_effect + heat_effect
                           + self.rng.normal(0, 0.1))
        self._t_indoor = float(np.clip(self._t_indoor, 10.0, 35.0))

        hvac_draw = 0.5 * abs(hvac_action)
        storage_draw = 2.0 * abs(dhw_action) + 5.0 * abs(elec_action) + hvac_draw
        net = float(np.clip(load + cooling - solar_gen + storage_draw,
                            -self._solar_cap * 2, self._base_load * 8.0))

        t_pred = [t_out + self.rng.normal(0, 0.5) for _ in range(3)]
        s_pred = [max(0, solar * 0.8 + self.rng.normal(0, 30)) for _ in range(3)]
        p_pred = [float(np.clip(price + self.rng.normal(0, 0.01), 0.05, 0.30)) for _ in range(2)]
        t_set = 22.0 + self.rng.normal(0, 0.5)

        obs = [
            float(day_type), float(hour), t_out, t_pred[0], t_pred[1], t_pred[2],
            solar * 0.6, s_pred[0], s_pred[1], s_pred[2],
            solar * 0.4, s_pred[0], s_pred[1], s_pred[2],
            carbon, self._t_indoor, load, solar_gen, self._soc_dhw, self._soc_elec,
            net, price, p_pred[0], p_pred[1], cooling, dhw, occupant, t_set,
        ]
        if heat_pump:
            # HEATPUMP_OBS_NAMES: heating set point, heating electricity.
            obs += [t_set - 2.0, 0.5 * max(0.0, hvac_action)]
        return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Supervisory heat-pump control: a set-point offset tracked by an inner loop
# ---------------------------------------------------------------------------
# CityLearn's HVAC action is the fraction of nameplate electrical power, and its
# LSTM temperature model responds to it almost statically and very strongly.
# Measured step responses (Travis, 8 houses): +0.25 heating raises the indoor
# temperature 1.5-2.5 degC in the first hour, +0.5 by 3.4-6.1 degC; -0.25 cooling
# lowers it 2-9 degC in summer and 4-13 degC in winter (the last is outside
# anything the model was trained on and is not physical). The plant gain is
# therefore ~8-12 degC per unit action for heating and up to ~50 for cooling.
#
# A controller acting on power directly must hold a 2 degC band through that
# gain at an hourly step: a proportional thermostat at 0.5 action/degC has a
# loop gain of 5-25 and oscillates between heating and cooling (measured: 66%
# discomfort at three times the energy), and an untrained policy does no better.
#
# In ``hvac_control="setpoint"`` the HVAC action is instead a set-point offset
# in [-1, 1] x HVAC_OFFSET_RANGE degC, and this integral loop turns the
# temperature error into the power command:
#
#     u <- clip(u + HVAC_LOOP_GAIN * e, -1, 1)
#
# with e the distance to the shifted comfort band (zero inside it, and inside a
# HVAC_DEADBAND). At the typical heating gain of 10 degC/unit the loop gain is
# 0.5 -- stable, converging in a few hours. With a zero offset it is a plain
# thermostat: measured 2.1% winter and 5.2% summer discomfort. The offset range
# is smaller than the 2 degC comfort tolerance, so a supervisory policy can
# pre-heat or coast without being able to leave the band in steady state.

HVAC_LOOP_GAIN = 0.05      # power-action change per degC of error per step
HVAC_DEADBAND = 0.3        # degC
HVAC_OFFSET_RANGE = 1.5    # degC of set-point shift at offset action = +-1


def thermostat_step(u: np.ndarray, t_in: np.ndarray, t_heat: np.ndarray,
                    t_cool: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """One step of the integral thermostat; returns the new power action (B,).

    ``t_heat``/``t_cool`` are the heating and cooling set points of the hour
    being decided and ``offset`` the supervisory action in [-1, 1]. Positive
    power heats, negative cools (CityLearn's sign convention).
    """
    shift = HVAC_OFFSET_RANGE * np.clip(offset, -1.0, 1.0)
    low, high = t_heat + shift, t_cool + shift
    error = np.where(t_in < low, low - t_in, np.where(t_in > high, high - t_in, 0.0))
    error = np.where(np.abs(error) < HVAC_DEADBAND, 0.0, error)
    return np.clip(u + HVAC_LOOP_GAIN * error, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Upstream CityLearn defect: the DHW storage action is a no-op
# ---------------------------------------------------------------------------
# ``citylearn.building.Building.update_dhw_storage`` (2.6.0b1, building.py:1562)
# scales the requested energy by ``heating_storage.capacity`` instead of
# ``dhw_storage.capacity``:
#
#     energy = action * self.heating_storage.capacity * ratio
#
# This schema has no heating storage tank, so ``heating_storage.capacity`` is
# 0.0 and *every* dhw_storage action resolves to zero energy -- the hot-water
# actuator silently does nothing. Verified directly: charging at action=0.8 for
# eight consecutive steps leaves ``dhw_storage_soc`` at exactly 0.0.
#
# The patch below is the one-word correction. It is applied loudly (banner +
# metadata stamp) and can be disabled with ``patch_dhw=False``, consistent with
# this project's no-silent-fallbacks rule: a simulator correction must be
# visible in the run record, never assumed.

def _patched_update_dhw_storage(self, action: float) -> None:
    """``Building.update_dhw_storage`` with ``dhw_storage.capacity`` (see above)."""
    from citylearn.energy_model import HeatPump

    energy = (action * self.dhw_storage.capacity
              * self.algorithm_action_based_time_step_hours_ratio)
    temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]

    if energy > 0.0:
        max_electric_power = self.downward_electrical_flexibility
        max_output = (
            self.dhw_device.get_max_output_power(
                temperature, heating=True, max_electric_power=max_electric_power)
            if isinstance(self.dhw_device, HeatPump)
            else self.dhw_device.get_max_output_power(
                max_electric_power=max_electric_power))
        energy = min(max_output, energy)
    else:
        demand = self.dhw_demand[self.time_step]
        energy = max(-demand, energy)

    self.dhw_storage.charge(self._convert_energy_for_storage(self.dhw_storage, energy))
    charged_energy = max(self.dhw_storage.energy_balance[self.time_step], 0.0)
    electricity_consumption = (
        self.dhw_device.get_input_power(charged_energy, temperature, heating=True)
        if isinstance(self.dhw_device, HeatPump)
        else self.dhw_device.get_input_power(charged_energy))
    self.dhw_device.update_electricity_consumption(electricity_consumption)


# ---------------------------------------------------------------------------
# Upstream CityLearn defect: thermal-device load counted twice at episode start
# ---------------------------------------------------------------------------
# At ``reset()``, ``Building.update_variables`` (2.6.0b1, building.py:2432) runs
# its ``time_step == 0`` branch and *sets* each thermal device's t=0
# electricity consumption from the uncontrolled (ideal) load. The first
# ``step()`` then runs ``update_energy_from_heating_device`` (building.py:1491),
# which checks the demand against
#
#     max_output = cop * (nominal_power - electricity_consumption[0])
#
# -- headroom that already has the same load subtracted -- before *adding* the
# controlled consumption on top. Any building whose episode-start heating need
# exceeds half its nameplate fails CityLearn's own demand assertion whatever the
# action. Measured on Travis building 120912, winter, t=0: need 3.84 kWe against
# a 4.96 kW heat pump, booked headroom 1.12 kW, demand 6.61 kW > output 1.93 kW.
# CityLearn fixed exactly this double count for the battery (the NOTE at
# building.py:2464) but not for the cooling, heating and DHW devices.
#
# The same shrunken headroom also *silently clips* storage charging on that
# first step, where CityLearn caps instead of asserting. Measured on building
# 400973 (Travis, winter): a DHW charge command of 0.87 drew 6.42 instead of the
# heater's full 6.76 kWe -- short by exactly the booked load. The tank state
# reconverged by t=2 and the 3-day episode's energy moved by 0.002%. The final
# t=0 accounting is otherwise correct: after ``apply_actions`` the same
# ``time_step == 0`` branch runs again and *overwrites* the booked value with the
# controlled one.
#
# The patch clears the reset-time booking immediately before the first step.
# Applied loudly, disable with ``patch_t0_double_count=False``.

# ---------------------------------------------------------------------------
# Upstream CityLearn defect: the indoor temperature observation ignores control
# ---------------------------------------------------------------------------
# ``LSTMDynamicsBuilding.apply_actions`` (2.6.0b1, building.py:2763) simulates the
# indoor temperature and writes it to ``indoor_dry_bulb_temperature[t]`` *before*
# the time step advances. The observation is then read at ``t + 1``
# (internal/building_ops.py, ``series[t]`` after advancing), which still holds the
# dataset's uncontrolled value. The source comment at building.py:2860 ("called
# after advancing to next timestep") describes an order the refactor no longer
# has. Measured over 216 controlled building-steps in winter: the observed
# temperature equals the uncontrolled ``indoor_dry_bulb_temperature_without_control
# [t+1]`` every time and the simulated one never; full heating drove the
# simulated house to 34 degC and full cooling to -15 degC while the controller
# observed 17-23 degC either way. A controller, its comfort reward and any
# comfort metric reading the observation are blind to their own HVAC actions.
#
# Separately, on an episode's final transition CityLearn applies the action but
# does not advance the buildings (internal/runtime.py:204), so every state
# observation repeats the previous hour.
#
# The correction reads each state observation for the hour the action was
# applied to straight from the simulator's own series. Elsewhere the values are
# identical to what CityLearn returns; exogenous observations (weather, price,
# set points, occupancy) are untouched and keep describing the hour about to be
# decided. Applied loudly, disable with ``patch_endogenous_obs=False``.

ENDOGENOUS_OBS = {
    "indoor_dry_bulb_temperature": lambda b: b.energy_simulation.indoor_dry_bulb_temperature,
    "net_electricity_consumption": lambda b: b.net_electricity_consumption,
    "electrical_storage_soc": lambda b: b.electrical_storage.soc,
    "dhw_storage_soc": lambda b: b.dhw_storage.soc,
    "heating_electricity_consumption": lambda b: b.heating_electricity_consumption,
}


def _wrap_apply_actions_t0(building) -> None:
    """Clear reset-booked thermal-device consumption before an episode's first step."""
    original = building.apply_actions

    def apply_actions(**kwargs):
        if building.time_step == 0:
            devices = {id(d): d for d in (building.cooling_device, building.heating_device,
                                          building.dhw_device) if d is not None}
            for device in devices.values():
                device.set_electricity_consumption(0.0, time_step=0)
        return original(**kwargs)

    building.apply_actions = apply_actions


class _MockCityLearnEnv:
    """Minimal mock of CityLearnEnv (8 buildings, full-year episodes)."""

    NUM_BUILDINGS = 8
    EPISODE_LEN = 8760

    def __init__(self, seed: int = 0, heat_pump: bool = False) -> None:
        self._heat_pump = heat_pump
        self._buildings = [_MockBuilding(np.random.default_rng(seed + i), i)
                           for i in range(self.NUM_BUILDINGS)]
        self._timestep = 0

    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        self._timestep = 0
        for b in self._buildings:
            b.reset()
        obs = [b.step(np.zeros(3), self._heat_pump) for b in self._buildings]
        return obs, {}

    def set_temp_offset(self, offset: float) -> None:
        for b in self._buildings:
            b._temp_offset = float(offset)

    def step(self, actions: List[np.ndarray]):
        self._timestep += 1
        obs = [b.step(a, self._heat_pump) for b, a in zip(self._buildings, actions)]
        rewards = [float(-o[20] * o[21]) for o in obs]
        done = self._timestep >= self.EPISODE_LEN
        return obs, rewards, done, False, {}

    @property
    def observation_names(self):
        return [list(OBS_NAMES)] * self.NUM_BUILDINGS

    @property
    def action_names(self):
        return [["dhw_storage", "electrical_storage", "cooling_or_heating_device"]
                ] * self.NUM_BUILDINGS

    @property
    def action_space(self):
        class _S:
            shape = (ACTION_DIM,)
            low = np.full(ACTION_DIM, -1.0)
            high = np.full(ACTION_DIM, 1.0)
        return [_S()] * self.NUM_BUILDINGS

    @property
    def buildings(self):
        out = []
        for i in range(self.NUM_BUILDINGS):
            b = type("_B", (), {})()
            b.name = f"Building_{i+1}"
            # Mock battery consistent with _MockBuilding.MOCK_SOC_RATE = 0.1:
            # nominal_power/capacity = 0.1 -> e.g. 1.0 / 10.0.
            b.electrical_storage = type("_E", (), {
                "capacity": 10.0, "nominal_power": 1.0, "efficiency": 1.0})()
            out.append(b)
        return out


# --------------------------------------------------------------------------
# STEMSEnvironment
# --------------------------------------------------------------------------

class STEMSEnvironment:
    """Wraps real CityLearn (default) or, only on explicit request, the mock.

    Parameters
    ----------
    schema : str | None
        Path to a CityLearn schema JSON (defaults to the Travis 8-building one).
    seed : int
        Seed (used by the mock; CityLearn seeding is in the schema).
    force_mock : bool
        Use the synthetic mock instead of real CityLearn. Prints a loud banner;
        results from this mode must never be reported as real.
    heat_pump : bool
        Enable bidirectional heat-pump physics/observations (Phase 3).
    """

    SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
    _BANNER = ("\n" + "*" * 64 +
               "\n***  SYNTHETIC MOCK ENVIRONMENT -- NOT FOR REPORTED RESULTS  ***\n" +
               "*" * 64 + "\n")

    def __init__(self, schema: Optional[str] = None, seed: int = 0,
                 force_mock: bool = False, heat_pump: bool = False,
                 patch_dhw: bool = True, max_ev_slots: Optional[int] = None,
                 allow_missing_obs: bool = False,
                 env_kwargs: Optional[Dict[str, Any]] = None,
                 allow_resized_simulation: bool = False,
                 patch_t0_double_count: bool = True,
                 patch_endogenous_obs: bool = True,
                 hvac_control: str = "power") -> None:
        self._seed = seed
        self._heat_pump = heat_pump
        self._comm_dropout = 0.0
        self._temp_offset = 0.0
        self._temp_gradient = 0.0
        self._patch_dhw = bool(patch_dhw)
        self._patch_t0 = bool(patch_t0_double_count)
        self._patch_endogenous = bool(patch_endogenous_obs)
        if hvac_control not in ("power", "setpoint"):
            raise ValueError(f"hvac_control must be 'power' or 'setpoint', got {hvac_control!r}")
        if hvac_control == "setpoint" and not (heat_pump and patch_endogenous_obs) :
            raise ValueError("hvac_control='setpoint' tracks the heating and cooling set "
                             "points against the simulated indoor temperature: it needs "
                             "heat_pump=True and patch_endogenous_obs=True")
        if hvac_control == "setpoint" and force_mock:
            raise ValueError("hvac_control='setpoint' is not available on the mock environment")
        self._hvac_control = hvac_control
        self._max_ev_slots = max_ev_slots
        self._allow_missing_obs = bool(allow_missing_obs)
        self._absent_obs: List[str] = []
        self._patches: List[str] = []
        # Overrides handed straight to CityLearnEnv, recorded in run metadata.
        # Select a period with ``episode_time_steps=[(start, end)]``; the guard
        # below explains why the simulation itself must not be shortened.
        self._env_kwargs: Dict[str, Any] = dict(env_kwargs or {})
        resizing = sorted(k for k in ("simulation_start_time_step", "simulation_end_time_step")
                          if k in self._env_kwargs)
        if resizing and not allow_resized_simulation:
            raise ValueError(
                f"env_kwargs sets {resizing}. Shortening the simulation makes CityLearn "
                "autosize every device from that shorter period, so the house changes with "
                "the window: a 7-day summer window sizes building 134795's heating heat pump "
                "at 0 kW instead of 8.68 kW, and CityLearn's own demand check then fails at "
                "the first step. Select the period with episode_time_steps=[(start, end)] "
                "inside the full-year simulation instead, which keeps full-year device sizes. "
                "Pass allow_resized_simulation=True only if a resized house is intended.")
        requested = schema or self.SCHEMA

        if force_mock:
            sys.stderr.write(self._BANNER)
            sys.stderr.flush()
            self._env = _MockCityLearnEnv(seed=seed, heat_pump=heat_pump)
            self._mock = True
        else:
            self._env = self._build_real_env(requested)
            self._mock = False

        self._configure_dims_and_indices()

    # ------------------------------------------------------------------
    # Construction helpers (fail loud)
    # ------------------------------------------------------------------

    def _build_real_env(self, requested_schema: str):
        if CityLearnEnv is None:
            raise RuntimeError(
                "Real CityLearn is required but could not be imported "
                f"({_CITYLEARN_IMPORT_ERROR!r}). Use the project .venv (Python 3.12) "
                "where torch/citylearn/cvxpy are installed, or pass force_mock=True "
                "for synthetic smoke tests only."
            )
        schema = self._resolve_schema(requested_schema)
        try:
            return CityLearnEnv(schema=schema, central_agent=False, **self._env_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to construct real CityLearn env from schema {schema!r}: {exc!r}. "
                "Regenerate it with `python -B setup_citylearn_8b.py --validate`. "
                "Refusing to silently fall back to a synthetic mock."
            ) from exc

    @staticmethod
    def _resolve_schema(schema: str) -> str:
        """Resolve a schema path, or the name of a bundled CityLearn dataset.

        A bare name such as ``citylearn_challenge_2022_phase_all_plus_evs`` is
        handed to CityLearn unchanged, which downloads and caches it. That makes
        the published EV and cold-climate datasets usable without copying them
        into the repository, while a local path still wins so the project's own
        generated schemas are never shadowed by a same-named dataset.
        """
        p = Path(schema)
        if p.is_file():
            return str(p)
        repo_local = Path(__file__).resolve().parents[1] / schema
        if repo_local.is_file():
            return str(repo_local)
        try:
            from citylearn.data import DataSet
            if schema in DataSet().get_dataset_names():
                return schema
        except Exception:
            pass
        raise RuntimeError(
            f"Schema {schema!r} not found (looked at {p}, {repo_local}, and the "
            "bundled CityLearn dataset names). "
            "Run `python -B setup_citylearn_8b.py --validate` to generate it."
        )

    def _configure_dims_and_indices(self) -> None:
        """Build a homogeneous STEMS view over possibly heterogeneous buildings.

        CityLearn buildings need not agree: in the EV challenge datasets the
        per-building action space has 1-3 entries and the observation vector
        28-42, because only some buildings own a charger and charger ids are
        baked into observation names. STEMS needs one shape for all buildings --
        the GCN stacks them into a single matrix and every actor is built to the
        same signature -- so this method constructs a *canonical* padded layout
        plus a per-building map into each building's own native indices.

        Devices a building does not have are simply absent from its map: their
        canonical observations read zero and their canonical actions are dropped
        before the vector reaches CityLearn. For a homogeneous schema such as
        ``tx_travis_8b`` every map is the identity and this reduces exactly to
        the previous behaviour.
        """
        self._num_buildings = len(self._env.observation_names)
        B = self._num_buildings
        raw_obs_names = [list(n) for n in self._env.observation_names]
        raw_action_names = getattr(self._env, "action_names", None)
        if raw_action_names:
            native_actions = [list(n) for n in raw_action_names]
        else:
            native_actions = [[f"action_{j}" for j in range(sp.shape[0])]
                              for sp in self._env.action_space]
        self._native_action_names = native_actions

        # -- charger discovery, per building -----------------------------
        self._charger_ids: List[List[str]] = []
        marker, suffix = "electric_vehicle_charger_", "_connected_state"
        for names in raw_obs_names:
            ids = [n[len(marker):-len(suffix)] for n in names
                   if n.startswith(marker) and n.endswith(suffix)]
            self._charger_ids.append(ids)
        found = max((len(ids) for ids in self._charger_ids), default=0)
        self._ev_slots = int(self._max_ev_slots) if self._max_ev_slots is not None else found
        if self._max_ev_slots is not None and found > self._ev_slots:
            raise RuntimeError(
                f"Schema exposes {found} chargers on some building but "
                f"max_ev_slots={self._ev_slots}; raise it or bays would be dropped.")

        # -- canonical observation layout --------------------------------
        obs_names = list(OBS_NAMES)
        if self._heat_pump:
            obs_names = obs_names + list(HEATPUMP_OBS_NAMES)
        for slot in range(self._ev_slots):
            obs_names += ev_slot_obs_names(slot)
        self._selected_obs_names = obs_names
        self._obs_dim = len(obs_names)

        base_count = len(obs_names) - self._ev_slots * len(EV_SLOT_FIELDS)
        self._obs_indices_per_building: List[List[Optional[int]]] = []
        for b in range(B):
            index = {n: i for i, n in enumerate(raw_obs_names[b])}
            row: List[Optional[int]] = [index.get(n) for n in obs_names[:base_count]]
            for slot in range(self._ev_slots):
                ids = self._charger_ids[b]
                if slot < len(ids):
                    native = ev_native_obs_names(ids[slot])
                    row += [index.get(native[f]) for f in EV_SLOT_FIELDS]
                else:
                    row += [None] * len(EV_SLOT_FIELDS)
            self._obs_indices_per_building.append(row)
        # Kept for backward compatibility: the first building's map.
        self._obs_indices = self._obs_indices_per_building[0]

        # A base feature absent from *every* building is a schema mismatch, not
        # padding. Fail loud unless explicitly allowed, and record it either way.
        absent = [n for j, n in enumerate(obs_names[:base_count])
                  if all(row[j] is None for row in self._obs_indices_per_building)]
        self._absent_obs = absent
        if absent and not self._mock:
            if not self._allow_missing_obs:
                raise RuntimeError(
                    f"Schema is missing required observations {absent}. "
                    "Regenerate the schema with them active, or pass "
                    "allow_missing_obs=True to zero-fill them (recorded in "
                    "metadata as absent_observations).")
            print("=" * 74)
            print(f"[STEMS] {len(absent)} STEMS observation(s) ABSENT from this "
                  "schema and zero-filled:")
            print(f"        {absent}")
            print("        Any reward or metric term reading them is meaningless "
                  "for this run.")
            print("=" * 74)

        # -- canonical action layout -------------------------------------
        canonical: List[str] = []
        for name in ("dhw_storage", "electrical_storage", "cooling_or_heating_device"):
            if any(name in names for names in native_actions):
                canonical.append(name)
        for slot in range(self._ev_slots):
            canonical.append(ev_slot_action_name(slot))
        if not canonical:
            raise RuntimeError(
                f"No recognised STEMS actions in {native_actions[0]!r}.")
        self._action_names = canonical
        self._action_dim = len(canonical)

        self._action_slot_to_native: List[Dict[int, int]] = []
        for b in range(B):
            names = native_actions[b]
            ev_native = [j for j, n in enumerate(names)
                         if n.startswith(EV_ACTION_PREFIX)]
            mapping: Dict[int, int] = {}
            for slot, cname in enumerate(canonical):
                if cname.startswith(EV_ACTION_PREFIX):
                    k = int(cname.rsplit("_", 1)[1])
                    if k < len(ev_native):
                        mapping[slot] = ev_native[k]
                elif cname in names:
                    mapping[slot] = names.index(cname)
            self._action_slot_to_native.append(mapping)

        self._electrical_storage_action_index = self._find_action_index(
            "electrical_storage", optional=True)
        self._hvac_action_index = self._find_action_index(
            "cooling_or_heating_device", optional=True)
        self._dhw_action_index = self._find_action_index("dhw_storage", optional=True)

        self._apply_citylearn_patches()
        self._battery_info = self._extract_battery_info()
        self._dhw_info = self._extract_dhw_info()
        self._heat_pump_info = self._extract_heat_pump_info()

    @property
    def ev_slots(self) -> int:
        """Number of EV charger slots in the canonical layout (0 when none)."""
        return self._ev_slots

    @property
    def absent_observations(self) -> List[str]:
        """STEMS base features this schema does not provide (zero-filled)."""
        return list(self._absent_obs)

    def building_has_action(self, building: int, slot: int) -> bool:
        """Whether ``building`` physically owns the device at canonical ``slot``."""
        return slot in self._action_slot_to_native[building]

    def action_presence_mask(self) -> np.ndarray:
        """(B, action_dim) mask: 1 where the building owns that actuator.

        A padded slot is not merely unused -- commanding it is meaningless, so
        the agent's output there is discarded rather than passed on.
        """
        mask = np.zeros((self._num_buildings, self._action_dim), dtype=np.float32)
        for b, mapping in enumerate(self._action_slot_to_native):
            for slot in mapping:
                mask[b, slot] = 1.0
        return mask

    def _find_action_index(self, name: str, optional: bool = False) -> int:
        try:
            return self._action_names.index(name)
        except ValueError:
            if optional:
                return -1
            raise RuntimeError(
                f"Action {name!r} not found in action space {self._action_names}.")

    def _find_action_indices(self, name: str) -> List[int]:
        """Resolve an action name to every index it matches, in order.

        An exact match wins. Failing that the name is treated as a prefix, which
        is what per-bay device naming requires: CityLearn calls a charger action
        ``electric_vehicle_storage_charger_1_1``, and a building may have several.
        Raises when nothing matches, so asking to control a device the schema
        does not have fails loudly instead of yielding an empty control set.
        """
        if name in self._action_names:
            return [self._action_names.index(name)]
        matches = [i for i, n in enumerate(self._action_names)
                   if n.startswith(name)]
        if not matches:
            raise RuntimeError(
                f"Action {name!r} not found (exact or prefix) in action space "
                f"{self._action_names}. This schema does not expose that device.")
        return matches

    # ------------------------------------------------------------------
    # Battery dynamics (calibrates the CBF -- the key fix vs the old code)
    # ------------------------------------------------------------------

    def _extract_battery_info(self) -> Dict[str, np.ndarray]:
        """Per-building battery parameters and the SOC change per unit action.

        CityLearn's battery moves SOC by ~ action * nominal_power / capacity per
        hour (verified empirically to 3 dp). We expose ``soc_rate = nominal_power
        / capacity`` -- a slight over-estimate of the true (efficiency-scaled)
        delta, which makes the CBF marginally conservative (safe) rather than
        optimistic (the old hard-coded 0.1 under-estimated it 2-5x and was the
        main cause of SOC violations).
        """
        B = self._num_buildings
        cap = np.ones(B, dtype=np.float32)
        nom = np.full(B, 0.1, dtype=np.float32)
        eff = np.ones(B, dtype=np.float32)
        try:
            for i, b in enumerate(self._env.buildings):
                es = b.electrical_storage
                cap[i] = float(getattr(es, "capacity", 1.0)) or 1.0
                nom[i] = float(getattr(es, "nominal_power", 0.1))
                eff[i] = float(getattr(es, "efficiency", 1.0))
        except Exception as exc:
            raise RuntimeError(f"Could not read battery parameters: {exc!r}") from exc
        soc_rate = np.clip(nom / np.maximum(cap, 1e-6), 1e-3, 1.0).astype(np.float32)
        return {"soc_rate": soc_rate, "capacity": cap,
                "nominal_power": nom, "efficiency": eff}

    def battery_info(self) -> Dict[str, np.ndarray]:
        """Return per-building battery dynamics (soc_rate, capacity, ...)."""
        return {k: v.copy() for k, v in self._battery_info.items()}

    def battery_model(self):
        """The batteries' one-step dynamics for the safety barrier.

        On real CityLearn this is the simulator's own model, built from each
        battery's curves and losses (``stems.battery.BatteryModel``); on the mock
        it is the mock's linear rule.
        """
        from stems.battery import BatteryModel

        if self._mock:
            return BatteryModel.linear(self._battery_info["soc_rate"])
        return BatteryModel.from_citylearn(
            self._env.buildings, float(self._env.seconds_per_time_step))

    def dhw_tank_model(self):
        """The hot-water tanks and their heaters as a one-step model
        (``stems.battery.TankModel``): what a storage action draws this hour."""
        from stems.battery import TankModel

        if self._mock:
            raise RuntimeError("the mock environment has no hot-water tank model")
        return TankModel.from_citylearn(self._env.buildings,
                                        float(self._env.seconds_per_time_step))

    # ------------------------------------------------------------------
    # Simulator corrections (applied loudly, recorded in metadata)
    # ------------------------------------------------------------------

    def _apply_citylearn_patches(self) -> None:
        """Correct upstream CityLearn defects (see the module-level notes).

        Every patched run announces itself and stamps ``citylearn_patches`` into
        the run metadata.
        """
        if self._mock:
            return
        self._patch_dhw_storage()
        self._patch_t0_thermal_double_count()
        self._announce_endogenous_obs()

    def _patch_t0_thermal_double_count(self) -> None:
        """Stop the first step of an episode checking demand against booked load."""
        if not self._patch_t0:
            return
        for b in self._env.buildings:
            _wrap_apply_actions_t0(b)
        self._patches.append("t0_thermal_double_count")
        print("=" * 74)
        print("[STEMS] PATCHED CityLearn defect 't0_thermal_double_count' on "
              f"{self._num_buildings}/{self._num_buildings} buildings.")
        print("        reset() books the ideal thermal load at t=0, then the first step")
        print("        limits devices to nominal power minus that same load (asserts on")
        print("        heating, silently clips storage charging). Cleared before step 1.")
        print("        Disable with STEMSEnvironment(patch_t0_double_count=False).")
        print("=" * 74)

    def _announce_endogenous_obs(self) -> None:
        """Record the state-observation correction (applied in ``step``)."""
        if not self._patch_endogenous:
            return
        self._patches.append("endogenous_obs_from_simulated_hour")
        print("=" * 74)
        print("[STEMS] PATCHED CityLearn defect 'endogenous_obs_from_simulated_hour'.")
        print("        The indoor temperature observation reads the uncontrolled dataset")
        print("        value, never the simulated one. State observations now come from")
        print("        the simulator's series for the hour the action was applied to.")
        print("        Disable with STEMSEnvironment(patch_endogenous_obs=False).")
        print("=" * 74)

    def _patch_dhw_storage(self) -> None:
        """Correct the no-op DHW storage action where the defect is present.

        Only applied to a building that has a DHW tank but no heating storage tank.
        """
        if not self._patch_dhw or self._dhw_action_index < 0:
            return
        import types

        affected = []
        for i, b in enumerate(self._env.buildings):
            dhw_cap = float(getattr(b.dhw_storage, "capacity", 0.0) or 0.0)
            heat_cap = float(getattr(b.heating_storage, "capacity", 0.0) or 0.0)
            if dhw_cap > 0.0 and heat_cap <= 0.0:
                b.update_dhw_storage = types.MethodType(_patched_update_dhw_storage, b)
                affected.append(i)
        if affected:
            self._patches.append("dhw_storage_capacity")
            print("=" * 74)
            print("[STEMS] PATCHED CityLearn defect 'dhw_storage_capacity' on "
                  f"{len(affected)}/{self._num_buildings} buildings.")
            print("        Building.update_dhw_storage scales the action by "
                  "heating_storage.capacity")
            print("        (= 0.0 in this schema), making the DHW action a no-op. "
                  "Now uses dhw_storage.capacity.")
            print("        Disable with STEMSEnvironment(patch_dhw=False).")
            print("=" * 74)

    @property
    def citylearn_patches(self) -> List[str]:
        """Simulator corrections active in this run (stamped into metadata)."""
        return list(self._patches)

    @property
    def env_kwargs(self) -> Dict[str, Any]:
        """CityLearnEnv overrides for this run (building subset, time window)."""
        return dict(self._env_kwargs)

    # ------------------------------------------------------------------
    # DHW tank + heat-pump parameters (calibrate the thermal study)
    # ------------------------------------------------------------------

    def _action_bounds(self) -> np.ndarray:
        """Per-building upper bound on each action, from the live action space."""
        B, A = self._num_buildings, self._action_dim
        highs = np.ones((B, A), dtype=np.float32)
        try:
            for i, space in enumerate(self._env.action_space):
                highs[i] = np.asarray(space.high, dtype=np.float32)
        except Exception:
            pass
        return highs

    def _extract_dhw_info(self) -> Dict[str, np.ndarray]:
        """Per-building DHW tank + heater parameters.

        CityLearn charges the tank by ``energy = action * capacity`` capped at the
        heater's hourly output, so the SOC gain per step is bounded by
        ``min(action_bound, P_nom * eta / C)`` -- the finite time-to-heat that
        motivates anticipatory pre-heating (see ``stems.thermal``).
        """
        B = self._num_buildings
        cap = np.ones(B, dtype=np.float32)
        nom = np.full(B, 1.0, dtype=np.float32)
        eff = np.ones(B, dtype=np.float32)
        loss = np.zeros(B, dtype=np.float32)
        if self._mock:
            cap[:] = 6.0
            nom[:] = 5.0
            eff[:] = 0.95
            loss[:] = 0.005
        else:
            try:
                for i, b in enumerate(self._env.buildings):
                    tank, heater = b.dhw_storage, b.dhw_device
                    cap[i] = float(getattr(tank, "capacity", 1.0)) or 1.0
                    loss[i] = float(getattr(tank, "loss_coefficient", 0.0))
                    nom[i] = float(getattr(heater, "nominal_power", 1.0))
                    eff[i] = float(getattr(heater, "efficiency", 1.0))
            except Exception as exc:
                raise RuntimeError(f"Could not read DHW device parameters: {exc!r}") from exc
        idx = self._dhw_action_index
        bounds = self._action_bounds()
        action_bound = (bounds[:, idx].astype(np.float32) if idx >= 0
                        else np.ones(B, dtype=np.float32))
        return {"capacity": cap, "nominal_power": nom, "efficiency": eff,
                "loss_coefficient": loss, "action_bound": action_bound}

    def _extract_heat_pump_info(self) -> Dict[str, np.ndarray]:
        """Per-building heat-pump efficiencies, supply temperatures and ratings.

        These feed the Carnot CoP model (``stems.thermal.CoPModel``), matching
        ``citylearn.energy_model.HeatPump.get_cop``.
        """
        B = self._num_buildings
        out = {k: np.zeros(B, dtype=np.float32) for k in
               ("efficiency_heat", "target_heat", "nominal_power_heat",
                "efficiency_cool", "target_cool", "nominal_power_cool")}
        if self._mock:
            out["efficiency_heat"][:] = 0.25
            out["target_heat"][:] = 45.0
            out["nominal_power_heat"][:] = 5.0
            out["efficiency_cool"][:] = 0.25
            out["target_cool"][:] = 8.0
            out["nominal_power_cool"][:] = 5.0
            return out
        try:
            for i, b in enumerate(self._env.buildings):
                hd, cd = b.heating_device, b.cooling_device
                out["efficiency_heat"][i] = float(getattr(hd, "efficiency", 0.25))
                out["target_heat"][i] = float(getattr(hd, "target_heating_temperature", 45.0))
                out["nominal_power_heat"][i] = float(getattr(hd, "nominal_power", 5.0))
                out["efficiency_cool"][i] = float(getattr(cd, "efficiency", 0.25))
                out["target_cool"][i] = float(getattr(cd, "target_cooling_temperature", 8.0))
                out["nominal_power_cool"][i] = float(getattr(cd, "nominal_power", 5.0))
        except Exception as exc:
            raise RuntimeError(f"Could not read heat-pump parameters: {exc!r}") from exc
        return out

    def ev_action_indices(self) -> List[int]:
        """Canonical action indices of the EV slots, or [] when there are none."""
        return [i for i, n in enumerate(self._action_names)
                if n.startswith(EV_ACTION_PREFIX)]

    def ev_obs_layout(self) -> List[Dict[str, int]]:
        """Canonical observation indices for each EV slot.

        Indices are positions in the STEMS observation vector, identical for
        every building by construction, so one layout drives all of them. An
        empty list means the schema has no chargers.
        """
        index = {n: i for i, n in enumerate(self._selected_obs_names)}
        layouts = []
        for slot in range(self._ev_slots):
            names = ev_slot_obs_names(slot)
            layout = {f: index[n] for f, n in zip(EV_SLOT_FIELDS, names)}
            layout["hour"] = index.get("hour", 1)
            layout["slot"] = slot
            layouts.append(layout)
        return layouts

    def ev_info(self) -> Dict[str, np.ndarray]:
        """Per-(building, slot) charger limits read from the live ``Charger``s.

        Mirrors ``battery_info`` / ``dhw_info``: the charge rate that calibrates
        the EV deadline barrier comes from the plant, never assumed. Buildings
        without a charger in a slot report zero power, which makes the slot
        inert rather than silently fast.
        """
        B, K = self._num_buildings, self._ev_slots
        power = np.zeros((B, K), dtype=np.float32)
        eff = np.ones((B, K), dtype=np.float32)
        if not self._mock and K > 0:
            for b, bld in enumerate(self._env.buildings):
                chargers = getattr(bld, "electric_vehicle_chargers", None) or []
                by_id = {getattr(c, "charger_id", str(k)): c
                         for k, c in enumerate(chargers)}
                for slot, cid in enumerate(self._charger_ids[b][:K]):
                    ch = by_id.get(cid)
                    if ch is not None:
                        power[b, slot] = float(getattr(ch, "max_charging_power", 0.0))
                        eff[b, slot] = float(getattr(ch, "efficiency", 1.0))
        return {"max_charging_power": power, "efficiency": eff,
                "charger_ids": self._charger_ids}

    def ev_fleet_model(self, slot: int = 0):
        """The chargers' and vehicles' one-step dynamics (``stems.fleet.EVFleetModel``),
        built from the live simulator."""
        from stems.fleet import EVFleetModel

        if self._mock or self._ev_slots == 0:
            raise RuntimeError("this environment has no EV chargers")
        return EVFleetModel.from_citylearn(self._env, slot)

    @property
    def ev_draw_kwh(self) -> np.ndarray:
        """(B,) grid-side energy each building's chargers took in the last step,
        from the simulator's own series."""
        return self._ev_draw_kwh.copy()

    @property
    def ev_departures(self) -> List[Dict[str, float]]:
        """Vehicles that left at the end of the last step.

        Each entry: ``building``, ``soc`` (the state of charge it left with,
        after that hour's charging), ``required_soc`` and ``capacity_kwh``. The
        charge delivered in a vehicle's last connected hour never appears in an
        observation -- the next one already shows an empty bay -- so a departure
        scored from observations ignores that hour; this is the simulator's own
        record of it.
        """
        return [dict(d) for d in self._ev_departures]

    def _record_ev_step(self, t: int) -> None:
        """Read charger draw and departures for hour ``t`` from the simulator."""
        self._ev_draw_kwh = np.zeros(self._num_buildings, dtype=np.float32)
        self._ev_departures = []
        if self._mock or self._ev_slots == 0:
            return
        vehicles = {ev.name: ev for ev in self._env.electric_vehicles}
        for b, bld in enumerate(self._env.buildings):
            for charger in (getattr(bld, "electric_vehicle_chargers", None) or []):
                self._ev_draw_kwh[b] += float(charger.electricity_consumption[t])
                sim = charger.charger_simulation
                state = np.asarray(sim.electric_vehicle_charger_state)
                if t + 1 >= len(state) or state[t] != 1 or state[t + 1] == 1:
                    continue
                ev = vehicles[str(np.asarray(sim.electric_vehicle_id)[t])]
                self._ev_departures.append({
                    "building": b, "soc": float(ev.battery.soc[t]),
                    "required_soc": float(np.asarray(sim.electric_vehicle_required_soc_departure)[t]),
                    "capacity_kwh": float(ev.battery.capacity)})

    def dhw_info(self) -> Dict[str, np.ndarray]:
        """Return per-building DHW tank/heater parameters."""
        return {k: v.copy() for k, v in self._dhw_info.items()}

    def heat_pump_info(self) -> Dict[str, np.ndarray]:
        """Return per-building heat-pump CoP parameters."""
        return {k: v.copy() for k, v in self._heat_pump_info.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_buildings(self) -> int:
        return self._num_buildings

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def using_mock(self) -> bool:
        return self._mock

    @property
    def env_type(self) -> str:
        return "mock" if self._mock else "CityLearn"

    @property
    def action_names(self) -> List[str]:
        return list(self._action_names)

    @property
    def obs_names(self) -> List[str]:
        return list(self._selected_obs_names)

    @property
    def electrical_storage_action_index(self) -> int:
        return self._electrical_storage_action_index

    @property
    def hvac_action_index(self) -> int:
        return self._hvac_action_index

    @property
    def dhw_action_index(self) -> int:
        return self._dhw_action_index

    def resolve_control_indices(self, isolate: str) -> Optional[List[int]]:
        """Map an isolation mode name (``ACTION_GROUPS`` key) to action indices.

        Returns ``None`` for ``"none"`` (full control, the default). Used to
        restrict ``STEMSAgent`` to a device subset (e.g. heat-pump-only
        studies) while keeping the rest of the pipeline (CBF, reward, env)
        untouched -- non-controlled actuators are simply held at no-op.
        """
        if isolate == "none":
            return None
        try:
            names = ACTION_GROUPS[isolate]
        except KeyError:
            raise ValueError(
                f"Unknown isolation mode {isolate!r}; choices: "
                f"{['none'] + list(ACTION_GROUPS)}") from None
        indices: List[int] = []
        for n in names:
            for idx in self._find_action_indices(n):
                if idx not in indices:
                    indices.append(idx)
        return sorted(indices)

    @property
    def heating_setpoint_idx(self) -> Optional[int]:
        """Index of the heating setpoint in the observation vector (heat-pump
        mode only); None when heating observations are not active."""
        name = "indoor_dry_bulb_temperature_heating_set_point"
        return self._selected_obs_names.index(name) if name in self._selected_obs_names else None

    # ------------------------------------------------------------------
    # Perturbations (extreme-weather / comm-disruption experiments)
    # ------------------------------------------------------------------

    def set_comm_disruption(self, dropout_prob: float) -> None:
        self._comm_dropout = float(np.clip(dropout_prob, 0.0, 1.0))

    def set_temp_offset(self, offset: float) -> None:
        self._temp_offset = float(offset)
        if self._mock:
            self._env.set_temp_offset(offset)

    def set_weather_front(self, gradient: float) -> None:
        """Perturb the outdoor-temperature *forecast* by ``gradient`` degC per hour.

        ``set_temp_offset`` shifts the current temperature and all three of its
        forecasts by the same amount, so it changes the temperature *level* but
        leaves the forecast *gradient* untouched. Any mechanism keyed on an
        approaching change -- such as the cold-front term of
        ``CoPModel.cop_drop`` -- is therefore invisible to it by construction.

        This applies an additive ramp ``gradient * lead`` to each prediction,
        where ``lead`` is its lead time in hours (``T_OUT_PRED_LEAD_H``: 6, 12,
        24), leaving the current reading alone, so a
        negative gradient presents the controller with "it is about to get
        colder". Like ``set_temp_offset`` this perturbs the observation stream,
        not CityLearn's internal physics: it probes whether the anticipation
        logic responds to a weather signal, and is not a claim about true energy
        use in a real cold snap.
        """
        self._temp_gradient = float(gradient)

    # ------------------------------------------------------------------
    # Core API (gymnasium-style: reset->(obs,info), step->(o,r,term,trunc,info))
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        result = self._env.reset()
        obs_list, info = result if isinstance(result, tuple) else (result, {})
        obs = self._extract_obs([np.asarray(o, dtype=np.float32) for o in obs_list])
        self._last_obs = obs
        self._hvac_u = np.zeros(self._num_buildings, dtype=np.float32)
        self._executed_actions = np.zeros((self._num_buildings, self._action_dim), dtype=np.float32)
        self._ev_draw_kwh = np.zeros(self._num_buildings, dtype=np.float32)
        self._ev_departures = []
        return obs, info

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0).copy()
        if self._hvac_control == "setpoint" and self._hvac_action_index >= 0:
            j = self._hvac_action_index
            col = lambda idx: np.array([o[idx] for o in self._last_obs], dtype=np.float32)
            self._hvac_u = thermostat_step(
                self._hvac_u, col(self._selected_obs_names.index("indoor_dry_bulb_temperature")),
                col(self.heating_setpoint_idx),
                col(self._selected_obs_names.index("indoor_dry_bulb_temperature_cooling_set_point")),
                actions[:, j])
            actions[:, j] = self._hvac_u
        action_list = self._remap_actions(actions)
        self._record_executed(actions, action_list)
        t_applied = None if self._mock else int(self._env.time_step)
        result = self._env.step(action_list)
        if len(result) == 5:
            obs_list, rewards, terminated, truncated, info = result
        else:
            obs_list, rewards, done, info = result
            terminated, truncated = done, False
        obs_list = self._extract_obs([np.asarray(o, dtype=np.float32) for o in obs_list])
        if t_applied is not None and self._patch_endogenous:
            self._read_simulated_hour(obs_list, t_applied)
        if t_applied is not None:
            self._record_ev_step(t_applied)
        self._last_obs = obs_list
        rewards = [float(r) for r in rewards]
        if self._comm_dropout > 0.0:
            for i in range(self._num_buildings):
                if np.random.random() < self._comm_dropout:
                    obs_list[i] = np.zeros_like(obs_list[i])
        return obs_list, rewards, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_executed(self, actions: np.ndarray, native: List[np.ndarray]) -> None:
        """Keep what was actually sent to the simulator, in the canonical layout.

        This differs from the caller's action in two ways: in set-point mode the
        HVAC column holds the inner loop's power command, and every column has
        been clipped to the device's own action bounds (CityLearn caps a storage
        action at nominal power / capacity, e.g. 0.54-0.87 for the hot-water
        tanks). KPIs and actuator checks about *devices* must read this.
        """
        executed = np.asarray(actions, dtype=np.float32).copy()
        if not self._mock:
            for i in range(self._num_buildings):
                for slot, j in self._action_slot_to_native[i].items():
                    if j < len(native[i]):
                        executed[i, slot] = native[i][j]
        self._executed_actions = executed

    @property
    def executed_actions(self) -> np.ndarray:
        """(B, action_dim) device commands of the last step (see ``_record_executed``)."""
        return self._executed_actions.copy()

    @property
    def hvac_control(self) -> str:
        """'power' (action = power fraction) or 'setpoint' (action = set-point offset)."""
        return self._hvac_control

    def _read_simulated_hour(self, obs_list: List[np.ndarray], t: int) -> None:
        """Overwrite the state observations with the simulator's values for hour ``t``.

        See the note on ``ENDOGENOUS_OBS``. ``t`` is the hour the action was
        applied to; every value written here is the one CityLearn itself stores
        for that hour.
        """
        for b, building in enumerate(self._env.buildings):
            for j, name in enumerate(self._selected_obs_names):
                if name not in ENDOGENOUS_OBS:
                    continue
                if self._obs_indices_per_building[b][j] is None:
                    continue          # the building lacks this device: padded slot
                obs_list[b][j] = float(np.asarray(ENDOGENOUS_OBS[name](building))[t])

    def _extract_obs(self, raw_obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """Select the STEMS feature subset from raw CityLearn observations."""
        if self._mock:
            result = [o.astype(np.float32) for o in raw_obs_list]
        else:
            result = []
            for b, raw in enumerate(raw_obs_list):
                obs = np.zeros(self._obs_dim, dtype=np.float32)
                for j, idx in enumerate(self._obs_indices_per_building[b]):
                    if idx is not None and idx < len(raw):
                        obs[j] = raw[idx]
                result.append(obs)
        if self._temp_offset != 0.0:
            for obs in result:
                for idx in (2, 3, 4, 5):
                    obs[idx] += self._temp_offset
        if getattr(self, "_temp_gradient", 0.0) != 0.0:
            # Forecast-only ramp: index 2 is "now", 3/4/5 lead by 6/12/24 hours.
            for obs in result:
                for idx, lead in zip((3, 4, 5), T_OUT_PRED_LEAD_H):
                    obs[idx] += self._temp_gradient * lead
        return result

    def _remap_actions(self, actions: np.ndarray) -> List[np.ndarray]:
        """Canonical actions -> each building's own native action vector.

        Slots for devices a building does not own are dropped rather than passed
        as zeros: CityLearn sizes the action array to the devices that exist, so
        a padded slot has no place to go.
        """
        if self._mock:
            return [actions[i] for i in range(self._num_buildings)]
        out: List[np.ndarray] = []
        for i in range(self._num_buildings):
            space = self._env.action_space[i]
            low = np.asarray(space.low, dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
            native_dim = int(space.shape[0])
            native = np.zeros(native_dim, dtype=np.float32)
            for slot, j in self._action_slot_to_native[i].items():
                if j < native_dim:
                    native[j] = actions[i][slot]
            for j in range(native_dim):
                lo = float(low[j]) if low.ndim > 0 else float(low)
                hi = float(high[j]) if high.ndim > 0 else float(high)
                native[j] = float(np.clip(native[j], lo, hi))
            out.append(native)
        return out

    # ------------------------------------------------------------------
    # Building metadata for the similarity graph (Eq. 10-11)
    # ------------------------------------------------------------------

    def get_building_info(self) -> Dict[str, Any]:
        """Return {'positions', 'features'} for BuildingGraph construction."""
        B = self._num_buildings
        if not self._mock:
            try:
                positions = np.zeros((B, 2), dtype=np.float32)
                feats = []
                for i, bld in enumerate(self._env.buildings):
                    lat = getattr(bld, "latitude", None) or (30.26 + 0.01 * i)
                    lon = getattr(bld, "longitude", None) or (-97.74 + 0.01 * i)
                    positions[i] = [float(lat), float(lon)]
                    cap = float(getattr(bld.electrical_storage, "capacity", 6.4))
                    area = float(getattr(bld, "floor_area", 150.0))
                    feats.append([cap, area])
                features = np.asarray(feats, dtype=np.float32)
                for c in range(features.shape[1]):
                    span = features[:, c].max() - features[:, c].min()
                    if span > 1e-6:
                        features[:, c] = (features[:, c] - features[:, c].min()) / span
                return {"positions": positions, "features": features}
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read CityLearn building metadata for the graph: {exc!r}"
                ) from exc
        # Mock layout (Travis-inspired): 5 residential, 2 commercial, 1 mixed.
        positions = np.array([[30.260, -97.740], [30.262, -97.738], [30.264, -97.742],
                              [30.258, -97.736], [30.266, -97.744], [30.275, -97.720],
                              [30.278, -97.718], [30.268, -97.730]], dtype=np.float32)[:B]
        features = np.array([[1, 0, 0, 0.3, 0.2], [1, 0, 0, 0.35, 0.25], [1, 0, 0, 0.25, 0.18],
                             [1, 0, 0, 0.4, 0.3], [1, 0, 0, 0.28, 0.22], [0, 1, 0, 0.8, 0.9],
                             [0, 1, 0, 0.75, 0.85], [0, 0, 1, 0.6, 0.5]], dtype=np.float32)[:B]
        return {"positions": positions, "features": features}
