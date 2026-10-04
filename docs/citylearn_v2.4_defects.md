# Three defects in CityLearn ≥ 2.4.0 from the `step()` reordering

*Draft upstream issue. Not filed. Reproduced on CityLearn 2.6.0b1 with the shipped dataset
`citylearn_challenge_2023_phase_2_local_evaluation`; v2.4.0–v2.5.0 checked by reading the tagged source.*

## Summary

Since v2.4.0, `CityLearnEnv.step` applies the actions **before** advancing the time step. Up to v2.3.1 it
advanced first. Code written for the old order was not updated, and three things broke:

| # | Defect | Effect |
|---|---|---|
| 1 | The `indoor_dry_bulb_temperature` observation is the dataset's uncontrolled value, never the simulated one | An agent cannot observe the effect of its own HVAC actions; any reward or logic built on the observation is blind |
| 2 | The `dhw_storage` action is scaled by `heating_storage.capacity` | With no heating storage tank (capacity 0) the action does nothing |
| 3 | Each thermal device's ideal load is booked at `t = 0` during `reset()`, then the first `step()` limits the device to `nominal_power` minus that load | Heating/cooling assert "demand is greater than … max output" when the first-step need exceeds half the nameplate; storage charging is silently clipped |

| Version | `step()` order | 1 | 2 | 3 |
|---|---|---|---|---|
| v2.1b9 – v2.3.1 | `next_time_step` → `apply_actions` → `update_variables` | ok | ok | ok (first action lands at `t = 1`) |
| v2.4.0 – v2.4.2 | `apply_actions` → `next_time_step` → `update_variables` | broken | broken | broken |
| v2.5.0, v2.6.0b1 | `apply_actions` → `update_variables` → `next_time_step` | broken | broken | broken |

## Reproduction

```bash
python -m experiments.citylearn_defects
```

The script uses only `citylearn.citylearn.CityLearnEnv` on the shipped dataset. Output on 2.6.0b1:

```
[1] indoor temperature observation vs simulated temperature
    cooling action 0: 144 controlled building-steps | observed == simulated[t] on 0.00 | observed == uncontrolled dataset[t+1] on 1.00 | simulated 23.1..38.9 C, observed 20.0..26.5 C
    cooling action 1: 144 controlled building-steps | observed == simulated[t] on 0.00 | observed == uncontrolled dataset[t+1] on 1.00 | simulated 8.4..25.5 C, observed 20.0..26.5 C
[2] dhw_storage action
    Building_1: dhw_storage.capacity=2.28 kWh, heating_storage.capacity=0.00 kWh, soc after 8 steps at action 0.8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Building_2: dhw_storage.capacity=1.66 kWh, heating_storage.capacity=0.00 kWh, soc after 8 steps at action 0.8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Building_3: dhw_storage.capacity=2.84 kWh, heating_storage.capacity=0.00 kWh, soc after 8 steps at action 0.8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[3] thermal-device load booked before the first step
    Building_1 cooling: nominal 4.11 kW, already booked at t=0 before any action 0.26 kW -> first step is limited to 3.85 kW
    Building_1 dhw: nominal 4.86 kW, already booked at t=0 before any action 0.06 kW -> first step is limited to 4.80 kW
    Building_3 cooling: nominal 2.78 kW, already booked at t=0 before any action 0.14 kW -> first step is limited to 2.64 kW
```

## Causes (2.6.0b1 line numbers)

1. **Temperature.** `LSTMDynamicsBuilding.apply_actions` (`building.py:2758`) calls
   `update_indoor_dry_bulb_temperature()`, which writes `energy_simulation.indoor_dry_bulb_temperature[self.time_step]`
   (`building.py:2863`). Its comment — "this function is called after advancing to next timestep" — describes the
   pre-2.4 order. The step then advances, and the observation reads `series[t]` at the *new* `t`
   (`internal/building_ops.py`, `_energy_simulation_observation_sources`), which still holds the dataset value.
   CityLearn's own KPIs read the simulated series and are unaffected; only what the agent sees is wrong.
2. **Hot water.** `Building.update_dhw_storage` (`building.py:1562`):
   `energy = action * self.heating_storage.capacity * …` should use `self.dhw_storage.capacity`.
3. **First step.** `Building.update_variables` has a `time_step == 0` branch (`building.py:2432`) that *sets* the
   cooling, heating and DHW device consumption from the ideal load; `reset()` runs it. The first `step()` then
   calls `update_energy_from_*_device`, whose limit is `get_max_output_power(…)` over
   `available_nominal_power = nominal_power − electricity_consumption[t]`. The note at `building.py:2464`
   shows the same double count was already fixed for the battery. On a Travis County building in winter:
   need 3.84 kW of a 4.96 kW heat pump at `t = 0` → booked headroom 1.12 kW →
   `AssertionError: demand is greater than heating_device max output`, whatever the action.

## Suggested fixes

1. Read the endogenous temperature observation at the step the action was applied to (as is already done for
   the storage SOCs via `endogenous_t`), or restore the pre-2.4 order.
2. `self.dhw_storage.capacity` in `update_dhw_storage`.
3. Do not book thermal-device consumption at `t = 0` in `reset()` (or clear it before the first
   `apply_actions`), as was done for the battery.

A separate, smaller observation: on an episode's final transition the buildings are not advanced
(`internal/runtime.py:204`), so the returned state observations repeat the previous step.

## Who is affected

Any work on CityLearn ≥ 2.4.0 that uses LSTM dynamics buildings and feeds the indoor temperature observation to a
controller or a reward, or that controls `dhw_storage` in a building without a heating storage tank. The 2023
Challenge itself ran on v2.1 and is not affected.
