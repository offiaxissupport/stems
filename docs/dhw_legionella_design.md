# Hot water with a heat pump, and the weekly Legionella cycle — design note

*Status: design, not implemented. Written 2026-10-04 for discussion with Manal.*

## Why the current hot-water model cannot carry a result

In CityLearn the hot-water device serves every draw directly, in the hour it occurs, and is sized so that it
always can (`Building.update_energy_from_dhw_device`; an electric heater of 4–7.4 kW on the Travis houses). The
tank is optional storage on top. Consequences, both verified:

- An empty tank costs nothing: no draw is ever unmet. The "readiness" KPI used so far (does the tank hold the
  forecast demand plus a margin?) therefore has **no physical consequence** in the simulator.
- That KPI was scored with the same rule the readiness barrier enforces, so a controller with the barrier wins it
  by construction.

A real heat-pump water heater is the opposite: its heat source is small (1–3 kW thermal) and cannot meet a
shower in real time, which is the whole reason the tank exists. The constraint only becomes real when the heat
source is power-limited.

## Proposed model

Keep CityLearn's energy accounting, and add a **shadow tank with a power-limited heat source** in
`STEMSEnvironment` (the same pattern as the set-point thermostat):

- State: stored energy `E` in `[0, C]`, per building. `C` is the CityLearn tank capacity.
- Heat source: at most `P_hp` kWh thermal per hour (a parameter; default from a realistic heat-pump size, e.g.
  the capacity that covers the mean daily demand in ~4 h — to be agreed with Manal), with CoP from the existing
  `CoPModel` when the source is a heat pump, or efficiency when it is an element.
- Each hour: `E ← E·(1 − loss) + heat_in − draw`, with `heat_in ≤ P_hp`.
- **Unmet hot water** `= max(0, draw − (E_before + heat_in))`: the service failure. This is the new headline
  hot-water KPI (kWh and share of draws), and the quantity a barrier must keep at zero.

Why this is equivalent to CityLearn when the limit does not bind: with a draw `d` and a tank charge `c` in the
same hour, CityLearn's heater supplies `d + c` and the tank gains `c`; a tank-served system supplies `d` from the
tank and puts `d + c` in — same electricity, same tank change. The two differ only when `d + c > P_hp`, which is
exactly the case the model is meant to capture.

Limitation to state in any result: CityLearn itself still serves the unmet draw with its oversized heater and
bills the electricity, so when the constraint is violated the energy total is slightly high. With a barrier that
keeps unmet demand at zero the two coincide.

## Legionella cycle

Manal's rule: once a week, on a schedule, heat the water to 60 °C (older units need the electric backup heater;
newer propane units reach 70 °C alone; in new units the interval is a user setting, 7–10 days).

In the energy model this is a **weekly deadline**: the tank must reach the state of charge corresponding to
60 °C (`soc_legionella`, e.g. 1.0 if the tank's rated capacity is defined at 60 °C and normal operation holds
~50 °C ≈ `(50 − T_cold)/(60 − T_cold)` of it) at least once in every 7-day window.

That is the same object as an EV charging deadline — a store that must hit a required level by a time, at a
finite rate — so it uses `stems/deadline.py::DeadlineStorageBarrier` with:

- `required_soc = soc_legionella`,
- `steps_to_deadline =` hours until the end of the current 7-day window (reset when the level is reached),
- `rate =` the power-limited charge rate above (conservative, as for the EV barrier).

The barrier does nothing while there is slack and forces charging from the latest feasible start. The slack is
the interesting part: *when* in the week to run the cycle is free, so a price-aware policy can place it in the
cheapest hours (and outside the 16:00–21:00 peak), while the barrier guarantees it happens.

Two variants worth comparing, matching what Manal described:

| Variant | Source for the 60 °C lift | Electricity for the cycle |
|---|---|---|
| Older unit | heat pump to its limit (~50–55 °C), electric element above | element at efficiency ≈ 1 for the top-up |
| Propane unit | heat pump alone | CoP at a high sink temperature (lower than normal operation) |

## What this would let us claim

- A hot-water constraint with a physical consequence (unmet draws), not a proxy.
- The Legionella cycle as a hard weekly constraint that a barrier guarantees and a policy schedules.
- Interaction with the shared power cap: the cycle is a large, deferrable load, i.e. the same coupled-deadline
  structure as EV charging (phase 3b).

## Open questions for Manal

1. Realistic `P_hp` and tank temperatures for the units in her benchmark (normal set point, Legionella set
   point, cold-water inlet).
2. CoP at 60 °C sink for the propane unit; element power for the older one.
3. Is "once per 7 days, on a schedule" a fixed weekday/hour in practice, or any time within the window?
4. Does her benchmark need the heat pump shared between space heating and hot water (priority switching)?
