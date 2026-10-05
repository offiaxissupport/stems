#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np

DATASET = "citylearn_challenge_2023_phase_2_local_evaluation"


def make_env(dataset: str):
    from citylearn.citylearn import CityLearnEnv

    return CityLearnEnv(dataset, central_agent=False)


def action_vectors(env, **values: float):
    out = []
    for b in env.buildings:
        out.append([float(values.get(name, 0.0)) for name in b.active_actions])
    return out


def defect_temperature(dataset: str, steps: int = 60) -> None:
    print("\n[1] indoor temperature observation vs simulated temperature")
    for level in (0.0, 1.0):
        env = make_env(dataset)
        env.reset()
        rows = []
        for _ in range(steps):
            t = env.time_step
            obs = env.step(action_vectors(env, cooling_device=level,
                                          cooling_or_heating_device=-level))[0]
            for b, o in zip(env.buildings, obs):
                if not getattr(b, "simulate_dynamics", False):
                    continue
                names = b.active_observations
                if "indoor_dry_bulb_temperature" not in names:
                    continue
                es = b.energy_simulation
                rows.append((o[names.index("indoor_dry_bulb_temperature")],
                             es.indoor_dry_bulb_temperature[t],
                             es.indoor_dry_bulb_temperature_without_control[t + 1]))
        r = np.array(rows, dtype=float)
        if not len(r):
            print("    no controlled building-steps (dataset without dynamics?)")
            return
        print(f"    cooling action {level:.0f}: {len(r)} controlled building-steps | "
              f"observed == simulated[t] on {np.mean(np.isclose(r[:, 0], r[:, 1], atol=1e-3)):.2f} | "
              f"observed == uncontrolled dataset[t+1] on "
              f"{np.mean(np.isclose(r[:, 0], r[:, 2], atol=1e-3)):.2f} | "
              f"simulated {r[:, 1].min():.1f}..{r[:, 1].max():.1f} C, "
              f"observed {r[:, 0].min():.1f}..{r[:, 0].max():.1f} C")


def defect_dhw(dataset: str, steps: int = 8) -> None:
    print("\n[2] dhw_storage action")
    env = make_env(dataset)
    env.reset()
    for _ in range(steps):
        env.step(action_vectors(env, dhw_storage=0.8))
    for b in env.buildings:
        if "dhw_storage" not in b.active_actions:
            continue
        print(f"    {b.name}: dhw_storage.capacity={b.dhw_storage.capacity:.2f} kWh, "
              f"heating_storage.capacity={b.heating_storage.capacity:.2f} kWh, "
              f"soc after {steps} steps at action 0.8 = "
              f"{np.round(np.asarray(b.dhw_storage.soc)[:steps], 3).tolist()}")


def defect_first_step(dataset: str) -> None:
    print("\n[3] thermal-device load booked before the first step")
    env = make_env(dataset)
    env.reset()
    for b in env.buildings:
        for label, device in (("cooling", b.cooling_device), ("heating", b.heating_device),
                              ("dhw", b.dhw_device)):
            booked = float(device.electricity_consumption[0])
            nominal = device.nominal_power
            if nominal and booked > 0:
                print(f"    {b.name} {label}: nominal {nominal:.2f} kW, already booked at t=0 "
                      f"before any action {booked:.2f} kW -> first step is limited to "
                      f"{device.available_nominal_power:.2f} kW "
                      f"({'ASSERTS' if booked > nominal / 2 else 'clips storage charging'} "
                      "if the step needs more)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce three CityLearn defects on plain CityLearn")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()
    import citylearn

    print(f"CityLearn {getattr(citylearn, '__version__', '?')} | dataset {args.dataset}")
    defect_temperature(args.dataset)
    defect_dhw(args.dataset)
    defect_first_step(args.dataset)


if __name__ == "__main__":
    main()
