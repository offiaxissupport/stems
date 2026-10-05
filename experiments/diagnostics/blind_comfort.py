import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from experiments.controllers import IdlePolicy, PlainController
from experiments.runner import _window_len, evaluate, make_config
from experiments.scenario import Scenario
from stems.environment import STEMSEnvironment

season = sys.argv[1] if len(sys.argv) > 1 else "winter"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 28
rows = []
for label, control, patched in (("heat pump off", "power", False),
                                ("heat pump off", "power", True),
                                ("thermostat at the set point", "setpoint", True)):
    sc = Scenario(season=season, days=days, hvac_control=control)
    kw = sc.env_kwargs("eval")
    env = STEMSEnvironment(schema=sc.schema_path(), seed=0, heat_pump=True, env_kwargs=kw,
                           hvac_control=control, patch_endogenous_obs=patched)
    ctrl = PlainController(IdlePolicy(env.num_buildings, env.action_dim))
    k = evaluate(ctrl, env, make_config(sc), _window_len(kw))["kpis"]
    b = env._env.buildings
    es = [x.energy_simulation for x in b]
    n = _window_len(kw) - 1
    simulated = np.array([np.asarray(x.indoor_dry_bulb_temperature)[:n] for x in es])
    dataset = np.array([np.asarray(x.indoor_dry_bulb_temperature_without_control)[:n] for x in es])
    rows.append((label, "patched" if patched else "as shipped", k["cost"], k["electricity_consumption"],
                 k["discomfort_rate"], float(simulated.mean()), float(simulated.min()),
                 float(dataset.mean())))

print(f"{season}, {days}-day evaluation window, eight houses, battery and tank idle\n")
print("| Heat pump | Simulator | Cost | Consumption [kWh] | Discomfort the KPI reports | "
      "Simulated indoor mean / min [degC] | Temperature in the dataset, mean [degC] |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.0f} | {100 * r[4]:.1f}% | {r[5]:.1f} / {r[6]:.1f} | {r[7]:.1f} |")
