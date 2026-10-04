"""Which hour do the load, solar and hot-water demand observations describe?

    .venv/Scripts/python experiments/diagnostics/observation_timing.py
"""
import os
import sys

import numpy as np

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from experiments.scenario import Scenario
from stems.environment import STEMSEnvironment

EV = "citylearn_schemas/tx_travis_8b_ev/schema.json"
sc = Scenario(schema=EV, season="winter", days=3)
env = STEMSEnvironment(schema=EV, seed=0, heat_pump=True, env_kwargs=sc.env_kwargs("eval"),
                       hvac_control="setpoint")
obs, _ = env.reset()
bs = env._env.buildings
done = False
acc = {k: {"acted": [], "previous": []} for k in ("non_shiftable_load", "solar_generation", "dhw_demand")}
IDX = {"non_shiftable_load": 16, "solar_generation": 17, "dhw_demand": 25}
while not done:
    ts = env._env.time_step
    for name, i in IDX.items():
        o = np.array([float(x[i]) for x in obs])
        series = lambda t: np.array([abs(getattr(b, name)[t]) for b in bs])
        acc[name]["acted"].append(np.abs(o - series(ts)).max())
        if ts > 0:
            acc[name]["previous"].append(np.abs(o - series(ts - 1)).max())
    nxt, _, term, trunc, _ = env.step(np.zeros((env.num_buildings, env.action_dim), dtype=np.float32))
    obs, done = nxt, bool(term or trunc)
for name, d in acc.items():
    print(f"{name:20s} max |obs - series[hour acted on]| = {max(d['acted']):8.4f}   "
          f"max |obs - series[previous hour]| = {max(d['previous'][1:]):8.4f}")
