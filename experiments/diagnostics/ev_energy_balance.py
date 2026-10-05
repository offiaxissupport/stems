import os
import sys

import numpy as np

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from experiments import ev_coupling as ec
from stems.fleet import BaseLoadForecaster, FleetShield

season, days, cap = sys.argv[1], 14, float(sys.argv[2])
replay = np.load(REPO / "results" / "ev_coupling" / f"base__{season}{days}d__idle.npy")
for policy, rule in (("asap", "llf"), ("asap", "lp"), ("none", "lp")):
    env = ec._env(season, days)
    model = env.ev_fleet_model()
    layout, e = env.ev_obs_layout()[0], env.ev_action_indices()[0]
    house = ec.HouseController(env, "idle")
    shield = FleetShield(model, layout, e, cap, rule, BaseLoadForecaster(env.num_buildings, replay=replay))
    obs, _ = env.reset()
    done, ev_kwh, deps = False, 0.0, []
    connected_soc_hours = 0.0
    capacity = None
    while not done:
        a = house(obs)
        col = lambda key: np.array([float(o[layout[key]]) for o in obs])
        conn = (col("connected_state") > 0.5) & model.has_ev
        wants = conn & (col("soc") < col("required_soc_departure"))
        if policy == "none":
            wants = np.zeros_like(wants)
        a[:, e] = np.where(wants, 1.0, 0.0)
        a = shield.project(a, obs)
        connected_soc_hours += float((np.where(conn, col("soc"), 0.0) * model.battery.capacity).sum())
        nxt = env.step(a)
        obs_next, done = nxt[0], nxt[2] or nxt[3]
        shield.observe(obs_next, env.ev_draw_kwh)
        ev_kwh += float(env.ev_draw_kwh.sum())
        deps += env.ev_departures
        obs = obs_next
    over = sum(max(d["soc"] - d["required_soc"], 0.0) * d["capacity_kwh"] for d in deps)
    short = sum(max(d["required_soc"] - d["soc"], 0.0) * d["capacity_kwh"] for d in deps)
    held = sum(d["soc"] * d["capacity_kwh"] for d in deps)
    standby = connected_soc_hours * float(model.battery.loss.max())
    print(f"{season} cap {cap:g} {policy:5s}+{rule:3s}: grid {ev_kwh:7.1f} kWh | departures {len(deps)} "
          f"| energy on board at departure {held:7.1f} | above requirement {over:6.1f} | short {short:5.1f} "
          f"| standby loss (model) {standby:6.1f}")
