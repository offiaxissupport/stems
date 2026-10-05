import os
import sys

import numpy as np

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from experiments.controllers import ARMS, build_controller
from experiments.runner import make_config
from experiments.scenario import Scenario
from stems.environment import STEMSEnvironment

EV = "citylearn_schemas/tx_travis_8b_ev/schema.json"
arm_name, season, days, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
pattern = int(sys.argv[5])
nohouse = len(sys.argv) > 6 and sys.argv[6] == "nohouse"
sc = Scenario(schema=EV, season=season, days=days, grid_cap_kw=cap)
NET, HOUR = 20, 1


def rollout(ctrl, phase, record):
    env = STEMSEnvironment(schema=EV, seed=0, heat_pump=True, env_kwargs=sc.env_kwargs(phase),
                           hvac_control="setpoint")
    obs, _ = env.reset()
    done, rows, deps = False, [], []
    while not done:
        hour = int(round(float(obs[0][HOUR])))
        a = ctrl.select_action(obs, None, explore=False)
        nxt, _, term, trunc, _ = env.step(a)
        ctrl.observe(nxt, env.ev_draw_kwh)
        net = np.array([float(o[NET]) for o in nxt])
        last = dict(ctrl.fleet_shield.last)
        rows.append((hour, float(np.maximum(net, 0).sum()), float(env.ev_draw_kwh.sum()),
                     float(np.maximum(net - env.ev_draw_kwh, 0).sum()),
                     last.get("predicted_import_kw"), last.get("margin_kw"),
                     last.get("storage_shed_kw", 0.0), last.get("feasible")))
        deps += env.ev_departures
        obs, done = nxt, bool(term or trunc)
    return rows, deps


env0 = STEMSEnvironment(schema=EV, seed=0, heat_pump=True, env_kwargs=sc.env_kwargs("eval"),
                        hvac_control="setpoint")
ctrl = build_controller(ARMS[arm_name], env0, make_config(sc))
ctrl.fleet_shield.forecaster.daily_pattern_days = pattern
if nohouse:
    ctrl.fleet_shield.house = None
rollout(ctrl, "train", False)
if hasattr(ctrl.base, "reset"):
    ctrl.base.reset()
rows, deps = rollout(ctrl, "eval", True)
imp = np.array([r[1] for r in rows])
house = np.array([r[3] for r in rows])
err = np.array([r[1] - r[4] for r in rows])
over = np.maximum(imp - cap, 0.0)
avoid = over[house <= cap]
missed = sum(d["soc"] + 1e-3 < d["required_soc"] for d in deps)
short = sum(max(d["required_soc"] - d["soc"], 0.0) * d["capacity_kwh"] for d in deps)
print(f"RESULT {arm_name} {season}{days}d cap {cap:g} pattern {pattern} house {not nohouse}: "
      f"over-cap {over.sum():6.1f} kWh in {int((over > 0.1).sum()):3d} h (max {over.max():5.1f} kW); "
      f"avoidable {avoid.sum():6.1f} kWh | house alone over in {int((house > cap).sum())} h | "
      f"missed {missed}/{len(deps)} short {short:5.1f} kWh | error mean {err.mean():5.2f} "
      f"|e| {np.abs(err).mean():4.2f} q95 {np.quantile(err, 0.95):5.2f} max {err.max():5.2f} | "
      f"margin {np.mean([r[5] for r in rows]):4.2f} | shed {sum(r[6] for r in rows):6.1f} kWh | "
      f"infeasible h {sum(r[7] is False for r in rows)}")
worst = sorted(range(len(rows)), key=lambda i: -over[i])[:6]
for i in worst:
    if over[i] > 0.1:
        r = rows[i]
        print(f"   t={i:3d} hour {r[0]:2d} import {r[1]:5.1f} ev {r[2]:5.1f} house {r[3]:5.1f} "
              f"predicted {r[4]:5.1f} margin {r[5]:4.1f}")
