"""How does a trained arm charge the cars? Requested vs executed charging, by hour of
day, from a replay of the evaluation with the saved policy (report, 6.7).

    .venv/Scripts/python experiments/diagnostics/ev_rl_requests.py rl+calibrated winter 0 [results/ev_rl_v1]
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from experiments.controllers import ARMS, build_controller
from experiments.runner import make_config
from experiments.scenario import Scenario
from stems.environment import STEMSEnvironment
from stems.utils import HistoryBuffer, set_seed

EV = "citylearn_schemas/tx_travis_8b_ev/schema.json"
arm_name, season, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
root = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("results/ev_rl_v1")
sc = Scenario(schema=EV, season=season, days=14, grid_cap_kw=40.0)
set_seed(seed)
learner = {"share_parameters": True}


def make_env(phase):
    return STEMSEnvironment(schema=EV, seed=seed, heat_pump=True, env_kwargs=sc.env_kwargs(phase),
                            hvac_control="setpoint")


env = make_env("eval")
config = make_config(sc, learner)
arm = ARMS[arm_name]
ctrl = build_controller(arm, env, config)
if arm.learns:
    ctrl.load(str(root / sc.key / f"{arm_name}__seed{seed}_model"))
fs = ctrl.fleet_shield
e = env.ev_action_indices()[0]
layout = env.ev_obs_layout()[0]


def rollout(environment, record):
    hist = HistoryBuffer(environment.num_buildings, environment.obs_dim, config.transformer.window_size)
    obs, _ = environment.reset()
    hist.update(obs)
    done, rows, deps = False, [], []
    while not done:
        hour = int(round(float(obs[0][1])))
        a = ctrl.select_action(obs, hist.get(), explore=False)
        nominal = getattr(ctrl, "_last_nominal_actions", getattr(ctrl, "_last_raw_actions", None))
        col = lambda key: np.array([float(o[layout[key]]) for o in obs])
        conn = (col("connected_state") > 0.5) & fs.model.has_ev
        asked = np.where(conn, np.clip(nominal[:, e], 0.0, 1.0), 0.0)
        asked_kw = fs.model.draw_kw(np.where(conn, col("soc"), 0.0), asked)
        price = float(obs[0][21])
        nxt, _, term, trunc, _ = environment.step(a)
        ctrl.observe(nxt, environment.ev_draw_kwh)
        rows.append((hour, float(asked_kw.sum()), float(environment.ev_draw_kwh.sum()), int(conn.sum()),
                     fs.last.get("feasible"), price, float(fs.last.get("storage_shed_kw", 0.0))))
        deps += environment.ev_departures
        hist.update(nxt)
        obs, done = nxt, bool(term or trunc)
    return rows, deps


rollout(make_env("train"), False)
for inner in (getattr(ctrl, "base", None), getattr(ctrl, "base_policy", None)):
    if inner is not None and hasattr(inner, "reset"):
        inner.reset()
rows, deps = rollout(env, True)
r = np.array([[x[0], x[1], x[2], x[3], 0.0 if x[4] is None else float(not x[4]), x[5], x[6]] for x in rows])
missed = [d for d in deps if d["soc"] + 1e-3 < d["required_soc"]]
short = sum((d["required_soc"] - d["soc"]) * d["capacity_kwh"] for d in missed)
peak = (r[:, 0] >= 17) & (r[:, 0] <= 21)
print(f"RESULT {arm_name} {season} seed {seed}: executed EV {r[:, 2].sum():.0f} kWh, requested {r[:, 1].sum():.0f} kWh "
      f"(requested/executed {r[:, 1].sum() / max(r[:, 2].sum(), 1e-9):.2f}); "
      f"EV energy in the tariff peak: requested {r[peak, 1].sum():.0f}, executed {r[peak, 2].sum():.0f} kWh; "
      f"mean price paid per EV kWh {(r[:, 2] * r[:, 5]).sum() / max(r[:, 2].sum(), 1e-9):.3f}; "
      f"missed {len(missed)}/{len(deps)} short {short:.1f} kWh; infeasible hours {int(r[:, 4].sum())}; "
      f"storage shed {r[:, 6].sum():.1f} kWh")
by_hour = {h: (r[r[:, 0] == h, 1].mean(), r[r[:, 0] == h, 2].mean(), r[r[:, 0] == h, 3].mean()) for h in range(1, 25)}
print("   hour: requested / executed kW (cars connected)")
print("   " + "  ".join(f"{h}:{v[0]:.0f}/{v[1]:.0f}({v[2]:.1f})" for h, v in by_hour.items()))
print("   missed:", [(d["building"], round(d["soc"], 3), round(d["required_soc"], 3)) for d in missed])
