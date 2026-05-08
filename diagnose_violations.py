"""
Deep diagnostic: probe real CityLearn physics to expose why violations are always 0.

Checks:
  1. Actual SOC delta per unit action (vs CBF's assumed 0.1)
  2. Actual net electricity ranges (vs P_building_max=80, P_grid_max=300)
  3. Whether CBF ever fires (modifies the nominal action)
  4. How often nominal action is already feasible vs needs projection
  5. Distribution of SOC values seen during the episode
"""
import numpy as np
import torch
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.config import STEMSConfig
from stems.metrics import MetricsCalculator
from stems.utils import HistoryBuffer, set_seed
from stems.cbf import CBFShield

config = STEMSConfig()
set_seed(1)
env = STEMSEnvironment(seed=1)
info = env.get_building_info()
graph = BuildingGraph(env.num_buildings, info["positions"], info["features"], config.graph)
agent = STEMSAgent(
    obs_dim=env.obs_dim, action_dim=env.action_dim,
    num_buildings=env.num_buildings, building_graph=graph,
    config=config, use_cbf=True,
)
agent.load("checkpoints/seed1/best/")

print(f"Using mock: {env.using_mock}")
print(f"Num buildings: {env.num_buildings}")
print()

hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
obs_list, _ = env.reset()
hist.update(obs_list)

# Accumulators
cbf_fired = 0          # steps where CBF changed the action
cbf_skipped = 0        # steps where nominal was already feasible
soc_vals = []          # post-step SOC
net_vals = []          # post-step net
soc_deltas = []        # actual SOC change per step
pre_soc_vals = []      # pre-step SOC
raw_action_vals = []   # nominal action[1] (elec storage)
cbf_action_vals = []   # post-CBF action[1]
action_delta_vals = [] # how much CBF changed action[1]

done = False
step = 0
while not done:
    pre_soc = np.array([obs[19] for obs in obs_list])
    pre_soc_vals.append(pre_soc.copy())

    # Get raw nominal action (before CBF)
    with torch.no_grad():
        x = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32)
        h = torch.tensor(hist.get(), dtype=torch.float32)
        repr_mat = agent.encoder(x, agent.adj, h)
        raw_actions = np.stack([
            agent.actors[i](repr_mat[i].unsqueeze(0)).squeeze(0).cpu().numpy()
            for i in range(agent.B)
        ], axis=0)

    raw_action_vals.append(raw_actions[:, 1].copy())  # elec storage action

    # Apply CBF projection
    nominal_scaled = raw_actions * config.training.action_scale
    cbf_safe = agent.cbf.project(nominal_scaled, obs_list)
    cbf_action_vals.append(cbf_safe[:, 1].copy())

    # Check if CBF fired
    action_diff = np.abs(cbf_safe - nominal_scaled).max()
    action_delta_vals.append(np.abs(cbf_safe[:, 1] - nominal_scaled[:, 1]).copy())
    if action_diff > 1e-6:
        cbf_fired += 1
    else:
        cbf_skipped += 1

    next_obs, _, term, trunc, _ = env.step(cbf_safe)
    done = term or trunc

    post_soc = np.array([o[19] for o in next_obs])
    soc_vals.append(post_soc.copy())
    net_vals.append(np.array([o[20] for o in next_obs]).copy())
    soc_deltas.append(post_soc - pre_soc)

    obs_list = next_obs
    hist.update(obs_list)
    step += 1

soc_arr = np.array(soc_vals)      # (T, B)
net_arr = np.array(net_vals)      # (T, B)
delta_arr = np.array(soc_deltas)  # (T, B)
pre_soc_arr = np.array(pre_soc_vals)
raw_act_arr = np.array(raw_action_vals)   # (T, B)
cbf_act_arr = np.array(cbf_action_vals)  # (T, B)
delta_act_arr = np.array(action_delta_vals)  # (T, B)

print("=" * 60)
print("CBF FIRING STATS")
print("=" * 60)
print(f"Total steps: {step}")
print(f"CBF fired (changed action): {cbf_fired} ({100*cbf_fired/step:.1f}%)")
print(f"CBF skipped (nominal feasible): {cbf_skipped} ({100*cbf_skipped/step:.1f}%)")
print()

print("=" * 60)
print("SOC ANALYSIS")
print("=" * 60)
print(f"SOC range:  min={soc_arr.min():.4f}  max={soc_arr.max():.4f}")
print(f"SOC bounds: [{config.cbf.SOC_min}, {config.cbf.SOC_max}]")
print(f"SOC violations (post-step): {((soc_arr < config.cbf.SOC_min) | (soc_arr > config.cbf.SOC_max)).sum()}")
print()
print(f"SOC delta stats (actual change per step):")
print(f"  mean={delta_arr.mean():.4f}  std={delta_arr.std():.4f}")
print(f"  min={delta_arr.min():.4f}   max={delta_arr.max():.4f}")
print()
print(f"Nominal elec action (raw policy):")
print(f"  mean={raw_act_arr.mean():.4f}  std={raw_act_arr.std():.4f}")
print(f"  min={raw_act_arr.min():.4f}   max={raw_act_arr.max():.4f}")
print()
print(f"CBF elec action (post-projection):")
print(f"  mean={cbf_act_arr.mean():.4f}  std={cbf_act_arr.std():.4f}")
print(f"  min={cbf_act_arr.min():.4f}   max={cbf_act_arr.max():.4f}")
print()
print(f"Action change by CBF (delta):")
print(f"  mean={delta_act_arr.mean():.4f}  std={delta_act_arr.std():.4f}")
print(f"  max={delta_act_arr.max():.6f}")
print()

# CBF model: predicted SOC delta = action * SOC_DELTA_RATE = action * 0.1
cbf_model_delta = cbf_act_arr * 0.1  # what CBF thinks the change will be
print(f"CBF MODEL (0.1 * action) vs ACTUAL SOC delta:")
print(f"  CBF model range: [{cbf_model_delta.min():.4f}, {cbf_model_delta.max():.4f}]")
print(f"  Actual delta range: [{delta_arr.min():.4f}, {delta_arr.max():.4f}]")
ratio = delta_arr / (cbf_model_delta + 1e-10)
print(f"  Ratio actual/model: mean={ratio.mean():.3f}, std={ratio.std():.3f}")
print()

print("=" * 60)
print("NET ELECTRICITY ANALYSIS")
print("=" * 60)
print(f"Net range (kWh per building per hour):")
print(f"  min={net_arr.min():.2f}  max={net_arr.max():.2f}  mean={net_arr.mean():.2f}")
print(f"P_building_max={config.cbf.P_building_max} kW")
print(f"Power violations (|net| > P_building_max): {(np.abs(net_arr) > config.cbf.P_building_max).sum()}")
print()
grid_net = net_arr.sum(axis=1)
print(f"Grid total net (sum across buildings):")
print(f"  min={grid_net.min():.2f}  max={grid_net.max():.2f}  mean={grid_net.mean():.2f}")
print(f"P_grid_max={config.cbf.P_grid_max} kW")
print(f"Grid violations (Σnet > P_grid_max): {(grid_net > config.cbf.P_grid_max).sum()}")
