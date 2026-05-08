"""5-step trace to inspect SOC deltas, CBF firing, net values."""
import numpy as np
import torch
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.config import STEMSConfig
from stems.utils import HistoryBuffer, set_seed

config = STEMSConfig()
set_seed(1)
env = STEMSEnvironment(seed=1)
info = env.get_building_info()
graph = BuildingGraph(env.num_buildings, info["positions"], info["features"], config.graph)
agent = STEMSAgent(obs_dim=env.obs_dim, action_dim=env.action_dim,
                   num_buildings=env.num_buildings, building_graph=graph,
                   config=config, use_cbf=True)
agent.load("checkpoints/seed1/best/")
hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
obs_list, _ = env.reset()
hist.update(obs_list)

print(f"Using mock: {env.using_mock}")
print(f"CBF SOC bounds: [{config.cbf.SOC_min}, {config.cbf.SOC_max}]")
print(f"CBF SOC_DELTA_RATE (assumed): {agent.cbf.SOC_DELTA_RATE}")
print(f"P_building_max: {config.cbf.P_building_max} kW")
print(f"P_grid_max: {config.cbf.P_grid_max} kW")
print()

for step_i in range(20):
    pre_soc = [obs[19] for obs in obs_list]
    pre_net = [obs[20] for obs in obs_list]

    with torch.no_grad():
        x = torch.tensor(np.stack(obs_list, 0), dtype=torch.float32)
        h = torch.tensor(hist.get(), dtype=torch.float32)
        agent.encoder.eval()
        repr_mat = agent.encoder(x, agent.adj, h)
        raw_actions = np.stack([
            agent.actors[i](repr_mat[i].unsqueeze(0)).squeeze(0).cpu().numpy()
            for i in range(agent.B)
        ], 0)

    nominal = raw_actions * config.training.action_scale
    cbf_safe = agent.cbf.project(nominal, obs_list)
    cbf_fired = np.abs(cbf_safe - nominal).max() > 1e-6

    next_obs, _, term, trunc, _ = env.step(cbf_safe)
    post_soc = [o[19] for o in next_obs]
    post_net = [o[20] for o in next_obs]

    delta_soc = [post_soc[i] - pre_soc[i] for i in range(3)]
    cbf_model_pred = [cbf_safe[i, 1] * agent.cbf.SOC_DELTA_RATE for i in range(3)]

    print(f"step={step_i+1:3d}  CBF_fired={str(cbf_fired):5s}")
    print(f"  pre_soc  = {[round(s,4) for s in pre_soc]}")
    print(f"  post_soc = {[round(s,4) for s in post_soc]}")
    print(f"  actual_delta = {[round(d,4) for d in delta_soc]}")
    print(f"  cbf_model_pred(action*0.1) = {[round(p,4) for p in cbf_model_pred]}")
    print(f"  raw_a[elec]  = {[round(float(raw_actions[i,1]),4) for i in range(3)]}")
    print(f"  cbf_a[elec]  = {[round(float(cbf_safe[i,1]),4) for i in range(3)]}")
    print(f"  net          = {[round(n,4) for n in post_net]}")
    print()

    obs_list = next_obs
    hist.update(obs_list)
    if term or trunc:
        break
