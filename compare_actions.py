"""
Print the actual actions sent to env.step() in mc_debug to compare with trace5.py.
"""
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
agent = STEMSAgent(
    obs_dim=env.obs_dim, action_dim=env.action_dim,
    num_buildings=env.num_buildings, building_graph=graph,
    config=config, use_cbf=True,
)
agent.load("checkpoints/seed1/best/")

hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
obs_list, _ = env.reset()
hist.update(obs_list)

print(f"Using mock: {env.using_mock}")
print(f"action_scale={config.training.action_scale}")
print()

for step_i in range(5):
    # --- PATH A: select_action ---
    actions_A = agent.select_action(obs_list, hist.get(), explore=False)

    # --- PATH B: manual cbf.project (same as trace5.py) ---
    with torch.no_grad():
        x = torch.tensor(np.stack(obs_list, 0), dtype=torch.float32)
        h = torch.tensor(hist.get(), dtype=torch.float32)
        agent.encoder.eval()
        repr_mat = agent.encoder(x, agent.adj, h)
        raw_B = np.stack([
            agent.actors[i](repr_mat[i].unsqueeze(0)).squeeze(0).cpu().numpy()
            for i in range(agent.B)
        ], 0)
    nominal_B = raw_B * config.training.action_scale
    actions_B = agent.cbf.project(nominal_B, obs_list)

    pre_soc = [o[19] for o in obs_list]

    print(f"Step {step_i+1}")
    print(f"  pre_soc     = {[round(s, 4) for s in pre_soc]}")
    print(f"  actions_A[elec] (select_action) = {actions_A[:, 1].tolist()}")
    print(f"  actions_B[elec] (manual cbf)    = {actions_B[:, 1].tolist()}")
    print(f"  match = {np.allclose(actions_A, actions_B, atol=1e-5)}")
    print(f"  diff max = {np.abs(actions_A - actions_B).max():.6f}")

    # Step with both and compare SOC
    next_A, _, _, _, _ = env.step(actions_A.copy())
    # NOTE: can't step twice — just show what actions_B would produce theoretically
    post_soc_A = [o[19] for o in next_A]

    print(f"  post_soc_A (select_action)   = {[round(s, 4) for s in post_soc_A]}")
    print(f"  actions_A raw  = {actions_A[:, 1].tolist()}")
    print(f"  raw_B (direct from actor) = {raw_B[:, 1].tolist()}")
    print()

    obs_list = next_A
    hist.update(obs_list)
