"""Quick multi-seed violation check for the STEMS checkpoint."""
import numpy as np
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.config import STEMSConfig
from stems.metrics import MetricsCalculator
from stems.utils import HistoryBuffer, set_seed

config = STEMSConfig()
costs, viols, soc_viols = [], [], []

for seed in [0, 1, 42]:
    set_seed(seed)
    env = STEMSEnvironment(seed=seed)
    info = env.get_building_info()
    graph = BuildingGraph(env.num_buildings, info["positions"], info["features"], config.graph)
    agent = STEMSAgent(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        num_buildings=env.num_buildings, building_graph=graph,
        config=config, use_cbf=True,
    )
    agent.load("checkpoints/seed1/best/")

    calc = MetricsCalculator(env.num_buildings, config.cbf)
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs_list, _ = env.reset()
    hist.update(obs_list)
    done = False
    while not done:
        actions = agent.select_action(obs_list, hist.get(), explore=False)
        next_obs, _, term, trunc, _ = env.step(actions)
        done = term or trunc
        calc.add_step(obs_list, actions, next_obs)
        obs_list = next_obs
        hist.update(obs_list)

    m = calc.compute_all()
    costs.append(m["cost"])
    viols.append(m["safety_violation_rate"])
    soc_viols.append(m["soc_violation_rate"])
    print(f"seed={seed}  cost={m['cost']:.2f}  viol={m['safety_violation_rate']:.4f}  soc_viol={m['soc_violation_rate']:.4f}  power_viol={m['power_violation_rate']:.4f}")

print(f"\nMean cost: {np.mean(costs):.2f} +/- {np.std(costs):.2f}")
print(f"Mean viol: {np.mean(viols):.4f} +/- {np.std(viols):.4f}")
print(f"Mean soc_viol: {np.mean(soc_viols):.4f} +/- {np.std(soc_viols):.4f}")
