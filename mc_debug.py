"""
Definitive test: run 10 steps using check_violations.py approach,
simultaneously print MetricsCalculator's internal SOC list and manual SOC,
to find the discrepancy.
"""
import numpy as np
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.config import STEMSConfig
from stems.metrics import MetricsCalculator
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

print(f"Using mock: {env.using_mock}")
print(f"SOC_min={config.cbf.SOC_min}  SOC_max={config.cbf.SOC_max}")
print()

calc = MetricsCalculator(env.num_buildings, config.cbf)
hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
obs_list, _ = env.reset()
hist.update(obs_list)

for step_i in range(10):
    actions = agent.select_action(obs_list, hist.get(), explore=False)
    next_obs, _, term, trunc, _ = env.step(actions)

    # What MetricsCalculator.add_step will store for SOC
    mc_soc = np.array([o[19] for o in next_obs])

    # Manual violation check on the same values
    manual_viol = (mc_soc < config.cbf.SOC_min) | (mc_soc > config.cbf.SOC_max)

    calc.add_step(obs_list, actions, next_obs)

    # Peek into MetricsCalculator's stored SOC list
    stored_soc = calc._soc_list[-1]  # last appended

    print(f"Step {step_i+1:2d}")
    print(f"  next_obs[19]  = {mc_soc.tolist()}")
    print(f"  stored in calc= {stored_soc.tolist()}")
    print(f"  match={np.allclose(mc_soc, stored_soc)}")
    print(f"  manual_viol   = {manual_viol.tolist()}  (any={manual_viol.any()})")
    print()

    obs_list = next_obs
    hist.update(obs_list)
    if term or trunc:
        break

m = calc.compute_all()
print(f"\nMetricsCalculator final: soc_viol={m['soc_violation_rate']:.4f}  safety_viol={m['safety_violation_rate']:.4f}")

# Manually recompute violations from the stored lists
import numpy as np
all_soc = np.stack(calc._soc_list, axis=0)
print(f"Stored SOC array: shape={all_soc.shape}  min={all_soc.min():.4f}  max={all_soc.max():.4f}")
manual_soc_violations = (all_soc < config.cbf.SOC_min) | (all_soc > config.cbf.SOC_max)
print(f"Manual SOC violations from stored data: {manual_soc_violations.sum()} / {all_soc.size} ({100*manual_soc_violations.mean():.1f}%)")
