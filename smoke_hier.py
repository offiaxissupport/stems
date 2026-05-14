import numpy as np
import torch
from stems.hierarchical import LargeGridEnv, HierarchicalSTEMSAgent
from stems.config import CBFConfig, LagrangianConfig
from stems.utils import ReplayBuffer

env = LargeGridEnv(num_buildings=10, seed=0)
agent = HierarchicalSTEMSAgent(
    num_buildings=10, obs_dim=28, action_dim=3,
    cbf_config=CBFConfig(), lagrangian_cfg=LagrangianConfig()
)
replay = ReplayBuffer(capacity=10_000)

obs, _ = env.reset()
for _ in range(64):
    acts = agent.select_action(obs)
    next_obs, rews, done, trunc, _ = env.step(acts)
    replay.add(obs, acts, rews, next_obs, done)
    obs = next_obs if not done else env.reset()[0]

batch = replay.sample(16)
losses = agent.update(batch)
print("update losses:", losses)

agent.save("/tmp/hier_test")
agent.load("/tmp/hier_test")
print("save/load OK")
