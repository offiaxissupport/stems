"""Short scale-test: 50 buildings, 200 steps, 3 updates. Verifies no crash."""
import numpy as np
from stems.hierarchical import LargeGridEnv, HierarchicalSTEMSAgent
from stems.utils import ReplayBuffer

B = 50
env = LargeGridEnv(num_buildings=B, seed=0, episode_len=200)
agent = HierarchicalSTEMSAgent(num_buildings=B, obs_dim=28, action_dim=3)
replay = ReplayBuffer(capacity=10_000)

obs, _ = env.reset()
total_steps = 0
for step in range(300):
    acts = agent.select_action(obs)
    next_obs, rews, done, trunc, _ = env.step(acts)
    replay.add(obs, acts, rews, next_obs, done)
    obs = next_obs if not (done or trunc) else env.reset()[0]
    total_steps += 1
    if total_steps >= 128 and total_steps % 50 == 0:
        batch = replay.sample(32)
        losses = agent.update(batch)
        print(f"step {total_steps}: {losses}")

print("50-building scale test PASSED")
