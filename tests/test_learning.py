"""The learner must learn: a known-answer bandit through the production update().

Reward r_b = -4 (a_hvac - 0.6)^2 - 4 (a_batt + 0.3)^2 has its optimum at
(0.6, -0.3) whatever the state. Before the entropy-temperature fix the
deterministic action plateaued near 0.27 on this problem; a learner that cannot
leave its initialisation fails here, not three hours into a grid. Mock
environment (synthetic): only the action matters to this reward.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.agent import STEMSAgent
from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed

STEPS, EPISODES = 120, 10


def _agent_and_env():
    set_seed(0)
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    cfg = STEMSConfig()
    info = env.get_building_info()
    graph = BuildingGraph(env.num_buildings, info["positions"], info["features"], cfg.graph)
    agent = STEMSAgent(env.obs_dim, env.action_dim, env.num_buildings, graph, config=cfg,
                       battery_info=env.battery_info(), use_cbf=False,
                       electrical_storage_action_index=env.electrical_storage_action_index,
                       hvac_action_index=env.hvac_action_index)
    return agent, env, cfg


def _episode(agent, env, cfg, explore, buf=None):
    h, e = env.hvac_action_index, env.electrical_storage_action_index
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.update(obs)
    for k in range(STEPS):
        w = hist.get()
        a = agent.select_action(obs, w, explore=explore)
        nxt = env.step(a)[0]
        r = [-4 * (a[i, h] - 0.6) ** 2 - 4 * (a[i, e] + 0.3) ** 2 for i in range(len(obs))]
        hist.update(nxt)
        if buf is not None:
            buf.add(obs=obs, actions=a, rewards=r, next_obs=nxt, done=(k == STEPS - 1),
                    history=w, next_history=hist.get(), raw_actions=agent._last_raw_actions,
                    safe_actions=agent._last_safe_actions, pre_tanh=agent._last_pre_tanh,
                    behaviour_log_probs=agent._last_log_probs,
                    constraint_costs=np.zeros((env.num_buildings, 3), np.float32))
        obs = nxt
    return agent._last_raw_actions[:, h].mean(), agent._last_raw_actions[:, e].mean()


def test_first_episode_only_fits_the_normaliser():
    agent, env, cfg = _agent_and_env()
    before = [p.detach().clone() for p in agent.actors.parameters()]
    buf = EpisodeBuffer()
    _episode(agent, env, cfg, explore=True, buf=buf)
    stats = agent.update(buf.get_batch())
    assert stats["warmup"] and stats["gradient_steps"] == 0
    assert int(agent.obs_normalizer.count) > 0
    assert all((a == b).all() for a, b in zip(before, agent.actors.parameters()))


def test_first_minibatch_ratio_is_one():
    """The recorded behaviour log-prob must equal the policy's own at update time."""
    agent, env, cfg = _agent_and_env()
    cfg.training.update_epochs, cfg.training.minibatch_size = 1, 10_000
    for expected_steps in (0, 1):                      # warm-up, then one real step
        buf = EpisodeBuffer()
        _episode(agent, env, cfg, explore=True, buf=buf)
        stats = agent.update(buf.get_batch())
        assert stats["gradient_steps"] == expected_steps
    assert abs(stats["approx_kl"]) < 1e-5 and stats["clip_frac"] == 0.0


def test_policy_moves_to_the_known_optimum():
    agent, env, cfg = _agent_and_env()
    for _ in range(EPISODES):
        buf = EpisodeBuffer()
        _episode(agent, env, cfg, explore=True, buf=buf)
        agent.update(buf.get_batch())
    hvac, batt = _episode(agent, env, cfg, explore=False)
    assert hvac > 0.35, f"HVAC action {hvac:+.3f} did not move toward 0.6"
    assert batt < -0.15, f"battery action {batt:+.3f} did not move toward -0.3"


def test_dual_rises_above_the_limit_and_relaxes_below_it():
    import torch

    agent, _, cfg = _agent_and_env()
    limit = cfg.lagrangian.cost_limit
    start = agent._lambdas.clone()
    for _ in range(5):
        agent._update_lambdas(torch.tensor([0.5, 0.0, limit]))
    lam = agent._lambdas
    assert lam[0] > start[0] + 1.0, "a 50% violation rate must raise its multiplier"
    assert lam[1] == 0.0, "a satisfied constraint's multiplier relaxes to zero"
    assert lam[2] == pytest.approx(start[2]), "at the limit the multiplier holds"
    high = lam[0].item()
    for _ in range(5):
        agent._update_lambdas(torch.tensor([0.0, 0.0, limit]))
    assert agent._lambdas[0] < high
