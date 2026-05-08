#!/usr/bin/env python3
"""
Evaluate STEMS and baselines, print Table I and Table II.

Usage:
    python evaluate.py [--checkpoint checkpoints/] [--episodes 1]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.baselines import (
    RuleBasedAgent, SingleAgentSAC, DMAPPOAgent,
    MPCAgent, MADDPGAgent, MARLISAAgent, MADCQAgent, MetaEMSAgent,
)
from stems.reward import STEMSReward
from stems.metrics import MetricsCalculator
from stems.utils import HistoryBuffer, EpisodeBuffer, ReplayBuffer, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate STEMS and baseline agents")
    p.add_argument("--checkpoint", type=str, default="checkpoints/",
                   help="Path to STEMS checkpoint directory")
    p.add_argument("--episodes", type=int, default=1, help="Evaluation episodes per agent")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--schema", type=str, default=None, help="CityLearn schema name or path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------

def _make_stems(env: STEMSEnvironment, checkpoint: str) -> STEMSAgent:
    config = STEMSConfig()
    info = env.get_building_info()
    graph = BuildingGraph(env.num_buildings, info["positions"], info["features"], config.graph)
    agent = STEMSAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=env.num_buildings,
        building_graph=graph,
        config=config,
        use_cbf=True,
    )
    if os.path.isdir(checkpoint) and os.path.exists(os.path.join(checkpoint, "encoder.pt")):
        agent.load(checkpoint)
        print(f"[eval] Loaded STEMS checkpoint from {checkpoint}")
    else:
        print("[eval] No checkpoint found – using untrained STEMS weights (run train.py first for best results)")
    return agent


def _make_sac(env: STEMSEnvironment) -> SingleAgentSAC:
    return SingleAgentSAC(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=env.num_buildings,
    )


def _make_ppo(env: STEMSEnvironment) -> DMAPPOAgent:
    return DMAPPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=env.num_buildings,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent: Any,
    env: STEMSEnvironment,
    config: STEMSConfig,
    explore: bool = False,
) -> MetricsCalculator:
    """Run one episode, return populated MetricsCalculator."""
    calc = MetricsCalculator(
        num_buildings=env.num_buildings,
        cbf_config=config.cbf,
    )
    history_buf = HistoryBuffer(
        num_buildings=env.num_buildings,
        obs_dim=env.obs_dim,
        window_size=config.transformer.window_size,
    )

    obs_list, _ = env.reset()
    history_buf.update(obs_list)
    done = False

    while not done:
        history = history_buf.get()
        actions = agent.select_action(obs_list, history, explore=explore)
        next_obs_list, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        calc.add_step(obs_list, actions, next_obs_list)
        obs_list = next_obs_list
        history_buf.update(obs_list)

    return calc


# ---------------------------------------------------------------------------
# Quick training for SAC/PPO baselines
# ---------------------------------------------------------------------------

def quick_train(
    agent: Any,
    env: STEMSEnvironment,
    config: STEMSConfig,
    episodes: int = 3,
) -> None:
    """Short training run so baselines have learned something."""
    episode_buffer = EpisodeBuffer()
    history_buf = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    reward_fn = STEMSReward(
        config=config.reward,
        num_buildings=env.num_buildings,
        P_grid_max=config.cbf.P_grid_max,
        P_building_max=config.cbf.P_building_max,
    )

    for _ in range(episodes):
        obs_list, _ = env.reset()
        history_buf.reset()
        history_buf.update(obs_list)
        episode_buffer.reset()
        done = False
        prev_net = [float(o[20]) for o in obs_list]
        while not done:
            obs_window = history_buf.get()
            actions = agent.select_action(obs_list, obs_window, explore=True)
            next_obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            rewards = reward_fn.compute(obs_list, actions, next_obs, prev_net)
            prev_net = [float(o[20]) for o in next_obs]
            history_buf.update(next_obs)
            next_obs_window = history_buf.get()
            # Store raw_actions if the agent tracks them, else use actions
            raw_actions = getattr(agent, "_last_raw_actions", actions)
            episode_buffer.add(
                obs=obs_list, actions=actions, rewards=rewards,
                next_obs=next_obs, done=done, history=obs_window,
                next_history=next_obs_window, raw_actions=raw_actions,
            )
            obs_list = next_obs
        batch = episode_buffer.get_batch()
        agent.update(batch)


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

def _table1_row(name: str, m: Dict[str, float]) -> str:
    return (
        f"  {name:<20s} | "
        f"{m.get('cost', 0):.3f} | "
        f"{m.get('emission', 0):.3f} | "
        f"{m.get('avg_daily_peak', 0):.3f} | "
        f"{m.get('electricity_consumption', 0):.3f} | "
        f"{m.get('ramping_rate', 0):.3f} | "
        f"{m.get('discomfort_rate', 0):.3f} | "
        f"{m.get('safety_violation_rate', 0):.3f}"
    )


def print_table1(metrics: Dict[str, Dict[str, float]]) -> None:
    header = (
        f"  {'Agent':<20s} | "
        f"{'Cost':>6} | "
        f"{'Emiss':>6} | "
        f"{'DayPk':>6} | "
        f"{'Consm':>6} | "
        f"{'Ramp':>6} | "
        f"{'Discom':>7} | "
        f"{'SafVio':>7}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  TABLE I – Normalised Performance (baseline = 1.0; lower is better for cols 1-5)")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, m in metrics.items():
        print(_table1_row(name, m))
    print(sep)


def print_table2(
    normal: Dict[str, Dict[str, float]],
    heatwave: Dict[str, Dict[str, float]],
    coldwave: Dict[str, Dict[str, float]],
) -> None:
    print("\n" + "=" * 80)
    print("  TABLE II – Extreme Weather & Outage Robustness (absolute metrics)")
    print("=" * 80)
    print(f"  {'Agent':<20s} | {'Normal':>10} | {'HeatWave':>10} | {'ColdWave':>10}")
    print("-" * 60)
    for name in normal:
        nc = normal[name].get("cost", 0)
        hc = heatwave.get(name, {}).get("cost", 0)
        cc = coldwave.get(name, {}).get("cost", 0)
        print(f"  {name:<20s} | {nc:10.1f} | {hc:10.1f} | {cc:10.1f}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    config = STEMSConfig()

    env = STEMSEnvironment(schema=args.schema, seed=args.seed)
    B = env.num_buildings

    print(f"[eval] Environment: {'mock' if env.using_mock else 'CityLearn'}, "
          f"buildings={B}, obs_dim={env.obs_dim}")

    # ---- build agents ----
    stems_agent = _make_stems(env, args.checkpoint)
    rule_agent = RuleBasedAgent(num_buildings=B)
    sac_agent = _make_sac(env)
    ppo_agent = _make_ppo(env)
    mpc_agent = MPCAgent(
        num_buildings=B, action_dim=env.action_dim,
        soc_min=config.cbf.SOC_min, soc_max=config.cbf.SOC_max,
        P_building_max=config.cbf.P_building_max, P_grid_max=config.cbf.P_grid_max,
    )
    maddpg_agent = MADDPGAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B)
    marlisa_agent = MARLISAAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B)
    madcq_agent = MADCQAgent(
        obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B,
        soc_min=config.cbf.SOC_min, soc_max=config.cbf.SOC_max,
    )
    metaems_agent = MetaEMSAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B)

    # Quick baseline training (few episodes so results are non-trivial)
    learnable_baselines = {
        "SingleSAC": (sac_agent, args.seed + 1),
        "DMAPPO": (ppo_agent, args.seed + 2),
        "MADDPG": (maddpg_agent, args.seed + 3),
        "MARLISA": (marlisa_agent, args.seed + 4),
        "MADCQ": (madcq_agent, args.seed + 5),
        "MetaEMS": (metaems_agent, args.seed + 6),
    }
    for bname, (bagent, bseed) in learnable_baselines.items():
        print(f"[eval] Training {bname} baseline ...")
        quick_train(bagent, STEMSEnvironment(schema=args.schema, seed=bseed), config, episodes=3)

    agents = {
        "STEMS": stems_agent,
        "RuleBased": rule_agent,
        "MPC": mpc_agent,
        "SingleSAC": sac_agent,
        "MADDPG": maddpg_agent,
        "MARLISA": marlisa_agent,
        "MADCQ": madcq_agent,
        "DMAPPO": ppo_agent,
        "MetaEMS": metaems_agent,
    }

    # ---- normal evaluation ----
    normal_raw: Dict[str, Dict[str, float]] = {}
    for name, agent in agents.items():
        print(f"[eval] Running {name} ...")
        calc = run_episode(agent, env, config, explore=False)
        normal_raw[name] = calc.compute_all()

    # Normalise against RuleBased baseline
    baseline = normal_raw.get("RuleBased", {})
    normal_norm: Dict[str, Dict[str, float]] = {}
    for name, m in normal_raw.items():
        normal_norm[name] = MetricsCalculator(B, config.cbf).compute_all.__func__(  # type: ignore
            type("_", (), {"_net_list": [], "_price_list": [], "_carbon_list": [],
                           "_t_in_list": [], "_t_set_list": [], "_occupant_list": [],
                           "_soc_list": [], "_action_list": [], "B": B, "cbf": config.cbf})(),
        ) if False else m   # placeholder – just use raw here for clarity

    # Normalise manually
    for name, m in normal_raw.items():
        norm_m = dict(m)
        for key in ["cost", "emission", "avg_daily_peak", "electricity_consumption", "ramping_rate"]:
            base_val = baseline.get(key, 1.0)
            norm_m[key] = m[key] / base_val if abs(base_val) > 1e-10 else 1.0
        normal_norm[name] = norm_m

    print_table1(normal_norm)

    # ---- extreme weather ----
    print("\n[eval] Running extreme weather evaluation ...")

    def run_extreme(temp_offset: float) -> Dict[str, Dict[str, float]]:
        """Run evaluation with forced outdoor temperature offsets."""
        results: Dict[str, Dict[str, float]] = {}
        for name, agent in agents.items():
            xenv = STEMSEnvironment(schema=args.schema, seed=args.seed)
            xenv.set_temp_offset(temp_offset)
            calc = run_episode(agent, xenv, config, explore=False)
            results[name] = calc.compute_all()
        return results

    heatwave_raw = run_extreme(temp_offset=10.0)
    coldwave_raw = run_extreme(temp_offset=-10.0)

    print_table2(normal_raw, heatwave_raw, coldwave_raw)

    # Save evaluation results to JSON for visualize.py
    eval_output = {}
    for name, m in normal_norm.items():
        eval_output[name] = {k: round(v, 4) for k, v in m.items()}
    eval_path = os.path.join(args.checkpoint, "eval_results.json")
    os.makedirs(args.checkpoint, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n[eval] Normalised results saved to {eval_path}")

    # Save Table II (extreme weather absolute costs) to JSON
    table2_output = {
        "normal": {n: round(v.get("cost", 0), 2) for n, v in normal_raw.items()},
        "heatwave": {n: round(v.get("cost", 0), 2) for n, v in heatwave_raw.items()},
        "coldwave": {n: round(v.get("cost", 0), 2) for n, v in coldwave_raw.items()},
    }
    table2_path = os.path.join(args.checkpoint, "eval_extreme_results.json")
    with open(table2_path, "w") as f:
        json.dump(table2_output, f, indent=2)
    print(f"[eval] Extreme weather results saved to {table2_path}")

    print("\n[eval] Evaluation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate(parse_args())
