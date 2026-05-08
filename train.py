#!/usr/bin/env python3
"""
Main training script for STEMS.
Implements Algorithm 2 from the paper.

Usage:
    python train.py [--episodes 15] [--save-dir checkpoints/] [--seed 42] [--no-cbf]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.reward import STEMSReward
from stems.metrics import MetricsCalculator
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STEMS agent (Algorithm 2)")
    p.add_argument("--episodes", type=int, default=15, help="Number of training episodes")
    p.add_argument("--save-dir", type=str, default="checkpoints/", help="Checkpoint directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-cbf", action="store_true", help="Disable CBF safety shield")
    p.add_argument("--schema", type=str, default=None, help="CityLearn schema name or path (default: Phase 2 local eval)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_episode(agent: STEMSAgent, env: STEMSEnvironment, config: STEMSConfig) -> float:
    """Run one evaluation episode and return total cost."""
    history_buf = HistoryBuffer(
        num_buildings=env.num_buildings,
        obs_dim=env.obs_dim,
        window_size=config.transformer.window_size,
    )
    obs_list, _ = env.reset()
    history_buf.update(obs_list)

    total_cost = 0.0
    done = False
    while not done:
        obs_window = history_buf.get()
        actions = agent.select_action(obs_list, obs_window, explore=False)
        next_obs_list, env_rewards, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        net = np.array([o[20] for o in next_obs_list])
        price = np.array([o[21] for o in next_obs_list])
        total_cost += float((np.maximum(net, 0.0) * price).sum())

        obs_list = next_obs_list
        history_buf.update(obs_list)

    return total_cost


# ---------------------------------------------------------------------------
# Main training loop (Algorithm 2)
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    config = STEMSConfig()
    config.training.episodes = args.episodes

    print(f"[STEMS] Initialising environment ...")
    env = STEMSEnvironment(schema=args.schema, seed=args.seed)
    eval_env = STEMSEnvironment(schema=args.schema, seed=args.seed + 1000)

    if env.using_mock:
        print("[STEMS] Using mock environment (CityLearn not installed)")
    else:
        print("[STEMS] Using real CityLearn environment")

    B = env.num_buildings
    print(f"[STEMS] Buildings: {B}, obs_dim: {env.obs_dim}, action_dim: {env.action_dim}")

    # Build building graph
    building_info = env.get_building_info()
    graph = BuildingGraph(
        num_buildings=B,
        positions=building_info["positions"],
        features=building_info["features"],
        config=config.graph,
    )

    # Initialise agent
    agent = STEMSAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=B,
        building_graph=graph,
        config=config,
        use_cbf=not args.no_cbf,
    )

    # Episode buffer (on-policy) and history buffer
    episode_buffer = EpisodeBuffer()
    history_buf = HistoryBuffer(
        num_buildings=B,
        obs_dim=env.obs_dim,
        window_size=config.transformer.window_size,
    )

    reward_fn = STEMSReward(
        config=config.reward,
        num_buildings=B,
        P_grid_max=config.cbf.P_grid_max,
        P_building_max=config.cbf.P_building_max,
    )

    # Training history log
    history: Dict[str, List[Any]] = {
        "episode": [],
        "total_reward": [],
        "eval_cost": [],
        "safety_violations": [],
        "actor_loss": [],
        "critic_loss": [],
        "cost_critic_loss": [],
        "lambda_soc": [],
        "lambda_power": [],
        "lambda_grid": [],
        "duration_s": [],
    }

    # Best checkpoint tracking
    best_cost = float("inf")
    best_ep = 0
    best_dir = os.path.join(args.save_dir, "best")

    print(f"\n[STEMS] Starting training for {config.training.episodes} episodes ...")
    print("-" * 65)

    for ep in range(1, config.training.episodes + 1):
        t0 = time.time()
        obs_list, _ = env.reset()
        history_buf.reset()
        history_buf.update(obs_list)

        ep_reward = 0.0
        ep_violations = 0
        ep_steps = 0
        prev_net = [float(o[20]) for o in obs_list]
        done = False

        # --- Phase 1: Collect full episode trajectory (Algorithm 2, lines 5-10) ---
        episode_buffer.reset()
        while not done:
            obs_window = history_buf.get()
            actions = agent.select_action(obs_list, obs_window, explore=True)
            next_obs_list, env_rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated

            # Compute STEMS reward
            stems_rewards = reward_fn.compute(obs_list, actions, next_obs_list, prev_net)
            prev_net = [float(o[20]) for o in next_obs_list]

            if ep_steps > 0 and ep_steps % 500 == 0:
                print(f"  [Ep{ep}] step={ep_steps}  reward={ep_reward:.1f}  "
                      f"viol_so_far={ep_violations}", flush=True)

            # Count safety violations
            violations = agent.cbf.check_violations(actions, obs_list)
            ep_violations += int(violations.sum())

            # Constraint cost signals (B, 3): binary violation per constraint per building.
            # k=0: SOC bounds (h1), k=1: per-building power (h2), k=2: grid power (h3)
            _IDX_SOC = 19
            _IDX_NET = 20
            soc_next = np.array([o[_IDX_SOC] for o in next_obs_list], dtype=np.float32)
            net_next = np.array([o[_IDX_NET] for o in next_obs_list], dtype=np.float32)
            c_soc = (
                (soc_next < config.cbf.SOC_min) | (soc_next > config.cbf.SOC_max)
            ).astype(np.float32)                                             # (B,)
            c_power = (np.abs(net_next) > config.cbf.P_building_max).astype(np.float32)  # (B,)
            c_grid = np.full(
                B, float(net_next.sum() > config.cbf.P_grid_max), dtype=np.float32
            )                                                                 # (B,)
            constraint_costs = np.stack([c_soc, c_power, c_grid], axis=-1)  # (B, 3)

            # Build next_history for storage (after updating with next_obs)
            history_buf.update(next_obs_list)
            next_obs_window = history_buf.get()

            # Store transition in episode buffer
            episode_buffer.add(
                obs=obs_list,
                actions=actions,
                rewards=stems_rewards,
                next_obs=next_obs_list,
                done=done,
                history=obs_window,
                next_history=next_obs_window,
                raw_actions=agent._last_raw_actions,
                safe_actions=agent._last_safe_actions,   # post-CBF, used for Eq 24
                constraint_costs=constraint_costs,        # (B, 3) Lagrangian cost signals
            )

            ep_reward += float(np.mean(stems_rewards))
            ep_steps += 1
            obs_list = next_obs_list

        # --- Phase 2: Single on-policy update on full trajectory (Algorithm 2, lines 11-13) ---
        batch = episode_buffer.get_batch()
        losses = agent.update(batch)
        ep_actor_loss = losses.get("actor_loss", 0.0)
        ep_critic_loss = losses.get("critic_loss", 0.0)
        ep_cost_critic_loss = losses.get("cost_critic_loss", 0.0)
        ep_lambdas = losses.get("lambdas", [0.0, 0.0, 0.0])

        # Evaluate agent (no noise, no exploration)
        eval_cost = evaluate_episode(agent, eval_env, config)

        duration = time.time() - t0
        viol_rate = ep_violations / max(ep_steps * B, 1)

        # Log
        history["episode"].append(ep)
        history["total_reward"].append(round(ep_reward, 3))
        history["eval_cost"].append(round(eval_cost, 3))
        history["safety_violations"].append(round(viol_rate, 4))
        history["actor_loss"].append(round(ep_actor_loss, 4))
        history["critic_loss"].append(round(ep_critic_loss, 4))
        history["cost_critic_loss"].append(round(ep_cost_critic_loss, 4))
        history["lambda_soc"].append(round(float(ep_lambdas[0]), 5))
        history["lambda_power"].append(round(float(ep_lambdas[1]), 5))
        history["lambda_grid"].append(round(float(ep_lambdas[2]), 5))
        history["duration_s"].append(round(duration, 1))

        # Save best checkpoint
        if eval_cost < best_cost:
            best_cost = eval_cost
            best_ep = ep
            os.makedirs(best_dir, exist_ok=True)
            agent.save(best_dir)

        star = " *" if eval_cost <= best_cost else ""
        print(
            f"[STEMS] Ep{ep:3d}: reward={ep_reward:8.2f}  "
            f"viol={viol_rate:.3f}  eval_cost={eval_cost:.1f}  "
            f"({duration:.1f}s){star}"
        )

    print("-" * 65)
    print("[STEMS] Training complete.")

    # Save final checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(args.save_dir)
    print(f"[STEMS] Final checkpoint saved to {args.save_dir}")

    # If we tracked a best checkpoint, report it
    if best_cost < float("inf"):
        print(f"[STEMS] Best checkpoint (ep {best_ep}, eval_cost={best_cost:.1f}) saved to {best_dir}")

    # Save training history
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[STEMS] Training history saved to {history_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())
