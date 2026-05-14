#!/usr/bin/env python3
"""
Training script for the Hierarchical STEMS agent.

Trains HierarchicalSTEMSAgent on either:
  - The default 3-building CityLearn / mock environment (same as STEMS)
  - A large-grid mock environment with configurable B buildings (--num-buildings 50)

Uses off-policy SAC with a replay buffer (Algorithm 2 variant):
  - Collect transitions into ReplayBuffer
  - Sample random mini-batches at every step once the buffer is warm
  - Evaluate on a separate env instance every --eval-every episodes

Usage:
    # Standard 3-building (matches CityLearn Phase 2):
    python train_hierarchical.py --episodes 50

    # Large-grid 50-building scale-out test:
    python train_hierarchical.py --num-buildings 50 --episodes 30 --save-dir checkpoints/hierarchical_50b/

    # Resume:
    python train_hierarchical.py --load checkpoints/hierarchical/best/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.hierarchical import HierarchicalSTEMSAgent, LargeGridEnv
from stems.metrics import MetricsCalculator
from stems.reward import STEMSReward
from stems.utils import HistoryBuffer, ReplayBuffer, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Hierarchical STEMS agent (off-policy SAC + Lagrangian)"
    )
    p.add_argument("--episodes",     type=int,   default=50,
                   help="Number of training episodes")
    p.add_argument("--num-buildings", type=int,  default=0,
                   help="Override number of buildings (0 = use environment default; "
                        "set to 50+ for scale-out experiment)")
    p.add_argument("--save-dir",     type=str,   default="checkpoints/hierarchical/",
                   help="Checkpoint directory")
    p.add_argument("--load",         type=str,   default=None,
                   help="Path to checkpoint directory to resume from")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval-every",   type=int,   default=5,
                   help="Run one evaluation episode every N training episodes")
    p.add_argument("--replay-capacity", type=int, default=100_000)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--warmup-steps", type=int,   default=1_024,
                   help="Fill replay buffer for this many steps before first update")
    p.add_argument("--cluster-size", type=int,   default=5,
                   help="Target number of buildings per cluster (K = ceil(B / cluster_size))")
    p.add_argument("--event-threshold", type=float, default=0.5,
                   help="Event-trigger Euclidean distance threshold")
    p.add_argument("--no-cbf",       action="store_true",
                   help="Disable CBF safety shield")
    p.add_argument("--schema",       type=str,   default=None,
                   help="CityLearn schema (ignored when --num-buildings > 0)")
    p.add_argument("--device",       type=str,   default="cpu",
                   help="PyTorch device (cpu / cuda)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_envs(args: argparse.Namespace):
    """Return (train_env, eval_env)."""
    if args.num_buildings > 0:
        print(f"[hier] Using LargeGridEnv with {args.num_buildings} buildings")
        train_env = LargeGridEnv(num_buildings=args.num_buildings, seed=args.seed)
        eval_env  = LargeGridEnv(num_buildings=args.num_buildings, seed=args.seed + 1000)
    else:
        train_env = STEMSEnvironment(schema=args.schema, seed=args.seed)
        eval_env  = STEMSEnvironment(schema=args.schema, seed=args.seed + 1000)
    return train_env, eval_env


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(env, args: argparse.Namespace) -> HierarchicalSTEMSAgent:
    B = env.num_buildings
    adj: Optional[np.ndarray] = None
    if hasattr(env, "get_building_info"):
        try:
            config = STEMSConfig()
            info   = env.get_building_info()
            graph  = BuildingGraph(B, info["positions"], info["features"], config.graph)
            adj    = graph.compute_edge_weights().numpy()
        except Exception:
            pass

    agent = HierarchicalSTEMSAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=B,
        adj=adj,
        cluster_size=args.cluster_size,
        event_threshold=args.event_threshold,
        use_cbf=not args.no_cbf,
        device=args.device,
    )
    print(
        f"[hier] Agent: B={B} buildings, K={agent.cluster.K} clusters, "
        f"feat_dim={agent._feat_dim}"
    )
    return agent


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_once(
    agent: HierarchicalSTEMSAgent,
    env,
    config: STEMSConfig,
) -> Dict[str, float]:
    """Run one evaluation episode and return metric dict."""
    calc = MetricsCalculator(
        num_buildings=env.num_buildings,
        cbf_config=config.cbf,
    )
    history_buf = HistoryBuffer(
        num_buildings=env.num_buildings,
        obs_dim=env.obs_dim,
        window_size=config.transformer.window_size,
    )
    agent.event_trigger.reset()

    obs_list, _ = env.reset()
    history_buf.update(obs_list)
    done = False

    while not done:
        history  = history_buf.get()
        actions  = agent.select_action(obs_list, history, explore=False)
        next_obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        calc.add_step(obs_list, actions, next_obs)
        obs_list = next_obs
        history_buf.update(obs_list)

    return calc.summary()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    config = STEMSConfig()

    train_env, eval_env = make_envs(args)
    agent = make_agent(train_env, args)

    if args.load and os.path.isdir(args.load):
        agent.load(args.load)
        print(f"[hier] Loaded checkpoint from {args.load}")

    replay = ReplayBuffer(capacity=args.replay_capacity)
    reward_fn = STEMSReward(
        config=config.reward,
        num_buildings=train_env.num_buildings,
        P_grid_max=config.cbf.P_grid_max,
        P_building_max=config.cbf.P_building_max,
    )
    history_buf = HistoryBuffer(
        num_buildings=train_env.num_buildings,
        obs_dim=train_env.obs_dim,
        window_size=config.transformer.window_size,
    )

    best_cost   = float("inf")
    total_steps = 0
    history_log: List[Dict] = []

    print(
        f"\n[hier] Training: episodes={args.episodes}, "
        f"batch={args.batch_size}, warmup={args.warmup_steps}, "
        f"replay_capacity={args.replay_capacity}"
    )
    print(f"[hier] Device: {args.device}\n")

    for ep in range(1, args.episodes + 1):
        t0 = time.time()
        obs_list, _ = train_env.reset()
        history_buf.reset()
        history_buf.update(obs_list)
        agent.event_trigger.reset()
        agent._cached_cluster_latents = None

        done          = False
        ep_reward     = 0.0
        ep_violations = 0
        ep_steps      = 0
        ep_losses: Dict[str, float] = {
            "actor_loss": 0.0, "q_loss": 0.0,
            "cost_loss": 0.0, "lambda_loss": 0.0,
        }
        prev_net = [float(o[20]) for o in obs_list]

        while not done:
            obs_window = history_buf.get()
            actions    = agent.select_action(obs_list, obs_window, explore=True)
            next_obs, _, terminated, truncated, _ = train_env.step(actions)
            done     = terminated or truncated
            rewards  = reward_fn.compute(obs_list, actions, next_obs, prev_net)
            prev_net = [float(o[20]) for o in next_obs]

            # Count raw CBF violations (before shield projection)
            if agent.cbf is not None:
                viols = agent.cbf.check_violations(actions, obs_list)
                ep_violations += int(viols.any())

            history_buf.update(next_obs)
            next_obs_window = history_buf.get()

            replay.add(
                obs=obs_list, actions=actions, rewards=rewards,
                next_obs=next_obs, done=done,
                history=obs_window, next_history=next_obs_window,
            )
            total_steps += 1
            ep_reward   += float(np.mean(rewards))
            ep_steps    += 1

            # Off-policy mini-batch update
            if total_steps >= args.warmup_steps and replay.is_ready:
                step_losses = agent.update(replay.sample(args.batch_size))
                for k, v in step_losses.items():
                    ep_losses[k] += v

            obs_list = next_obs

        # Normalise episode losses
        if ep_steps > 0:
            ep_losses = {k: v / ep_steps for k, v in ep_losses.items()}

        elapsed = time.time() - t0
        viol_rate = ep_violations / max(ep_steps, 1)
        lam = agent.log_lambdas.exp().detach().cpu().tolist()

        print(
            f"  ep {ep:3d}/{args.episodes} | "
            f"steps={ep_steps:5d} | reward={ep_reward:+8.2f} | "
            f"viol={viol_rate:.3f} | "
            f"λ=[{', '.join(f'{v:.3f}' for v in lam)}] | "
            f"Ql={ep_losses['q_loss']:.4f} | "
            f"Al={ep_losses['actor_loss']:.4f} | "
            f"{elapsed:.0f}s"
        )

        history_log.append({
            "episode": ep,
            "reward": ep_reward,
            "violation_rate": viol_rate,
            "losses": ep_losses,
            "lambdas": lam,
        })

        # ---- Save latest ----
        agent.save(args.save_dir)
        with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
            json.dump(history_log, f, indent=2)

        # ---- Evaluation ----
        if ep % args.eval_every == 0:
            metrics = evaluate_once(agent, eval_env, config)
            cost    = float(metrics.get("cost", float("inf")))
            print(
                f"  [eval ep {ep}] cost={cost:.4f} | "
                f"emission={metrics.get('emission',0):.4f} | "
                f"safety_viol={metrics.get('safety_violation_rate',0):.4f}"
            )
            if cost < best_cost:
                best_cost = cost
                best_dir  = os.path.join(args.save_dir, "best")
                agent.save(best_dir)
                eval_result = {"episode": ep, **metrics}
                with open(os.path.join(best_dir, "eval_results.json"), "w") as f:
                    json.dump(eval_result, f, indent=2)
                print(f"  [eval] New best (cost={best_cost:.4f}) saved to {best_dir}")

    print(f"\n[hier] Training complete. Best eval cost: {best_cost:.4f}")
    print(f"[hier] Checkpoints: {args.save_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
