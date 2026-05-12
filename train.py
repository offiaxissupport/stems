#!/usr/bin/env python3
"""
Main training script for STEMS.
Implements Algorithm 2 from the paper.

Usage:
    python train.py [--episodes 15] [--save-dir checkpoints/] [--seed 42] [--no-cbf] [--strict-paper-mode]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn.functional as F

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.reward import STEMSReward
from stems.metrics import MetricsCalculator
from stems.paper_mode import validate_strict_paper_mode
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
    p.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help=(
            "Enforce paper-relatable protocol: fail on mock env and require 8-building setup."
        ),
    )
    p.add_argument("--mock", action="store_true",
                   help="Force mock environment (8 buildings, matches paper setup) even if CityLearn is installed")
    p.add_argument("--nf-pretrain-after", type=int, default=5,
                   help="Episode after which to pretrain neural filter (0 = disable)")
    p.add_argument("--nf-pretrain-steps", type=int, default=200,
                   help="Gradient steps for neural filter pretraining")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Neural filter pretraining
# ---------------------------------------------------------------------------

def pretrain_neural_filter(
    agent: "STEMSAgent",
    nf_buffer: list,
    steps: int = 200,
) -> float:
    """Train the NeuralSafetyFilter on CBF oracle labels collected during rollouts.

    Parameters
    ----------
    agent      : STEMSAgent with neural_filter and neural_filter_optimizer
    nf_buffer  : list of (obs_B, a_nom_B, a_safe_B) tuples; each array is (B, dim)
    steps      : number of gradient steps

    Returns
    -------
    final_loss : float
    """
    if not nf_buffer:
        return 0.0

    device = agent.device
    # Flatten across timesteps and buildings
    obs_all   = np.concatenate([d[0] for d in nf_buffer], axis=0)   # (N*B, obs_dim)
    a_nom_all = np.concatenate([d[1] for d in nf_buffer], axis=0)   # (N*B, action_dim)
    a_safe_all= np.concatenate([d[2] for d in nf_buffer], axis=0)   # (N*B, action_dim)

    obs_t    = torch.tensor(obs_all,    dtype=torch.float32, device=device)
    a_nom_t  = torch.tensor(a_nom_all,  dtype=torch.float32, device=device)
    a_safe_t = torch.tensor(a_safe_all, dtype=torch.float32, device=device)

    N = obs_t.shape[0]
    batch = min(256, N)
    soc_min = agent.cfg.cbf.SOC_min
    soc_max = agent.cfg.cbf.SOC_max
    P_building_max = agent.cfg.cbf.P_building_max
    _IDX_SOC = 19
    _IDX_NET = 20
    # Approximate kW change per unit battery action (matches mock env's 5 kW * action)
    _BATTERY_KW = 5.0

    agent.neural_filter.train()
    final_loss = 0.0
    for _ in range(steps):
        idx = torch.randint(N, (batch,), device=device)
        obs_b    = obs_t[idx]
        a_nom_b  = a_nom_t[idx]
        a_safe_b = a_safe_t[idx]

        a_pred = agent.neural_filter(obs_b, a_nom_b)

        # Imitation loss: match the CBF QP oracle
        mse = F.mse_loss(a_pred, a_safe_b)

        # Differentiable SOC safety penalty (constraint k=0)
        soc = obs_b[:, _IDX_SOC : _IDX_SOC + 1]
        delta_soc = a_pred[:, 1:2] * agent.neural_filter.SOC_DELTA_RATE
        new_soc = soc + delta_soc
        soc_penalty = (
            F.relu(soc_min - new_soc) + F.relu(new_soc - soc_max)
        ).mean()

        # Differentiable building power penalty (constraint k=1)
        net_current = obs_b[:, _IDX_NET : _IDX_NET + 1]
        predicted_net = net_current + a_pred[:, 1:2] * _BATTERY_KW
        power_penalty = (
            F.relu(torch.abs(predicted_net) - P_building_max)
        ).mean()

        loss = mse + 0.1 * soc_penalty + 0.05 * power_penalty
        agent.neural_filter_optimizer.zero_grad()
        loss.backward()
        agent.neural_filter_optimizer.step()
        final_loss = loss.item()

    agent.neural_filter.eval()
    return final_loss


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
    env = STEMSEnvironment(schema=args.schema, seed=args.seed, force_mock=args.mock)
    eval_env = STEMSEnvironment(schema=args.schema, seed=args.seed + 1000, force_mock=args.mock)

    if args.strict_paper_mode:
        validate_strict_paper_mode(env, context="training")
        validate_strict_paper_mode(eval_env, context="training eval env")
        print("[STEMS] Strict paper mode enabled")

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
        "alpha": [],
        "duration_s": [],
    }

    # Best checkpoint tracking
    best_cost = float("inf")
    best_ep = 0
    best_dir = os.path.join(args.save_dir, "best")

    # Neural filter pretraining buffer: accumulates (obs, a_nom, a_safe) tuples
    nf_buffer: List[Any] = []
    nf_pretrain_after = args.nf_pretrain_after if hasattr(args, "nf_pretrain_after") else 5
    nf_pretrain_steps = args.nf_pretrain_steps if hasattr(args, "nf_pretrain_steps") else 200

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

            # Constraint cost signals (B, 3): binary violation per constraint per building.
            # k=0: SOC bounds (h1), k=1: per-building power (h2), k=2: grid power (h3)
            # Uses next_obs to capture what actually happened after the action.
            _IDX_SOC = 19
            _IDX_NET = 20
            soc_next = np.array([o[_IDX_SOC] for o in next_obs_list], dtype=np.float32)
            net_next = np.array([o[_IDX_NET] for o in next_obs_list], dtype=np.float32)
            c_soc = (
                (soc_next < config.cbf.SOC_min) | (soc_next > config.cbf.SOC_max)
            ).astype(np.float32)                                             # (B,)
            c_power = (np.abs(net_next) > config.cbf.P_building_max).astype(np.float32)  # (B,)
            grid_violated = float(np.maximum(net_next, 0.0).sum() > config.cbf.P_grid_max)
            c_grid = np.full(B, grid_violated, dtype=np.float32)             # (B,)
            constraint_costs = np.stack([c_soc, c_power, c_grid], axis=-1)  # (B, 3)

            # Count actual post-step violations (consistent with constraint_costs).
            # This replaces the pre-step check_violations which measures a different
            # time point and could disagree with what the environment actually observed.
            ep_violations += int(c_soc.sum()) + int(c_power.sum()) + int(grid_violated)

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

            # Collect oracle labels BEFORE advancing obs_list so we store the obs
            # that was actually used to generate _last_raw_actions/_last_qp_safe_actions.
            if not args.no_cbf and nf_pretrain_after > 0:
                nf_buffer.append((
                    np.stack(obs_list, axis=0).copy(),          # (B, obs_dim) current obs
                    agent._last_raw_actions.copy(),              # (B, action_dim) nominal
                    agent._last_qp_safe_actions.copy(),          # (B, action_dim) QP oracle
                ))

            ep_reward += float(np.mean(stems_rewards))
            ep_steps += 1
            obs_list = next_obs_list

        # --- Phase 1.5: Pretrain neural filter after warm-up episodes ---
        if (
            not args.no_cbf
            and nf_pretrain_after > 0
            and ep == nf_pretrain_after
            and not agent.use_neural_filter
            and nf_buffer
        ):
            print(f"[STEMS] Pretraining neural safety filter on {len(nf_buffer)} steps ...")
            nf_loss = pretrain_neural_filter(agent, nf_buffer, steps=nf_pretrain_steps)
            agent.use_neural_filter = True
            nf_buffer.clear()
            print(f"[STEMS] Neural filter pretrained (final loss={nf_loss:.4f}); "
                  "switching to differentiable safety path.")

        # --- Phase 2: Single on-policy update on full trajectory (Algorithm 2, lines 11-13) ---
        batch = episode_buffer.get_batch()
        losses = agent.update(batch)
        ep_actor_loss = losses.get("actor_loss", 0.0)
        ep_critic_loss = losses.get("critic_loss", 0.0)
        ep_cost_critic_loss = losses.get("cost_critic_loss", 0.0)
        ep_lambdas = losses.get("lambdas", [0.0, 0.0, 0.0])
        ep_alpha = losses.get("alpha", 1.0)

        # Evaluate agent (no noise, no exploration)
        eval_cost = evaluate_episode(agent, eval_env, config)

        duration = time.time() - t0
        # Violation rate: ep_violations counts unique constraint events per step
        # (one SOC viol = 1, one power viol = 1, one grid viol = 1, not B copies).
        # Denominator is ep_steps * (B + B + 1) possible violations (SOC×B, power×B, grid×1).
        max_possible = ep_steps * (2 * B + 1)
        viol_rate = ep_violations / max(max_possible, 1)

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
        history["alpha"].append(round(float(ep_alpha), 5))
        history["duration_s"].append(round(duration, 1))

        # Compute star BEFORE updating best_cost so it only marks genuine new bests
        star = " *" if eval_cost < best_cost else ""

        # Save best checkpoint
        if eval_cost < best_cost:
            best_cost = eval_cost
            best_ep = ep
            os.makedirs(best_dir, exist_ok=True)
            agent.save(best_dir)
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
