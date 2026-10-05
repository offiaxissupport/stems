#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from stems.config import STEMSConfig
from stems.environment import ACTION_GROUPS, STEMSEnvironment
from stems.thermal import build_thermal_stack
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.reward import STEMSReward
from stems.metrics import MetricsCalculator
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed

_IDX_SOC, _IDX_NET, _IDX_PRICE = 19, 20, 21


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STEMS (Algorithm 2)")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--save-dir", type=str, default="checkpoints/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schema", type=str, default=None)
    p.add_argument("--no-cbf", action="store_true", help="Disable the CBF safety shield")
    p.add_argument("--mock", action="store_true",
                   help="Use the synthetic mock (loud banner; NOT for reported results)")
    p.add_argument("--heat-pump", action="store_true",
                   help="Enable bidirectional heat-pump observations/physics (Phase 3)")
    p.add_argument("--preheat", action="store_true",
                   help="Enable the anticipatory hot-water readiness barrier (h4) "
                        "and the weather-aware CoP power guard (stems.thermal).")
    p.add_argument("--preheat-horizon", type=int, default=None,
                   help="Pre-heat horizon L in hours (default: ThermalConfig).")
    p.add_argument("--isolate", type=str, default="none",
                   choices=["none"] + list(ACTION_GROUPS),
                   help="Restrict the agent to a device subset; other actuators are held "
                        "at no-op. 'heatpump'/'thermal' imply --heat-pump physics.")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Truncate each episode to N steps (0 = full year). Use for smoke tests.")
    p.add_argument("--smoke", action="store_true",
                   help="Fast pipeline check: 2 episodes x 300 steps, eval every episode")
    p.add_argument("--eval-interval", type=int, default=0,
                   help="Run a deterministic eval episode every N episodes (0 = only at end)")
    p.add_argument("--no-pid", action="store_true",
                   help="Ablation: plain dual ascent instead of PID-Lagrangian")
    return p.parse_args()


def evaluate_episode(agent: STEMSAgent, env: STEMSEnvironment, config: STEMSConfig,
                     soc_rate: np.ndarray, max_steps: int = 0,
                     count_soc: bool = True) -> Dict[str, float]:
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    metrics = MetricsCalculator(env.num_buildings, config.cbf, soc_rate=soc_rate,
                                heating_setpoint_idx=env.heating_setpoint_idx,
                                count_soc=count_soc)
    obs_list, _ = env.reset()
    hist.update(obs_list)
    done, steps = False, 0
    while not done:
        actions = agent.select_action(obs_list, hist.get(), explore=False)
        next_obs, _, terminated, truncated, _ = env.step(actions)
        agent.observe(next_obs)
        metrics.add_step(obs_list, actions, next_obs)
        obs_list = next_obs
        hist.update(obs_list)
        steps += 1
        done = terminated or truncated or (max_steps and steps >= max_steps)
    return metrics.compute_all()


def train(args: argparse.Namespace) -> None:
    if args.smoke:
        args.episodes, args.max_steps, args.eval_interval = 2, 300, 1
    set_seed(args.seed)

    config = STEMSConfig()
    config.training.episodes = args.episodes
    if args.no_pid:
        config.lagrangian.use_pid = False
    heat_pump = args.heat_pump or (
        "cooling_or_heating_device" in ACTION_GROUPS.get(args.isolate, []))
    if heat_pump:
        config.heat_pump.enabled = True

    print("[STEMS] Initialising environment ...")
    env = STEMSEnvironment(schema=args.schema, seed=args.seed,
                           force_mock=args.mock, heat_pump=heat_pump)
    eval_env = STEMSEnvironment(schema=args.schema, seed=args.seed + 1000,
                                force_mock=args.mock, heat_pump=heat_pump)
    B = env.num_buildings
    battery = env.battery_info()
    soc_rate = battery["soc_rate"]
    control_indices = env.resolve_control_indices(args.isolate)
    count_soc = (control_indices is None
                or env.electrical_storage_action_index in control_indices)
    print(f"[STEMS] env_type={env.env_type}  B={B}  obs_dim={env.obs_dim}  "
          f"action_dim={env.action_dim}")
    print(f"[STEMS] tricks_active={config.tricks_active()}  use_cbf={not args.no_cbf}")
    print(f"[STEMS] per-building soc_rate={np.round(soc_rate, 3).tolist()}")
    print(f"[STEMS] isolate={args.isolate}  control_indices={control_indices}")

    if args.preheat_horizon is not None:
        config.thermal.preheat_horizon = args.preheat_horizon
    dhw_barrier, cop_model = build_thermal_stack(env, config.thermal, enable=args.preheat)
    if args.preheat:
        print(f"[STEMS] pre-heat barrier ON: L={config.thermal.preheat_horizon}h  "
              f"time_to_heat={np.round(dhw_barrier.dyn.time_to_heat_h, 2).tolist()}h")

    info = env.get_building_info()
    graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
    agent = STEMSAgent(env.obs_dim, env.action_dim, B, graph, config=config,
                       battery_info=battery, use_cbf=not args.no_cbf,
                       electrical_storage_action_index=env.electrical_storage_action_index,
                       control_indices=control_indices,
                       dhw_barrier=dhw_barrier, cop_model=cop_model,
                       hvac_action_index=env.hvac_action_index)

    reward_fn = STEMSReward(config.reward, B, config.cbf.P_grid_max, config.cbf.P_building_max,
                            heating_setpoint_idx=env.heating_setpoint_idx)
    episode_buffer = EpisodeBuffer()
    hist = HistoryBuffer(B, env.obs_dim, config.transformer.window_size)

    history: Dict[str, List[Any]] = {k: [] for k in (
        "episode", "total_reward", "safety_violation_rate", "soc_violation_rate",
        "power_violation_rate", "grid_violation_rate", "eval_cost", "eval_safety",
        "policy_loss", "value_loss", "cost_value_loss", "entropy", "approx_kl",
        "clip_frac", "lambdas", "duration_s")}
    history["meta"] = {
        "env_type": env.env_type, "num_buildings": B, "obs_dim": env.obs_dim,
        "action_dim": env.action_dim, "seed": args.seed, "use_cbf": not args.no_cbf,
        "heat_pump": heat_pump, "tricks_active": config.tricks_active(),
        "soc_rate": np.round(soc_rate, 4).tolist(), "max_steps": args.max_steps,
        "isolate": args.isolate, "control_indices": control_indices, "count_soc": count_soc,
        "preheat": args.preheat,
        "preheat_horizon": config.thermal.preheat_horizon if args.preheat else None,
        "citylearn_patches": env.citylearn_patches,
    }

    print(f"\n[STEMS] Training {args.episodes} episode(s)"
          f"{f' x {args.max_steps} steps' if args.max_steps else ' (full year)'} ...")
    print("-" * 70)
    for ep in range(1, args.episodes + 1):
        t0 = time.time()
        obs_list, _ = env.reset()
        hist.reset(); hist.update(obs_list)
        episode_buffer.reset()
        ep_metrics = MetricsCalculator(B, config.cbf, soc_rate=soc_rate,
                                       heating_setpoint_idx=env.heating_setpoint_idx,
                                       count_soc=count_soc)
        ep_reward = 0.0
        prev_net = [float(o[_IDX_NET]) for o in obs_list]
        done, steps = False, 0

        while not done:
            obs_window = hist.get()
            actions = agent.select_action(obs_list, obs_window, explore=True)
            next_obs, _, terminated, truncated, _ = env.step(actions)
            agent.observe(next_obs)
            steps += 1
            done = terminated or truncated or (args.max_steps and steps >= args.max_steps)

            rewards = reward_fn.compute(obs_list, actions, next_obs, prev_net)
            prev_net = [float(o[_IDX_NET]) for o in next_obs]
            ep_metrics.add_step(obs_list, actions, next_obs)

            soc_n = np.array([o[_IDX_SOC] for o in next_obs], dtype=np.float32)
            net_n = np.array([o[_IDX_NET] for o in next_obs], dtype=np.float32)
            c_soc = ((soc_n < config.cbf.SOC_min) | (soc_n > config.cbf.SOC_max)).astype(np.float32)
            c_pow = (np.abs(net_n) > config.cbf.P_building_max).astype(np.float32)
            c_grid = np.full(B, float(np.maximum(net_n, 0).sum() > config.cbf.P_grid_max), np.float32)
            constraint_costs = np.stack([c_soc, c_pow, c_grid], axis=-1)

            hist.update(next_obs)
            episode_buffer.add(obs=obs_list, actions=actions, rewards=rewards,
                               next_obs=next_obs, done=done, history=obs_window,
                               next_history=hist.get(), raw_actions=agent._last_raw_actions,
                               safe_actions=agent._last_safe_actions,
                               pre_tanh=agent._last_pre_tanh,
                               behaviour_log_probs=agent._last_log_probs,
                               constraint_costs=constraint_costs)
            ep_reward += float(np.mean(rewards))
            obs_list = next_obs

        losses = agent.update(episode_buffer.get_batch())
        m = ep_metrics.compute_all()
        do_eval = args.eval_interval and ep % args.eval_interval == 0
        ev = evaluate_episode(agent, eval_env, config, soc_rate, args.max_steps,
                             count_soc=count_soc) if do_eval else None
        dur = time.time() - t0

        history["episode"].append(ep)
        history["total_reward"].append(round(ep_reward, 3))
        for key in ("safety_violation_rate", "soc_violation_rate",
                    "power_violation_rate", "grid_violation_rate"):
            history[key].append(round(m[key], 5))
        history["eval_cost"].append(round(ev["cost"], 3) if ev else None)
        history["eval_safety"].append(round(ev["safety_violation_rate"], 5) if ev else None)
        for key, src in (("policy_loss", "policy"), ("value_loss", "value"),
                         ("cost_value_loss", "cost_value"), ("entropy", "entropy"),
                         ("approx_kl", "approx_kl"), ("clip_frac", "clip_frac")):
            history[key].append(round(losses[src], 5))
        history["lambdas"].append([round(x, 4) for x in losses["lambdas"]])
        history["duration_s"].append(round(dur, 1))

        evtxt = f"  eval_cost={ev['cost']:.1f} eval_viol={ev['safety_violation_rate']:.4f}" if ev else ""
        print(f"[STEMS] Ep{ep:3d}: reward={ep_reward:8.2f}  "
              f"viol={m['safety_violation_rate']:.4f} (soc={m['soc_violation_rate']:.4f})"
              f"{evtxt}  ({dur:.1f}s)")

    print("-" * 70)
    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(args.save_dir)
    with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[STEMS] Saved checkpoint + training_history.json to {args.save_dir}")


if __name__ == "__main__":
    train(parse_args())
