#!/usr/bin/env python3
"""
Evaluate STEMS and baselines, print Table I and Table II.

Usage:
    python evaluate.py [--checkpoint checkpoints/] [--episodes 3] [--baseline-train-episodes 15] [--strict-paper-mode]
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
from stems.paper_mode import validate_strict_paper_mode
from stems.utils import HistoryBuffer, EpisodeBuffer, ReplayBuffer, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate STEMS and baseline agents")
    p.add_argument("--checkpoint", type=str, default="checkpoints/",
                   help="Path to STEMS checkpoint directory")
    p.add_argument("--episodes", type=int, default=3, help="Evaluation episodes per agent (averaged)")
    p.add_argument(
        "--baseline-train-episodes", type=int, default=50,
        help="Training episodes for learnable baselines before evaluation (should match STEMS training episodes for fair comparison)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--schema", type=str, default=None, help="CityLearn schema name or path")
    p.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help=(
            "Enforce paper-relatable protocol: fail on mock env, require 8 buildings, "
            "and skip synthetic extreme-weather perturbation."
        ),
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "List of training seeds to aggregate over (e.g. --seeds 0 1 42). "
            "If given, evaluates checkpoints/seed{s}/best/ for each seed and "
            "reports mean ± std in Table I. Overrides --checkpoint for STEMS."
        ),
    )
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


def _mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of metric dictionaries key-wise."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m.get(k, 0.0) for m in metrics_list])) for k in keys}


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


def _table1_row_stats(
    name: str,
    means: Dict[str, float],
    stds: Optional[Dict[str, float]] = None,
) -> str:
    """Print a Table I row with optional ± std columns."""

    def _fmt(key: str) -> str:
        mu = means.get(key, 0.0)
        if stds is None:
            return f"{mu:.3f}"
        sd = stds.get(key, 0.0)
        return f"{mu:.3f}±{sd:.3f}"

    w = 10  # column width when stds are shown
    if stds is not None:
        return (
            f"  {name:<20s} | "
            f"{_fmt('cost'):>{w}} | "
            f"{_fmt('emission'):>{w}} | "
            f"{_fmt('avg_daily_peak'):>{w}} | "
            f"{_fmt('electricity_consumption'):>{w}} | "
            f"{_fmt('ramping_rate'):>{w}} | "
            f"{_fmt('discomfort_rate'):>{w}} | "
            f"{_fmt('safety_violation_rate'):>{w}}"
        )
    return _table1_row(name, means)


def print_table1(
    metrics: Dict[str, Dict[str, float]],
    stds: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    col_w = 10 if stds else 6
    header = (
        f"  {'Agent':<20s} | "
        f"{'Cost':>{col_w}} | "
        f"{'Emiss':>{col_w}} | "
        f"{'DayPk':>{col_w}} | "
        f"{'Consm':>{col_w}} | "
        f"{'Ramp':>{col_w}} | "
        f"{'Discom':>{col_w}} | "
        f"{'SafVio':>{col_w}}"
    )
    sep = "-" * len(header)
    if stds is not None:
        print("\n  (values shown as mean±std across seeds)")
    print("\n" + "=" * len(header))
    print("  TABLE I – Normalised Performance (baseline = 1.0; lower is better for cols 1-5)")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, m in metrics.items():
        sd = stds.get(name) if stds else None
        print(_table1_row_stats(name, m, sd))
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

    if args.strict_paper_mode:
        validate_strict_paper_mode(env, context="evaluation")
        print("[eval] Strict paper mode enabled")

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
        quick_train(
            bagent,
            STEMSEnvironment(schema=args.schema, seed=bseed),
            config,
            episodes=args.baseline_train_episodes,
        )

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
        per_ep = [
            run_episode(agent, env, config, explore=False).compute_all()
            for _ in range(max(1, args.episodes))
        ]
        normal_raw[name] = _mean_metrics(per_ep)

    # Normalise against RuleBased baseline
    baseline = normal_raw.get("RuleBased", {})
    normal_norm: Dict[str, Dict[str, float]] = {}

    # Normalise manually
    for name, m in normal_raw.items():
        norm_m = dict(m)
        for key in ["cost", "emission", "avg_daily_peak", "electricity_consumption", "ramping_rate"]:
            base_val = baseline.get(key, 1.0)
            norm_m[key] = m[key] / base_val if abs(base_val) > 1e-10 else 1.0
        normal_norm[name] = norm_m

    print_table1(normal_norm)

    # ---- extreme weather ----
    if args.strict_paper_mode:
        print("\n[eval] Strict paper mode: skipping synthetic extreme-weather evaluation.")
        print("[eval] Table II is disabled in strict mode to avoid non-paper perturbations.")
    else:
        print("\n[eval] Running extreme weather evaluation ...")

    def run_extreme(temp_offset: float) -> Dict[str, Dict[str, float]]:
        """Run evaluation with forced outdoor temperature offsets."""
        results: Dict[str, Dict[str, float]] = {}
        for name, agent in agents.items():
            xenv = STEMSEnvironment(schema=args.schema, seed=args.seed)
            xenv.set_temp_offset(temp_offset)
            per_ep = [
                run_episode(agent, xenv, config, explore=False).compute_all()
                for _ in range(max(1, args.episodes))
            ]
            results[name] = _mean_metrics(per_ep)
        return results

    heatwave_raw: Dict[str, Dict[str, float]] = {}
    coldwave_raw: Dict[str, Dict[str, float]] = {}
    if not args.strict_paper_mode:
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
    if not args.strict_paper_mode:
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
# Multi-seed aggregation helper
# ---------------------------------------------------------------------------

def _aggregate_seeds(
    results_per_seed: List[Dict[str, Dict[str, float]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Return (means, stds) dicts for each agent across seeds."""
    all_agents = list(results_per_seed[0].keys())
    all_keys = list(results_per_seed[0][all_agents[0]].keys())

    means: Dict[str, Dict[str, float]] = {}
    stds:  Dict[str, Dict[str, float]] = {}
    for agent in all_agents:
        per_key: Dict[str, List[float]] = {k: [] for k in all_keys}
        for seed_result in results_per_seed:
            if agent not in seed_result:
                continue
            for k in all_keys:
                per_key[k].append(seed_result[agent].get(k, 0.0))
        means[agent] = {k: float(np.mean(per_key[k])) for k in all_keys}
        stds[agent]  = {k: float(np.std(per_key[k], ddof=1) if len(per_key[k]) > 1 else 0.0)
                        for k in all_keys}
    return means, stds


def evaluate_multiseed(args: argparse.Namespace) -> None:
    """Run evaluation for multiple training seeds and report mean ± std."""
    seeds: List[int] = args.seeds
    config = STEMSConfig()

    # We keep baselines constant (train once, evaluate once per seed iteration)
    env0 = STEMSEnvironment(schema=args.schema, seed=args.seed)
    B = env0.num_buildings
    if args.strict_paper_mode:
        validate_strict_paper_mode(env0, context="multiseed evaluation")
        print("[eval] Strict paper mode enabled")
    print(f"[eval] Multi-seed STEMS evaluation over seeds {seeds}")
    print(f"[eval] Environment: {'mock' if env0.using_mock else 'CityLearn'}, "
          f"buildings={B}, obs_dim={env0.obs_dim}")

    # Train baselines once
    rule_agent   = RuleBasedAgent(num_buildings=B)
    sac_agent    = _make_sac(env0)
    ppo_agent    = _make_ppo(env0)
    mpc_agent    = MPCAgent(
        num_buildings=B, action_dim=env0.action_dim,
        soc_min=config.cbf.SOC_min, soc_max=config.cbf.SOC_max,
        P_building_max=config.cbf.P_building_max, P_grid_max=config.cbf.P_grid_max,
    )
    maddpg_agent  = MADDPGAgent(obs_dim=env0.obs_dim, action_dim=env0.action_dim, num_buildings=B)
    marlisa_agent = MARLISAAgent(obs_dim=env0.obs_dim, action_dim=env0.action_dim, num_buildings=B)
    madcq_agent   = MADCQAgent(
        obs_dim=env0.obs_dim, action_dim=env0.action_dim, num_buildings=B,
        soc_min=config.cbf.SOC_min, soc_max=config.cbf.SOC_max,
    )
    metaems_agent = MetaEMSAgent(obs_dim=env0.obs_dim, action_dim=env0.action_dim, num_buildings=B)

    for bname, (bagent, bseed) in {
        "SingleSAC": (sac_agent,  args.seed + 1),
        "DMAPPO":    (ppo_agent,  args.seed + 2),
        "MADDPG":    (maddpg_agent, args.seed + 3),
        "MARLISA":   (marlisa_agent, args.seed + 4),
        "MADCQ":     (madcq_agent,   args.seed + 5),
        "MetaEMS":   (metaems_agent, args.seed + 6),
    }.items():
        print(f"[eval] Training {bname} baseline ...")
        quick_train(
            bagent,
            STEMSEnvironment(schema=args.schema, seed=bseed),
            config,
            episodes=args.baseline_train_episodes,
        )

    # Shared baseline agents evaluated once (against seed 0 env)
    baseline_agents = {
        "RuleBased": rule_agent,
        "MPC":        mpc_agent,
        "SingleSAC":  sac_agent,
        "MADDPG":     maddpg_agent,
        "MARLISA":    marlisa_agent,
        "MADCQ":      madcq_agent,
        "DMAPPO":     ppo_agent,
        "MetaEMS":    metaems_agent,
    }
    baseline_raw: Dict[str, Dict[str, float]] = {}
    for bname, bagent in baseline_agents.items():
        per_ep = [
            run_episode(bagent, env0, config, explore=False).compute_all()
            for _ in range(max(1, args.episodes))
        ]
        baseline_raw[bname] = _mean_metrics(per_ep)

    # Evaluate STEMS for each seed
    stems_per_seed: List[Dict[str, float]] = []
    for s in seeds:
        ckpt = os.path.join("checkpoints", f"seed{s}", "best")
        if not os.path.isdir(ckpt):
            ckpt = args.checkpoint   # fall back to default checkpoint
            print(f"[eval] seed{s}: checkpoint not found at expected path, using {ckpt}")
        env_s = STEMSEnvironment(schema=args.schema, seed=s)
        stems = _make_stems(env_s, ckpt)
        per_ep = [
            run_episode(stems, env_s, config, explore=False).compute_all()
            for _ in range(max(1, args.episodes))
        ]
        stems_per_seed.append(_mean_metrics(per_ep))
        print(f"[eval] seed{s}: cost={stems_per_seed[-1].get('cost',0):.3f}")

    # Aggregate STEMS statistics
    all_keys = list(stems_per_seed[0].keys())
    stems_mean = {k: float(np.mean([r[k] for r in stems_per_seed])) for k in all_keys}
    stems_std  = {k: float(np.std([r[k] for r in stems_per_seed], ddof=1)
                           if len(stems_per_seed) > 1 else 0.0) for k in all_keys}

    # Merge: STEMS with stats + baselines
    normal_raw: Dict[str, Dict[str, float]] = {"STEMS": stems_mean, **baseline_raw}

    # Normalise against RuleBased
    baseline_vals = baseline_raw.get("RuleBased", {})
    normal_norm: Dict[str, Dict[str, float]] = {}
    normal_stds: Dict[str, Dict[str, float]] = {}

    for name, m in normal_raw.items():
        norm_m = dict(m)
        norm_std: Dict[str, float] = {}
        sd_src = stems_std if name == "STEMS" else {}
        for key in ["cost", "emission", "avg_daily_peak", "electricity_consumption", "ramping_rate"]:
            bv = baseline_vals.get(key, 1.0)
            if abs(bv) > 1e-10:
                norm_m[key] = m[key] / bv
                if key in sd_src:
                    norm_std[key] = sd_src[key] / abs(bv)
            else:
                norm_m[key] = 1.0
                norm_std[key] = 0.0
        # Non-normalised metrics
        for key in ["discomfort_rate", "safety_violation_rate"]:
            norm_std[key] = sd_src.get(key, 0.0)
        normal_norm[name] = norm_m
        normal_stds[name] = norm_std

    print_table1(normal_norm, stds=normal_stds)

    # Save aggregated results
    save_dir = args.checkpoint
    os.makedirs(save_dir, exist_ok=True)
    agg_path = os.path.join(save_dir, "eval_multiseed_results.json")
    with open(agg_path, "w") as f:
        json.dump(
            {
                "means": {n: {k: round(v, 4) for k, v in m.items()}
                          for n, m in normal_norm.items()},
                "stds":  {n: {k: round(v, 4) for k, v in sd.items()}
                          for n, sd in normal_stds.items()},
                "seeds": seeds,
            },
            f, indent=2,
        )
    print(f"\n[eval] Multi-seed aggregated results saved to {agg_path}")
    print("[eval] Evaluation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _args = parse_args()
    if _args.seeds:
        evaluate_multiseed(_args)
    else:
        evaluate(_args)
