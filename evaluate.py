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
import subprocess
import sys
from datetime import datetime, timezone
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
from stems.hierarchical import HierarchicalSTEMSAgent
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
    p.add_argument("--episodes", type=int, default=None,
                   help="Evaluation episodes per agent (default: 3, or 1 in paper mode)")
    p.add_argument(
        "--baseline-train-episodes", type=int, default=50,
        help="Training episodes for learnable baselines before evaluation (should match STEMS training episodes for fair comparison)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--schema", type=str, default=None, help="CityLearn schema name or path")
    p.add_argument(
        "--mode",
        choices=["standard", "paper", "extended-safety"],
        default="standard",
        help=(
            "Evaluation protocol: standard keeps the current workflow, paper enforces "
            "paper-comparable settings, extended-safety enables added safety modules."
        ),
    )
    p.add_argument(
        "--paper-reproduction",
        action="store_true",
        help="Alias for --mode paper; enforces real CityLearn, 5 seeds, no extensions.",
    )
    p.add_argument(
        "--extended-safety",
        action="store_true",
        help="Alias for --mode extended-safety; enables neural-filter reporting when available.",
    )
    p.add_argument(
        "--enable-neural-filter",
        action="store_true",
        help="Enable a loaded neural safety filter during STEMS evaluation if neural_filter.pt exists.",
    )
    p.add_argument(
        "--table2-outage",
        action="store_true",
        help="Run paper-style outage robustness Table II with unserved energy rate.",
    )
    p.add_argument(
        "--skip-legacy-extreme-weather",
        action="store_true",
        help="Skip the older synthetic temperature-offset cost table in standard mode.",
    )
    p.add_argument(
        "--no-baseline-training",
        action="store_true",
        help="Evaluate learnable baselines without the quick local training pass.",
    )
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


PAPER_SEEDS = [0, 1, 2, 3, 4]
OUTAGE_SCENARIOS = ["year_round", "heat_wave", "cold_wave"]


def configure_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply protocol defaults after parsing CLI arguments."""
    if args.paper_reproduction:
        args.mode = "paper"
    if args.extended_safety:
        args.mode = "extended-safety"

    if args.mode == "paper":
        args.strict_paper_mode = True
        args.table2_outage = True
        args.skip_legacy_extreme_weather = True
        args.enable_neural_filter = False
        if args.seeds is None:
            args.seeds = PAPER_SEEDS.copy()
        if args.episodes is None:
            args.episodes = 1
    else:
        if args.episodes is None:
            args.episodes = 3
        if args.mode == "extended-safety":
            args.enable_neural_filter = True
            args.table2_outage = True

    return args


# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------

def _make_stems(
    env: STEMSEnvironment,
    checkpoint: str,
    enable_neural_filter: bool = False,
) -> STEMSAgent:
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
        electrical_storage_action_index=env.electrical_storage_action_index,
    )
    if os.path.isdir(checkpoint) and os.path.exists(os.path.join(checkpoint, "encoder.pt")):
        agent.load(checkpoint)
        print(f"[eval] Loaded STEMS checkpoint from {checkpoint}")
        nf_path = os.path.join(checkpoint, "neural_filter.pt")
        if enable_neural_filter:
            if os.path.exists(nf_path):
                agent.use_neural_filter = True
                print("[eval] Neural safety filter enabled for STEMS")
            else:
                print("[eval] Neural filter requested but neural_filter.pt was not found; using CBF QP")
    else:
        print("[eval] No checkpoint found – using untrained STEMS weights (run train.py first for best results)")
    return agent


def _make_hierarchical(
    env: STEMSEnvironment, checkpoint: str
) -> HierarchicalSTEMSAgent:
    config = STEMSConfig()
    B = env.num_buildings
    adj = None
    try:
        info  = env.get_building_info()
        graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
        adj   = graph.compute_edge_weights().numpy()
    except Exception:
        pass
    agent = HierarchicalSTEMSAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=B,
        adj=adj,
        use_cbf=True,
        electrical_storage_action_index=env.electrical_storage_action_index,
    )
    hier_ckpt = os.path.join(checkpoint, "hierarchical", "best")
    if os.path.isdir(hier_ckpt) and os.path.exists(
        os.path.join(hier_ckpt, "hier_encoder.pt")
    ):
        agent.load(hier_ckpt)
        print(f"[eval] Loaded Hierarchical checkpoint from {hier_ckpt}")
    else:
        print("[eval] No Hierarchical checkpoint – using untrained weights (run train_hierarchical.py first)")
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
    if hasattr(agent, "event_trigger"):
        agent.event_trigger.reset()
    if hasattr(agent, "_cached_cluster_latents"):
        agent._cached_cluster_latents = None
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


def _std_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Sample standard deviation for a list of metric dictionaries."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: float(np.std([m.get(k, 0.0) for m in metrics_list], ddof=1))
        if len(metrics_list) > 1 else 0.0
        for k in keys
    }


def _normalise_against_rulebased(
    raw_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Normalise Table I metrics against the RuleBased row."""
    baseline = raw_metrics.get("RuleBased", {})
    normalised: Dict[str, Dict[str, float]] = {}
    for name, metrics in raw_metrics.items():
        norm_m = dict(metrics)
        for key in ["cost", "emission", "avg_daily_peak",
                    "electricity_consumption", "ramping_rate"]:
            base_val = float(baseline.get(key, 1.0))
            norm_m[key] = metrics[key] / base_val if abs(base_val) > 1e-10 else 1.0
        normalised[name] = norm_m
    return normalised


def _git_commit() -> str:
    """Return the current git commit hash if available."""
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_eval_manifest(
    args: argparse.Namespace,
    env: STEMSEnvironment,
    output_dir: str,
    agents: List[str],
    seeds: Optional[List[int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the protocol metadata needed to interpret evaluation results."""
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "schema": args.schema or env.SCHEMA,
        "environment": "mock" if env.using_mock else "CityLearn",
        "using_mock": env.using_mock,
        "num_buildings": env.num_buildings,
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim,
        "seed": args.seed,
        "seeds": seeds if seeds is not None else args.seeds,
        "episodes": args.episodes,
        "baseline_train_episodes": args.baseline_train_episodes,
        "strict_paper_mode": args.strict_paper_mode,
        "neural_filter_enabled": args.enable_neural_filter,
        "table2_outage": args.table2_outage,
        "agents": agents,
        "git_commit": _git_commit(),
    }
    if extra:
        manifest.update(extra)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "eval_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[eval] Manifest saved to {manifest_path}")


def _build_agents(
    env: STEMSEnvironment,
    args: argparse.Namespace,
    config: STEMSConfig,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[Any, int]]]:
    """Construct evaluation agents and the learnable-baseline training plan."""
    B = env.num_buildings
    stems_agent = _make_stems(
        env, args.checkpoint,
        enable_neural_filter=args.enable_neural_filter,
    )
    rule_agent = RuleBasedAgent(num_buildings=B)
    sac_agent = _make_sac(env)
    ppo_agent = _make_ppo(env)
    mpc_agent = MPCAgent(
        num_buildings=B, action_dim=env.action_dim,
        soc_min=config.cbf.SOC_min, soc_max=config.cbf.SOC_max,
        P_building_max=config.cbf.P_building_max,
        P_grid_max=config.cbf.P_grid_max,
    )
    maddpg_agent = MADDPGAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B)
    marlisa_agent = MARLISAAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B)
    madcq_agent = MADCQAgent(
        obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B,
        soc_min=config.cbf.SOC_min, soc_max=config.cbf.SOC_max,
    )
    metaems_agent = MetaEMSAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, num_buildings=B)

    agents: Dict[str, Any] = {"STEMS": stems_agent}
    if args.mode != "paper":
        agents["HierSTEMS"] = _make_hierarchical(env, args.checkpoint)
    agents.update({
        "RuleBased": rule_agent,
        "MPC": mpc_agent,
        "SingleSAC": sac_agent,
        "MADDPG": maddpg_agent,
        "MARLISA": marlisa_agent,
        "MADCQ": madcq_agent,
        "DMAPPO": ppo_agent,
        "MetaEMS": metaems_agent,
    })

    learnable_baselines = {
        "SingleSAC": (sac_agent, args.seed + 1),
        "DMAPPO": (ppo_agent, args.seed + 2),
        "MADDPG": (maddpg_agent, args.seed + 3),
        "MARLISA": (marlisa_agent, args.seed + 4),
        "MADCQ": (madcq_agent, args.seed + 5),
        "MetaEMS": (metaems_agent, args.seed + 6),
    }
    return agents, learnable_baselines


def _train_learnable_baselines(
    learnable_baselines: Dict[str, Tuple[Any, int]],
    args: argparse.Namespace,
    config: STEMSConfig,
) -> None:
    """Train baselines with the same local budget before evaluation."""
    if args.no_baseline_training:
        print("[eval] Skipping quick baseline training (--no-baseline-training)")
        return
    for bname, (bagent, bseed) in learnable_baselines.items():
        print(f"[eval] Training {bname} baseline ...")
        quick_train(
            bagent,
            STEMSEnvironment(schema=args.schema, seed=bseed),
            config,
            episodes=args.baseline_train_episodes,
        )


# ---------------------------------------------------------------------------
# Quick training for SAC/PPO baselines
# ---------------------------------------------------------------------------

# Off-policy agents use a replay buffer with per-step mini-batch updates;
# on-policy agents collect a full episode then update once.
_OFF_POLICY_TYPES: tuple = (SingleAgentSAC, MADDPGAgent, MARLISAAgent, MADCQAgent)

_REPLAY_CAPACITY = 50_000
_REPLAY_BATCH    = 256
_WARMUP_STEPS    = 512   # fill buffer before first update


def quick_train(
    agent: Any,
    env: STEMSEnvironment,
    config: STEMSConfig,
    episodes: int = 3,
) -> None:
    """Short training run so baselines have learned something.

    Off-policy agents (SingleSAC, MADDPG, MARLISA, MADCQ) now use a proper
    replay buffer with random mini-batch sampling at every environment step,
    instead of collecting a full episode on-policy and calling update once.
    This was the missing piece that prevented these agents from learning.
    """
    is_off_policy = isinstance(agent, _OFF_POLICY_TYPES)

    history_buf = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    reward_fn = STEMSReward(
        config=config.reward,
        num_buildings=env.num_buildings,
        P_grid_max=config.cbf.P_grid_max,
        P_building_max=config.cbf.P_building_max,
    )

    if is_off_policy:
        replay = ReplayBuffer(capacity=_REPLAY_CAPACITY)
    else:
        episode_buffer = EpisodeBuffer()

    total_steps = 0

    for _ in range(episodes):
        obs_list, _ = env.reset()
        history_buf.reset()
        history_buf.update(obs_list)
        if not is_off_policy:
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
            raw_actions = getattr(agent, "_last_raw_actions", actions)

            if is_off_policy:
                replay.add(
                    obs=obs_list, actions=actions, rewards=rewards,
                    next_obs=next_obs, done=done, history=obs_window,
                    next_history=next_obs_window, raw_actions=raw_actions,
                )
                total_steps += 1
                # Sample a random mini-batch once the buffer has enough transitions
                if total_steps >= _WARMUP_STEPS and replay.is_ready:
                    agent.update(replay.sample(_REPLAY_BATCH))
            else:
                episode_buffer.add(
                    obs=obs_list, actions=actions, rewards=rewards,
                    next_obs=next_obs, done=done, history=obs_window,
                    next_history=next_obs_window, raw_actions=raw_actions,
                )
            obs_list = next_obs

        # On-policy agents update at episode end with the full trajectory
        if not is_off_policy:
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


def print_extreme_cost_table(
    normal: Dict[str, Dict[str, float]],
    heatwave: Dict[str, Dict[str, float]],
    coldwave: Dict[str, Dict[str, float]],
) -> None:
    print("\n" + "=" * 80)
    print("  LEGACY EXTREME WEATHER – Synthetic Temperature Offset Costs")
    print("=" * 80)
    print(f"  {'Agent':<20s} | {'Normal':>10} | {'HeatWave':>10} | {'ColdWave':>10}")
    print("-" * 60)
    for name in normal:
        nc = normal[name].get("cost", 0)
        hc = heatwave.get(name, {}).get("cost", 0)
        cc = coldwave.get(name, {}).get("cost", 0)
        print(f"  {name:<20s} | {nc:10.1f} | {hc:10.1f} | {cc:10.1f}")
    print("-" * 60)


def _outage_eligible(obs_list: List[np.ndarray], scenario: str) -> bool:
    """Return True when the current weather matches a Table II scenario."""
    t_out = float(np.mean([obs[2] for obs in obs_list]))
    if scenario == "heat_wave":
        return t_out >= 30.0
    if scenario == "cold_wave":
        return t_out <= 0.0
    return True


def _estimate_unserved_energy(
    obs_list: List[np.ndarray],
    config: STEMSConfig,
) -> Tuple[float, float]:
    """Estimate unmet demand during a grid outage from CityLearn observations.

    CityLearn does not expose a universal outage hook through the wrapper, so
    Table II is computed as an evaluation overlay: positive grid import during
    an outage is treated as unmet load, normalized by building demand.  The
    denominator uses non-shiftable, cooling, and DHW demand when available.
    """
    unserved = 0.0
    demand = 0.0
    for obs in obs_list:
        non_shiftable = max(0.0, float(obs[16]))
        cooling = max(0.0, float(obs[24])) if len(obs) > 24 else 0.0
        dhw = max(0.0, float(obs[25])) if len(obs) > 25 else 0.0
        total_demand = max(non_shiftable + cooling + dhw, 1e-6)
        grid_import = max(0.0, float(obs[20]))
        unserved += min(total_demand, grid_import)
        demand += total_demand
    return unserved, demand


def run_outage_episode(
    agent: Any,
    env: STEMSEnvironment,
    config: STEMSConfig,
    scenario: str,
    seed: int,
) -> Dict[str, float]:
    """Evaluate one episode with stochastic paper-style outage overlay."""
    calc = MetricsCalculator(env.num_buildings, config.cbf)
    if hasattr(agent, "event_trigger"):
        agent.event_trigger.reset()
    if hasattr(agent, "_cached_cluster_latents"):
        agent._cached_cluster_latents = None
    history_buf = HistoryBuffer(
        num_buildings=env.num_buildings,
        obs_dim=env.obs_dim,
        window_size=config.transformer.window_size,
    )
    rng = np.random.default_rng(seed)
    # Paper uses SAIFI=1.436 events/year and CAIDI=331.2 minutes/event.
    expected_events_per_year = 1.436
    outage_duration_steps = int(np.ceil(331.2 / 60.0))
    scenario_fraction = {
        "year_round": 1.0,
        "heat_wave": 0.148,
        "cold_wave": 0.012,
    }.get(scenario, 1.0)
    eligible_hours = max(1.0, 8760.0 * scenario_fraction)
    start_probability = min(1.0, expected_events_per_year / eligible_hours)
    outage_remaining = 0
    outage_steps = 0
    outage_events = 0
    unserved_total = 0.0
    demand_total = 0.0

    obs_list, _ = env.reset()
    history_buf.update(obs_list)
    done = False
    while not done:
        eligible = _outage_eligible(obs_list, scenario)
        if outage_remaining <= 0 and eligible and rng.random() < start_probability:
            outage_remaining = outage_duration_steps
            outage_events += 1
        outage_active = outage_remaining > 0 and eligible

        history = history_buf.get()
        actions = agent.select_action(obs_list, history, explore=False)
        next_obs_list, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        calc.add_step(obs_list, actions, next_obs_list)

        if outage_active:
            unserved, demand = _estimate_unserved_energy(next_obs_list, config)
            unserved_total += unserved
            demand_total += demand
            outage_steps += 1
            outage_remaining -= 1

        obs_list = next_obs_list
        history_buf.update(obs_list)

    metrics = calc.compute_all()
    return {
        "unserved_energy_rate": float(unserved_total / demand_total) if demand_total > 1e-10 else 0.0,
        "safety_violation_rate": metrics.get("safety_violation_rate", 0.0),
        "outage_steps": float(outage_steps),
        "outage_events": float(outage_events),
    }


def run_outage_table2(
    agents: Dict[str, Any],
    args: argparse.Namespace,
    config: STEMSConfig,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run outage robustness evaluation for all agents and scenarios."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario_idx, scenario in enumerate(OUTAGE_SCENARIOS):
        scenario_results: Dict[str, Dict[str, float]] = {}
        for agent_idx, (name, agent) in enumerate(agents.items()):
            per_ep = []
            for ep in range(max(1, args.episodes)):
                env_seed = args.seed + 10_000 + 1_000 * scenario_idx + ep
                xenv = STEMSEnvironment(schema=args.schema, seed=env_seed)
                per_ep.append(run_outage_episode(
                    agent, xenv, config, scenario,
                    seed=env_seed + 97 * (agent_idx + 1),
                ))
            scenario_results[name] = _mean_metrics(per_ep)
        results[scenario] = scenario_results
    return results


def print_table2(outage_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Print paper-style Table II outage robustness results."""
    agents = list(next(iter(outage_results.values())).keys()) if outage_results else []
    header = f"  {'Scenario':<12s} | {'Metric':<22s} | " + " | ".join(
        f"{name:>10s}" for name in agents
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  TABLE II – Power Outage Robustness under Different Weather Conditions")
    print("=" * len(header))
    print(header)
    print(sep)
    labels = {
        "year_round": "Year-round",
        "heat_wave": "Heat Wave",
        "cold_wave": "Cold Wave",
    }
    for scenario in OUTAGE_SCENARIOS:
        rows = outage_results.get(scenario, {})
        unserved = [rows.get(agent, {}).get("unserved_energy_rate", 0.0) for agent in agents]
        safety = [rows.get(agent, {}).get("safety_violation_rate", 0.0) for agent in agents]
        print(f"  {labels.get(scenario, scenario):<12s} | {'Unserved Energy Rate':<22s} | "
              + " | ".join(f"{v:10.3f}" for v in unserved))
        print(f"  {'':<12s} | {'Safety Viol. Rate':<22s} | "
              + " | ".join(f"{v:10.3f}" for v in safety))
    print(sep)


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
    agents, learnable_baselines = _build_agents(env, args, config)
    _train_learnable_baselines(learnable_baselines, args, config)

    # ---- normal evaluation ----
    normal_raw: Dict[str, Dict[str, float]] = {}
    for name, agent in agents.items():
        print(f"[eval] Running {name} ...")
        per_ep = [
            run_episode(agent, env, config, explore=False).compute_all()
            for _ in range(max(1, args.episodes))
        ]
        normal_raw[name] = _mean_metrics(per_ep)

    normal_norm = _normalise_against_rulebased(normal_raw)

    print_table1(normal_norm)

    # ---- paper Table II outage robustness ----
    outage_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    if args.table2_outage:
        print("\n[eval] Running paper-style outage Table II ...")
        outage_raw = run_outage_table2(agents, args, config)
        print_table2(outage_raw)

    # ---- legacy synthetic extreme weather ----
    run_legacy_extreme = (
        not args.strict_paper_mode
        and not args.skip_legacy_extreme_weather
        and args.mode == "standard"
    )
    if run_legacy_extreme:
        print("\n[eval] Running legacy synthetic temperature-offset evaluation ...")
    elif args.strict_paper_mode:
        print("\n[eval] Strict/paper mode: skipping synthetic temperature-offset evaluation.")

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
    if run_legacy_extreme:
        heatwave_raw = run_extreme(temp_offset=10.0)
        coldwave_raw = run_extreme(temp_offset=-10.0)
        print_extreme_cost_table(normal_raw, heatwave_raw, coldwave_raw)

    # Save evaluation results to JSON for visualize.py
    eval_output = {}
    for name, m in normal_norm.items():
        eval_output[name] = {k: round(v, 4) for k, v in m.items()}
    eval_path = os.path.join(args.checkpoint, "eval_results.json")
    os.makedirs(args.checkpoint, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n[eval] Normalised results saved to {eval_path}")

    # Save paper Table II outage results to JSON
    if outage_raw:
        table2_path = os.path.join(args.checkpoint, "eval_outage_results.json")
        with open(table2_path, "w") as f:
            json.dump(outage_raw, f, indent=2)
        print(f"[eval] Outage results saved to {table2_path}")

    # Save legacy extreme weather absolute costs to JSON
    if run_legacy_extreme:
        table2_output = {
            "normal": {n: round(v.get("cost", 0), 2) for n, v in normal_raw.items()},
            "heatwave": {n: round(v.get("cost", 0), 2) for n, v in heatwave_raw.items()},
            "coldwave": {n: round(v.get("cost", 0), 2) for n, v in coldwave_raw.items()},
        }
        legacy_path = os.path.join(args.checkpoint, "eval_extreme_results.json")
        with open(legacy_path, "w") as f:
            json.dump(table2_output, f, indent=2)
        print(f"[eval] Legacy extreme weather results saved to {legacy_path}")

    _write_eval_manifest(
        args, env, args.checkpoint, list(agents.keys()),
        extra={
            "result_files": {
                "table_i": "eval_results.json",
                "table_ii_outage": "eval_outage_results.json" if outage_raw else None,
                "legacy_extreme": "eval_extreme_results.json" if run_legacy_extreme else None,
            }
        },
    )

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


def _aggregate_outage_tables(
    outage_per_seed: List[Dict[str, Dict[str, Dict[str, float]]]],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Aggregate nested Table II outage dictionaries across seeds."""
    if not outage_per_seed:
        return {}, {}
    means: Dict[str, Dict[str, Dict[str, float]]] = {}
    stds: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario in outage_per_seed[0].keys():
        means[scenario] = {}
        stds[scenario] = {}
        for agent in outage_per_seed[0][scenario].keys():
            means[scenario][agent] = {}
            stds[scenario][agent] = {}
            for metric in outage_per_seed[0][scenario][agent].keys():
                vals = [table[scenario][agent].get(metric, 0.0) for table in outage_per_seed]
                means[scenario][agent][metric] = float(np.mean(vals))
                stds[scenario][agent][metric] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return means, stds


def evaluate_multiseed(args: argparse.Namespace) -> None:
    """Run a fair multi-seed evaluation for every reported agent."""
    seeds: List[int] = args.seeds
    config = STEMSConfig()
    original_seed = args.seed
    original_checkpoint = args.checkpoint
    per_seed_norm: List[Dict[str, Dict[str, float]]] = []
    per_seed_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    outage_per_seed: List[Dict[str, Dict[str, Dict[str, float]]]] = []
    manifest_env: Optional[STEMSEnvironment] = None

    print(f"[eval] Multi-seed evaluation over seeds {seeds}")

    for seed in seeds:
        print(f"\n[eval] === Seed {seed} ===")
        set_seed(seed)
        args.seed = seed
        seed_ckpt = os.path.join(original_checkpoint, f"seed{seed}", "best")
        if os.path.isdir(seed_ckpt):
            args.checkpoint = seed_ckpt
        elif len(seeds) == 1 and os.path.exists(os.path.join(original_checkpoint, "encoder.pt")):
            args.checkpoint = original_checkpoint
        else:
            if args.strict_paper_mode:
                raise FileNotFoundError(
                    f"Missing paper-mode STEMS checkpoint for seed {seed}: {seed_ckpt}\n"
                    "Train every requested seed first, for example: "
                    f"python train.py --paper-reproduction --episodes 15 --save-dir {os.path.join(original_checkpoint, f'seed{seed}')} --seed {seed}\n"
                    "For a single-seed smoke evaluation, add `--seeds 0` and point --checkpoint at that seed's best checkpoint."
                )
            args.checkpoint = original_checkpoint
            print(f"[eval] seed{seed}: checkpoint not found at {seed_ckpt}; using {args.checkpoint}")

        env = STEMSEnvironment(schema=args.schema, seed=seed)
        if manifest_env is None:
            manifest_env = env
        if args.strict_paper_mode:
            validate_strict_paper_mode(env, context=f"multiseed evaluation seed {seed}")
        print(f"[eval] Environment: {'mock' if env.using_mock else 'CityLearn'}, "
              f"buildings={env.num_buildings}, obs_dim={env.obs_dim}")

        agents, learnable_baselines = _build_agents(env, args, config)
        _train_learnable_baselines(learnable_baselines, args, config)

        raw_metrics: Dict[str, Dict[str, float]] = {}
        for name, agent in agents.items():
            print(f"[eval] Running {name} (seed {seed}) ...")
            per_ep = []
            for ep in range(max(1, args.episodes)):
                ep_env = STEMSEnvironment(schema=args.schema, seed=seed + ep)
                if args.strict_paper_mode:
                    validate_strict_paper_mode(ep_env, context=f"seed {seed} episode {ep}")
                per_ep.append(run_episode(agent, ep_env, config, explore=False).compute_all())
            raw_metrics[name] = _mean_metrics(per_ep)

        per_seed_raw[str(seed)] = raw_metrics
        seed_norm = _normalise_against_rulebased(raw_metrics)
        per_seed_norm.append(seed_norm)
        print(f"[eval] seed{seed}: STEMS cost={seed_norm['STEMS'].get('cost', 0):.3f}, "
              f"safety={seed_norm['STEMS'].get('safety_violation_rate', 0):.3f}")

        if args.table2_outage:
            print(f"[eval] Running outage Table II for seed {seed} ...")
            outage_per_seed.append(run_outage_table2(agents, args, config))

    args.seed = original_seed
    args.checkpoint = original_checkpoint

    normal_means, normal_stds = _aggregate_seeds(per_seed_norm)
    print_table1(normal_means, stds=normal_stds)

    outage_means: Dict[str, Dict[str, Dict[str, float]]] = {}
    outage_stds: Dict[str, Dict[str, Dict[str, float]]] = {}
    if outage_per_seed:
        outage_means, outage_stds = _aggregate_outage_tables(outage_per_seed)
        print_table2(outage_means)

    save_dir = original_checkpoint
    os.makedirs(save_dir, exist_ok=True)
    agg_path = os.path.join(save_dir, "eval_multiseed_results.json")
    with open(agg_path, "w") as f:
        json.dump(
            {
                "means": {n: {k: round(v, 4) for k, v in m.items()}
                          for n, m in normal_means.items()},
                "stds": {n: {k: round(v, 4) for k, v in sd.items()}
                         for n, sd in normal_stds.items()},
                "per_seed_raw": per_seed_raw,
                "seeds": seeds,
            },
            f, indent=2,
        )
    print(f"\n[eval] Multi-seed aggregated results saved to {agg_path}")

    if outage_means:
        outage_path = os.path.join(save_dir, "eval_outage_multiseed_results.json")
        with open(outage_path, "w") as f:
            json.dump({"means": outage_means, "stds": outage_stds, "seeds": seeds}, f, indent=2)
        print(f"[eval] Multi-seed outage results saved to {outage_path}")

    if manifest_env is None:
        manifest_env = STEMSEnvironment(schema=args.schema, seed=original_seed)
    _write_eval_manifest(
        args, manifest_env, save_dir,
        list(normal_means.keys()), seeds=seeds,
        extra={
            "result_files": {
                "table_i_multiseed": "eval_multiseed_results.json",
                "table_ii_outage_multiseed": "eval_outage_multiseed_results.json" if outage_means else None,
            }
        },
    )
    print("[eval] Evaluation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        _args = configure_protocol_args(parse_args())
        if _args.seeds:
            evaluate_multiseed(_args)
        else:
            evaluate(_args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[eval] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
