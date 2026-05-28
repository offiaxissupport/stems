#!/usr/bin/env python3
"""
Ablation study comparing STEMS variants (Table IV in the paper).

Usage:
    python ablation.py [--episodes 15] [--load-full-checkpoint] [--strict-paper-mode]

Variants:
    Full STEMS            – complete architecture with all components
    w/o GCN-Transformer   – replaces encoder with identity (raw observations)
    w/o Spatial Graph     – removes GCN, uses only Temporal Transformer
    w/o Temporal Attention – removes Transformer, uses only GCN
    w/o CBF Safety        – full encoder but no safety shield
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent, Actor, Critic
from stems.encoder import STEncoder, SpatialGCN, TemporalTransformer
from stems.cbf import CBFShield
from stems.metrics import MetricsCalculator
from stems.paper_mode import validate_strict_paper_mode
from stems.reward import STEMSReward
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STEMS ablation study (Table IV)")
    p.add_argument("--episodes", type=int, default=15,
                   help="Training episodes per ablated variant (default 15, same as main training)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schema", type=str, default=None,
                   help="CityLearn schema name or path")
    p.add_argument("--checkpoint", type=str, default="checkpoints/seed1/best/",
                   help="Pre-trained Full STEMS checkpoint; ablated variants train from scratch")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds for fair mean±std ablation aggregation")
    p.add_argument("--fair", action="store_true",
                   help="Train every variant from scratch with the same episode budget")
    p.add_argument("--paper-reproduction", action="store_true",
                   help="Strict paper ablation protocol: real CityLearn, 5 seeds, fair training")
    p.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help=(
            "Enforce paper-relatable protocol: fail on mock env and require 8-building setup."
        ),
    )
    p.add_argument(
        "--load-full-checkpoint",
        action="store_true",
        help=(
            "Load pre-trained Full STEMS checkpoint instead of training it from scratch. "
            "Disable for a strictly fair ablation protocol where all variants train equally."
        ),
    )
    return p.parse_args()


PAPER_SEEDS = [0, 1, 2, 3, 4]


def configure_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply ablation protocol defaults."""
    if args.paper_reproduction:
        args.strict_paper_mode = True
        args.fair = True
        args.load_full_checkpoint = False
        if args.seeds is None:
            args.seeds = PAPER_SEEDS.copy()
    elif args.fair:
        args.load_full_checkpoint = False
        if args.seeds is None:
            args.seeds = [args.seed]
    return args


# ---------------------------------------------------------------------------
# Ablated encoder variants
# ---------------------------------------------------------------------------

class _IdentityEncoder(nn.Module):
    """No encoder – passes raw observations through a linear projection."""

    def __init__(self, obs_dim: int, output_dim: int, num_buildings: int) -> None:
        super().__init__()
        self.proj = nn.Linear(obs_dim, output_dim)
        self.out_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        return torch.relu(self.proj(x))   # (B, output_dim)

    def batch_forward(
        self,
        x_nb: torch.Tensor,
        adj: torch.Tensor,
        history_nb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.relu(self.proj(x_nb))   # (N, B, output_dim)


class _GCNOnlyEncoder(nn.Module):
    """Spatial GCN only – no Temporal Transformer."""

    def __init__(self, obs_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.gcn = SpatialGCN(obs_dim, hidden_dim, num_layers)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.out_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        h = self.gcn(x, adj)
        return torch.relu(self.proj(h))

    def batch_forward(
        self,
        x_nb: torch.Tensor,
        adj: torch.Tensor,
        history_nb: torch.Tensor,
    ) -> torch.Tensor:
        N, B, _ = x_nb.shape
        out = []
        for n in range(N):
            out.append(torch.relu(self.proj(self.gcn(x_nb[n], adj))))
        return torch.stack(out, dim=0)   # (N, B, output_dim)


class _TransformerOnlyEncoder(nn.Module):
    """Temporal Transformer only – no Spatial GCN."""

    def __init__(self, obs_dim: int, embed_dim: int, num_heads: int,
                 window_size: int, output_dim: int) -> None:
        super().__init__()
        self.transformer = TemporalTransformer(obs_dim, embed_dim, num_heads, window_size)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.out_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        z = self.transformer(history)
        return torch.relu(self.proj(z))

    def batch_forward(
        self,
        x_nb: torch.Tensor,
        adj: torch.Tensor,
        history_nb: torch.Tensor,
    ) -> torch.Tensor:
        N, B, T, obs_dim = history_nb.shape
        hist_flat = history_nb.view(N * B, T, obs_dim)
        z_flat = self.transformer(hist_flat)    # (N*B, embed_dim)
        z_nb = z_flat.view(N, B, -1)           # (N, B, embed_dim)
        return torch.relu(self.proj(z_nb))


# ---------------------------------------------------------------------------
# Build variant agents
# ---------------------------------------------------------------------------

def _make_base_agent(
    env: STEMSEnvironment,
    config: STEMSConfig,
    use_cbf: bool = True,
    encoder_override: Optional[nn.Module] = None,
) -> STEMSAgent:
    info = env.get_building_info()
    graph = BuildingGraph(
        env.num_buildings, info["positions"], info["features"], config.graph
    )
    agent = STEMSAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=env.num_buildings,
        building_graph=graph,
        config=config,
        use_cbf=use_cbf,
        electrical_storage_action_index=env.electrical_storage_action_index,
    )
    if encoder_override is not None:
        agent.encoder = encoder_override.to(agent.device)
        # Rebuild optimisers with new encoder (3 separate optimisers)
        lr = config.actor_critic.lr
        agent.encoder_optimizer = torch.optim.Adam(
            agent.encoder.parameters(), lr=lr
        )
        agent.actor_optimizer = torch.optim.Adam(
            agent.actors.parameters(), lr=lr
        )
        agent.critic_optimizer = torch.optim.Adam(
            agent.critics.parameters(), lr=lr
        )
    return agent


def build_variants(
    env: STEMSEnvironment, config: STEMSConfig
) -> Dict[str, STEMSAgent]:
    obs_dim = env.obs_dim
    cfg = config

    variants: Dict[str, STEMSAgent] = {}

    # 1. Full STEMS
    variants["Full STEMS"] = _make_base_agent(env, config, use_cbf=True)

    # 2. w/o GCN-Transformer (identity encoder)
    id_enc = _IdentityEncoder(obs_dim, cfg.fusion.output_dim, env.num_buildings)
    variants["w/o GCN-Transformer"] = _make_base_agent(env, config, use_cbf=True,
                                                        encoder_override=id_enc)

    # 3. w/o Spatial Graph (Transformer only)
    t_enc = _TransformerOnlyEncoder(
        obs_dim, cfg.transformer.embed_dim, cfg.transformer.num_heads,
        cfg.transformer.window_size, cfg.fusion.output_dim,
    )
    variants["w/o Spatial Graph"] = _make_base_agent(env, config, use_cbf=True,
                                                      encoder_override=t_enc)

    # 4. w/o Temporal Attention (GCN only)
    g_enc = _GCNOnlyEncoder(obs_dim, cfg.gcn.hidden_dim, cfg.fusion.output_dim,
                             cfg.gcn.num_layers)
    variants["w/o Temporal Attn"] = _make_base_agent(env, config, use_cbf=True,
                                                      encoder_override=g_enc)

    # 5. w/o CBF Safety (full encoder, no shield)
    variants["w/o CBF Safety"] = _make_base_agent(env, config, use_cbf=False)

    return variants


# ---------------------------------------------------------------------------
# Evaluation-only loop (for the pre-trained Full STEMS checkpoint)
# ---------------------------------------------------------------------------

def _eval_only(agent: STEMSAgent, env: STEMSEnvironment, config: STEMSConfig) -> Dict[str, float]:
    """Run one evaluation episode without any training."""
    calc = MetricsCalculator(env.num_buildings, config.cbf)
    hist_buf = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs_list, _ = env.reset()
    hist_buf.reset()
    hist_buf.update(obs_list)
    done = False
    while not done:
        actions = agent.select_action(obs_list, hist_buf.get(), explore=False)
        next_obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        calc.add_step(obs_list, actions, next_obs)
        obs_list = next_obs
        hist_buf.update(obs_list)
    return calc.compute_all()


# ---------------------------------------------------------------------------
# Training + evaluation loop
# ---------------------------------------------------------------------------

def train_and_eval(
    agent: STEMSAgent,
    env: STEMSEnvironment,
    config: STEMSConfig,
    n_episodes: int,
) -> Dict[str, float]:
    """Train agent for n_episodes, then evaluate for 1 episode."""
    reward_fn = STEMSReward(
        config=config.reward,
        num_buildings=env.num_buildings,
        P_grid_max=config.cbf.P_grid_max,
        P_building_max=config.cbf.P_building_max,
    )
    episode_buffer = EpisodeBuffer()
    hist_buf = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)

    # Training (on-policy, Algorithm 2)
    for _ in range(n_episodes):
        obs_list, _ = env.reset()
        hist_buf.reset()
        hist_buf.update(obs_list)
        episode_buffer.reset()
        done = False
        prev_net = [float(o[20]) for o in obs_list]
        while not done:
            obs_window = hist_buf.get()
            actions = agent.select_action(obs_list, obs_window, explore=True)
            next_obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            stems_rewards = reward_fn.compute(obs_list, actions, next_obs, prev_net)
            prev_net = [float(o[20]) for o in next_obs]
            hist_buf.update(next_obs)
            next_obs_window = hist_buf.get()
            episode_buffer.add(
                obs=obs_list, actions=actions, rewards=stems_rewards,
                next_obs=next_obs, done=done, history=obs_window,
                next_history=next_obs_window, raw_actions=agent._last_raw_actions,
            )
            obs_list = next_obs
        # Single on-policy update per episode
        batch = episode_buffer.get_batch()
        agent.update(batch)

    # Evaluation
    calc = MetricsCalculator(env.num_buildings, config.cbf)
    obs_list, _ = env.reset()
    hist_buf.reset()
    hist_buf.update(obs_list)
    done = False
    while not done:
        actions = agent.select_action(obs_list, hist_buf.get(), explore=False)
        next_obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        calc.add_step(obs_list, actions, next_obs)
        obs_list = next_obs
        hist_buf.update(obs_list)

    return calc.compute_all()


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table4(results: Dict[str, Dict[str, float]]) -> None:
    header = (
        f"  {'Variant':<22s} | "
        f"{'Cost':>7} | "
        f"{'Emiss':>7} | "
        f"{'DayPk':>7} | "
        f"{'Consm':>7} | "
        f"{'Ramp':>7} | "
        f"{'Discom':>8} | "
        f"{'SafVio':>8}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  TABLE IV – Ablation Study (absolute metrics)")
    print("=" * len(header))
    print(header)
    print(sep)


def _format_stat(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}"


def print_table4_stats(
    means: Dict[str, Dict[str, float]],
    stds: Dict[str, Dict[str, float]],
) -> None:
    """Print mean±std Table IV for fair multi-seed ablations."""
    header = (
        f"  {'Variant':<22s} | "
        f"{'Cost':>13} | "
        f"{'Emiss':>13} | "
        f"{'DayPk':>13} | "
        f"{'Consm':>13} | "
        f"{'Ramp':>13} | "
        f"{'Discom':>13} | "
        f"{'SafVio':>13}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  TABLE IV – Fair Ablation Study (mean±std across seeds)")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, m in means.items():
        sd = stds.get(name, {})
        row = (
            f"  {name:<22s} | "
            f"{_format_stat(m.get('cost', 0), sd.get('cost', 0)):>13} | "
            f"{_format_stat(m.get('emission', 0), sd.get('emission', 0)):>13} | "
            f"{_format_stat(m.get('avg_daily_peak', 0), sd.get('avg_daily_peak', 0)):>13} | "
            f"{_format_stat(m.get('electricity_consumption', 0), sd.get('electricity_consumption', 0)):>13} | "
            f"{_format_stat(m.get('ramping_rate', 0), sd.get('ramping_rate', 0)):>13} | "
            f"{_format_stat(m.get('discomfort_rate', 0), sd.get('discomfort_rate', 0)):>13} | "
            f"{_format_stat(m.get('safety_violation_rate', 0), sd.get('safety_violation_rate', 0)):>13}"
        )
        print(row)
    print(sep)


def _aggregate_variant_results(
    results_per_seed: List[Dict[str, Dict[str, float]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Return mean/std dictionaries for per-seed ablation results."""
    variants = list(results_per_seed[0].keys())
    metrics = list(results_per_seed[0][variants[0]].keys())
    means: Dict[str, Dict[str, float]] = {}
    stds: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        means[variant] = {}
        stds[variant] = {}
        for metric in metrics:
            vals = [r[variant].get(metric, 0.0) for r in results_per_seed]
            means[variant][metric] = float(np.mean(vals))
            stds[variant][metric] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return means, stds


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_ablation_manifest(
    args: argparse.Namespace,
    env: STEMSEnvironment,
    output_dir: str,
    result_file: str,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "schema": args.schema or env.SCHEMA,
        "environment": "mock" if env.using_mock else "CityLearn",
        "using_mock": env.using_mock,
        "num_buildings": env.num_buildings,
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim,
        "seed": args.seed,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "fair": args.fair,
        "load_full_checkpoint": args.load_full_checkpoint,
        "strict_paper_mode": args.strict_paper_mode,
        "paper_reproduction": args.paper_reproduction,
        "result_file": result_file,
        "git_commit": _git_commit(),
    }
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "ablation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ablation] Manifest saved to {manifest_path}")
    for name, m in results.items():
        row = (
            f"  {name:<22s} | "
            f"{m.get('cost', 0):7.2f} | "
            f"{m.get('emission', 0):7.3f} | "
            f"{m.get('avg_daily_peak', 0):7.2f} | "
            f"{m.get('electricity_consumption', 0):7.2f} | "
            f"{m.get('ramping_rate', 0):7.3f} | "
            f"{m.get('discomfort_rate', 0):8.4f} | "
            f"{m.get('safety_violation_rate', 0):8.4f}"
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablation_once(args: argparse.Namespace, seed: int) -> Tuple[Dict[str, Dict[str, float]], STEMSEnvironment]:
    """Run one full ablation pass for a single seed."""
    set_seed(seed)
    config = STEMSConfig()

    print(f"[ablation] Building environment ...")
    env = STEMSEnvironment(schema=args.schema, seed=seed)
    if args.strict_paper_mode:
        validate_strict_paper_mode(env, context=f"ablation seed {seed}")
        print("[ablation] Strict paper mode enabled")
    print(f"[ablation] Buildings: {env.num_buildings}, obs_dim: {env.obs_dim}")

    print(f"[ablation] Building 5 variants ...")
    variants = build_variants(env, config)

    results: Dict[str, Dict[str, float]] = {}
    for name, agent in variants.items():
        variant_env = STEMSEnvironment(schema=args.schema, seed=seed)
        if args.strict_paper_mode:
            validate_strict_paper_mode(variant_env, context=f"ablation seed {seed} variant {name}")

        if (
            name == "Full STEMS"
            and args.load_full_checkpoint
            and not args.fair
            and os.path.isdir(args.checkpoint)
            and os.path.exists(os.path.join(args.checkpoint, "encoder.pt"))
        ):
            # Use pre-trained converged checkpoint — no further training needed
            agent.load(args.checkpoint)
            print(f"[ablation] '{name}': loaded pre-trained checkpoint from {args.checkpoint}")
            metrics = _eval_only(agent, variant_env, config)
        else:
            print(f"[ablation] Training/evaluating '{name}' for {args.episodes} episodes ...")
            metrics = train_and_eval(agent, variant_env, config, args.episodes)

        results[name] = metrics
        print(f"           cost={metrics['cost']:.2f}  "
              f"discomfort={metrics['discomfort_rate']:.4f}  "
              f"safety_viol={metrics['safety_violation_rate']:.4f}")

    return results, env


def main(args: argparse.Namespace) -> None:
    args = configure_protocol_args(args)

    if args.seeds:
        per_seed_results: List[Dict[str, Dict[str, float]]] = []
        manifest_env: Optional[STEMSEnvironment] = None
        for seed in args.seeds:
            print(f"\n[ablation] === Seed {seed} ===")
            results, env = run_ablation_once(args, seed)
            per_seed_results.append(results)
            if manifest_env is None:
                manifest_env = env

        means, stds = _aggregate_variant_results(per_seed_results)
        print_table4_stats(means, stds)

        out_path = os.path.join(args.checkpoint, "ablation_multiseed_results.json")
        os.makedirs(args.checkpoint, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "means": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in means.items()},
                    "stds": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in stds.items()},
                    "per_seed": per_seed_results,
                    "seeds": args.seeds,
                },
                f,
                indent=2,
            )
        print(f"\n[ablation] Fair multi-seed results saved to {out_path}")
        if manifest_env is not None:
            _write_ablation_manifest(args, manifest_env, args.checkpoint, "ablation_multiseed_results.json")
        print("\n[ablation] Ablation study complete.")
        return

    results, env = run_ablation_once(args, args.seed)

    print_table4(results)

    # Save results to JSON for downstream use
    out_path = os.path.join(args.checkpoint, "ablation_results.json")
    os.makedirs(args.checkpoint, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\n[ablation] Results saved to {out_path}")
    _write_ablation_manifest(args, env, args.checkpoint, "ablation_results.json")
    print("\n[ablation] Ablation study complete.")


if __name__ == "__main__":
    main(parse_args())
