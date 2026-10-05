#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    _SEABORN = True
except ImportError:
    _SEABORN = False

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.paper_mode import validate_strict_paper_mode
from stems.utils import HistoryBuffer, set_seed
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate STEMS paper visualizations")
    p.add_argument("--checkpoint", type=str, default="checkpoints/")
    p.add_argument("--output-dir", type=str, default="plots/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-real-data", action="store_true",
                   help="Fail if training/evaluation JSON files are missing instead of using demo placeholders")
    p.add_argument("--paper-reproduction", action="store_true",
                   help="Require real CityLearn and real result files for paper figures")
    return p.parse_args()


def _load_agent(checkpoint: str, env: STEMSEnvironment) -> STEMSAgent:
    config = STEMSConfig()
    info = env.get_building_info()
    graph = BuildingGraph(env.num_buildings, info["positions"], info["features"], config.graph)
    agent = STEMSAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_buildings=env.num_buildings,
        building_graph=graph,
        config=config,
        electrical_storage_action_index=env.electrical_storage_action_index,
    )
    if os.path.isdir(checkpoint) and os.path.exists(os.path.join(checkpoint, "encoder.pt")):
        agent.load(checkpoint)
    return agent


def plot_training_curves(
    history: Dict[str, List[Any]],
    output_dir: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("STEMS Training Curves", fontsize=14, fontweight="bold")

    episodes = history.get("episode", list(range(1, len(history.get("total_reward", [])) + 1)))

    def _smooth(vals: List[float], window: int = 3) -> List[float]:
        if len(vals) < window:
            return vals
        return [float(np.mean(vals[max(0, i - window + 1): i + 1])) for i in range(len(vals))]

    ax = axes[0, 0]
    rewards = history.get("total_reward", [0.0] * len(episodes))
    ax.plot(episodes, _smooth(rewards), color="steelblue", linewidth=2)
    ax.fill_between(episodes, _smooth(rewards), alpha=0.15, color="steelblue")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Episode Reward")
    ax.set_title("Reward (higher is better)")

    ax = axes[0, 1]
    viols = history.get("safety_violations", [0.0] * len(episodes))
    ax.plot(episodes, _smooth(viols), color="crimson", linewidth=2)
    ax.fill_between(episodes, _smooth(viols), alpha=0.15, color="crimson")
    ax.set_xlabel("Episode"); ax.set_ylabel("Violation Rate")
    ax.set_title("Safety Violation Rate (lower is better)")

    ax = axes[1, 0]
    a_loss = history.get("actor_loss", [0.0] * len(episodes))
    ax.plot(episodes, _smooth(a_loss), color="darkorange", linewidth=2)
    ax.fill_between(episodes, _smooth(a_loss), alpha=0.15, color="darkorange")
    ax.set_xlabel("Episode"); ax.set_ylabel("Actor Loss")
    ax.set_title("Actor Loss (convergence indicator)")

    ax = axes[1, 1]
    eval_cost = history.get("eval_cost", [0.0] * len(episodes))
    ax.plot(episodes, _smooth(eval_cost), color="forestgreen", linewidth=2)
    ax.fill_between(episodes, _smooth(eval_cost), alpha=0.15, color="forestgreen")
    ax.set_xlabel("Episode"); ax.set_ylabel("Evaluation Cost ($)")
    ax.set_title("Evaluation Cost (lower is better)")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def plot_radar_chart(
    metrics: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    categories = ["Cost", "Emission", "Daily Peak", "Consumption", "Ramping"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = ["steelblue", "crimson", "darkorange", "forestgreen"]

    for (name, m), color in zip(metrics.items(), colors):
        values = [
            m.get("cost", 1.0),
            m.get("emission", 1.0),
            m.get("avg_daily_peak", 1.0),
            m.get("electricity_consumption", 1.0),
            m.get("ramping_rate", 1.0),
        ]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 2)
    ax.set_title("Normalised Performance Comparison\n(baseline = 1.0, lower is better)",
                 pad=20, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def plot_discomfort_bar(
    metrics: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    agents = list(metrics.keys())
    values = [metrics[a].get("discomfort_rate", 0.0) * 100.0 for a in agents]
    colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, values, color=colors, edgecolor="grey", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("Discomfort Rate (%)")
    ax.set_title("Thermal Discomfort Rate by Agent (Fig 4)\n(lower is better)", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3 + 0.1)

    path = os.path.join(output_dir, "discomfort_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def plot_safety_bar(
    metrics: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    agents = list(metrics.keys())
    values = [metrics[a].get("safety_violation_rate", 0.0) * 100.0 for a in agents]
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(agents)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, values, color=colors, edgecolor="grey", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_ylabel("Safety Violation Rate (%)")
    ax.set_title("Safety Violation Rate by Agent (Fig 5)\n(lower is better; STEMS ≈ 0)",
                 fontweight="bold")
    ax.set_ylim(0, max(max(values), 0.1) * 1.4)

    path = os.path.join(output_dir, "safety_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def plot_adjacency_heatmap(
    agent: STEMSAgent,
    output_dir: str,
) -> None:
    adj = agent.adj.cpu().numpy()
    B = adj.shape[0]
    labels = [f"B{i+1}" for i in range(B)]

    fig, ax = plt.subplots(figsize=(max(5, B), max(4, B - 1)))
    im = ax.imshow(adj, cmap="YlOrRd", vmin=0.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(B)); ax.set_xticklabels(labels)
    ax.set_yticks(range(B)); ax.set_yticklabels(labels)
    ax.set_title("Building Similarity Graph Adjacency Weights (Fig 6)", fontweight="bold")

    for i in range(B):
        for j in range(B):
            ax.text(j, i, f"{adj[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if adj[i, j] < 0.5 else "white")

    path = os.path.join(output_dir, "adjacency_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def plot_attention_weights(
    agent: STEMSAgent,
    obs: List[np.ndarray],
    output_dir: str,
) -> None:
    config = STEMSConfig()
    T = config.transformer.window_size
    B = agent.B

    history = np.stack([obs] * T, axis=1) if len(np.array(obs).shape) == 2 else \
              np.tile(np.array(obs)[None, :], (1, T, 1))
    h_tensor = torch.tensor(history, dtype=torch.float32).to(agent.device)

    transformer = agent.encoder.temporal_transformer
    with torch.no_grad():
        x = transformer.input_proj(h_tensor) + transformer.pos_enc[:, :T, :]
        _, attn_weights = transformer.attn(x, x, x, need_weights=True, average_attn_weights=True)

    avg_attn = attn_weights.mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(avg_attn, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Key timestep (hours ago)")
    ax.set_ylabel("Query timestep")
    ax.set_title(f"Temporal Attention Weights ({T}×{T}) – Averaged over Buildings (Fig 7)",
                 fontweight="bold")

    ticks = list(range(0, T, 6))
    tick_labels = [f"t-{T-1-i}h" for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)

    path = os.path.join(output_dir, "attention_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    env = STEMSEnvironment(seed=args.seed)
    if args.paper_reproduction:
        args.require_real_data = True
        validate_strict_paper_mode(env, context="visualization")
    agent = _load_agent(args.checkpoint, env)

    hist_path = os.path.join(args.checkpoint, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
    else:
        if args.require_real_data:
            raise FileNotFoundError(
                f"Missing training history at {hist_path}; run train.py before paper visualization."
            )
        print("[viz] No training history found – generating synthetic history for demo")
        n = 15
        history = {
            "episode": list(range(1, n + 1)),
            "total_reward": list(np.linspace(-500, -100, n) + np.random.randn(n) * 20),
            "safety_violations": list(np.linspace(0.15, 0.02, n) + np.random.rand(n) * 0.01),
            "actor_loss": list(np.linspace(2.0, 0.3, n) + np.random.rand(n) * 0.1),
            "eval_cost": list(np.linspace(800, 400, n) + np.random.randn(n) * 30),
        }

    eval_path = os.path.join(args.checkpoint, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            sample_metrics = json.load(f)
        print(f"[viz] Loaded evaluation metrics from {eval_path}")
    else:
        if args.require_real_data:
            raise FileNotFoundError(
                f"Missing evaluation metrics at {eval_path}; run evaluate.py before paper visualization."
            )
        print("[viz] No eval_results.json found – run evaluate.py first for real data")
        print("[viz] Using placeholder metrics for demo")
        sample_metrics = {
            "STEMS":      {"cost": 0.82, "emission": 0.79, "avg_daily_peak": 0.85,
                           "electricity_consumption": 0.88, "ramping_rate": 0.80,
                           "discomfort_rate": 0.034, "safety_violation_rate": 0.001},
            "RuleBased":  {"cost": 1.00, "emission": 1.00, "avg_daily_peak": 1.00,
                           "electricity_consumption": 1.00, "ramping_rate": 1.00,
                           "discomfort_rate": 0.089, "safety_violation_rate": 0.052},
        }

    plot_training_curves(history, args.output_dir)
    plot_radar_chart(sample_metrics, args.output_dir)
    plot_discomfort_bar(sample_metrics, args.output_dir)
    plot_safety_bar(sample_metrics, args.output_dir)
    plot_adjacency_heatmap(agent, args.output_dir)

    obs_list, _ = env.reset()
    plot_attention_weights(agent, obs_list, args.output_dir)

    print(f"\n[viz] All figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main(parse_args())
